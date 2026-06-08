"""
Live Trend Following Bot (first version).

A simple ADX based trend-following strategy for BTC futures, kept as the initial
reference implementation. In walk-forward testing on the same data and splits it
outperformed the PPO/SAC reinforcement learning agent (see walkforward_comparison.csv
and test_baselines_walkforward.py). On the four splits where the RL agent was also
evaluated it averaged +161.11% versus the agent's -36.75%.

Strategy:
1. Compute ADX (trend strength).
2. If ADX > 20: long 2x when price is above the 20 period SMA, short 2x when below.
3. If ADX < 20: stay flat.

Reads Binance API credentials from bot_config.json (a template is created on first
run). Test on a paper or testnet account first. This is research code, not
financial advice.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trend_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrendFollowingBot:
    """Live trend following bot using ADX"""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        adx_threshold: float = 20,
        leverage: float = 2.0,
        max_position_size: float = 0.9  # 90% of balance
    ):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        self.symbol = symbol
        self.timeframe = timeframe
        self.adx_threshold = adx_threshold
        self.leverage = leverage
        self.max_position_size = max_position_size

        self.current_position = 0.0  # -2.0 to +2.0
        self.last_check_time = None

        logger.info(f"Bot initialized: {symbol} on {timeframe}")
        logger.info(f"Strategy: Trend Following (ADX>{adx_threshold})")
        logger.info(f"Max leverage: {leverage}x")

    def fetch_ohlcv(self, limit: int = 100) -> pd.DataFrame:
        """Fetch recent OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return None

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX"""
        if len(df) < period + 1:
            return 0.0

        data = df.tail(period + 1).copy()

        # True Range
        data['h-l'] = data['high'] - data['low']
        data['h-pc'] = abs(data['high'] - data['close'].shift(1))
        data['l-pc'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)

        # Directional Movement
        data['up_move'] = data['high'] - data['high'].shift(1)
        data['down_move'] = data['low'].shift(1) - data['low']

        data['plus_dm'] = np.where(
            (data['up_move'] > data['down_move']) & (data['up_move'] > 0),
            data['up_move'], 0
        )
        data['minus_dm'] = np.where(
            (data['down_move'] > data['up_move']) & (data['down_move'] > 0),
            data['down_move'], 0
        )

        # ATR and DI
        atr = data['tr'].rolling(period).mean().iloc[-1]
        if atr == 0:
            return 0.0

        plus_di = (data['plus_dm'].rolling(period).mean().iloc[-1] / atr) * 100
        minus_di = (data['minus_dm'].rolling(period).mean().iloc[-1] / atr) * 100

        # DX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100

        return dx

    def calculate_signal(self, df: pd.DataFrame) -> float:
        """Calculate trading signal"""
        # Calculate ADX
        adx = self.calculate_adx(df)

        if adx < self.adx_threshold:
            # No strong trend - stay flat
            return 0.0

        # Strong trend - determine direction
        sma_20 = df['close'].tail(20).mean()
        current_price = df['close'].iloc[-1]

        if current_price > sma_20:
            # Uptrend
            return self.leverage
        else:
            # Downtrend
            return -self.leverage

    def get_current_position(self) -> float:
        """Get current futures position"""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    contracts = float(pos['contracts'])
                    side = pos['side']

                    if contracts > 0:
                        if side == 'long':
                            return abs(contracts)
                        elif side == 'short':
                            return -abs(contracts)

            return 0.0

        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return 0.0

    def execute_trade(self, target_position: float):
        """Execute trade to reach target position"""
        try:
            # Get current position
            current_pos = self.get_current_position()

            # Calculate position change needed
            position_change = target_position - current_pos

            if abs(position_change) < 0.01:
                logger.info(f"No trade needed (current: {current_pos:.2f}, target: {target_position:.2f})")
                return

            # Get account balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            # Calculate order size
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']

            # Position change in USDT
            position_change_usdt = abs(position_change) * usdt_balance * self.max_position_size

            # Convert to contracts
            contracts = position_change_usdt / current_price

            # Determine side
            if position_change > 0:
                side = 'buy'
            else:
                side = 'sell'

            logger.info(f"Executing {side} {contracts:.4f} BTC @ ${current_price:.2f}")

            # Execute market order
            order = self.exchange.create_market_order(
                symbol=self.symbol,
                side=side,
                amount=contracts
            )

            logger.info(f"Order executed: {order['id']}")
            self.current_position = target_position

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def run_once(self):
        """Run one iteration of the bot"""
        try:
            logger.info("="*60)
            logger.info("Running bot iteration...")

            # Fetch data
            df = self.fetch_ohlcv(limit=100)
            if df is None or len(df) < 50:
                logger.error("Insufficient data")
                return

            # Calculate signal
            signal = self.calculate_signal(df)

            # Get current metrics
            adx = self.calculate_adx(df)
            current_price = df['close'].iloc[-1]
            sma_20 = df['close'].tail(20).mean()

            logger.info(f"Price: ${current_price:.2f}")
            logger.info(f"SMA20: ${sma_20:.2f}")
            logger.info(f"ADX: {adx:.2f}")
            logger.info(f"Signal: {signal:+.2f}x")

            # Execute trade if signal changed
            if abs(signal - self.current_position) > 0.1:
                logger.info(f"Position change: {self.current_position:.2f} -> {signal:.2f}")
                self.execute_trade(signal)
            else:
                logger.info(f"Holding current position: {self.current_position:.2f}")

            self.last_check_time = datetime.now()

        except Exception as e:
            logger.error(f"Error in bot iteration: {e}")

    def run_loop(self, check_interval: int = 3600):
        """Run bot in continuous loop"""
        logger.info(f"Starting bot loop (checking every {check_interval}s)")

        while True:
            try:
                self.run_once()

                # Wait until next check
                logger.info(f"Waiting {check_interval}s until next check...")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def load_config(config_file: str = 'bot_config.json') -> dict:
    """Load bot configuration"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_file}")
        logger.info("Creating template config file...")

        template = {
            "api_key": "YOUR_BINANCE_API_KEY",
            "api_secret": "YOUR_BINANCE_API_SECRET",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "adx_threshold": 20,
            "leverage": 2.0,
            "max_position_size": 0.9,
            "check_interval": 3600
        }

        with open(config_file, 'w') as f:
            json.dump(template, f, indent=2)

        logger.info(f"Template created at {config_file}")
        logger.info("Please edit with your API credentials and run again")
        exit(1)


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Create and run bot
    bot = TrendFollowingBot(
        api_key=config['api_key'],
        api_secret=config['api_secret'],
        symbol=config['symbol'],
        timeframe=config['timeframe'],
        adx_threshold=config['adx_threshold'],
        leverage=config['leverage'],
        max_position_size=config['max_position_size']
    )

    # Run bot
    bot.run_loop(check_interval=config['check_interval'])
