#!/usr/bin/env python3
"""
PPO BTC Trading Environment - SENIOR DEV CANONICAL FIX
USD Notional Position System - Fixed Cost Calculations
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading environment configuration - SENIOR DEV CANONICAL VERSION"""
    initial_balance: float = 10_000.0
    max_leverage: float = 3.0                   # L_max
    maker_fee: float = 0.0002                   # 2 bps
    taker_fee: float = 0.0004                   # 4 bps
    funding_rate_8h: float = 0.0001             # 1 bp / 8h
    step_hours: float = 1.0                     # env step size
    slippage_bp: float = 1.0                    # 1 bp per turnover
    turnover_cap: float = 6.0                   # |ΔN| ≤ cap * E_pre (allow full leverage changes)
    cost_cap_frac: float = 0.10                 # per-trade sanity cap
    drawdown_stop: float = 0.50                 # training kill switch

class PPOTradingEnvironment(gym.Env):
    """
    FIXED: USD Notional Position System
    
    Position N = signed USD notional (NOT leverage ratio)
    Leverage L = |N|/E
    PnL = N[t-1] * return[t]
    Costs = turnover * fee_rate
    """
    
    # Class-level data cache
    _shared_data = None
    _data_loaded = False
    
    def __init__(self, config: TradingConfig = None, data_path: str = None):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.data = self._get_shared_data(data_path)
        
        # Setup features and observation space
        self._setup_features_and_prices()
        
        # FIXED: Observation space includes L=|N|/E, turnover/E, dd, r
        obs_dim = len(self.feature_columns) + 6  # features + N/E + L + dd + prev_r + turnover_ratio + equity_norm
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info("PPO Trading Environment (FIXED USD NOTIONAL):")
        logger.info(f"  Data shape: {self.data.shape}")
        logger.info(f"  Price column: {self.price_column}")
        logger.info(f"  Feature columns: {len(self.feature_columns)}")
        logger.info("  Action space: Continuous [-1, 1] → target leverage")
        logger.info(f"  Observation space: {self.observation_space.shape}")
        logger.info("  Position system: USD notional N, Leverage L = |N|/E")
    
    def _setup_features_and_prices(self):
        """Setup price column and deterministic feature list"""
        # Find price column
        price_cols = [col for col in self.data.columns if 'close' in col.lower() or 'price' in col.lower()]
        if not price_cols:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.price_column = numeric_cols[0]
        else:
            self.price_column = price_cols[0]
            
        # Store prices array
        self.prices = self.data[self.price_column].to_numpy()
        
        # Define deterministic feature list
        exclude_cols = {self.price_column, 'return_1h'}
        self.feature_columns = [col for col in self.data.columns 
                               if col not in exclude_cols and self.data[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        logger.info(f"✅ Features setup: {len(self.feature_columns)} features, price column: {self.price_column}")
    
    @classmethod
    def _get_shared_data(cls, data_path: str = None) -> pd.DataFrame:
        """Get shared data using singleton pattern"""
        if not cls._data_loaded:
            logger.info("🔄 Loading data (first environment only)...")
            cls._shared_data = cls._load_and_validate_data_static(data_path)
            cls._data_loaded = True
            logger.info(f"✅ Data loaded and cached for all environments: {cls._shared_data.shape}")
        else:
            logger.info("✅ Using cached data (no reload needed)")
        return cls._shared_data.copy()

    @staticmethod
    def _load_and_validate_data_static(data_path: str = None) -> pd.DataFrame:
        """Load and validate data with comprehensive checks"""
        if data_path and data_path.startswith('gs://'):
            # Download from cloud storage
            import subprocess
            local_path = '/tmp/btc_data.csv'
            subprocess.run(['gsutil', 'cp', data_path, local_path], check=True)
            data_path = local_path

        try:
            if data_path:
                data = pd.read_csv(data_path)
                logger.info(f"✅ Loaded raw data: {data.shape}")
            else:
                logger.info("No data path provided, generating synthetic data")
                data = PPOTradingEnvironment._generate_synthetic_data()
            
            logger.info("🔍 Starting data validation...")
            
            # Find price column
            price_cols = [col for col in data.columns if 'close' in col.lower() or 'price' in col.lower()]
            if price_cols:
                price_column = price_cols[0]
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                price_column = numeric_cols[0]
                
            logger.info(f"Using price column: {price_column}")
            
            # Remove NaN prices
            initial_len = len(data)
            data = data.dropna(subset=[price_column])
            removed_nan = initial_len - len(data)
            if removed_nan > 0:
                logger.info(f"Removed {removed_nan} rows with NaN prices")
            
            # Handle zero/negative prices
            if np.any(data[price_column] <= 0):
                logger.warning("Found zero or negative prices - replacing with forward fill")
                data[price_column] = data[price_column].replace([0, np.inf, -np.inf], np.nan)
                data[price_column] = data[price_column].fillna(method='ffill').fillna(1.0)
                data[price_column] = np.maximum(data[price_column], 1e-8)
            
            # Calculate returns for validation
            prices = data[price_column].values
            returns = np.log(prices[1:] / prices[:-1])
            
            # Remove first row to match returns length
            data = data.iloc[1:].copy()
            data.reset_index(drop=True, inplace=True)
            data['return_1h'] = returns
            
            # Validation checks
            return_std = np.std(returns)
            non_zero_features = sum(1 for col in data.columns 
                                  if col != price_column and data[col].dtype in [np.float64, np.float32, np.int64, np.int32] 
                                  and not np.all(data[col] == 0))
            
            logger.info("✅ Data validation complete:")
            logger.info(f"  Final shape: {data.shape}")
            logger.info(f"  Return std: {return_std:.6f}")
            logger.info(f"  Non-zero features: {non_zero_features}")
            
            return data
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            logger.info("Falling back to synthetic data generation")
            return PPOTradingEnvironment._generate_synthetic_data()

    @staticmethod
    def _generate_synthetic_data(n_steps: int = 50000) -> pd.DataFrame:
        """Generate synthetic trading data for testing"""
        logger.info(f"Generating {n_steps} steps of synthetic BTC data...")
        
        np.random.seed(42)  # Reproducible
        
        # Generate realistic price series with volatility clustering
        returns = []
        vol = 0.02  # Initial volatility
        
        for _ in range(n_steps):
            # Volatility clustering
            vol = 0.9 * vol + 0.1 * 0.02 + 0.05 * np.random.randn() * vol
            vol = np.clip(vol, 0.005, 0.10)
            
            # Return with fat tails
            ret = np.random.normal(0, vol) + 0.1 * np.random.standard_t(3) * vol
            returns.append(ret)
            
        returns = np.array(returns)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        # Create feature matrix
        data = pd.DataFrame({
            'price_change_pct': prices,  # Use consistent naming
            'volume': np.random.lognormal(15, 1, n_steps),
            'volatility': pd.Series(returns).rolling(24).std().fillna(0.02),
            'momentum_24h': pd.Series(returns).rolling(24).mean().fillna(0),
            'rsi': np.random.uniform(20, 80, n_steps),
        })
        
        # Add technical features
        for period in [12, 24, 48]:
            data[f'sma_{period}'] = data['price_change_pct'].rolling(period).mean().fillna(data['price_change_pct'])
            data[f'return_{period}h'] = np.log(data['price_change_pct'] / data['price_change_pct'].shift(period)).fillna(0)
            
        logger.info(f"Generated synthetic data: {data.shape}")
        return data
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with USD notional position system"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # FIXED: USD notional position system
        self.current_step = 0
        self.equity = self.config.initial_balance  # E
        self.N = 0.0  # signed USD notional
        self.price = self.prices[0]  # current price
        self.peak_equity = self.equity
        
        # Risk tracking
        self.daily_start_equity = self.equity
        self.steps_since_daily_reset = 0
        
        # Cost tracking & telemetry
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.total_slippage = 0.0
        self.total_turnover = 0.0
        self.last_trading_cost = 0.0
        
        # Previous values for observation
        self.prev_return = 0.0
        self.prev_turnover_ratio = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _target_notional_from_action(self, a: float, E_pre: float) -> float:
        """SENIOR DEV FIX: a ∈ [-1,1] → target leverage in [-Lmax, Lmax]"""
        L_target = float(a) * self.config.max_leverage
        return L_target * E_pre

    def _exec_trade(self, N_target: float, E_pre: float) -> Tuple[float, float]:
        """SENIOR DEV FIX: Execute trade with turnover-based costs"""
        # turnover limit (prevents pathological costs)
        delta_N_req = N_target - self.N
        max_turnover = self.config.turnover_cap * E_pre
        delta_N = float(np.clip(delta_N_req, -max_turnover, max_turnover))
        traded_notional = abs(delta_N)

        # fees + slippage on turnover
        fee = traded_notional * self.config.taker_fee
        slip = traded_notional * (self.config.slippage_bp / 1e4)
        trading_cost = fee + slip

        # sanity cap
        cap = self.config.cost_cap_frac * E_pre
        if trading_cost > cap:
            logger.warning(f"🚨 Trading cost {trading_cost:.2f} capped at {cap:.2f} (10% of equity)")
            trading_cost = cap

        # commit trade
        self.N += delta_N
        
        # Track costs
        self.total_fees += fee
        self.total_slippage += slip
        self.total_turnover += traded_notional
        
        return traded_notional, trading_cost

    def _step_accounting(self, new_price: float) -> Dict:
        """SENIOR DEV FIX: Step accounting with proper temporal ordering"""
        r = (new_price / self.price) - 1.0

        E_pre = self.equity
        # PnL from prior notional (FIXED: use N from previous step)
        pnl = self.N * r

        # funding (linear perp approx): pro-rated by step size
        funding_step = self.config.funding_rate_8h * (self.config.step_hours / 8.0)
        funding = abs(self.N) * funding_step  # Always charge funding (conservative)

        # Update equity
        self.equity = E_pre + pnl - self.last_trading_cost - funding
        self.price = new_price
        self.peak_equity = max(self.peak_equity, self.equity)

        # hard risk: liquidation guard
        L = abs(self.N) / max(self.equity, 1e-12)
        if L > self.config.max_leverage:
            # force reduce to margin-safe notional
            N_safe = np.sign(self.N) * self.config.max_leverage * self.equity
            _, cost = self._exec_trade(N_safe, self.equity)
            self.last_trading_cost += cost
            logger.warning(f"🚨 Forced liquidation: L={L:.2f} > {self.config.max_leverage}, reduced to {abs(N_safe):.0f}")

        # Track funding
        self.total_funding += funding
        
        # Daily loss tracking (24 steps = 24 hours)
        self.steps_since_daily_reset += 1
        if self.steps_since_daily_reset >= 24:
            self.daily_start_equity = self.equity
            self.steps_since_daily_reset = 0

        dd = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 1.0
        return {"r": r, "pnl": pnl, "funding": funding, "drawdown": dd}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """SENIOR DEV FIX: Step with USD notional position system"""
        action_signal = np.clip(float(action[0]), -1.0, 1.0)
        
        # 1) mark-to-market on current step
        new_price = self.prices[min(self.current_step + 1, len(self.prices) - 1)]
        acct = self._step_accounting(new_price)

        # 2) translate action → target notional from *pre-trade* equity
        E_pre = self.equity
        N_target = self._target_notional_from_action(action_signal, E_pre)

        # 3) execute and record costs
        traded, cost = self._exec_trade(N_target, E_pre)
        self.last_trading_cost = cost

        # 4) reward (risk-aware log equity) with turnover penalty
        old_E = E_pre
        new_E = self.equity
        log_ret = np.log(max(new_E, 1e-12) / max(old_E, 1e-12))
        turnover_pen = 0.5e-3 * (traded / max(old_E, 1.0))  # 0.05% per 1x turnover
        reward = log_ret - turnover_pen
        
        # Clip reward to prevent exploding negatives
        reward = np.clip(reward, -0.1, 0.1)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # 5) termination and training tripwires
        dd = acct["drawdown"]
        terminated = (self.equity <= 0.0) or (dd >= self.config.drawdown_stop)
        
        # Store for next observation
        self.prev_return = acct["r"]
        self.prev_turnover_ratio = traded / max(old_E, 1.0)

        observation = self._get_observation() if not (done or terminated) else np.zeros(self.observation_space.shape)
        info = {
            "equity": self.equity, 
            "N": self.N, 
            "cost": cost, 
            "pnl": acct["pnl"],
            "funding": acct["funding"], 
            "turnover": traded, 
            "dd": dd,
            "leverage": abs(self.N) / max(self.equity, 1e-12),
            "cost_ratio": cost / max(old_E, 1e-12),
            "gross_return": acct["pnl"] / max(old_E, 1e-12),
            "net_return": (acct["pnl"] - cost) / max(old_E, 1e-12)
        }
        
        return observation, reward, terminated, done, info

    def _get_observation(self) -> np.ndarray:
        """Get observation with enhanced features"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        # Market features
        features = []
        for col in self.feature_columns:
            val = self.data.iloc[self.current_step][col]
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            features.append(val)
        
        # ENHANCED: Add position and risk features
        L = abs(self.N) / max(self.equity, 1e-12)  # Current leverage
        N_norm = self.N / max(self.config.initial_balance, 1e-12)  # Normalized notional
        dd = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        equity_norm = self.equity / self.config.initial_balance
        
        # Add enhanced features
        features.extend([
            N_norm,  # Normalized notional position
            L,  # Current leverage
            dd,  # Drawdown
            self.prev_return,  # Previous price return
            self.prev_turnover_ratio,  # Previous turnover ratio
            equity_norm  # Normalized equity
        ])
        
        return np.array(features, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get comprehensive info dict"""
        L = abs(self.N) / max(self.equity, 1e-12)
        return {
            "equity": self.equity,
            "N": self.N,
            "leverage": L,
            "drawdown": 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_turnover": self.total_turnover,
            "step": self.current_step,
            "price": self.price
        }