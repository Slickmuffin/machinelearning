"""
Test Simple Baselines with PROPER Walk-Forward Validation

Compare apples to apples:
- Same data as the RL agent (Dec 2023 to Dec 2025)
- Same 7 walk-forward splits
- Same validation methodology

Both approaches are measured the same way, with no cherry-picked periods.
"""

import pandas as pd
import numpy as np
from typing import Dict


class SimpleStrategy:
    """Base class for simple strategies"""

    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0
        self.equity_history = []

    def execute_trade(self, new_position: float):
        """Execute trade with costs"""
        position_change = abs(new_position - self.position)

        if position_change < 0.01:
            return

        # Costs: 3bp fee + 0.5bp slippage
        notional = position_change * self.balance
        costs = notional * 0.00035
        self.balance -= costs

        self.position = new_position

    def update_equity(self, price_change_pct: float):
        """Update equity from price movement"""
        pnl = self.position * self.balance * (price_change_pct / 100)
        self.balance += pnl
        self.equity_history.append(self.balance)

    def get_return(self) -> float:
        """Calculate total return"""
        return (self.balance - self.initial_balance) / self.initial_balance * 100


class TrendFollowing(SimpleStrategy):
    """Trend following using ADX"""

    def calculate_adx(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate ADX"""
        if idx < period + 1:
            return 0.0

        data = df.iloc[max(0, idx-period-1):idx+1].copy()

        data['h-l'] = data['high'] - data['low']
        data['h-pc'] = abs(data['high'] - data['close'].shift(1))
        data['l-pc'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)

        data['up_move'] = data['high'] - data['high'].shift(1)
        data['down_move'] = data['low'].shift(1) - data['low']

        data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
        data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)

        atr = data['tr'].rolling(period).mean().iloc[-1]
        if atr == 0:
            return 0.0

        plus_di = (data['plus_dm'].rolling(period).mean().iloc[-1] / atr) * 100
        minus_di = (data['minus_dm'].rolling(period).mean().iloc[-1] / atr) * 100

        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8) * 100

        return dx

    def get_signal(self, df: pd.DataFrame, idx: int, threshold: float = 25) -> float:
        """Get trading signal"""
        adx = self.calculate_adx(df, idx)

        if adx < threshold:
            return 0.0  # No trend, stay flat

        # Determine direction
        if idx < 20:
            return 0.0

        sma_20 = df['close'].iloc[max(0, idx-20):idx+1].mean()
        current_price = df['close'].iloc[idx]

        if current_price > sma_20:
            return 2.0  # Uptrend, go long
        else:
            return -2.0  # Downtrend, go short


class MACrossover(SimpleStrategy):
    """MA Crossover strategy"""

    def get_signal(self, df: pd.DataFrame, idx: int, fast: int = 20, slow: int = 50) -> float:
        """Get trading signal"""
        if idx < slow:
            return 0.0

        prices = df['close'].iloc[max(0, idx-slow):idx+1]
        fast_ma = prices.tail(fast).mean()
        slow_ma = prices.tail(slow).mean()

        if fast_ma > slow_ma:
            return 1.5  # Long
        else:
            return -1.5  # Short


def test_strategy_on_split(strategy_class, df: pd.DataFrame, test_start: str, test_end: str, name: str, **kwargs) -> Dict:
    """Test strategy on one walk-forward split"""

    # Filter to test period
    df_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy().reset_index(drop=True)

    if len(df_test) < 50:
        return {'strategy': name, 'return': 0.0, 'error': 'Insufficient data'}

    strategy = strategy_class()

    for i in range(len(df_test)):
        # Generate signal (using all data up to this point for indicators)
        signal = strategy.get_signal(df_test, i, **kwargs)

        # Execute trade
        strategy.execute_trade(signal)

        # Update equity
        if i > 0:
            price_change = (df_test['close'].iloc[i] - df_test['close'].iloc[i-1]) / df_test['close'].iloc[i-1] * 100
            strategy.update_equity(price_change)

    return {
        'strategy': name,
        'return': strategy.get_return()
    }


def run_walkforward_comparison():
    """Run walk-forward comparison: ML vs Simple Baselines"""

    # Load data
    df = pd.read_csv('data/sac_training/btc_1h_2years.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Load splits
    splits = pd.read_csv('data/sac_training/walk_forward_splits.csv')

    print("="*80)
    print("WALK-FORWARD VALIDATION: ML vs SIMPLE BASELINES")
    print("="*80)
    print(f"Data: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Splits: {len(splits)}")

    # ML results (from failed training)
    ml_results = [-47.96, -29.37, -33.98, -35.68, None, None, None]

    # Test strategies
    strategies = [
        (TrendFollowing, "Trend Following (ADX>25)", {'threshold': 25}),
        (TrendFollowing, "Trend Following (ADX>20)", {'threshold': 20}),
        (MACrossover, "MA Crossover (20/50)", {'fast': 20, 'slow': 50}),
    ]

    all_results = []

    for idx, row in splits.iterrows():
        split_num = idx + 1
        print(f"\n{'='*80}")
        print(f"SPLIT {split_num}/{len(splits)}: {row['test_start']} to {row['test_end']}")
        print(f"{'='*80}")

        split_results = {'split': split_num}

        # ML result
        ml_return = ml_results[idx]
        if ml_return is not None:
            print(f"ML (SAC):                       {ml_return:+7.2f}%")
            split_results['ML'] = ml_return
        else:
            print("ML (SAC):                       Not tested")
            split_results['ML'] = None

        # Test each baseline
        for strategy_class, name, kwargs in strategies:
            result = test_strategy_on_split(
                strategy_class, df,
                row['test_start'], row['test_end'],
                name, **kwargs
            )

            if 'error' not in result:
                print(f"{name:30s} {result['return']:+7.2f}%")
                split_results[name] = result['return']
            else:
                print(f"{name:30s} ERROR")
                split_results[name] = None

        all_results.append(split_results)

    # Calculate averages
    print(f"\n{'='*80}")
    print("AVERAGE RESULTS (4 splits where ML was tested)")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)

    for col in results_df.columns:
        if col == 'split':
            continue

        # Only first 4 splits (where ML was tested)
        values = results_df[col].iloc[:4].dropna()

        if len(values) > 0:
            avg = values.mean()
            print(f"{col:30s} {avg:+7.2f}%")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    # Get averages
    ml_avg = results_df['ML'].iloc[:4].mean()

    best_strategy = None
    best_avg = ml_avg

    for col in results_df.columns:
        if col in ['split', 'ML']:
            continue
        values = results_df[col].iloc[:4].dropna()
        if len(values) > 0:
            avg = values.mean()
            if avg > best_avg:
                best_avg = avg
                best_strategy = col

    if best_strategy:
        print(f"WINNER: {best_strategy}")
        print(f"Return: {best_avg:+.2f}% vs ML {ml_avg:+.2f}%")
        print(f"\nSimple rules beat ML by {best_avg - ml_avg:.2f}%")
    else:
        print("ML was best (or everything failed)")
        print(f"ML: {ml_avg:+.2f}%")

    # Save results
    results_df.to_csv('walkforward_comparison.csv', index=False)
    print("\nResults saved to walkforward_comparison.csv")


if __name__ == "__main__":
    run_walkforward_comparison()
