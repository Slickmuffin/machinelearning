#!/usr/bin/env python3
"""
Verify the compounding bug is fixed by simulating a simple scenario.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_trading_ai'))

from ppo_trading_ai.trading_env_fixed import FixedTradingEnvironment, TradingConfig
import numpy as np
import pandas as pd

print("="*80)
print("VERIFYING FIX: Testing if compounding leverage bug is resolved")
print("="*80)

# Create environment
config = TradingConfig(initial_balance=10000.0, max_leverage=3.0)
env = FixedTradingEnvironment(
    data_path='/tmp/btc_data_1year.csv',
    config=config,
    step_minutes=60
)

# Test scenario: Hold 100% long position for 200 steps
obs, info = env.reset()

# Load prices to calculate expected returns
df = pd.read_csv('/tmp/btc_data_1year.csv')
start_idx = env.episode_start
end_idx = start_idx + 200

start_price = df['close'].iloc[start_idx]
end_price = df['close'].iloc[end_idx]
price_change_pct = (end_price / start_price - 1) * 100

print("\nTest Episode:")
print(f"Start price: ${start_price:,.2f}")
print(f"End price: ${end_price:,.2f}")
print(f"Price change: {price_change_pct:+.2f}%")

# Simulate holding max long position
action = np.array([1.0])  # Max long
equity_trace = []
position_trace = []

for step in range(200):
    obs, reward, terminated, truncated, step_info = env.step(action)
    equity_trace.append(step_info['equity'])
    position_trace.append(step_info['position'])
    
    if terminated or truncated:
        break

final_equity = equity_trace[-1]
pnl_pct = (final_equity - 10000) / 10000 * 100

print("\nResults:")
print(f"Final equity: ${final_equity:,.2f}")
print(f"PnL: {pnl_pct:+.2f}%")

# Expected result with 3x fixed leverage (ignoring costs)
expected_pnl_pct = price_change_pct * 3
print(f"\nExpected PnL (3x leverage): {expected_pnl_pct:+.2f}%")

# Check for compounding
position_changes = np.diff(position_trace)
unique_positions = len(set(np.round(position_trace, 4)))

print("\nPosition stability:")
print(f"Unique positions: {unique_positions} (should be ~1 if not compounding)")
print(f"Position range: {min(position_trace):.4f} to {max(position_trace):.4f} BTC")

# Verdict
if unique_positions > 10:
    print("\nWARNING: Position is changing too much - possible compounding!")
elif abs(pnl_pct - expected_pnl_pct) > 50:
    print(f"\nWARNING: PnL ({pnl_pct:+.2f}%) differs significantly from expected ({expected_pnl_pct:+.2f}%)!")
else:
    print("\nSUCCESS: Fix works! PnL is realistic and position doesn't compound.")
    print(f"Difference from expected: {abs(pnl_pct - expected_pnl_pct):.2f}% (within tolerance)")

