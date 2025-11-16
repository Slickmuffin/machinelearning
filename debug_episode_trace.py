#!/usr/bin/env python3
"""
Debug script to trace exactly what's happening in Solution 2 evaluation.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_trading_ai'))

from ppo_trading_ai.trading_env_sequential import SequentialTradingEnvironment, SequentialConfig
from ppo_trading_ai.ppo_agent_lstm import LSTMPPOAgent, LSTMPPOConfig


def trace_episode():
    """Trace a single episode step-by-step"""

    # Setup
    data_path = '/tmp/btc_data_1year.csv'
    env_config = SequentialConfig(sequence_length=50, episode_overlap=0)
    env = SequentialTradingEnvironment(data_path=data_path, config=env_config, step_minutes=60)

    # Load model
    obs_dim = env.observation_space.shape[1]
    lstm_config = LSTMPPOConfig()
    agent = LSTMPPOAgent(obs_dim=obs_dim, config=lstm_config)
    agent.load('./checkpoints_lstm_temporal/final_model.pt')
    agent.network.eval()

    # Run episode 5 (which reported +1104% PnL)
    episode_start = 1685
    env.episode_start = episode_start
    obs, info = env.reset()

    df = pd.read_csv(data_path)

    print(f"\n{'='*80}")
    print(f"EPISODE TRACE: Start={episode_start}")
    print(f"Initial BTC Price: ${df['close'].iloc[episode_start]:,.2f}")
    print(f"{'='*80}\n")

    episode_return = 0.0
    steps = 0
    trades = 0
    prev_position = 0.0

    equity_trace = []
    price_trace = []
    position_trace = []

    while True:
        # Get action
        action = agent.get_action(obs, deterministic=True)

        # Count trades
        if abs(action[0] - prev_position) > 0.1:
            trades += 1
            print(f"\nStep {steps}: TRADE #{trades}")
            print(f"  Action: {action[0]:+.4f}")
            print(f"  Prev Position: {prev_position:+.4f}")
        prev_position = action[0]

        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        episode_return += reward
        steps += 1

        # Track equity and price
        current_price_idx = episode_start + steps
        if current_price_idx < len(df):
            price = df['close'].iloc[current_price_idx]
        else:
            price = df['close'].iloc[-1]

        equity_trace.append(step_info.get('equity', 10000))
        price_trace.append(price)
        position_trace.append(step_info.get('position', 0.0))

        # Print every 50 steps
        if steps % 50 == 0 or terminated or truncated:
            print(f"\nStep {steps}:")
            print(f"  Price: ${price:,.2f}")
            print(f"  Equity: ${step_info.get('equity', 10000):,.2f}")
            print(f"  Position: {step_info.get('position', 0.0):+.4f}")
            print(f"  Reward: {reward:+.6f}")

        if terminated or truncated:
            final_equity = step_info.get('equity', 10000)
            break

    # Summary
    pnl_pct = (final_equity - 10000) / 10000 * 100
    end_price = df['close'].iloc[episode_start + steps]
    price_change_pct = (end_price / df['close'].iloc[episode_start] - 1) * 100

    print(f"\n{'='*80}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Steps: {steps}")
    print(f"Trades: {trades}")
    print(f"Start Price: ${df['close'].iloc[episode_start]:,.2f}")
    print(f"End Price: ${end_price:,.2f}")
    print(f"Price Change: {price_change_pct:+.2f}%")
    print(f"\nInitial Equity: $10,000")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"PnL: {pnl_pct:+.2f}%")
    print(f"\nReturn (log-equity): {episode_return:+.6f}")

    # Check for bugs
    if abs(price_change_pct) < 1.0 and abs(pnl_pct) > 100:
        print(f"\n{'='*80}")
        print(f"BUG DETECTED!")
        print(f"{'='*80}")
        print(f"Price barely moved ({price_change_pct:+.2f}%) but PnL is {pnl_pct:+.2f}%")
        print(f"This is mathematically impossible with realistic leverage.")

        # Analyze equity trace
        equity_array = np.array(equity_trace)
        print(f"\nEquity Statistics:")
        print(f"  Min: ${equity_array.min():,.2f}")
        print(f"  Max: ${equity_array.max():,.2f}")
        print(f"  Mean: ${equity_array.mean():,.2f}")
        print(f"  Std: ${equity_array.std():,.2f}")

        # Check for unrealistic jumps
        equity_changes = np.diff(equity_array)
        large_jumps = np.where(np.abs(equity_changes) > 1000)[0]
        if len(large_jumps) > 0:
            print(f"\nLarge equity jumps detected: {len(large_jumps)}")
            for idx in large_jumps[:5]:  # Show first 5
                print(f"  Step {idx}: ${equity_array[idx]:,.2f} → ${equity_array[idx+1]:,.2f} (change: ${equity_changes[idx]:+,.2f})")


if __name__ == "__main__":
    trace_episode()
