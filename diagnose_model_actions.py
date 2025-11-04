#!/usr/bin/env python3
"""
Diagnostic: Step-by-step analysis of model actions
Shows exactly what the model is doing during evaluation
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Import from PPO trading system
sys.path.insert(0, str(Path(__file__).parent / 'ppo_trading_ai'))
from data_pipeline import load_prices_or_fail
from trading_env_senior_dev import PPOTradingEnvironmentSeniorDev, TradingConfig
from ppo_agent import PPOAgent, PPOConfig

def diagnose_model(model_path: str, data_path: str, max_steps: int = 50):
    """Run model step-by-step and log all actions/states"""

    print("="*80)
    print("MODEL DIAGNOSTIC - STEP-BY-STEP ANALYSIS")
    print("="*80)

    # Load environment
    config = TradingConfig()
    env = PPOTradingEnvironmentSeniorDev(config=config, data_path=data_path, step_minutes=1440)
    print(f"\nEnvironment: {len(env.prices)} timesteps available")

    # Load model
    obs_dim = env.observation_space.shape[0]
    ppo_config = PPOConfig()
    agent = PPOAgent(obs_dim=obs_dim, config=ppo_config)
    agent.load(model_path)
    agent.device = torch.device('cpu')
    agent.network = agent.network.to('cpu')
    agent.network.eval()
    print(f"Model loaded: {obs_dim} features, {sum(p.numel() for p in agent.network.parameters())} parameters\n")

    # Run diagnostic
    obs, info = env.reset()

    step_data = []

    for step in range(max_steps):
        print(f"\n{'='*80}")
        print(f"STEP {step}")
        print(f"{'='*80}")

        # Get model action
        with torch.no_grad():
            action, log_prob, value = agent.get_action(obs, deterministic=True)

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Log everything
        equity = info['equity']
        position_N = info['N']
        price = env.prices[env.current_step-1]  # Current price (numpy array)
        leverage = position_N / equity if equity > 0 else 0
        drawdown = info.get('drawdown', 0)

        print(f"\nSTATE:")
        print(f"  Equity: ${equity:,.2f}")
        print(f"  Position (N): ${position_N:,.2f}")
        print(f"  Leverage: {leverage:.2f}x")
        print(f"  Price: ${price:,.2f}")
        print(f"  Drawdown: {drawdown:.2%}")

        print(f"\nACTION:")
        print(f"  Raw action: {action[0]:.4f} (range: -1 to +1)")
        print(f"  Target leverage: {action[0] * 3:.2f}x")  # max_leverage = 3
        print(f"  Value estimate: {float(value):.4f}")

        print(f"\nRESULT:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Cost: ${info.get('cost', 0):.2f}")
        print(f"  PnL: ${info.get('pnl', 0):.2f}")
        print(f"  Turnover: ${info.get('turnover', 0):,.2f}")

        step_data.append({
            'step': step,
            'equity': equity,
            'position_N': position_N,
            'leverage': leverage,
            'price': price,
            'action': action[0],
            'reward': reward,
            'value_est': float(value),
            'cost': info.get('cost', 0),
            'pnl': info.get('pnl', 0),
            'drawdown': drawdown,
            'done': done
        })

        obs = next_obs

        if done:
            print(f"\n{'='*80}")
            print(f"EPISODE TERMINATED AT STEP {step}")
            print(f"{'='*80}")
            print(f"Final Equity: ${info['equity']:,.2f}")
            print(f"Total Return: {(info['equity'] - 10000) / 10000:.2%}")
            print(f"Termination reason:")
            if equity <= 0:
                print("  - BANKRUPTCY (equity <= 0)")
            if drawdown >= 0.5:
                print(f"  - EXCESSIVE DRAWDOWN ({drawdown:.2%} >= 50%)")
            break

    # Summary statistics
    df = pd.DataFrame(step_data)
    print(f"\n\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    print(f"\nSteps completed: {len(df)}")
    print(f"Final equity: ${df.iloc[-1]['equity']:,.2f}")
    print(f"Total return: {(df.iloc[-1]['equity'] - 10000) / 10000:.2%}")
    print(f"Max drawdown: {df['drawdown'].max():.2%}")
    print(f"\nAction statistics:")
    print(f"  Mean action: {df['action'].mean():.4f}")
    print(f"  Std action: {df['action'].std():.4f}")
    print(f"  Min action: {df['action'].min():.4f}")
    print(f"  Max action: {df['action'].max():.4f}")
    print(f"\nLeverage statistics:")
    print(f"  Mean leverage: {df['leverage'].abs().mean():.2f}x")
    print(f"  Max leverage: {df['leverage'].abs().max():.2f}x")
    print(f"\nReward statistics:")
    print(f"  Total reward: {df['reward'].sum():.6f}")
    print(f"  Mean reward: {df['reward'].mean():.6f}")
    print(f"  Min reward: {df['reward'].min():.6f}")
    print(f"  Max reward: {df['reward'].max():.6f}")

    # Save detailed log
    df.to_csv('model_diagnostic_log.csv', index=False)
    print(f"\n\nDetailed log saved to: model_diagnostic_log.csv")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose PPO model behavior')
    parser.add_argument('--model', type=str, default='checkpoints_daily_enriched/final_model.pt')
    parser.add_argument('--data', type=str, default='data/4h_processed/btc_daily_REAL_FIXED.csv')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps to run')
    args = parser.parse_args()

    diagnose_model(args.model, args.data, args.max_steps)
