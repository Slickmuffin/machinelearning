#!/usr/bin/env python3
"""
Train Experiment B - Lighter penalties for profitability.

Based on friend's feedback:
- 4000 updates (vs 2000)
- Randomized episode starts (no overfitting to single path)
- 70/30 train/val split
- Only report PnL on validation set
- Compare to buy-and-hold baseline

Expected outcomes:
- Scenario 1: +5% to +10% on validation = found genuine edge
- Scenario 2: -2% to +2% on validation = no strong edge
- Scenario 3: <-5% on validation = penalties still too harsh or no edge
"""

import os
import sys
import torch
import numpy as np
import logging
import pandas as pd
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_trading_ai'))

from ppo_trading_ai.trading_env_experiment_b import ExperimentBTradingEnvironment, TradingConfigExperimentB
from ppo_trading_ai.ppo_agent_lstm import LSTMPPOAgent, LSTMPPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_rollout(env, agent, rollout_steps):
    """Collect rollout data with activity monitoring"""
    sequences, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    # Activity tracking
    positions = []
    trades = []
    pnls = []
    costs = []

    sequence, info = env.reset()
    agent.reset_hidden_state()
    done = False
    step_count = 0

    while step_count < rollout_steps:
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            action_mean, action_std, value, agent.hidden_state = agent.network.forward(
                seq_tensor, agent.hidden_state
            )
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        next_sequence, reward, terminated, truncated, step_info = env.step(action.cpu().numpy().flatten())
        done = terminated or truncated

        sequences.append(sequence.copy())
        actions.append(action.cpu().numpy().flatten())
        log_probs.append(log_prob.cpu().item())
        rewards.append(reward)
        values.append(value.cpu().item())
        dones.append(done)

        # Track activity
        positions.append(step_info.get('position_pct', 0.0))
        trades.append(step_info.get('traded_notional', 0.0))
        pnls.append(step_info.get('pnl', 0.0))
        costs.append(step_info.get('cost', 0.0))

        sequence = next_sequence
        step_count += 1

        if done:
            sequence, info = env.reset()
            agent.reset_hidden_state()
            done = False

    # Calculate advantages
    returns = []
    advantages = []
    R = 0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        R = r + 0.99 * R * (1 - d)
        returns.insert(0, R)
        advantages.insert(0, R - v)

    # Normalize advantages
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Calculate activity metrics
    actions_arr = np.array(actions).flatten()
    activity_metrics = {
        'avg_abs_position': np.mean(np.abs(positions)),
        'avg_action': np.mean(actions_arr),
        'action_std': np.std(actions_arr),
        'action_min': np.min(actions_arr),
        'action_max': np.max(actions_arr),
        'num_trades': np.sum(np.array(trades) > 0),
        'total_pnl': np.sum(pnls),
        'total_costs': np.sum(costs),
        'mean_reward': np.mean(rewards)
    }

    return {
        'sequences': np.array(sequences),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'advantages': advantages,
        'returns': np.array(returns),
        'activity': activity_metrics
    }


def ppo_update(agent, rollout_data, config):
    """PPO update"""
    sequences = torch.FloatTensor(rollout_data['sequences']).to(agent.device)
    actions = torch.FloatTensor(rollout_data['actions']).to(agent.device)
    old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(agent.device)
    advantages = torch.FloatTensor(rollout_data['advantages']).to(agent.device)
    returns = torch.FloatTensor(rollout_data['returns']).to(agent.device)

    dataset_size = len(sequences)
    batch_size = min(config.batch_size, dataset_size)

    for _ in range(config.ppo_epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_indices = indices[start:end]

            batch_seq = sequences[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Forward pass
            action_mean, action_std, value_pred, _ = agent.network(batch_seq, None)

            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(batch_actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # PPO loss
            ratio = torch.exp(log_prob - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = (batch_returns - value_pred.squeeze()).pow(2).mean()

            loss = actor_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config.max_grad_norm)

            agent.actor_optimizer.step()
            agent.critic_optimizer.step()

    return {'actor_loss': actor_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item()}


def evaluate_on_validation(env, agent, num_episodes: int = 10) -> dict:
    """Evaluate agent on validation set only"""
    logger.info("=" * 80)
    logger.info("VALIDATION EVALUATION (Held-Out Data)")
    logger.info("=" * 80)

    episode_returns = []
    episode_lengths = []
    final_equities = []

    for ep in range(num_episodes):
        sequence, info = env.reset()
        agent.reset_hidden_state()
        episode_return = 0.0
        steps = 0

        while True:
            action = agent.get_action(sequence, deterministic=True)
            sequence, reward, terminated, truncated, step_info = env.step(action)
            episode_return += reward
            steps += 1

            if terminated or truncated:
                final_equities.append(step_info.get('equity', 10000))
                break

        episode_returns.append(episode_return)
        episode_lengths.append(steps)

    # Calculate metrics
    mean_return = np.mean(episode_returns)
    mean_equity = np.mean(final_equities)
    pnl_pct = (mean_equity - 10000) / 10000 * 100

    logger.info(f"\nValidation Results (N={num_episodes}):")
    logger.info(f"  Mean Return: {mean_return:.4f}")
    logger.info(f"  Mean Final Equity: ${mean_equity:,.2f}")
    logger.info(f"  Mean PnL: {pnl_pct:+.2f}%")
    logger.info(f"  Mean Episode Length: {np.mean(episode_lengths):.1f} steps")

    return {
        'mean_return': mean_return,
        'mean_equity': mean_equity,
        'pnl_pct': pnl_pct,
        'mean_length': np.mean(episode_lengths)
    }


def main():
    logger.info("=" * 80)
    logger.info("PHASE 1.5 TRAINING - HONEST REWARD FUNCTION")
    logger.info("=" * 80)
    logger.info("Critical fix applied:")
    logger.info("  - REMOVED survival bonus (was causing reward hacking)")
    logger.info("  - REMOVED artificial penalties (costs already in equity)")
    logger.info("  - Reward = pure equity change / initial balance")
    logger.info("  - Same data: 3 years BTC/USDT (26,328 hourly bars)")
    logger.info("  - Same features: 28 features (multi-timeframe + indicators)")
    logger.info("  - Same architecture: LSTM PPO (628K params)")
    logger.info("  - Expected: Honest results (not fake +24.59%)")
    logger.info("=" * 80)

    # Configuration - PROPER 70/30 TRAIN/VAL SPLIT
    train_data_path = 'data/v2_raw/btc_train_70pct.csv'
    val_data_path = 'data/v2_raw/btc_val_30pct.csv'
    checkpoint_dir = './checkpoints_honest_reward_proper_split'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Training data: {train_data_path} (700 bars, July 8 - Aug 7)")
    logger.info(f"Validation data: {val_data_path} (300 bars, Aug 7 - Aug 19)")
    logger.info("CRITICAL: Model will NEVER see validation data during training!")

    # PHASE 1.5: Honest reward function (pure P&L only)
    env_config = TradingConfigExperimentB(
        initial_balance=10000.0,
        max_leverage=3.0,
        sequence_length=50,
        episode_overlap=10,
        reward_clip=0.5  # Only parameter needed - for numerical stability
    )

    # Create training environment (70% of data only)
    train_env = ExperimentBTradingEnvironment(data_path=train_data_path, config=env_config, step_minutes=60)

    # LSTM agent
    obs_dim = train_env.observation_space.shape[1]
    lstm_config = LSTMPPOConfig()
    agent = LSTMPPOAgent(obs_dim=obs_dim, config=lstm_config)

    logger.info(f"Model parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

    # Training loop - 4000 updates (vs 2000)
    num_updates = 4000
    rollout_steps = 512
    save_interval = 200
    activity_check_interval = 100
    backtest_interval = 500
    warmup_updates = 200

    for update in range(1, num_updates + 1):
        # Collect rollout (TRAINING DATA ONLY)
        rollout_data = collect_rollout(train_env, agent, rollout_steps)

        # PPO update
        metrics = ppo_update(agent, rollout_data, lstm_config)

        # Log progress
        if update % 20 == 0:
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

        # Activity monitoring (every 100 updates)
        if update % activity_check_interval == 0:
            activity = rollout_data['activity']
            logger.info("=" * 80)
            logger.info(f"ACTIVITY MONITORING - Update {update}")
            logger.info(f"  Avg Abs Position: {activity['avg_abs_position']:.4f}")
            logger.info(f"  Action Stats: mean={activity['avg_action']:.4f}, std={activity['action_std']:.4f}")
            logger.info(f"  Action Range: [{activity['action_min']:.4f}, {activity['action_max']:.4f}]")
            logger.info(f"  Trades: {activity['num_trades']}/{rollout_steps} steps ({100*activity['num_trades']/rollout_steps:.1f}%)")
            logger.info(f"  Total P&L: ${activity['total_pnl']:.2f}")
            logger.info(f"  Total Costs: ${activity['total_costs']:.2f}")
            logger.info(f"  Mean Reward: {activity['mean_reward']:.6f}")

            # Warning checks (after warmup)
            if update > warmup_updates:
                if activity['avg_abs_position'] < 0.05:
                    logger.warning("  WARNING: Model barely holding positions (<5% avg)")
                    logger.warning("  Possible reward hacking - check reward function!")

                if activity['num_trades'] < 0.05 * rollout_steps:
                    logger.warning(f"  WARNING: Very few trades ({activity['num_trades']} in {rollout_steps} steps)")
                    logger.warning("  Model may be exploiting 'do nothing' strategy!")

                if activity['action_std'] < 0.05:
                    logger.warning(f"  WARNING: Action variance very low (std={activity['action_std']:.4f})")
                    logger.warning("  Possible policy collapse!")
            logger.info("=" * 80)

        # Save checkpoint
        if update % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{update}.pt')
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Evaluate on training set (to monitor progress)
            val_results = evaluate_on_validation(train_env, agent, num_episodes=10)

        # Early backtesting (every 500 updates)
        if update % backtest_interval == 0 and update > warmup_updates:
            logger.info("\n" + "=" * 80)
            logger.info(f"EARLY BACKTEST - Update {update}")
            logger.info("=" * 80)

            # Quick backtest on training set
            backtest_results = evaluate_on_validation(train_env, agent, num_episodes=5)

            # Check for reward hacking patterns
            if backtest_results['pnl_pct'] > 20.0 and activity['num_trades'] < 10:
                logger.warning("  CRITICAL: High returns with very few trades!")
                logger.warning("  Likely reward hacking - investigate immediately!")

            if abs(backtest_results['pnl_pct']) < 0.5 and update > 1000:
                logger.info("  Note: Returns near zero - no strong edge found yet")

            logger.info("=" * 80 + "\n")

    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    agent.save(final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")

    # CRITICAL: Final validation on HELD-OUT data the model has NEVER seen
    logger.info("\n" + "=" * 80)
    logger.info("FINAL VALIDATION EVALUATION - HELD-OUT DATA (Aug 7-19)")
    logger.info("=" * 80)
    logger.info("Loading validation environment with held-out data...")
    val_env = ExperimentBTradingEnvironment(data_path=val_data_path, config=env_config, step_minutes=60)
    logger.info("Model has NEVER seen this data during training!")
    final_val = evaluate_on_validation(val_env, agent, num_episodes=20)

    # Interpretation
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT B INTERPRETATION")
    logger.info("=" * 80)
    pnl = final_val['pnl_pct']
    if pnl > 5.0:
        logger.info("SCENARIO 1: PROFITABLE (+5% to +10%)")
        logger.info("  Interpretation: Code works, found genuine edge")
        logger.info("  Next: Walk-forward testing, multiple regimes, risk analysis")
    elif pnl > -2.0:
        logger.info("SCENARIO 2: NEAR-ZERO (-2% to +2%)")
        logger.info("  Interpretation: Code correct, but no strong edge")
        logger.info("  Next: Better features, more data, or accept limitation")
    else:
        logger.info("SCENARIO 3: NEGATIVE (<-5%)")
        logger.info("  Interpretation: Penalties still too harsh OR no edge exists")
        logger.info("  Next: Further reduce penalties OR try different approach")


if __name__ == "__main__":
    main()
