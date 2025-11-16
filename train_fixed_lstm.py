#!/usr/bin/env python3
"""
Train LSTM model with FIXED environment (no compounding leverage bug).
"""

import os
import sys
import torch
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_trading_ai'))

from ppo_trading_ai.trading_env_fixed import FixedTradingEnvironment, TradingConfig
from ppo_trading_ai.ppo_agent_lstm import LSTMPPOAgent, LSTMPPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_rollout(env, agent, rollout_steps):
    """Collect rollout data"""
    sequences, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

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

    return {
        'sequences': np.array(sequences),
        'actions': np.array(actions),
        'log_probs': np.array(log_probs),
        'advantages': advantages,
        'returns': np.array(returns)
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


def main():
    logger.info("="*80)
    logger.info("TRAINING WITH FIXED ENVIRONMENT (NO COMPOUNDING BUG)")
    logger.info("="*80)

    # Configuration
    data_path = '/tmp/btc_data_1year.csv'
    checkpoint_dir = './checkpoints_fixed_lstm'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fixed environment
    env_config = TradingConfig(
        initial_balance=10000.0,
        max_leverage=3.0,
        sequence_length=50,
        episode_overlap=10
    )
    env = FixedTradingEnvironment(data_path=data_path, config=env_config, step_minutes=60)

    # LSTM agent
    obs_dim = env.observation_space.shape[1]
    lstm_config = LSTMPPOConfig()
    agent = LSTMPPOAgent(obs_dim=obs_dim, config=lstm_config)

    logger.info(f"Model parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

    # Training loop
    num_updates = 2000
    rollout_steps = 512
    save_interval = 100

    for update in range(1, num_updates + 1):
        # Collect rollout
        rollout_data = collect_rollout(env, agent, rollout_steps)

        # PPO update
        metrics = ppo_update(agent, rollout_data, lstm_config)

        # Log progress
        if update % 10 == 0:
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Actor Loss: {metrics['actor_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

        # Save checkpoint
        if update % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{update}.pt')
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    agent.save(final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")


if __name__ == "__main__":
    main()
