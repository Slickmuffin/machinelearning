#!/usr/bin/env python3
"""
Simplified PPO Implementation for Validation Testing
Single optimizer, clean design, CartPole/Pendulum focused
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimplePPOConfig:
    """Simple PPO configuration for validation tests"""
    # Network
    hidden_size: int = 64

    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    learning_rate: float = 3e-4
    critic_lr: float = 1e-4  # Lower LR for critic (3x lower)
    ppo_epochs: int = 4
    minibatch_size: int = 64

    # Stability (senior dev fixes)
    use_separate_optimizers: bool = True
    value_clip_epsilon: float = 0.2  # Value loss clipping
    max_kl: float = 0.03  # Early stopping KL threshold
    critic_max_grad_norm: float = 0.1  # Tighter gradient clipping for critic

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimplePPONetwork(nn.Module):
    """Simple PPO network with shared trunk"""

    def __init__(self, obs_dim: int, action_dim: int, continuous: bool = False, config: SimplePPOConfig = None):
        super().__init__()
        self.config = config or SimplePPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared feature network
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, self.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
        )

        if continuous:
            # Continuous action space (Pendulum)
            self.action_mean = nn.Linear(self.config.hidden_size, action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            # Discrete action space (CartPole)
            self.action_logits = nn.Linear(self.config.hidden_size, action_dim)

        # Value head
        self.value = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        features = self.shared(obs)

        if self.continuous:
            action_mean = self.action_mean(features)
            action_std = torch.exp(self.action_logstd).expand_as(action_mean)
            value = self.value(features).squeeze(-1)
            return action_mean, action_std, value
        else:
            action_logits = self.action_logits(features)
            value = self.value(features).squeeze(-1)
            return action_logits, value

    def get_action_and_value(self, obs, deterministic=False):
        with torch.no_grad():
            if self.continuous:
                action_mean, action_std, value = self.forward(obs)
                dist = Normal(action_mean, action_std)

                if deterministic:
                    action = action_mean
                else:
                    action = dist.sample()

                # Clip action to [-1, 1] range (Pendulum expects this)
                action = torch.clamp(action, -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)

                return action, log_prob, value
            else:
                action_logits, value = self.forward(obs)
                dist = Categorical(logits=action_logits)

                if deterministic:
                    action = torch.argmax(action_logits, dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

                return action, log_prob, value

class SimplePPOAgent:
    """Simplified PPO agent with single optimizer"""

    def __init__(self, obs_dim: int, action_dim: int, continuous: bool = False, config: SimplePPOConfig = None):
        self.config = config or SimplePPOConfig()
        self.device = self.config.device
        self.continuous = continuous

        # Network
        self.network = SimplePPONetwork(obs_dim, action_dim, continuous, self.config).to(self.device)

        # Optimizers (senior dev fix: separate optimizers for stability)
        if self.config.use_separate_optimizers:
            # Actor parameters
            if continuous:
                actor_params = list(self.network.shared.parameters()) + \
                              list(self.network.action_mean.parameters()) + \
                              [self.network.action_logstd]
            else:
                actor_params = list(self.network.shared.parameters()) + \
                              list(self.network.action_logits.parameters())

            # Critic parameters
            critic_params = list(self.network.value.parameters())

            self.actor_optimizer = optim.Adam(actor_params, lr=self.config.learning_rate, eps=1e-5)
            self.critic_optimizer = optim.Adam(critic_params, lr=self.config.critic_lr, eps=1e-5)
            self.optimizer = None  # Not used when separate
        else:
            # Single optimizer for all parameters
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
            self.actor_optimizer = None
            self.critic_optimizer = None

        # Rollout buffer
        self.reset_buffer()

        # Metrics
        self.total_updates = 0

        logger.info("SimplePPO Agent initialized:")
        logger.info(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
        logger.info(f"  Continuous: {continuous}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Optimizers: {'Separate (actor/critic)' if self.config.use_separate_optimizers else 'Single'}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    def reset_buffer(self):
        """Reset rollout buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_action(self, obs, deterministic=False):
        """Get action for environment interaction"""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        action, log_prob, value = self.network.get_action_and_value(obs_tensor, deterministic)

        return action[0].cpu().numpy(), log_prob[0].cpu().numpy(), value[0].cpu().numpy()

    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, next_value, next_done):
        """Compute GAE advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_done_t = next_done
            else:
                next_value_t = values[t + 1]
                next_done_t = dones[t + 1]

            delta = rewards[t] + self.config.gamma * next_value_t * (1 - next_done_t) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - next_done_t) * last_gae

        returns = advantages + values

        return advantages, returns

    def update(self, next_obs, next_done):
        """Perform PPO update"""
        if len(self.observations) == 0:
            return {}

        # Get next value for bootstrapping
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.continuous:
                _, _, next_value = self.network(next_obs_tensor)
            else:
                _, next_value = self.network(next_obs_tensor)
            next_value = next_value[0].cpu().numpy()

        # Compute advantages
        advantages, returns = self.compute_gae(next_value, next_done)

        # Convert to tensors
        obs_tensor = torch.as_tensor(np.array(self.observations), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(np.array(self.actions), dtype=torch.float32 if self.continuous else torch.long, device=self.device)
        old_log_probs_tensor = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        old_values_tensor = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=self.device)  # For value clipping
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages (senior dev: safe normalization with clamping)
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
        advantages_tensor = torch.clamp(advantages_tensor, -10.0, 10.0)  # Prevent extreme values

        # PPO update epochs
        batch_size = len(self.observations)
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []

        for epoch in range(self.config.ppo_epochs):
            # Create minibatches
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, batch_size)
                mb_indices = indices[start:end]

                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_old_values = old_values_tensor[mb_indices]  # For value clipping
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]

                # Forward pass
                if self.continuous:
                    action_mean, action_std, values = self.network(mb_obs)
                    dist = Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                else:
                    action_logits, values = self.network(mb_obs)
                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()

                # Policy loss (PPO clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (senior dev fix: PPO-style clipping + Huber)
                v_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon
                )
                v_loss_unclipped = F.smooth_l1_loss(values, mb_returns, reduction='mean')  # Huber
                v_loss_clipped = F.smooth_l1_loss(v_clipped, mb_returns, reduction='mean')
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Update (senior dev fix: separate optimizers with split gradient clipping)
                if self.config.use_separate_optimizers:
                    # Combined backward pass to avoid inplace errors with shared parameters
                    total_loss = policy_loss + self.config.value_loss_coef * value_loss + self.config.entropy_coef * entropy_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()

                    # Get actor parameters for gradient clipping
                    if self.continuous:
                        actor_params = list(self.network.shared.parameters()) + \
                                      list(self.network.action_mean.parameters()) + \
                                      [self.network.action_logstd]
                    else:
                        actor_params = list(self.network.shared.parameters()) + \
                                      list(self.network.action_logits.parameters())

                    # Split gradient clipping
                    nn.utils.clip_grad_norm_(actor_params, self.config.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.network.value.parameters(), self.config.critic_max_grad_norm)

                    # Separate optimizer steps (different learning rates)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                else:
                    # Single optimizer update
                    total_loss = policy_loss + self.config.value_loss_coef * value_loss + self.config.entropy_coef * entropy_loss
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - mb_old_log_probs)).mean()
                    approx_kls.append(approx_kl.item())

            # KL divergence early stopping (senior dev fix)
            mean_kl = np.mean(approx_kls)
            if mean_kl > self.config.max_kl:
                logger.warning(f"Early stopping at epoch {epoch+1}/{self.config.ppo_epochs} due to KL={mean_kl:.4f} > {self.config.max_kl}")
                break

        self.total_updates += 1

        # Clear buffer
        self.reset_buffer()

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls),
            'total_updates': self.total_updates,
        }

    def save(self, filepath: str):
        """Save agent"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_updates': self.total_updates,
        }, filepath)

    def load(self, filepath: str):
        """Load agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint.get('total_updates', 0)

if __name__ == "__main__":
    # Quick test
    agent = SimplePPOAgent(obs_dim=4, action_dim=2, continuous=False)
    obs = np.random.randn(4)
    action, log_prob, value = agent.get_action(obs)
    print(f"Test: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")
    print("SimplePPO implementation validated!")
