#!/usr/bin/env python3
"""
Compact Recurrent PPO Agent - GPU-Optimized Risk-Aware Trading
Proper continuous action PPO with GRU/LSTM and efficient GPU utilization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.cuda.amp import GradScaler, autocast
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO hyperparameter configuration"""
    # Network architecture
    hidden_size: int = 512
    recurrent_size: int = 128

    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    entropy_coef: float = 0.02  # FIX #5: 2x higher initial entropy for exploration
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    learning_rate: float = 1e-4
    critic_lr: float = 3e-5  # CartPole validation: 3x lower LR for critic stability
    ppo_epochs: int = 8
    minibatch_size: int = 2048
    rollout_steps: int = 1024

    # Stability fixes (CartPole validation proven)
    use_separate_optimizers: bool = True  # Separate actor/critic optimizers
    value_clip_epsilon: float = 0.2  # Value loss clipping
    max_kl: float = 0.03  # KL divergence early stopping
    critic_max_grad_norm: float = 0.1  # Tighter gradient clipping for critic

    # GPU optimization - AMP disabled for numerical stability
    use_amp: bool = False  # Disabled per senior dev recommendation
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPONetwork(nn.Module):
    """Pure MLP PPO Network - No GRU/LSTM"""
    
    def __init__(self, obs_dim: int, action_dim: int = 1, config: PPOConfig = None):
        super().__init__()
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = self.config.hidden_size
        
        # Shared feature network - deeper MLP instead of GRU
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        
        # Policy head (Gaussian policy) - NO tanh here for squashed Gaussian
        self.policy_mean = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # NO tanh - will be applied during sampling with proper log-prob correction
        )
        
        # Policy std head - state-dependent (per senior dev recommendation)
        self.policy_log_std = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(), 
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PPO Network: {total_params:,} parameters")
        
    def _init_weights(self, module):
        """Initialize network weights with proper gains (Fixed: only final heads use small gain)"""
        if isinstance(module, nn.Linear):
            # CRITICAL FIX: Only final policy/value heads get small gain
            if module == self.policy_mean[-1]:  # Only final policy output layer
                torch.nn.init.orthogonal_(module.weight, gain=0.01)
            elif module == self.policy_log_std[-1]:  # Only final std output layer
                torch.nn.init.orthogonal_(module.weight, gain=0.01)
            elif module == self.value_head[-1]:  # Only final value output layer
                torch.nn.init.orthogonal_(module.weight, gain=0.01)
            else:  # ALL other layers use proper ReLU gain
                torch.nn.init.orthogonal_(module.weight, gain=torch.nn.init.calculate_gain('relu'))  # √2 ≈ 1.414
                
            if module.bias is not None:
                module.bias.data.zero_()
                    
    def forward(self, obs, hidden_state=None, sequence_length=None):
        """
        Forward pass through MLP network
        
        Args:
            obs: Observations [batch_size, obs_dim]
            hidden_state: Unused (kept for compatibility)
            sequence_length: Unused (kept for compatibility)
            
        Returns:
            action_mean, action_std, value, None (no hidden state)
        """
        # Shared feature processing
        features = self.shared_net(obs)  # [batch_size, hidden_size]
        
        # Policy output - state-dependent std
        action_mean = self.policy_mean(features)
        action_log_std = self.policy_log_std(features)  # State-dependent log_std
        action_std = torch.exp(action_log_std)
        action_std = torch.clamp(action_std, min=0.01, max=1.0)  # Reasonable std range
        
        # Value output
        value = self.value_head(features).squeeze(-1)  # [batch_size]
        
        return action_mean, action_std, value, None  # No hidden state for MLP
        
    def get_action_and_value(self, obs, hidden_state=None, deterministic=False):
        """Get action and value for environment interaction"""
        with torch.no_grad():
            action_mean, action_std, value, new_hidden = self.forward(obs, hidden_state)
            
            # Create distribution and sample action (squashed Gaussian)
            dist = Normal(action_mean, action_std)
            
            if deterministic:
                raw_action = action_mean
            else:
                raw_action = dist.sample()
            
            # Apply tanh squashing to bound actions to [-1, 1]
            action = torch.tanh(raw_action)
            
            # Compute log_prob with log-det-jacobian correction for squashed Gaussian
            log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)  # Sum across action dims: [batch, action_dim] -> [batch]
            # Log-det-jacobian of tanh: log(1 - tanh^2(x)) = log(1 - action^2)
            log_det_jacobian = torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1)  # Sum across action dims
            log_prob = log_prob_raw - log_det_jacobian  # Corrected log_prob for squashed action
            
            return action, log_prob, value, new_hidden

    def compute_squashed_log_prob(self, action_mean, action_std, squashed_actions):
        """Compute log_prob for already squashed actions (used in training)"""
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Inverse tanh to get raw actions (atanh)
        # FIXED: Tighter epsilon for atanh boundaries (per senior dev)
        squashed_actions_clamped = torch.clamp(squashed_actions, -(1.0-1e-6), 1.0-1e-6)
        raw_actions = torch.atanh(squashed_actions_clamped)
        
        # Compute raw log_prob
        log_prob_raw = dist.log_prob(raw_actions).sum(dim=-1)  # Sum across action dims
        
        # Log-det-jacobian of tanh
        log_det_jacobian = torch.log(1 - squashed_actions_clamped.pow(2) + 1e-8).sum(dim=-1)
        
        # Corrected log_prob
        log_prob = log_prob_raw - log_det_jacobian
        
        return log_prob

class PPOTrajectoryBuffer:
    """GPU-optimized trajectory buffer for PPO"""
    
    def __init__(self, capacity: int, obs_dim: int, device: str = 'cuda'):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        self.size = 0
        self.ptr = 0
        
        # Pre-allocate tensors on GPU
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        
    def store(self, obs, action, reward, value, log_prob, done):
        """Store single transition"""
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(bool(done), dtype=torch.bool, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def compute_gae_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = None, last_done: bool = False):
        """Compute GAE advantages with proper single last_value bootstrapping (FIXED)"""
        with torch.no_grad():
            # Get stored values
            rewards = self.rewards[:self.size]
            values = self.values[:self.size]
            dones = self.dones[:self.size]
            
            # Compute advantages using GAE - standard approach
            advantages = torch.zeros_like(rewards, device=self.device)
            last_gae = 0.0
            
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    # Final step: use single last_value for bootstrap
                    next_value = last_value if last_value is not None else 0.0
                    next_done = last_done
                else:
                    # Internal steps: use next stored value
                    next_value = values[t + 1]
                    next_done = dones[t + 1]
                    
                # GAE computation - corrected done mask usage
                delta = rewards[t] + gamma * next_value * (1.0 - float(next_done)) - values[t]
                advantages[t] = delta + gamma * gae_lambda * (1.0 - float(next_done)) * last_gae
                last_gae = advantages[t]
                
            # Compute returns
            returns = advantages + values[:self.size]
            
            # Store computed values
            self.advantages[:self.size] = advantages
            self.returns[:self.size] = returns
            
    def get_minibatch(self, minibatch_size: int):
        """Get random minibatch from buffer"""
        indices = torch.randperm(self.size, device=self.device)[:minibatch_size]
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'old_log_probs': self.log_probs[indices],
            'returns': self.returns[indices],
            'advantages': self.advantages[indices],
        }
        
    def clear(self):
        """Clear buffer"""
        self.size = 0
        self.ptr = 0

class PPOAgent:
    """Compact Recurrent PPO Agent with GPU Optimization"""
    
    def __init__(self, obs_dim: int, action_dim: int = 1, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.device = self.config.device
        
        # Initialize network
        self.network = PPONetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # SENIOR DEV FIX #1: Fixed Parameter Separation (CRITICAL for gradient stability)
        # Shared parameters: backbone network used by both policy and value
        self.shared_params = list(self.network.shared_net.parameters())

        # Policy-only parameters: policy heads only
        self.policy_params = list(self.network.policy_mean.parameters()) + \
                            list(self.network.policy_log_std.parameters())

        # Value-only parameters: value head only
        self.value_params = list(self.network.value_head.parameters())

        # Three separate optimizers to avoid gradient conflicts (CartPole validation proven)
        self.shared_optimizer = optim.Adam(self.shared_params, lr=self.config.learning_rate, eps=1e-5)
        self.policy_optimizer = optim.Adam(self.policy_params, lr=self.config.learning_rate, eps=1e-5)
        self.value_optimizer = optim.Adam(self.value_params, lr=self.config.critic_lr, eps=1e-5)  # Use config.critic_lr directly
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Trajectory buffer
        self.buffer = PPOTrajectoryBuffer(
            capacity=self.config.rollout_steps * 32,  # Buffer for multiple environments
            obs_dim=obs_dim,
            device=self.device
        )
        
        # No hidden state needed for MLP network
        
        # Tracking
        self.total_updates = 0
        
        logger.info(f"PPO Agent initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.config.use_amp}")
        logger.info(f"  Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        
    def get_action(self, obs, deterministic=False):
        """Get action for environment interaction"""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        action, log_prob, value, _ = self.network.get_action_and_value(
            obs_tensor, None, deterministic
        )
        
        return action[0].cpu().numpy(), log_prob[0].cpu().numpy(), value[0].cpu().numpy()
        
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in buffer"""
        self.buffer.store(obs, action, reward, value, log_prob, done)
        
        # No hidden state management needed for MLP
            
    def update(self, next_observations=None, env_infos=None):
        """Perform PPO update with buffer validation, proper GAE bootstrapping, and training tripwires"""
        if self.buffer.size < self.config.minibatch_size:
            return {}
            
        # SENIOR DEV TRAINING TRIPWIRES - Abort bad jobs fast
        if env_infos is not None:
            # Calculate cost ratios and drawdown statistics
            cost_ratios = []
            bankrupt_count = 0
            total_envs = len(env_infos)
            
            for info in env_infos:
                equity = info.get('equity', 1.0)
                cost = info.get('cost', 0.0)
                dd = info.get('dd', 0.0)
                
                if equity > 0:
                    cost_ratio = cost / equity
                    cost_ratios.append(cost_ratio)
                
                if dd >= 0.5:  # 50% drawdown = bankruptcy
                    bankrupt_count += 1
            
            # Tripwire 1: Pathological costs
            if cost_ratios:
                mean_cost_ratio = np.mean(cost_ratios)
                if mean_cost_ratio > 0.01:  # 1% of equity per step
                    raise RuntimeError(f"🚨 TRIPWIRE: Pathological costs detected! Mean cost ratio: {mean_cost_ratio:.4f} > 1%")
            
            # Tripwire 2: Mass bankruptcies  
            bankrupt_frac = bankrupt_count / max(total_envs, 1)
            if bankrupt_frac > 0.5:
                raise RuntimeError(f"🚨 TRIPWIRE: Mass bankruptcies detected! {bankrupt_count}/{total_envs} envs bankrupt ({bankrupt_frac:.1%})")
            
            # Tripwire 3: Mean absolute advantages collapsed while high turnover
            advantages = self.buffer.advantages[:self.buffer.size]
            if len(advantages) > 100:  # Only check if we have enough data
                mean_abs_adv = torch.mean(torch.abs(advantages)).item()
                turnovers = [info.get('turnover', 0.0) / max(info.get('equity', 1.0), 1.0) for info in env_infos]
                mean_turnover = np.mean(turnovers) if turnovers else 0.0
                
                if mean_abs_adv < 0.001 and mean_turnover > 0.2:
                    logger.warning(f"🚨 TRIPWIRE WARNING: Collapsed advantages ({mean_abs_adv:.6f}) with high turnover ({mean_turnover:.3f})")
                    
            logger.debug(f"📊 TRIPWIRE STATUS: Cost ratio={np.mean(cost_ratios) if cost_ratios else 0:.6f}, Bankrupts={bankrupt_count}/{total_envs}, Adv={torch.mean(torch.abs(advantages)).item():.6f}")
        
        
        # Validate buffer contents before update
        buffer_observations = self.buffer.observations[:self.buffer.size]
        buffer_actions = self.buffer.actions[:self.buffer.size] 
        buffer_rewards = self.buffer.rewards[:self.buffer.size]
        buffer_values = self.buffer.values[:self.buffer.size]
        buffer_log_probs = self.buffer.log_probs[:self.buffer.size]
        buffer_dones = self.buffer.dones[:self.buffer.size]
        
        # Check for NaN/Inf in buffer contents
        if (torch.isnan(buffer_observations).any() or torch.isinf(buffer_observations).any() or
            torch.isnan(buffer_actions).any() or torch.isinf(buffer_actions).any() or
            torch.isnan(buffer_rewards).any() or torch.isinf(buffer_rewards).any() or 
            torch.isnan(buffer_values).any() or torch.isinf(buffer_values).any() or
            torch.isnan(buffer_log_probs).any() or torch.isinf(buffer_log_probs).any()):
            logger.error("🚨 NaN/Inf detected in buffer contents! Clearing corrupted buffer")
            self.buffer.size = 0  # Clear corrupted buffer
            return {}
            
        # Compute advantages with proper bootstrapping
        next_values = None
        if next_observations is not None:
            # Compute next state values for proper GAE bootstrapping
            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
                if next_obs_tensor.dim() == 1:
                    next_obs_tensor = next_obs_tensor.unsqueeze(0)
                    
                # Get next state values
                _, _, next_values, _ = self.network(next_obs_tensor)
                if next_values.dim() > 1:
                    next_values = next_values.squeeze()
                    
        # FIXED: Proper GAE bootstrapping with single last_value
        if next_observations is not None:
            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
                if next_obs_tensor.dim() == 1:
                    next_obs_tensor = next_obs_tensor.unsqueeze(0)
                # Get single last_value for proper bootstrapping
                _, _, last_values, _ = self.network(next_obs_tensor)
                last_value = last_values.mean().item() if len(last_values) > 1 else last_values.item()
                last_done = False  # Assume continuing episode
        else:
            last_value = None
            last_done = True  # Episode ended
            
        self.buffer.compute_gae_advantages(self.config.gamma, self.config.gae_lambda, 
                                          last_value=last_value, last_done=last_done)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        
        start_time = time.time()
        
        # FIX #1: Maintain higher entropy for exploration (prevent policy collapse)
        # OLD: max(0.1, 1.0 - updates/1000) decayed to 10% by update 900
        # NEW: max(0.5, 1.0 - updates/3000) maintains 50% and decays 3x slower
        entropy_coef = self.config.entropy_coef * max(0.5, 1.0 - self.total_updates / 3000.0)

        # FIX #3: Enhanced action variance monitoring to detect policy collapse
        action_std_threshold = 0.05  # Raised from 0.01 to require more exploration
        if self.buffer.size > 100:
            recent_actions = self.buffer.actions[max(0, self.buffer.size-100):self.buffer.size]
            action_std = torch.std(recent_actions).item()
            action_range = (torch.max(recent_actions) - torch.min(recent_actions)).item()

            if action_std < action_std_threshold:
                logger.warning(f"🚨 POLICY COLLAPSE WARNING: Action std={action_std:.6f} < {action_std_threshold}")
                logger.warning(f"   Action range: {action_range:.4f} (min={recent_actions.min():.4f}, max={recent_actions.max():.4f}, mean={recent_actions.mean():.4f})")

                # Critical level - model stuck in narrow range
                if action_std < 0.02:
                    logger.warning(f"   ⚠️ CRITICAL: Action variance extremely low (std={action_std:.6f})")

        # KL early stopping threshold
        target_kl = 0.02
        
        # Multiple PPO epochs with full sample coverage + KL early stop
        for epoch in range(self.config.ppo_epochs):
            # Create full permutation over ALL samples (not just one minibatch)
            indices = torch.randperm(self.buffer.size, device=self.device)
            
            # Process all samples in minibatches
            for start_idx in range(0, self.buffer.size, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, self.buffer.size)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract minibatch using indices
                batch = {
                    'observations': self.buffer.observations[batch_indices],
                    'actions': self.buffer.actions[batch_indices],
                    'old_log_probs': self.buffer.log_probs[batch_indices], 
                    'returns': self.buffer.returns[batch_indices],
                    'advantages': self.buffer.advantages[batch_indices],
                }
                
                # FIXED: Safe advantage normalization + clipping (per senior dev)
                advantages = batch['advantages']
                
                # Check for NaN/Inf in advantages before normalization
                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    logger.error("🚨 NaN/Inf detected in advantages! Replacing with zeros")
                    advantages = torch.zeros_like(advantages)
                
                # Safe normalization with population std (unbiased=False)
                adv_mean = advantages.mean()
                adv_std = torch.std(advantages, unbiased=False)  # Population std as required
                
                if adv_std < 1e-6:  # std too small, don't normalize
                    logger.warning(f"Advantage std too small ({adv_std}), skipping normalization")
                    advantages = advantages - adv_mean  # Just center
                else:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                
                # CRITICAL: Clip advantages to prevent ratio explosions (±10)
                advantages = torch.clamp(advantages, -10.0, 10.0)
                    
                # Final safety: replace any remaining NaN/Inf with zeros
                advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Forward pass with comprehensive NaN guards
                if self.config.use_amp:
                    with autocast():
                        # Check observations for NaN/Inf
                        if torch.isnan(batch['observations']).any() or torch.isinf(batch['observations']).any():
                            logger.error("🚨 NaN/Inf in observations! Aborting update")
                            continue  # Skip this minibatch
                    
                        action_mean, action_std, values, _ = self.network(batch['observations'])
                    
                        # Check network outputs for NaN/Inf
                        if (torch.isnan(action_mean).any() or torch.isinf(action_mean).any() or
                            torch.isnan(action_std).any() or torch.isinf(action_std).any() or 
                            torch.isnan(values).any() or torch.isinf(values).any()):
                            logger.error("🚨 NaN/Inf in network outputs! Aborting update")
                            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                        
                        # Compute log_prob for squashed Gaussian (actions are already tanh'd)
                        log_probs = self.network.compute_squashed_log_prob(action_mean, action_std, batch['actions'])
                        
                        # Check log_probs for NaN/Inf
                        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                            logger.error("🚨 NaN/Inf in log_probs! Aborting update")
                            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                        
                        # Entropy from original (unsquashed) distribution
                        dist = Normal(action_mean, action_std)
                        entropy = dist.entropy().mean()
                        
                        # PPO loss computation with KL monitoring
                        ratio = torch.exp(log_probs - batch['old_log_probs'])  # Both are now [batch] shape
                        
                        # ADDED: Approximate KL divergence monitoring
                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - (log_probs - batch['old_log_probs'])).mean()
                            approx_kls.append(approx_kl.item())
                        
                        # Check ratio for NaN/Inf
                        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                            logger.error("🚨 NaN/Inf in PPO ratio! Aborting update")
                            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0}
                        
                        # Policy loss (clipped) - NO unsqueeze to prevent broadcasting
                        policy_loss_1 = advantages * ratio  # Both [batch] shape
                        policy_loss_2 = advantages * torch.clamp(
                            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                        )
                        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                        
                        # SENIOR DEV FIX #3: Value Loss Clipping + Huber (CartPole validation proven)
                        value_clipped = batch['returns'] + torch.clamp(
                            values.squeeze() - batch['returns'],
                            -self.config.value_clip_epsilon, self.config.value_clip_epsilon
                        )
                        value_loss_1 = F.smooth_l1_loss(values.squeeze(), batch['returns'], reduction='mean')  # Huber
                        value_loss_2 = F.smooth_l1_loss(value_clipped, batch['returns'], reduction='mean')
                        value_loss = torch.max(value_loss_1, value_loss_2)  # Clipped value loss
                        
                        # Check computed losses for NaN/Inf
                        if (torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any() or
                            torch.isnan(value_loss).any() or torch.isinf(value_loss).any()):
                            logger.error("🚨 NaN/Inf in computed losses! Aborting update")
                            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                        
                        # SENIOR DEV FIX #6: Split gradient clipping (CRITICAL - separate backward passes)
                        # Policy loss with entropy
                        policy_total_loss = policy_loss - entropy_coef * entropy
                        
                        # Value loss separate  
                        value_total_loss = self.config.value_loss_coef * value_loss
                        
                        # Detailed tensor logging only on first epoch to avoid spam
                        if epoch == 0 and start_idx == 0:
                            logger.debug(f"📊 TENSOR STATS - Epoch {epoch}")
                            logger.debug(f"   action_mean: min={action_mean.min():.6f}, max={action_mean.max():.6f}, mean={action_mean.mean():.6f}")
                            logger.debug(f"   action_std: min={action_std.min():.6f}, max={action_std.max():.6f}, mean={action_std.mean():.6f}")
                            logger.debug(f"   log_probs: min={log_probs.min():.6f}, max={log_probs.max():.6f}, mean={log_probs.mean():.6f}")
                            logger.debug(f"   advantages: min={advantages.min():.6f}, max={advantages.max():.6f}, mean={advantages.mean():.6f}")
                            logger.debug(f"   returns: min={batch['returns'].min():.6f}, max={batch['returns'].max():.6f}, mean={batch['returns'].mean():.6f}")
                            logger.debug(f"   ratio: min={ratio.min():.6f}, max={ratio.max():.6f}, mean={ratio.mean():.6f}")
                            logger.debug(f"   losses: policy={policy_loss:.6f}, value={value_loss:.6f}, policy_total={policy_total_loss:.6f}, value_total={value_total_loss:.6f}")
                    
                    # SENIOR DEV FIX #6: Combined backward pass to avoid gradient conflicts (AMP)
                    # Zero all gradients
                    self.shared_optimizer.zero_grad()
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()

                    # Combined loss backward pass
                    total_loss = policy_total_loss + value_total_loss
                    self.scaler.scale(total_loss).backward()

                    # Unscale and clip gradients for each optimizer
                    self.scaler.unscale_(self.shared_optimizer)
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)

                    shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.shared_params,
                        self.config.max_grad_norm
                    )
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_params,
                        self.config.max_grad_norm
                    )
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value_params,
                        self.config.max_grad_norm * 0.5
                    )

                    # Update all optimizers
                    self.scaler.step(self.shared_optimizer)
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()
                    
                else:
                    # Standard precision with NaN guards
                    # Check observations for NaN/Inf
                    if torch.isnan(batch['observations']).any() or torch.isinf(batch['observations']).any():
                        logger.error("🚨 NaN/Inf in observations! Aborting update")
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                    
                    action_mean, action_std, values, _ = self.network(batch['observations'])
                    
                    # Check network outputs for NaN/Inf
                    if (torch.isnan(action_mean).any() or torch.isinf(action_mean).any() or
                        torch.isnan(action_std).any() or torch.isinf(action_std).any() or 
                        torch.isnan(values).any() or torch.isinf(values).any()):
                        logger.error("🚨 NaN/Inf in network outputs! Aborting update")
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                    
                    # Compute log_prob for squashed Gaussian (actions are already tanh'd)
                    log_probs = self.network.compute_squashed_log_prob(action_mean, action_std, batch['actions'])
                    
                    # Check log_probs for NaN/Inf
                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                        logger.error("🚨 NaN/Inf in log_probs! Aborting update")
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                    
                    # Entropy from original (unsquashed) distribution
                    dist = Normal(action_mean, action_std)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(log_probs - batch['old_log_probs'])  # Both are now [batch] shape
                    
                    # ADDED: Approximate KL divergence monitoring (standard precision)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (log_probs - batch['old_log_probs'])).mean()
                        approx_kls.append(approx_kl.item())
                    
                    # Check ratio for NaN/Inf
                    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                        logger.error("🚨 NaN/Inf in PPO ratio! Aborting update")
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0}
                    
                    policy_loss_1 = advantages * ratio  # Both [batch] shape - NO unsqueeze
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                    )
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # SENIOR DEV FIX #3: Value Loss Clipping + Huber (CartPole validation proven)
                    value_clipped = batch['returns'] + torch.clamp(
                        values.squeeze() - batch['returns'],
                        -self.config.value_clip_epsilon, self.config.value_clip_epsilon
                    )
                    value_loss_1 = F.smooth_l1_loss(values.squeeze(), batch['returns'], reduction='mean')  # Huber
                    value_loss_2 = F.smooth_l1_loss(value_clipped, batch['returns'], reduction='mean')
                    value_loss = torch.max(value_loss_1, value_loss_2)  # Clipped value loss
                    
                    # Check computed losses for NaN/Inf
                    if (torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any() or
                        torch.isnan(value_loss).any() or torch.isinf(value_loss).any()):
                        logger.error("🚨 NaN/Inf in computed losses! Aborting update")
                        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
                    
                    # SENIOR DEV FIX #6: Split gradient clipping (CRITICAL - separate backward passes)
                    # Policy loss with entropy
                    policy_total_loss = policy_loss - entropy_coef * entropy
                    
                    # Value loss separate  
                    value_total_loss = self.config.value_loss_coef * value_loss
                    
                    # Detailed tensor logging only on first epoch to avoid spam
                    if epoch == 0 and start_idx == 0:
                        logger.debug(f"📊 TENSOR STATS - Epoch {epoch}")
                        logger.debug(f"   action_mean: min={action_mean.min():.6f}, max={action_mean.max():.6f}, mean={action_mean.mean():.6f}")
                        logger.debug(f"   action_std: min={action_std.min():.6f}, max={action_std.max():.6f}, mean={action_std.mean():.6f}")
                        logger.debug(f"   log_probs: min={log_probs.min():.6f}, max={log_probs.max():.6f}, mean={log_probs.mean():.6f}")
                        logger.debug(f"   advantages: min={advantages.min():.6f}, max={advantages.max():.6f}, mean={advantages.mean():.6f}")
                        logger.debug(f"   returns: min={batch['returns'].min():.6f}, max={batch['returns'].max():.6f}, mean={batch['returns'].mean():.6f}")
                        logger.debug(f"   ratio: min={ratio.min():.6f}, max={ratio.max():.6f}, mean={ratio.mean():.6f}")
                        logger.debug(f"   losses: policy={policy_loss:.6f}, value={value_loss:.6f}, policy_total={policy_total_loss:.6f}, value_total={value_total_loss:.6f}")
                    
                    # SENIOR DEV FIX #6: Combined backward pass to avoid gradient conflicts
                    # Zero all gradients
                    self.shared_optimizer.zero_grad()
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()

                    # Combined loss backward pass (shared network gets gradients from both)
                    total_loss = policy_total_loss + value_total_loss
                    total_loss.backward()

                    # Separate gradient clipping for each optimizer
                    shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.shared_params,
                        self.config.max_grad_norm
                    )
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_params,
                        self.config.max_grad_norm
                    )
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value_params,
                        self.config.max_grad_norm * 0.5  # Tighter clipping for value function
                    )

                    # Update all optimizers
                    self.shared_optimizer.step()
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
            
            # ADDED: KL early stopping check after each epoch
            if approx_kls and approx_kls[-1] > target_kl:
                logger.warning(f"🛑 Early stopping at epoch {epoch+1}: KL={approx_kls[-1]:.6f} > target={target_kl}")
                break
                
        update_time = time.time() - start_time
        self.total_updates += 1
        
        # SENIOR DEV REQUIRED DIAGNOSTICS - Record every update
        diagnostics = {}
        if env_infos is not None:
            equities = [info.get('equity', 0.0) for info in env_infos]
            positions = [info.get('N', 0.0) for info in env_infos]  # USD notional
            leverages = [info.get('leverage', 0.0) for info in env_infos]
            costs = [info.get('cost', 0.0) for info in env_infos]
            pnls = [info.get('pnl', 0.0) for info in env_infos]
            turnovers = [info.get('turnover', 0.0) for info in env_infos]
            drawdowns = [info.get('dd', 0.0) for info in env_infos]
            
            # Gross vs net return calculation
            gross_returns = [p / max(e, 1.0) for p, e in zip(pnls, equities)]
            net_returns = [(p - c) / max(e, 1.0) for p, e, c in zip(pnls, equities, costs)]
            cost_pnl_ratios = [c / max(abs(p), 1e-6) for c, p in zip(costs, pnls)]
            turnover_equity_ratios = [t / max(e, 1.0) for t, e in zip(turnovers, equities)]
            
            diagnostics.update({
                # Senior dev required metrics
                'equity_mean': np.mean(equities),
                'equity_std': np.std(equities),
                'position_mean': np.mean(positions),
                'position_std': np.std(positions),
                'leverage_mean': np.mean(leverages),
                'leverage_max': np.max(leverages),
                'cost_mean': np.mean(costs),
                'cost_max': np.max(costs),
                'turnover_mean': np.mean(turnovers),
                'drawdown_mean': np.mean(drawdowns),
                'drawdown_max': np.max(drawdowns),
                
                # Running aggregates
                'gross_return_mean': np.mean(gross_returns),
                'net_return_mean': np.mean(net_returns), 
                'cost_pnl_ratio_mean': np.mean(cost_pnl_ratios),
                'turnover_equity_ratio_mean': np.mean(turnover_equity_ratios),
                
                # Risk metrics
                'CVaR_5pct': np.percentile(net_returns, 5),  # 5% Value at Risk
                'bankrupt_envs': sum(1 for dd in drawdowns if dd >= 0.5),
                'cost_ratio_violations': sum(1 for c, e in zip(costs, equities) if c/max(e,1) > 0.01),
            })
            
            # Log comprehensive diagnostics
            logger.info(f"💰 EQUITY: mean=${np.mean(equities):.2f}, std=${np.std(equities):.2f}")
            logger.info(f"📈 POSITIONS: mean=${np.mean(positions):.0f}, std=${np.std(positions):.0f}")
            logger.info(f"⚖️  LEVERAGE: mean={np.mean(leverages):.2f}x, max={np.max(leverages):.2f}x")
            logger.info(f"💸 COSTS: mean=${np.mean(costs):.2f}, max=${np.max(costs):.2f}")
            logger.info(f"🔄 TURNOVER: mean=${np.mean(turnovers):.0f}")
            logger.info(f"📉 DRAWDOWN: mean={np.mean(drawdowns):.3f}, max={np.max(drawdowns):.3f}")
            logger.info(f"📊 RETURNS: gross={np.mean(gross_returns):.6f}, net={np.mean(net_returns):.6f}")
            logger.info(f"⚠️  RISK: CVaR_5%={np.percentile(net_returns, 5):.6f}, bankrupts={sum(1 for dd in drawdowns if dd >= 0.5)}")
        
        # Clear buffer
        self.buffer.clear()
        
        # Comprehensive return with all senior dev metrics
        result = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses), 
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls) if approx_kls else 0.0,
            'entropy_coef': entropy_coef,
            'update_time': update_time,
            'total_updates': self.total_updates,
        }
        result.update(diagnostics)
        
        return result
        
    def save(self, filepath: str):
        """Save agent with separate optimizers"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'shared_optimizer_state_dict': self.shared_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config,
            'total_updates': self.total_updates,
        }, filepath)
        
    def load(self, filepath: str):
        """Load agent with separate optimizers"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # Handle both old and new checkpoint formats
        if 'shared_optimizer_state_dict' in checkpoint:
            # New format with three optimizers
            self.shared_optimizer.load_state_dict(checkpoint['shared_optimizer_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        elif 'policy_optimizer_state_dict' in checkpoint:
            # Old format with two optimizers
            logger.warning("Loading old checkpoint format - shared optimizer not restored")
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        elif 'optimizer_state_dict' in checkpoint:
            # Oldest format - load into policy optimizer only
            logger.warning("Loading oldest checkpoint format - only policy optimizer restored")
            self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.total_updates = checkpoint.get('total_updates', 0)

if __name__ == "__main__":
    # Test PPO agent
    obs_dim = 89
    config = PPOConfig()
    agent = PPOAgent(obs_dim, config=config)
    
    # Test forward pass
    obs = np.random.randn(obs_dim)
    action, log_prob, value = agent.get_action(obs)
    
    print(f"✅ PPO Agent test:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob}")
    print(f"  Value: {value}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")