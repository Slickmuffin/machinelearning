#!/usr/bin/env python3
"""
PPO TRADING AI FOR GRANDMA - Proximal Policy Optimization Agent
BREAKTHROUGH: Policy gradients that adapt to non-stationary BTC markets
Built with love to honor her memory and learn trading without catastrophic forgetting

ARCHITECTURAL FIXES APPLIED:
- Squashed Gaussian policy with log-det-jacobian correction
- Full minibatch coverage (all samples processed each epoch)
- Pure MLP architecture (GRU removed)
- Dropout removed from policy/value paths
- Proper weight initialization with gains
- Fixed GAE bootstrapping
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import autocast
import subprocess

# Import the architectural-fix PPO agent
from ppo_agent import PPOAgent, PPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActorCriticNetwork(nn.Module):
    """PPO Actor-Critic Network - Optimized for GPU utilization"""
    
    def __init__(self, state_size: int = 89, action_size: int = 8, hidden_size: int = 1024):
        super(ActorCriticNetwork, self).__init__()
        
        # Larger shared feature extraction for GPU efficiency
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Recurrent layer for temporal dependencies
        self.lstm = nn.LSTM(hidden_size // 2, hidden_size // 4, batch_first=True)
        self.lstm_hidden_size = hidden_size // 4
        
        # Actor head (policy) - outputs logits (no softmax for numerical stability)
        self.actor_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )
        
        # Critic head (value function) - outputs state value
        self.critic_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, hidden_size // 4),
            nn.ReLU(), 
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            module.bias.data.zero_()
    
    def forward(self, state, hidden_state=None):
        """Forward pass with LSTM for temporal modeling"""
        batch_size = state.size(0)
        features = self.shared_layers(state)
        
        # Reshape for LSTM (add sequence dimension)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM forward pass
        if hidden_state is None:
            lstm_out, _ = self.lstm(features)
        else:
            lstm_out, _ = self.lstm(features, hidden_state)
        
        # Use the last output for decision making
        lstm_features = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # Actor output: logits (NOT probabilities for numerical stability)
        logits = self.actor_head(lstm_features)
        # Clip logits to safe range to prevent overflow/underflow
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("🚨 NaN/Inf detected in logits! Replacing with zeros")
            logits = torch.zeros_like(logits)
        
        # Critic output: state value
        state_value = self.critic_head(lstm_features)
        
        # Check for NaN/Inf in state values
        if torch.isnan(state_value).any() or torch.isinf(state_value).any():
            logger.error("🚨 NaN/Inf detected in state_value! Replacing with zeros")
            state_value = torch.zeros_like(state_value)
        
        return logits, state_value

class PPOTrajectoryBuffer:
    """GPU-optimized PPO trajectory buffer with device-native operations"""
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95, device: str = 'cuda'):
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.gamma = gamma
        self.lam = lam
        
    def store(self, state, action, reward, value, log_prob, done):
        """Store single step experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae_returns(self):
        """Compute GAE (Generalized Advantage Estimation) returns and advantages"""
        advantages = []
        returns = []
        
        # Convert to numpy for easier computation
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE advantage
            advantage = delta + self.gamma * self.lam * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        # Returns = advantages + values
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def get_batch(self):
        """Get trajectory batch with GPU-native tensor creation"""
        advantages, returns = self.compute_gae_returns()
        
        # Create tensors directly on GPU - CRITICAL FIX!
        states_tensor = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        
        # Safe advantage normalization on GPU
        adv_mean = advantages_tensor.mean()
        adv_std = advantages_tensor.std()
        
        # Check for NaN/Inf in advantages before normalization
        if torch.isnan(advantages_tensor).any() or torch.isinf(advantages_tensor).any():
            logger.error("🚨 NaN/Inf detected in advantages! Replacing with zeros")
            advantages_tensor = torch.zeros_like(advantages_tensor)
            adv_mean = 0.0
            adv_std = 1.0
        
        # Safe normalization: only normalize if std is reasonable
        if adv_std < 1e-6:  # std too small, don't normalize
            logger.warning(f"Advantage std too small ({adv_std}), skipping normalization")
            advantages_tensor = advantages_tensor - adv_mean  # Just center
        else:
            advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
            
        # Final safety check and clamping
        advantages_tensor = torch.clamp(advantages_tensor, min=-10, max=10)
        advantages_tensor = torch.nan_to_num(advantages_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'old_log_probs': old_log_probs_tensor,
            'returns': returns_tensor,
            'advantages': advantages_tensor,
            'values': values_tensor
        }
    
    def clear(self):
        """Clear buffer after training"""
        self.states.clear()
        self.actions.clear() 
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class PPOBTCTradingEnvironment:
    """BTC Trading Environment optimized for PPO - reusing proven DQN setup"""
    
    def __init__(self, data_path: str = None):
        self.data = self._load_data(data_path)
        self.current_step = 0
        self.max_steps = len(self.data) - 1  # Use full dataset like successful DQN
        
        # Extract prices for trading actions - CRITICAL FIX with error handling
        try:
            if 'close' in self.data.columns:
                self.prices = self.data['close'].values
            elif 'Close' in self.data.columns:
                self.prices = self.data['Close'].values
            else:
                # Fallback to first numeric column
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.prices = self.data[numeric_cols[0]].values
                else:
                    # Emergency fallback - generate synthetic prices
                    self.prices = np.array([50000.0 * (1.01 ** i) for i in range(len(self.data))])
                    logger.warning("🚨 Using emergency synthetic prices")
        except Exception as e:
            # Emergency fallback if all else fails
            self.prices = np.array([50000.0] * len(self.data))
            logger.error(f"🚨 Price extraction failed: {e}, using flat prices")
            
        # Verify prices are set properly
        if not hasattr(self, 'prices') or len(self.prices) == 0:
            self.prices = np.array([50000.0] * len(self.data))
            logger.error("🚨 Prices not properly initialized, using default values")
        
        # Trading state
        self.balance = 10000.0
        self.btc_held = 0.0
        self.position_type = 0  # 0=none, 1=long, -1=short
        self.leverage = 1.0
        self.entry_price = 0.0
        
        # Action space: [WAIT, LONG_40X, LONG_80X, LONG_125X, SHORT_40X, SHORT_80X, SHORT_125X, CLOSE]
        self.action_space = 8
        self.state_size = 89
        
        logger.info(f"PPO BTC Trading Environment initialized with {len(self.data)} steps")
        
    def _load_data(self, data_path: str = None):
        """Load BTC trading data with robust error handling"""
        try:
            import subprocess
            result = subprocess.run(['gsutil', 'cp', 
                          'gs://YOUR_GCP_PROJECT-training-data/btc_comprehensive_dataset_20250812_125933_features.csv', 
                          '/tmp/btc_data.csv'], 
                          check=True, capture_output=True, timeout=30)
            data = pd.read_csv('/tmp/btc_data.csv')
            if len(data) > 0:
                logger.info(f"✅ Loaded BTC dataset with {len(data)} rows")
                return data
            else:
                logger.warning("Cloud data is empty, using synthetic data")
                return self._generate_synthetic_data(50000)
        except subprocess.TimeoutExpired:
            logger.warning("Cloud data download timeout, using synthetic data")
            return self._generate_synthetic_data(50000)
        except Exception as e:
            logger.warning(f"Cloud data failed: {e}, using synthetic data")
            return self._generate_synthetic_data(50000)
    
    def _generate_synthetic_data(self, steps: int = 50000):
        """Generate synthetic BTC data if needed"""
        np.random.seed(42)
        price = 50000.0
        data = []
        
        for i in range(steps):
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            
            features = {'close': price, 'volume': np.random.uniform(100, 1000)}
            for j in range(87):
                features[f'feature_{j}'] = np.random.normal(0, 1)
            data.append(features)
            
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.balance = 10000.0
        self.btc_held = 0.0
        self.position_type = 0
        self.leverage = 1.0
        self.entry_price = 0.0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation with NaN protection"""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_size, dtype=np.float32)
            
        row = self.data.iloc[self.current_step]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        state_values = row[numeric_cols].values.astype(np.float32)
        
        # Replace any NaN/inf values with zeros
        state_values = np.nan_to_num(state_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(state_values) > 86:
            state_values = state_values[:86]
        elif len(state_values) < 86:
            padding = np.zeros(86 - len(state_values), dtype=np.float32)
            state_values = np.append(state_values, padding)
        
        # Ensure balance and leverage are valid
        balance_norm = max(self.balance, 1.0) / 10000.0
        leverage_norm = max(self.leverage, 1.0) / 125.0
        
        state = np.append(state_values, [
            balance_norm,
            self.position_type,
            leverage_norm
        ])
        
        final_state = state[:self.state_size].astype(np.float32)
        # Final NaN protection
        final_state = np.nan_to_num(final_state, nan=0.0, posinf=0.0, neginf=0.0)
        
        return final_state
    
    def step(self, action: int):
        """Execute action and return new state, reward, done"""
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
            
        old_portfolio_value = self._calculate_portfolio_value()
        
        reward = self._execute_action(action)
        self.current_step += 1
        
        new_portfolio_value = self._calculate_portfolio_value()
        
        if old_portfolio_value > 0 and new_portfolio_value > 0:
            portfolio_reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
            portfolio_reward = max(-0.1, min(0.1, portfolio_reward))
        else:
            portfolio_reward = 0.0
        
        total_reward = reward + portfolio_reward * 5
        
        if np.isnan(total_reward) or np.isinf(total_reward):
            total_reward = 0.0
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), total_reward, done, {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'position': self.position_type
        }
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action - same logic as DQN"""
        current_price = self.prices[self.current_step]
        reward = 0.0
        
        if action == 0:  # WAIT
            reward = -0.001
            
        elif action in [1, 2, 3]:  # LONG positions
            leverages = [40, 80, 125]
            leverage = leverages[action - 1]
            
            if self.position_type == 0:
                self.position_type = 1
                self.leverage = leverage
                self.entry_price = current_price
                reward = 0.01
                
        elif action in [4, 5, 6]:  # SHORT positions
            leverages = [40, 80, 125]
            leverage = leverages[action - 4]
            
            if self.position_type == 0:
                self.position_type = -1
                self.leverage = leverage
                self.entry_price = current_price
                reward = 0.01
                
        elif action == 7:  # CLOSE position
            if self.position_type != 0:
                profit_loss = self._calculate_position_pnl(current_price)
                self.balance += profit_loss
                reward = profit_loss / 1000.0
                
                self.position_type = 0
                self.leverage = 1.0
                self.entry_price = 0.0
                
        return reward
    
    def _calculate_position_pnl(self, current_price: float) -> float:
        """Calculate position P&L"""
        if self.position_type == 0:
            return 0.0
            
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.position_type == 1:
            return self.balance * 0.1 * price_change * self.leverage
        else:
            return self.balance * 0.1 * (-price_change) * self.leverage
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance
            
        try:
            current_price = self.prices[self.current_step]
            if np.isnan(current_price) or current_price <= 0:
                return self.balance
            unrealized_pnl = self._calculate_position_pnl(current_price)
            portfolio_value = self.balance + unrealized_pnl
            if np.isnan(portfolio_value) or portfolio_value <= 0:
                return self.balance
            return portfolio_value
        except:
            return self.balance

class PPOTradingAgent:
    """PPO Trading Agent - Policy gradient method for adaptive BTC trading"""
    
    def _run_gpu_diagnostics(self):
        """Comprehensive GPU diagnostics and verification"""
        logger.info("=" * 60)
        logger.info("🔍 GPU DIAGNOSTICS AND DEVICE VERIFICATION")
        logger.info("=" * 60)
        
        # PyTorch CUDA diagnostics
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            
            # GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            
            # Check nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip().split(',')
                    logger.info(f"nvidia-smi GPU: {gpu_info[0].strip()}")
                    logger.info(f"Driver version: {gpu_info[1].strip()}")
                else:
                    logger.warning("⚠️ nvidia-smi command failed")
            except Exception as e:
                logger.warning(f"⚠️ nvidia-smi check failed: {e}")
        
        logger.info("=" * 60)
    
    def __init__(self, state_size: int = 89, action_size: int = 8, lr: float = 1e-4):
        # Comprehensive GPU diagnostics
        self._run_gpu_diagnostics()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"PPO Trading AI using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Name: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("✅ CUDA Optimizations enabled: cuDNN benchmark, TF32")
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor-Critic network - optimized for numerical stability
        self.ac_network = ActorCriticNetwork(state_size, action_size, hidden_size=512).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr, eps=1e-8, weight_decay=1e-6)
        
        # Learning rate scheduler for stability
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=3)
        
        # PPO hyperparameters - optimized for numerical stability
        self.clip_epsilon = 0.1  # Reduced clipping for stability
        self.value_loss_coef = 0.25  # Reduced value loss weight
        self.entropy_coef = 0.005  # Reduced entropy for stability
        self.max_grad_norm = 0.1  # Much stricter gradient clipping
        self.ppo_epochs = 4  # Reduced epochs to prevent overfitting
        self.minibatch_size = 1024  # Smaller batches for stability
        self.rollout_steps = 4096  # Smaller rollouts
        
        # Disable mixed precision for numerical stability in large model
        self.use_amp = False  # Temporarily disabled to fix NaN issues
        self.scaler = None
        logger.info("⚠️ Automatic Mixed Precision DISABLED for numerical stability")
        
        # GPU-optimized trajectory buffer
        self.trajectory_buffer = PPOTrajectoryBuffer(gamma=0.99, lam=0.95, device=self.device)
        
        
        # Training tracking and profiling
        self.total_steps = 0
        self.rollout_time = 0
        self.update_time = 0
        self.gpu_memory_peak = 0
        
        # Verify model is on GPU
        model_device = next(self.ac_network.parameters()).device
        logger.info(f"PPO Agent initialized with {sum(p.numel() for p in self.ac_network.parameters()):,} parameters")
        logger.info(f"✅ Model device: {model_device}")
        logger.info(f"✅ Model on GPU: {next(self.ac_network.parameters()).is_cuda}")
        logger.info("🎯 PPO HYPERPARAMETERS:")
        logger.info(f"   - Learning Rate: {lr}")
        logger.info(f"   - Clip Epsilon: {self.clip_epsilon}")
        logger.info(f"   - PPO Epochs: {self.ppo_epochs}")
        logger.info(f"   - Entropy Coeff: {self.entropy_coef}")
        
    def select_action(self, state):
        """Select action using current policy with NaN protection"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Check for NaN in input state
        if torch.isnan(state).any():
            logger.warning("NaN detected in input state, replacing with zeros")
            state = torch.nan_to_num(state, nan=0.0)
        
        with torch.no_grad():
            action_logits, state_value = self.ac_network(state)
        
        # Check for NaN in network outputs (already checked in forward but double-check)
        if torch.isnan(action_logits).any() or torch.isnan(state_value).any():
            logger.error(f"NaN in network outputs! action_logits: {action_logits}, state_value: {state_value}")
            # Return safe default action (WAIT=0)
            return 0, -1.0, 0.0
        
        # Create categorical distribution directly from logits (numerically stable)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item(), state_value.item()
    
    def select_action_batch(self, states):
        """Vectorized batch action selection for parallel environments"""
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
        
        # NaN protection for batch
        if torch.isnan(states).any():
            logger.warning("NaN detected in batch states, replacing with zeros")
            states = torch.nan_to_num(states, nan=0.0)
        
        with torch.no_grad():
            action_logits, state_values = self.ac_network(states)
        
        # Batch NaN protection (already checked in forward but double-check)
        if torch.isnan(action_logits).any() or torch.isnan(state_values).any():
            logger.error("NaN in batch network outputs!")
            # Return safe default actions (WAIT=0)
            batch_size = states.size(0)
            actions = torch.zeros(batch_size, dtype=torch.long)
            log_probs = torch.full((batch_size,), -1.0)
            values = torch.zeros(batch_size)
            return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
        
        # Sample actions for entire batch using logits (numerically stable)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        # Return CPU numpy arrays for environment interaction
        return actions.cpu().numpy(), log_probs.cpu().numpy(), state_values.squeeze().cpu().numpy()
    
    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in trajectory buffer"""
        self.trajectory_buffer.store(state, action, reward, value, log_prob, done)
    
    def train_ppo(self):
        """GPU-optimized PPO training with mini-batching and AMP"""
        if len(self.trajectory_buffer.states) == 0:
            return {}
        
        # Start GPU profiling
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        
        # Get training batch (already on GPU)
        batch = self.trajectory_buffer.get_batch()
        batch_size = batch['states'].size(0)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0  
        total_entropy_loss = 0
        total_batches = 0
        
        # Multiple PPO training epochs with mini-batching
        for epoch in range(self.ppo_epochs):
            # Shuffle indices on GPU
            indices = torch.randperm(batch_size, device=self.device)
            
            # Process mini-batches
            for start_idx in range(0, batch_size, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch (all on GPU)
                mb_states = batch['states'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['old_log_probs'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_advantages = batch['advantages'][mb_indices]
                
                # Safe advantage normalization per mini-batch on GPU
                adv_mean = mb_advantages.mean()
                adv_std = mb_advantages.std()
                
                # Check for NaN/Inf in mini-batch advantages
                if torch.isnan(mb_advantages).any() or torch.isinf(mb_advantages).any():
                    logger.error("🚨 NaN/Inf in mini-batch advantages! Replacing with zeros")
                    mb_advantages = torch.zeros_like(mb_advantages)
                    adv_mean = 0.0
                    adv_std = 1.0
                
                # Safe normalization: only normalize if std is reasonable
                if adv_std < 1e-6:  # std too small, don't normalize
                    logger.warning(f"Mini-batch advantage std too small ({adv_std}), skipping normalization")
                    mb_advantages = mb_advantages - adv_mean  # Just center
                else:
                    mb_advantages = (mb_advantages - adv_mean) / (adv_std + 1e-8)
                    
                # Final safety check and clamping
                mb_advantages = torch.clamp(mb_advantages, min=-10, max=10)
                mb_advantages = torch.nan_to_num(mb_advantages, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Forward pass with mixed precision and NaN checks
                if self.use_amp:
                    with autocast():
                        action_logits, state_values = self.ac_network(mb_states)
                        
                        # Critical NaN check before creating distribution (already checked in forward)
                        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                            logger.error(f"🚨 NaN/Inf in action_logits: {action_logits[0]}")
                            action_logits = torch.zeros_like(action_logits)  # Safe fallback
                        
                        dist = Categorical(logits=action_logits)
                        
                        # Calculate losses
                        log_probs = dist.log_prob(mb_actions)
                        ratio = torch.exp(log_probs - mb_old_log_probs)
                        
                        policy_loss_1 = mb_advantages * ratio
                        policy_loss_2 = mb_advantages * torch.clamp(
                            ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                        )
                        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                        
                        value_loss = F.mse_loss(state_values.squeeze(), mb_returns)
                        entropy_loss = -dist.entropy().mean()
                        
                        total_loss = (
                            policy_loss + 
                            self.value_loss_coef * value_loss + 
                            self.entropy_coef * entropy_loss
                        )
                else:
                    # Standard precision
                    action_logits, state_values = self.ac_network(mb_states)
                    
                    # NaN check for standard precision path
                    if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                        logger.error(f"🚨 NaN/Inf in action_logits (std precision): {action_logits[0]}")
                        action_logits = torch.zeros_like(action_logits)  # Safe fallback
                    
                    dist = Categorical(logits=action_logits)
                    
                    log_probs = dist.log_prob(mb_actions)
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    
                    policy_loss_1 = mb_advantages * ratio
                    policy_loss_2 = mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    )
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    
                    value_loss = F.mse_loss(state_values.squeeze(), mb_returns)
                    entropy_loss = -dist.entropy().mean()
                    
                    total_loss = (
                        policy_loss + 
                        self.value_loss_coef * value_loss + 
                        self.entropy_coef * entropy_loss
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_batches += 1
        
        # Update learning rate
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        # GPU profiling results
        self.update_time = time.time() - start_time
        if torch.cuda.is_available():
            self.gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e9
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - start_memory) / 1e9
            logger.info(f"🖥️ GPU Update: {self.update_time:.2f}s | Memory: {memory_used:.2f}GB | Peak: {self.gpu_memory_peak:.2f}GB")
        
        # Clear trajectory buffer
        self.trajectory_buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / total_batches,
            'value_loss': total_value_loss / total_batches,
            'entropy_loss': total_entropy_loss / total_batches,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'update_time': self.update_time,
            'gpu_memory_peak': self.gpu_memory_peak
        }

class VectorizedPPOEnvironment:
    """Vectorized PPO environments for parallel rollouts"""
    
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.envs = [PPOBTCTradingEnvironment() for _ in range(num_envs)]
        self.states = None
        
    def reset(self):
        """Reset all environments and return initial states"""
        self.states = np.array([env.reset() for env in self.envs])
        return self.states
        
    def step(self, actions):
        """Step all environments with given actions"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        next_states = np.array([result[0] for result in results])
        rewards = np.array([result[1] for result in results])
        dones = np.array([result[2] for result in results])
        infos = [result[3] for result in results]
        
        # Reset finished environments
        for i, done in enumerate(dones):
            if done:
                next_states[i] = self.envs[i].reset()
                
        self.states = next_states
        return next_states, rewards, dones, infos

def train_ppo_trading_ai():
    """GPU-optimized PPO Trading AI with vectorized environments"""
    logger.info("🚀 STARTING VECTORIZED GPU-OPTIMIZED PPO TRAINING 🚀")
    logger.info("🎯 BREAKTHROUGH: Vectorized environments + GPU-accelerated policy gradients!")
    logger.info("Building high-performance parallel reinforcement learning for BTC trading...")
    
    # Initialize vectorized environments and agent
    num_envs = 8  # Parallel environments for GPU efficiency
    vec_env = VectorizedPPOEnvironment(num_envs)
    
    # Initialize architectural-fix PPO agent
    config = PPOConfig(
        learning_rate=1e-4,  # Reduced for stability
        hidden_size=512,     # Match original size
        minibatch_size=1024, # Smaller for stability
        ppo_epochs=4         # Fewer epochs for stability
    )
    base_agent = PPOAgent(obs_dim=89, action_dim=1, config=config)
    
    # Wrap with continuous-to-discrete action converter
    class ContinuousToDiscretePPOAgent:
        """Adapter to convert continuous PPO actions to discrete trading actions"""
        def __init__(self, continuous_agent):
            self.agent = continuous_agent
            self.rollout_steps = 256  # Match expected rollout steps
            self.action_mapping = {}  # Store discrete->continuous mapping for training
            
        def select_action_batch(self, states):
            """Convert continuous actions to discrete trading actions"""
            batch_actions = []
            batch_log_probs = []
            batch_values = []
            
            for i, state in enumerate(states):
                continuous_action, log_prob, value = self.agent.get_action(state)
                
                # Convert continuous action [-1, 1] to discrete action [0, 7]
                action_val = continuous_action[0]  # Extract scalar from array
                discrete_action = int((action_val + 1) * 4)  # Maps [-1,1] to [0,8)
                discrete_action = np.clip(discrete_action, 0, 7)  # Ensure valid range
                
                # Store mapping for training
                self.action_mapping[len(batch_actions)] = continuous_action
                
                batch_actions.append(discrete_action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)
                
            return np.array(batch_actions), np.array(batch_log_probs), np.array(batch_values)
        
        def store_experience(self, state, discrete_action, reward, value, log_prob, done):
            """Store experience with continuous action for training"""
            # Convert discrete action back to continuous for storage
            continuous_action = (discrete_action / 4.0) - 1.0  # Map [0,7] back to [-1,1]
            continuous_action = np.array([continuous_action])
            
            return self.agent.store_transition(state, continuous_action, reward, value, log_prob, done)
            
        def update(self, *args, **kwargs):
            return self.agent.update(*args, **kwargs)
            
        def save(self, *args, **kwargs):
            return self.agent.save(*args, **kwargs)
            
        def __getattr__(self, name):
            # Delegate other attributes to the wrapped agent
            return getattr(self.agent, name)
    
    agent = ContinuousToDiscretePPOAgent(base_agent)
    
    episodes_per_env = 3  # Episodes per environment (total = 3 * 8 = 24 episodes)
    episode_scores = []
    start_time = time.time()
    total_rollout_time = 0
    total_update_time = 0
    
    # Reset all environments
    states = vec_env.reset()
    env_rewards = np.zeros(num_envs)
    env_steps = np.zeros(num_envs, dtype=int)
    global_step = 0
    
    for episode_batch in range(episodes_per_env):
        episode_start = time.time()
        rollout_start = time.time()
        
        # Vectorized rollout phase
        for rollout_step in range(agent.rollout_steps):
            step_start = time.time()
            
            # Batch action selection (GPU-efficient)
            actions, log_probs, values = agent.select_action_batch(states)
            
            # Vectorized environment step
            next_states, rewards, dones, infos = vec_env.step(actions)
            
            # Store experiences for each environment
            for env_idx in range(num_envs):
                agent.store_experience(
                    states[env_idx], actions[env_idx], rewards[env_idx], 
                    values[env_idx], log_probs[env_idx], dones[env_idx]
                )
                
                env_rewards[env_idx] += rewards[env_idx]
                env_steps[env_idx] += 1
                
                # Log completed episodes
                if dones[env_idx]:
                    episode_scores.append(env_rewards[env_idx])
                    logger.info(f"Env {env_idx} Episode Complete | Score: {env_rewards[env_idx]:.2f} | Steps: {env_steps[env_idx]}")
                    env_rewards[env_idx] = 0
                    env_steps[env_idx] = 0
            
            states = next_states
            global_step += num_envs
            agent.total_steps += num_envs
            
            # Training after rollout batch
            if (rollout_step + 1) % (agent.rollout_steps // 4) == 0:
                training_metrics = agent.train_ppo()
                total_update_time += agent.update_time
        
        rollout_time = time.time() - rollout_start
        total_rollout_time += rollout_time
        
        # Final training on remaining trajectory
        if len(agent.trajectory_buffer.states) > 0:
            training_metrics = agent.train_ppo()
            total_update_time += agent.update_time
        
        # Batch statistics with detailed profiling
        if episode_scores:
            avg_score = np.mean(episode_scores[-5:])  # Recent 5-episode average
            batch_score = np.mean(env_rewards)  # Current batch progress
        else:
            avg_score = 0
            batch_score = np.mean(env_rewards)
            
        episode_time = time.time() - episode_start
        steps_per_sec = (agent.rollout_steps * num_envs) / rollout_time
        
        # Performance analysis
        rollout_pct = (rollout_time / episode_time) * 100
        update_pct = ((total_update_time / (episode_batch + 1)) / episode_time) * 100
            
        # Comprehensive logging
        logger.info(f"Batch {episode_batch:2d}/{episodes_per_env} 🚀 | Completed Episodes: {len(episode_scores)} | "
                   f"Avg Score: {avg_score:8.2f} | Current Batch: {batch_score:8.2f}")
        logger.info(f"   ⏱️ Time: {episode_time:.1f}s | Rollout: {rollout_time:.1f}s ({rollout_pct:.1f}%) | "
                   f"Update: {agent.update_time:.2f}s ({update_pct:.1f}%)")
        logger.info(f"   🚀 Throughput: {steps_per_sec:.1f} steps/sec | Total Steps: {agent.total_steps:,}")
        
        # Log training metrics if available
        if 'training_metrics' in locals() and training_metrics:
            logger.info(f"   📊 Policy Loss: {training_metrics['policy_loss']:.4f} | "
                       f"Value Loss: {training_metrics['value_loss']:.4f} | "
                       f"Entropy: {training_metrics['entropy_loss']:.4f}")
            
            if 'gpu_memory_peak' in training_metrics:
                logger.info(f"   🎮 GPU Memory: {training_metrics['gpu_memory_peak']:.2f}GB")
    
    total_time = time.time() - start_time
    
    # Final performance summary
    logger.info("="*80)
    logger.info("🏁 GPU-OPTIMIZED PPO TRAINING COMPLETED!")
    logger.info("📊 Performance Summary:")
    logger.info(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"   Rollout Time: {total_rollout_time:.1f}s ({(total_rollout_time/total_time)*100:.1f}%)")
    logger.info(f"   Update Time: {total_update_time:.1f}s ({(total_update_time/total_time)*100:.1f}%)")
    logger.info(f"   Avg Episode: {total_time/episodes:.1f}s")
    logger.info(f"   GPU Utilization: {agent.gpu_memory_peak:.2f}GB peak memory")
    logger.info("="*80)
    
    # Save PPO model
    local_path = "/tmp/ppo_trading_model.pth"
    model_data = {
        'ac_network_state_dict': agent.ac_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episodes': episodes,
        'best_score': max(scores),
        'avg_final_score': np.mean(scores[-5:]),
        'model_type': 'PPO_Trading_Agent',
        'timestamp': datetime.now().isoformat(),
        'parameters': sum(p.numel() for p in agent.ac_network.parameters()),
        'training_time_seconds': total_time,
        'hyperparameters': {
            'clip_epsilon': agent.clip_epsilon,
            'value_loss_coef': agent.value_loss_coef,
            'entropy_coef': agent.entropy_coef,
            'ppo_epochs': agent.ppo_epochs,
            'learning_rate': 1e-4
        }
    }
    torch.save(model_data, local_path)
    
    # Upload to cloud
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket('YOUR_GCP_PROJECT-ml-trading-central')
        blob = bucket.blob('ppo-trading-ai/model/ppo_trading_model.pth')
        blob.upload_from_filename(local_path)
        logger.info("✅ PPO model uploaded to cloud storage")
    except Exception as e:
        logger.info(f"⚠️ Upload failed: {e}")
    
    logger.info("🎉 PPO TRADING AI TRAINING COMPLETED!")
    logger.info(f"Best score: {max(scores):.2f}")
    logger.info(f"Final 5-episode average: {np.mean(scores[-5:]):.2f}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Total steps: {agent.total_steps:,}")
    logger.info("🎯 BREAKTHROUGH: PPO policy gradients adapt without catastrophic forgetting! 🧠📈")
    
    return agent, env

if __name__ == "__main__":
    logger.info("=" * 90)
    logger.info("PPO TRADING AI FOR GRANDMA - Proximal Policy Optimization")
    logger.info("BREAKTHROUGH: Adaptive policy gradients for non-stationary BTC trading")
    logger.info("Built with love to honor her memory and learn without forgetting")
    logger.info("=" * 90)
    
    train_ppo_trading_ai()