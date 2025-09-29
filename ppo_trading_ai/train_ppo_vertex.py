#!/usr/bin/env python3
"""
GPU-Optimized PPO Training for Vertex AI
Vectorized environments with proper GPU utilization and risk-aware training
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import subprocess
from typing import Dict, Any
import argparse
from dataclasses import asdict

# Import our modules
from ppo_env import PPOTradingEnvironment, VectorizedTradingEnv, TradingConfig
from ppo_agent import PPOAgent, PPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_gpu_diagnostics():
    """Comprehensive GPU diagnostics with fail-fast behavior"""
    logger.info("=" * 60)
    logger.info("🔍 CRITICAL GPU VERIFICATION - SENIOR DEV MANDATED")
    logger.info("=" * 60)
    
    # PyTorch CUDA diagnostics
    logger.info(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    # FAIL FAST: Don't slip into CPU training
    if not cuda_available:
        logger.error("❌ FATAL: CUDA NOT AVAILABLE - ABORTING JOB")
        logger.error("❌ This job requires GPU. CPU training is not acceptable.")
        logger.error("❌ Check container image and machine type configuration.")
        raise RuntimeError("CUDA not available - failing fast per senior dev requirement")
    
    # Detailed GPU verification 
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    cuda_version = torch.version.cuda
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_version = torch.backends.cudnn.version()
    
    logger.info(f"✅ CUDA device count: {device_count}")
    logger.info(f"✅ Current device: {current_device}")
    logger.info(f"✅ PyTorch CUDA version: {cuda_version}")
    logger.info(f"✅ cuDNN enabled: {cudnn_enabled}")
    logger.info(f"✅ cuDNN version: {cudnn_version}")
    
    # GPU hardware info
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    
    logger.info(f"✅ GPU memory: {gpu_memory_gb:.1f} GB")
    logger.info(f"✅ GPU name: {gpu_name}")
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("✅ CUDA optimizations enabled")
    
    # Quick GPU test - ensure we can create tensors and compute
    try:
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.mm(test_tensor, test_tensor.t())
        gpu_memory_alloc = torch.cuda.memory_allocated() / 1e6
        logger.info(f"✅ GPU compute test passed - {gpu_memory_alloc:.1f}MB allocated")
    except Exception as e:
        logger.error(f"❌ FATAL: GPU compute test failed: {e}")
        raise RuntimeError(f"GPU compute verification failed: {e}")
    
    # Check nvidia-smi for driver info
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', 
                              '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            logger.info(f"✅ nvidia-smi GPU: {gpu_info[0].strip()}")
            logger.info(f"✅ Driver version: {gpu_info[1].strip()}")
        else:
            logger.warning(f"⚠️  nvidia-smi returned code {result.returncode}")
    except Exception as e:
        logger.warning(f"⚠️  nvidia-smi check failed: {e}")
        
    logger.info("=" * 60)
    logger.info("🎯 GPU VERIFICATION COMPLETE - PROCEEDING WITH TRAINING")
    logger.info("=" * 60)

def create_env_fn(config: TradingConfig):
    """Create environment factory function"""
    def _create_env():
        return PPOTradingEnvironment(config)
    return _create_env

def train_ppo(args):
    """Main PPO training function"""
    logger.info("🚀 STARTING RISK-AWARE PPO TRAINING WITH GPU OPTIMIZATION 🚀")
    logger.info("🎯 BREAKTHROUGH: Continuous actions + proper risk controls + GPU acceleration!")
    
    # GPU diagnostics
    run_gpu_diagnostics()
    
    # Configurations
    trading_config = TradingConfig(
        initial_balance=10000.0,
        max_leverage=3.0,  # Use canonical fix leverage limit
        drawdown_stop=0.50,  # 50% drawdown stop (matches canonical fix)
    )
    
    ppo_config = PPOConfig(
        hidden_size=512,
        recurrent_size=128,
        learning_rate=1e-4,  # Reduced from 3e-4 per senior dev recommendation
        clip_epsilon=0.15,
        entropy_coef=0.01,
        ppo_epochs=8,
        minibatch_size=2048,
        rollout_steps=1024,
        use_amp=False,  # Disabled per senior dev recommendation for numerical stability
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create single environment to get dimensions
    env = PPOTradingEnvironment(trading_config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Environment setup:")
    logger.info(f"  Observation dim: {obs_dim}")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  Action space: Continuous [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    
    # Create vectorized environments
    num_envs = args.num_envs
    env_fn = create_env_fn(trading_config)
    vec_env = VectorizedTradingEnv(env_fn, num_envs)
    
    logger.info(f"Vectorized environments: {num_envs}")
    
    # Create PPO agent
    agent = PPOAgent(obs_dim, action_dim, ppo_config)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    start_time = time.time()
    
    # Training loop
    num_updates = args.num_updates
    log_interval = 10
    
    logger.info(f"Starting training for {num_updates} updates...")
    
    # Reset environments
    observations, infos = vec_env.reset()
    episode_rewards_buffer = np.zeros(num_envs)
    episode_lengths_buffer = np.zeros(num_envs)
    
    for update in range(num_updates):
        update_start_time = time.time()
        rollout_start_time = time.time()
        
        # Rollout collection
        for step in range(ppo_config.rollout_steps):
            # Get actions from agent (vectorized)
            actions = []
            log_probs = []
            values = []
            
            for env_idx in range(num_envs):
                action, log_prob, value = agent.get_action(observations[env_idx])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                
            actions = np.array(actions)
            log_probs = np.array(log_probs)
            values = np.array(values)
            
            # Environment step
            next_observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            
            # Store transitions
            for env_idx in range(num_envs):
                agent.store_transition(
                    observations[env_idx],
                    actions[env_idx],
                    rewards[env_idx],
                    values[env_idx],
                    log_probs[env_idx],
                    terminateds[env_idx] or truncateds[env_idx]
                )
                
                # Track episode metrics
                episode_rewards_buffer[env_idx] += rewards[env_idx]
                episode_lengths_buffer[env_idx] += 1
                
                # Episode completed
                if terminateds[env_idx] or truncateds[env_idx]:
                    episode_rewards.append(episode_rewards_buffer[env_idx])
                    episode_lengths.append(episode_lengths_buffer[env_idx])
                    episode_rewards_buffer[env_idx] = 0
                    episode_lengths_buffer[env_idx] = 0
                    
            observations = next_observations
            total_steps += num_envs
            
        rollout_time = time.time() - rollout_start_time
        
        # PPO update with proper next state values for GAE
        ppo_start_time = time.time()
        update_info = agent.update(next_observations=observations)  # Pass current observations as next states
        ppo_update_time = time.time() - ppo_start_time
        
        total_update_time = time.time() - update_start_time
        
        # Logging
        if update % log_interval == 0 or update == num_updates - 1:
            # Performance metrics
            steps_per_sec = (ppo_config.rollout_steps * num_envs) / rollout_time
            rollout_pct = (rollout_time / total_update_time) * 100
            update_pct = (ppo_update_time / total_update_time) * 100
            
            # Episode metrics
            if episode_rewards:
                mean_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
                mean_length = np.mean(episode_lengths[-100:])
                reward_std = np.std(episode_rewards[-100:])
            else:
                mean_reward = 0.0
                mean_length = 0.0
                reward_std = 0.0
                
            logger.info(f"Update {update:4d}/{num_updates}")
            logger.info(f"  📊 Episodes: {len(episode_rewards):4d} | Reward: {mean_reward:8.3f} ± {reward_std:6.3f} | Length: {mean_length:6.1f}")
            logger.info(f"  ⏱️  Rollout: {rollout_time:5.2f}s ({rollout_pct:4.1f}%) | Update: {ppo_update_time:5.2f}s ({update_pct:4.1f}%)")
            logger.info(f"  🚀 Throughput: {steps_per_sec:7.1f} steps/sec | Total: {total_steps:,}")
            
            if update_info:
                logger.info(f"  📈 Policy: {update_info['policy_loss']:7.4f} | Value: {update_info['value_loss']:7.4f} | Entropy: {update_info['entropy_loss']:7.4f}")
                
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"  🎮 GPU Memory: {gpu_memory:.2f}GB (max: {gpu_memory_max:.2f}GB)")
                
        # Save checkpoint periodically
        if update > 0 and update % (num_updates // 5) == 0:
            save_path = f'/tmp/ppo_checkpoint_update_{update}.pth'
            agent.save(save_path)
            logger.info(f"💾 Saved checkpoint: {save_path}")
            
            # Copy to cloud storage if available
            try:
                cloud_path = f'gs://YOUR_GCP_PROJECT-ml-trading-central/ppo_checkpoint_update_{update}.pth'
                subprocess.run(['gsutil', 'cp', save_path, cloud_path], check=True)
                logger.info(f"☁️  Uploaded to: {cloud_path}")
            except Exception as e:
                logger.warning(f"Cloud upload failed: {e}")
                
    # Final save
    final_save_path = '/tmp/ppo_final_model.pth'
    agent.save(final_save_path)
    logger.info(f"💾 Saved final model: {final_save_path}")
    
    try:
        cloud_path = 'gs://YOUR_GCP_PROJECT-ml-trading-central/ppo_final_model.pth'
        subprocess.run(['gsutil', 'cp', final_save_path, cloud_path], check=True)
        logger.info(f"☁️  Final model uploaded to: {cloud_path}")
    except Exception as e:
        logger.warning(f"Final upload failed: {e}")
        
    # Training summary
    total_time = time.time() - start_time
    avg_steps_per_sec = total_steps / total_time
    
    logger.info("=" * 80)
    logger.info("🏁 PPO TRAINING COMPLETED!")
    logger.info(f"  Total time: {total_time/60:.2f} minutes")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Average throughput: {avg_steps_per_sec:.1f} steps/sec")
    logger.info(f"  Episodes completed: {len(episode_rewards)}")
    
    if episode_rewards:
        logger.info(f"  Final reward: {np.mean(episode_rewards[-10:]):.3f}")
        logger.info(f"  Best reward: {np.max(episode_rewards):.3f}")
        
    logger.info("✅ Risk-aware PPO training with GPU optimization complete!")
    
    vec_env.close()

def main():
    parser = argparse.ArgumentParser(description='PPO BTC Trading Training')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--num_updates', type=int, default=1000, help='Number of PPO updates')
    parser.add_argument('--config', type=str, help='Config file path (optional)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PPO BTC TRADING AI - RISK-AWARE GPU-OPTIMIZED TRAINING")
    logger.info("Built for proper continuous action PPO with realistic cost structure")
    logger.info("=" * 80)
    
    train_ppo(args)

if __name__ == "__main__":
    main()