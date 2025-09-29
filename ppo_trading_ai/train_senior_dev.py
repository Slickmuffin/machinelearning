#!/usr/bin/env python3
"""
PPO Training with Complete Senior Dev Economic Fixes
All specifications implemented and tested
"""

import os
import sys
import logging
import torch
import numpy as np
import argparse
from dataclasses import asdict
from typing import Dict, Any

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add current directory to path for Vertex AI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try different import patterns for robustness
try:
    from trading_env_senior_dev import PPOTradingEnvironmentSeniorDev, TradingConfig
    from ppo_agent import PPOAgent, PPOConfig
    logger.info("✅ Direct imports successful")
except ImportError:
    try:
        from ppo_trading_ai.trading_env_senior_dev import PPOTradingEnvironmentSeniorDev, TradingConfig
        from ppo_trading_ai.ppo_agent import PPOAgent, PPOConfig
        logger.info("✅ Module imports successful")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error(f"Current path: {os.getcwd()}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Files in current dir: {os.listdir('.')}")
        raise SystemExit(f"Import error: {e}")

def setup_gpu_diagnostics():
    """Senior Dev GPU Diagnostics - fail fast if no GPU"""
    logger.info("🚀 SENIOR DEV PPO TRAINING WITH ECONOMIC FIXES")
    logger.info("=" * 60)
    
    logger.info("🔧 GPU Diagnostics:")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        logger.info(f"  ✅ CUDA Available: {device_count} device(s)")
        logger.info(f"  📱 Current Device: {current_device} ({device_name})")
        logger.info(f"  💾 Memory: {memory_total:.1f}GB total")
        
        # Test GPU compute
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor.T).sum().item()
        logger.info(f"  🧮 GPU Compute Test: {result:.2e} (working)")
        
        return f"cuda:{current_device}"
    else:
        logger.error("🚨 CUDA NOT AVAILABLE - Cannot train without GPU")
        raise SystemExit("GPU required for training")

def validate_economic_setup():
    """Validate senior dev economic specifications are correctly implemented"""
    logger.info("🧪 Validating Senior Dev Economic Setup...")
    
    config = TradingConfig()
    
    # Test fee math (senior dev test case)
    E, N0, N1 = 10_000, 0.5*10_000, 3.0*10_000
    dN = abs(N1-N0)  # 25k
    fee = dN * config.taker_fee  # 3 bps
    slip = dN * (config.slippage_bp / 1e4)  # 0.5 bp
    total_cost = fee + slip
    
    assert round(fee, 2) == 7.5, f"Fee calculation wrong: {fee}"
    assert round(slip, 2) == 1.25, f"Slip calculation wrong: {slip}"
    assert round(total_cost, 2) == 8.75, f"Total cost wrong: {total_cost}"
    
    # Test funding
    funding_per_8h = 30_000 * config.funding_rate_8h
    assert round(funding_per_8h, 2) == 3.0, f"8h funding wrong: {funding_per_8h}"
    
    # Test thresholds
    assert config.taker_fee == 0.0003, "Taker fee not 3bp"
    assert config.slippage_bp == 0.5, "Slippage not 0.5bp"
    assert config.funding_rate_8h == 0.0001, "Funding rate not 1bp/8h"
    
    logger.info("  ✅ Fee Math: 3bp + 0.5bp = 3.5bp per execution")
    logger.info("  ✅ Funding: 1bp per 8 hours")
    logger.info("  ✅ Economic parameters validated")

def create_environment(data_path: str = None) -> PPOTradingEnvironmentSeniorDev:
    """Create environment with senior dev specifications"""
    logger.info("🏗️ Creating Senior Dev Trading Environment...")
    
    try:
        config = TradingConfig()
        env = PPOTradingEnvironmentSeniorDev(config=config, data_path=data_path)
        
        logger.info(f"  ✅ Environment created successfully")
        logger.info(f"  📊 Observation space: {env.observation_space.shape}")
        logger.info(f"  🎯 Action space: {env.action_space.shape}")
        
        return env
    except Exception as e:
        logger.error(f"🚨 Environment creation failed: {e}")
        raise SystemExit(f"Environment creation failed: {e}")

def test_environment_step(env):
    """Test environment produces valid rewards (not all zeros)"""
    logger.info("🧪 Testing Environment Step Logic...")
    
    obs, info = env.reset()
    total_reward = 0.0
    non_zero_rewards = 0
    
    for step in range(10):
        action = np.array([np.random.uniform(-0.5, 0.5)])  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if abs(reward) > 1e-8:
            non_zero_rewards += 1
            
        logger.info(f"  Step {step+1}: reward={reward:.6f}, equity=${info['equity']:.2f}, N=${info['N']:.2f}")
        
        if terminated or truncated:
            logger.info(f"  Episode ended at step {step+1}")
            break
    
    if non_zero_rewards == 0:
        logger.error("🚨 ALL REWARDS ARE ZERO - Environment broken!")
        raise SystemExit("Environment test failed: all rewards zero")
    
    logger.info(f"  ✅ Environment test passed: {non_zero_rewards}/10 non-zero rewards")

def train_ppo(args):
    """Main training loop with senior dev specifications"""
    device = setup_gpu_diagnostics()
    validate_economic_setup()
    
    # Create environment
    env = create_environment(args.data_path)
    test_environment_step(env)
    
    # Create PPO agent with senior dev stability specifications
    ppo_config = PPOConfig(
        learning_rate=3e-4,      # Policy LR (value will be 10x higher automatically)
        use_amp=False,           # Disabled per senior dev for numerical stability
        device=device,
        ppo_epochs=4,            # Conservative epochs to prevent overtraining
        clip_epsilon=0.2,        # Standard PPO clipping
        entropy_coef=0.01,       # Standard entropy coefficient
        value_loss_coef=0.5,     # Standard value loss weighting
        max_grad_norm=0.5,       # Conservative gradient clipping
        rollout_steps=256,       # Further reduced for faster completion
        minibatch_size=256       # Smaller minibatches for stability
    )
    
    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim=obs_dim, config=ppo_config)
    
    logger.info(f"🤖 PPO Agent created:")
    logger.info(f"  📊 Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    logger.info(f"  📚 Learning rate: {ppo_config.learning_rate}")
    logger.info(f"  🎯 Target: {args.max_updates} updates, {args.max_steps:,} steps")
    
    # Training metrics
    total_steps = 0
    total_episodes = 0
    best_reward = float('-inf')
    
    try:
        for update in range(args.max_updates):
            # Collect rollouts
            obs, info = env.reset()
            episode_rewards = []
            episode_length = 0
            episode_costs = []
            episode_turnovers = []
            
            rollout_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            }
            
            # Collect rollout
            for step in range(ppo_config.rollout_steps):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action, log_prob, value, _ = agent.network.get_action_and_value(obs_tensor)
                    
                rollout_data['observations'].append(obs.copy())
                rollout_data['actions'].append(action.cpu().numpy().flatten())
                rollout_data['values'].append(value.item())
                rollout_data['log_probs'].append(log_prob.item())
                
                # Environment step
                obs, reward, terminated, truncated, step_info = env.step(action.cpu().numpy())
                
                rollout_data['rewards'].append(reward)
                rollout_data['dones'].append(terminated or truncated)
                
                episode_rewards.append(reward)
                episode_length += 1
                episode_costs.append(step_info.get('cost', 0.0))
                episode_turnovers.append(step_info.get('traded', 0.0))
                total_steps += 1
                
                if terminated or truncated:
                    total_episodes += 1
                    obs, info = env.reset()
                    episode_length = 0
                    
                    # Log episode results
                    if len(episode_rewards) > 0:
                        ep_return = sum(episode_rewards)
                        ep_cost = sum(episode_costs)
                        ep_turnover = sum(episode_turnovers)
                        
                        if ep_return > best_reward:
                            best_reward = ep_return
                        
                        logger.info(f"Episode {total_episodes}: return={ep_return:.6f}, cost={ep_cost:.2f}, turnover={ep_turnover:.2f}")
                    
                    episode_rewards.clear()
                    episode_costs.clear()
                    episode_turnovers.clear()
            
            # Convert to tensors
            observations = torch.FloatTensor(np.array(rollout_data['observations'])).to(device)
            actions = torch.FloatTensor(np.array(rollout_data['actions'])).to(device)
            rewards = torch.FloatTensor(rollout_data['rewards']).to(device)
            values = torch.FloatTensor(rollout_data['values']).to(device)
            log_probs = torch.FloatTensor(rollout_data['log_probs']).to(device)
            dones = torch.BoolTensor(rollout_data['dones']).to(device)
            
            # Store all step data in agent buffer for PPO update
            for i in range(len(rollout_data['observations'])):
                agent.store_transition(
                    rollout_data['observations'][i],
                    rollout_data['actions'][i], 
                    rollout_data['rewards'][i],
                    rollout_data['values'][i],
                    rollout_data['log_probs'][i],
                    rollout_data['dones'][i]
                )
            
            # PPO update - pass next observations for proper GAE bootstrapping and env info for tripwires
            env_infos = [step_info] if step_info else None
            if not (terminated or truncated):
                # Episode continuing - pass current obs as next_obs for bootstrapping
                train_info = agent.update(next_observations=obs, env_infos=env_infos)
            else:
                # Episode ended - no bootstrap needed
                train_info = agent.update(next_observations=None, env_infos=env_infos)
            
            # Logging with critical value function monitoring
            if update % 10 == 0:
                mean_reward = rewards.mean().item()
                mean_value = values.mean().item()
                
                logger.info(f"Update {update+1}/{args.max_updates}")
                logger.info(f"  📊 Steps: {total_steps:,} | Episodes: {total_episodes}")
                logger.info(f"  💰 Mean Reward: {mean_reward:.6f} | Best: {best_reward:.6f}")
                logger.info(f"  📈 Mean Value: {mean_value:.6f}")
                logger.info(f"  🎯 Policy Loss: {train_info.get('policy_loss', 0.0):.6f}")
                logger.info(f"  📉 Value Loss: {train_info.get('value_loss', 0.0):.6f}")
                logger.info(f"  🎲 Entropy: {train_info.get('entropy_loss', 0.0):.6f}")
                logger.info(f"  🔍 Approx KL: {train_info.get('approx_kl', 0.0):.6f}")
                
                # CRITICAL: Value function stability monitoring
                value_loss = train_info.get('value_loss', 0.0)
                if value_loss > 1000:
                    logger.warning(f"⚠️ HIGH VALUE LOSS DETECTED: {value_loss:.2f} > 1000")
                elif value_loss > 500:
                    logger.warning(f"⚠️ HIGH VALUE LOSS: {value_loss:.2f} - approaching danger zone")
                
                # GPU memory
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    logger.info(f"  🖥️ GPU Memory: {memory_used:.2f}GB")
            
            # Economic tripwire check
            if update > 50:  # After warmup
                recent_rewards = rollout_data['rewards'][-100:] if len(rollout_data['rewards']) >= 100 else rollout_data['rewards']
                if recent_rewards:
                    mean_recent_reward = np.mean(recent_rewards)
                    if mean_recent_reward < -0.1 and update > 100:
                        logger.warning(f"⚠️ Consistently negative rewards: {mean_recent_reward:.6f}")
            
            if total_steps >= args.max_steps:
                logger.info(f"✅ Reached maximum steps: {total_steps:,}")
                break
                
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
    except Exception as e:
        logger.error(f"🚨 Training failed: {e}")
        raise
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Final stats: {total_steps:,} steps, {total_episodes} episodes, best reward: {best_reward:.6f}")
    
    # Save the trained model
    model_path = "/tmp/senior_dev_stable_model.pt"
    agent.save(model_path)
    logger.info(f"💾 Model saved to: {model_path}")
    
    # Upload to cloud storage if available
    try:
        import subprocess
        result = subprocess.run([
            "gsutil", "cp", model_path, 
            "gs://YOUR_GCP_PROJECT-vertex-training/models/"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("☁️ Model uploaded to cloud storage")
        else:
            logger.warning(f"⚠️ Cloud upload failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"⚠️ Cloud upload not available: {e}")

def main():
    parser = argparse.ArgumentParser(description='PPO Trading with Senior Dev Economic Fixes')
    parser.add_argument('--data-path', type=str, default='/tmp/btc_data.csv', help='Path to BTC data')
    parser.add_argument('--max-updates', type=int, default=200, help='Maximum PPO updates')
    parser.add_argument('--max-steps', type=int, default=1_000_000, help='Maximum environment steps')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting PPO Training with Senior Dev Economic Fixes")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        train_ppo(args)
        return True
    except Exception as e:
        logger.error(f"🚨 Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)