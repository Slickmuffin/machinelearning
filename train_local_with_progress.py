#!/usr/bin/env python3
"""
LOCAL PPO TRAINING WITH REAL-TIME PROGRESS MONITORING
Complete visual feedback system with progress bars and metrics
"""

import os
import sys
import time
import logging
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime, timedelta
from tqdm import tqdm
from colorama import init, Fore, Style, Back

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Add ppo_trading_ai to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppo_trading_ai'))

from ppo_trading_ai.trading_env_senior_dev import PPOTradingEnvironmentSeniorDev, TradingConfig
from ppo_trading_ai.ppo_agent import PPOAgent, PPOConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_local.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Track training metrics across updates"""
    total_steps: int = 0
    total_episodes: int = 0
    best_reward: float = float('-inf')
    episode_rewards: List[float] = None
    episode_costs: List[float] = None
    value_losses: List[float] = None
    policy_losses: List[float] = None

    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_costs is None:
            self.episode_costs = []
        if self.value_losses is None:
            self.value_losses = []
        if self.policy_losses is None:
            self.policy_losses = []

class ProgressMonitor:
    """Real-time training progress monitor with visual feedback"""

    def __init__(self, max_updates: int, rollout_steps: int):
        self.max_updates = max_updates
        self.rollout_steps = rollout_steps
        self.start_time = time.time()
        self.update_times = []

    def print_header(self):
        """Print training session header"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}  PPO TRADING AI - LOCAL TRAINING SESSION")
        print(f"{Fore.CYAN}  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.CYAN}{'='*80}\n")

    def get_eta(self, current_update: int) -> str:
        """Calculate estimated time remaining"""
        if current_update == 0 or not self.update_times:
            return "Calculating..."

        avg_time_per_update = np.mean(self.update_times[-10:])  # Use last 10 updates
        remaining_updates = self.max_updates - current_update
        eta_seconds = avg_time_per_update * remaining_updates

        return str(timedelta(seconds=int(eta_seconds)))

    def get_elapsed(self) -> str:
        """Get elapsed time"""
        elapsed_seconds = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed_seconds)))

    def print_update_header(self, update: int, metrics: TrainingMetrics):
        """Print header for each update"""
        pct_complete = (update / self.max_updates) * 100
        elapsed = self.get_elapsed()
        eta = self.get_eta(update)

        print(f"\n{Fore.YELLOW}{'─'*80}")
        print(f"{Fore.YELLOW}Update {update}/{self.max_updates} ({pct_complete:.1f}% Complete)")
        print(f"{Fore.YELLOW}Elapsed: {elapsed} | ETA: {eta}")
        print(f"{Fore.YELLOW}{'─'*80}")

    def print_metrics_summary(self, metrics: TrainingMetrics, train_info: Dict):
        """Print detailed metrics summary"""
        # Episode metrics
        if metrics.episode_rewards:
            recent_rewards = metrics.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)

            reward_color = Fore.GREEN if avg_reward > 0 else Fore.RED
            print(f"\n{Fore.CYAN}Episode Metrics:")
            print(f"  {reward_color}Avg Reward (last 10): {avg_reward:.6f}")
            print(f"  {Fore.WHITE}Best Reward: {metrics.best_reward:.6f}")
            print(f"  {Fore.WHITE}Total Episodes: {metrics.total_episodes}")

        # Training metrics
        value_loss = train_info.get('value_loss', 0.0)
        policy_loss = train_info.get('policy_loss', 0.0)
        entropy = train_info.get('entropy_loss', 0.0)
        kl = train_info.get('approx_kl', 0.0)

        # Color-code value loss based on danger level
        if value_loss > 1000:
            value_color = Fore.RED + Back.RED + Style.BRIGHT
        elif value_loss > 500:
            value_color = Fore.YELLOW + Style.BRIGHT
        else:
            value_color = Fore.GREEN

        print(f"\n{Fore.CYAN}Training Metrics:")
        print(f"  {value_color}Value Loss: {value_loss:.4f}")
        print(f"  {Fore.WHITE}Policy Loss: {policy_loss:.6f}")
        print(f"  {Fore.WHITE}Entropy: {entropy:.6f}")
        print(f"  {Fore.WHITE}Approx KL: {kl:.6f}")

        # Cost metrics
        if metrics.episode_costs:
            avg_cost = np.mean(metrics.episode_costs[-10:])
            print(f"\n{Fore.CYAN}Economic Metrics:")
            print(f"  {Fore.WHITE}Avg Cost (last 10): {avg_cost:.4f}")

        # GPU metrics
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"\n{Fore.CYAN}GPU Metrics:")
            print(f"  {Fore.WHITE}Memory Used: {memory_used:.2f}GB")
            print(f"  {Fore.WHITE}Memory Cached: {memory_cached:.2f}GB")

    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}WARNING: {message}")

    def print_error(self, message: str):
        """Print error message"""
        print(f"{Fore.RED}ERROR: {message}")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{Fore.GREEN}SUCCESS: {message}")

def setup_gpu():
    """Setup and validate GPU"""
    logger.info("Checking GPU availability...")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        logger.info(f"GPU detected: {device_name}")
        logger.info(f"GPU memory: {memory_total:.1f}GB")

        # Test GPU compute
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor.T).sum().item()
        logger.info(f"GPU compute test passed: {result:.2e}")

        return f"cuda:{current_device}"
    else:
        logger.warning("No GPU detected - training on CPU (will be slow)")
        return "cpu"

def validate_economic_setup():
    """Validate senior dev economic specifications"""
    config = TradingConfig()

    # Test fee math (senior dev test case)
    E, N0, N1 = 10_000, 0.5*10_000, 3.0*10_000
    dN = abs(N1-N0)
    fee = dN * config.taker_fee
    slip = dN * (config.slippage_bp / 1e4)
    total_cost = fee + slip

    assert round(fee, 2) == 7.5, f"Fee calculation wrong: {fee}"
    assert round(slip, 2) == 1.25, f"Slip calculation wrong: {slip}"
    assert round(total_cost, 2) == 8.75, f"Total cost wrong: {total_cost}"

    logger.info("Economic specifications validated")

def train_ppo(
    data_path: str = "/tmp/btc_data.csv",
    max_updates: int = 1000,
    max_steps: int = 1_000_000,
    rollout_steps: int = 256,
    save_interval: int = 100,
    checkpoint_dir: str = "./checkpoints"
):
    """
    Main training loop with comprehensive progress monitoring

    Args:
        data_path: Path to BTC price data
        max_updates: Maximum number of PPO updates
        max_steps: Maximum environment steps
        rollout_steps: Steps per rollout
        save_interval: Save checkpoint every N updates
        checkpoint_dir: Directory to save checkpoints
    """

    # Setup
    device = setup_gpu()
    validate_economic_setup()

    # Create progress monitor
    monitor = ProgressMonitor(max_updates, rollout_steps)
    monitor.print_header()

    # Create environment
    logger.info(f"Loading data from {data_path}")
    env = PPOTradingEnvironmentSeniorDev(data_path=data_path)
    logger.info(f"Environment created: {env.observation_space.shape[0]} features")

    # Create PPO agent
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        use_amp=False,
        device=device,
        ppo_epochs=4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=rollout_steps,
        minibatch_size=256
    )

    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim=obs_dim, config=ppo_config)

    param_count = sum(p.numel() for p in agent.network.parameters())
    logger.info(f"PPO Agent created: {param_count:,} parameters")

    # Training metrics
    metrics = TrainingMetrics()

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Main training loop
    try:
        # Overall progress bar
        with tqdm(total=max_updates, desc="Training Progress",
                  unit="update", position=0, leave=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            for update in range(max_updates):
                update_start_time = time.time()

                # Print update header
                if update % 10 == 0:
                    monitor.print_update_header(update, metrics)

                # Collect rollouts
                obs, info = env.reset()
                episode_rewards = []
                episode_costs = []

                rollout_data = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'values': [],
                    'log_probs': [],
                    'dones': []
                }

                # Rollout progress bar
                rollout_desc = f"Update {update+1} - Rollout"
                with tqdm(total=rollout_steps, desc=rollout_desc,
                         unit="step", position=1, leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as rollout_pbar:

                    for step in range(rollout_steps):
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
                        episode_costs.append(step_info.get('cost', 0.0))
                        metrics.total_steps += 1

                        if terminated or truncated:
                            metrics.total_episodes += 1
                            obs, info = env.reset()

                            # Log episode results
                            if episode_rewards:
                                ep_return = sum(episode_rewards)
                                ep_cost = sum(episode_costs)

                                metrics.episode_rewards.append(ep_return)
                                metrics.episode_costs.append(ep_cost)

                                if ep_return > metrics.best_reward:
                                    metrics.best_reward = ep_return
                                    monitor.print_success(f"New best reward: {ep_return:.6f}")

                                # Update rollout progress bar with metrics
                                rollout_pbar.set_postfix({
                                    'episodes': metrics.total_episodes,
                                    'reward': f"{ep_return:.4f}",
                                    'cost': f"{ep_cost:.4f}"
                                })

                                episode_rewards.clear()
                                episode_costs.clear()

                        rollout_pbar.update(1)

                # Store transitions in agent buffer
                for i in range(len(rollout_data['observations'])):
                    agent.store_transition(
                        rollout_data['observations'][i],
                        rollout_data['actions'][i],
                        rollout_data['rewards'][i],
                        rollout_data['values'][i],
                        rollout_data['log_probs'][i],
                        rollout_data['dones'][i]
                    )

                # PPO update
                if not (terminated or truncated):
                    train_info = agent.update(next_observations=obs, env_infos=None)
                else:
                    train_info = agent.update(next_observations=None, env_infos=None)

                # Track losses
                metrics.value_losses.append(train_info.get('value_loss', 0.0))
                metrics.policy_losses.append(train_info.get('policy_loss', 0.0))

                # Check for value function instability
                value_loss = train_info.get('value_loss', 0.0)
                if value_loss > 1000:
                    monitor.print_error(f"CRITICAL: Value loss explosion: {value_loss:.2f}")
                    monitor.print_error("Consider stopping training or reducing learning rate")
                elif value_loss > 500:
                    monitor.print_warning(f"High value loss detected: {value_loss:.2f}")

                # Print metrics every 10 updates
                if update % 10 == 0:
                    monitor.print_metrics_summary(metrics, train_info)

                # Save checkpoint
                if (update + 1) % save_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update+1}.pt")
                    agent.save(checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Update timing
                update_elapsed = time.time() - update_start_time
                monitor.update_times.append(update_elapsed)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'episodes': metrics.total_episodes,
                    'steps': f"{metrics.total_steps:,}",
                    'best_reward': f"{metrics.best_reward:.4f}",
                    'value_loss': f"{value_loss:.2f}"
                })

                # Check max steps
                if metrics.total_steps >= max_steps:
                    logger.info(f"Reached maximum steps: {metrics.total_steps:,}")
                    break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        monitor.print_warning("Training interrupted - saving final checkpoint")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Save final model
        final_path = os.path.join(checkpoint_dir, "final_model.pt")
        agent.save(final_path)
        logger.info(f"Final model saved: {final_path}")

        # Print final summary
        elapsed = monitor.get_elapsed()
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}TRAINING COMPLETE")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.WHITE}Total Time: {elapsed}")
        print(f"{Fore.WHITE}Total Steps: {metrics.total_steps:,}")
        print(f"{Fore.WHITE}Total Episodes: {metrics.total_episodes}")
        print(f"{Fore.WHITE}Best Reward: {metrics.best_reward:.6f}")

        if metrics.episode_rewards:
            avg_reward = np.mean(metrics.episode_rewards[-100:])
            print(f"{Fore.WHITE}Avg Reward (last 100): {avg_reward:.6f}")

        if metrics.value_losses:
            final_value_loss = metrics.value_losses[-1]
            print(f"{Fore.WHITE}Final Value Loss: {final_value_loss:.4f}")

        print(f"{Fore.CYAN}{'='*80}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Local PPO Training with Progress Monitoring')
    parser.add_argument('--data', type=str, default='/tmp/btc_data.csv',
                       help='Path to BTC data CSV')
    parser.add_argument('--updates', type=int, default=1000,
                       help='Maximum number of PPO updates')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Maximum environment steps')
    parser.add_argument('--rollout-steps', type=int, default=256,
                       help='Steps per rollout')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save checkpoint every N updates')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    logger.info("Starting local PPO training with progress monitoring")
    logger.info(f"Configuration: {vars(args)}")

    train_ppo(
        data_path=args.data,
        max_updates=args.updates,
        max_steps=args.steps,
        rollout_steps=args.rollout_steps,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir
    )
