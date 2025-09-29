#!/usr/bin/env python3
"""
PPO Training with SENIOR DEV TRIPWIRES
Fixed USD notional position system with pathological cost detection
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(__file__))

from ppo_env_fixed import PPOTradingEnvironment, TradingConfig
from ppo_agent import PPOAgent, PPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TripwireConfig:
    """Training safety tripwires"""
    max_cost_ratio: float = 0.01  # 1% of equity per update
    max_bankrupt_frac: float = 0.5  # 50% of environments
    min_advantage_threshold: float = 0.01  # Minimum advantage magnitude
    max_turnover_ratio: float = 0.2  # 20% of equity
    cost_history_window: int = 10  # Updates to average over
    
class PPOTrainerWithTripwires:
    """PPO trainer with senior dev tripwires to prevent pathological training"""
    
    def __init__(self, num_envs: int = 16, num_updates: int = 128):
        self.num_envs = num_envs
        self.num_updates = num_updates
        
        # Configs
        self.trading_config = TradingConfig()
        self.ppo_config = PPOConfig(
            # SENIOR DEV PPO KNOBS after economics are fixed
            ppo_epochs=4,  # Reduced from 8
            minibatch_size=1024,  # Reduced from 2048  
            learning_rate=3e-4,  # Standard
            clip_epsilon=0.15,
            value_loss_coef=0.5,
            entropy_coef=0.005  # Reduced
        )
        self.tripwire_config = TripwireConfig()
        
        # Create environments
        logger.info(f"🚀 Initializing {num_envs} PPO environments with FIXED USD notional system...")
        self.envs = [PPOTradingEnvironment(self.trading_config) for _ in range(num_envs)]
        
        # Create agent
        obs_dim = self.envs[0].observation_space.shape[0]
        self.agent = PPOAgent(obs_dim, config=self.ppo_config)
        
        # Tripwire tracking
        self.cost_history = []
        self.update_count = 0
        
        logger.info(f"✅ PPO Trainer initialized:")
        logger.info(f"  Environments: {num_envs}")
        logger.info(f"  Observation dim: {obs_dim}")
        logger.info(f"  PPO epochs: {self.ppo_config.ppo_epochs}")
        logger.info(f"  Learning rate: {self.ppo_config.learning_rate}")
        logger.info(f"  Tripwires enabled: cost_ratio<{self.tripwire_config.max_cost_ratio}, bankruptcy<{self.tripwire_config.max_bankrupt_frac}")
        
    def collect_rollouts(self) -> Dict[str, Any]:
        """Collect rollouts with tripwire monitoring"""
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'infos': []
        }
        
        # Reset environments
        observations = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=42 + i)
            observations.append(obs)
        
        # Collect rollout_steps of experience
        for step in range(self.ppo_config.rollout_steps):
            # Environment steps with individual agent calls
            next_observations = []
            step_infos = []
            
            for i, env in enumerate(self.envs):
                # Get action for this environment
                with torch.no_grad():
                    action, log_prob, value = self.agent.get_action(observations[i])
                
                obs, reward, terminated, done, info = env.step(action)
                
                rollout_data['observations'].append(observations[i])
                rollout_data['actions'].append(action)
                rollout_data['rewards'].append(reward)
                rollout_data['values'].append(value)
                rollout_data['log_probs'].append(log_prob)
                rollout_data['dones'].append(bool(terminated or done))
                rollout_data['infos'].append(info)
                
                if terminated or done:
                    obs, info = env.reset(seed=42 + i + step * self.num_envs)
                
                next_observations.append(obs)
                step_infos.append(info)
            
            observations = next_observations
            
            # TRIPWIRE: Check for pathological costs every 50 steps
            if step % 50 == 0:
                self._check_rollout_tripwires(step_infos, step)
        
        return rollout_data
    
    def _check_rollout_tripwires(self, infos: List[Dict], step: int):
        """Check tripwires during rollout collection"""
        if not infos:
            return
            
        # Calculate metrics
        cost_ratios = [info.get('cost_ratio', 0.0) for info in infos]
        bankruptcies = sum(1 for info in infos if info.get('equity', 1.0) <= 1.0)
        turnovers = [info.get('turnover', 0.0) / max(info.get('equity', 1.0), 1.0) for info in infos]
        
        mean_cost_ratio = np.mean(cost_ratios)
        bankruptcy_frac = bankruptcies / len(infos)
        mean_turnover = np.mean(turnovers)
        
        # Check tripwires
        if mean_cost_ratio > self.tripwire_config.max_cost_ratio:
            logger.error(f"🚨 TRIPWIRE: Cost ratio {mean_cost_ratio:.4f} > {self.tripwire_config.max_cost_ratio}")
            raise RuntimeError(f"Tripwire: pathological costs - mean cost ratio {mean_cost_ratio:.4f}")
        
        if bankruptcy_frac > self.tripwire_config.max_bankrupt_frac:
            logger.error(f"🚨 TRIPWIRE: Bankruptcy fraction {bankruptcy_frac:.2f} > {self.tripwire_config.max_bankrupt_frac}")
            raise RuntimeError(f"Tripwire: mass drawdowns - {bankruptcy_frac:.1%} environments bankrupt")
        
        if step % 200 == 0:  # Log every 200 steps
            logger.info(f"  Step {step}: cost_ratio={mean_cost_ratio:.4f}, bankruptcies={bankruptcy_frac:.1%}, turnover={mean_turnover:.4f}")
    
    def _check_update_tripwires(self, rollout_data: Dict, advantages: np.ndarray):
        """Check tripwires after PPO update"""
        self.update_count += 1
        
        # Calculate update metrics
        infos = rollout_data['infos']
        cost_ratios = [info.get('cost_ratio', 0.0) for info in infos]
        mean_cost_ratio = np.mean(cost_ratios)
        
        bankruptcies = sum(1 for info in infos if info.get('equity', 1.0) <= 1.0)
        bankruptcy_frac = bankruptcies / len(infos)
        
        mean_abs_advantage = np.mean(np.abs(advantages))
        
        # Track cost history
        self.cost_history.append(mean_cost_ratio)
        if len(self.cost_history) > self.tripwire_config.cost_history_window:
            self.cost_history.pop(0)
        
        running_cost_ratio = np.mean(self.cost_history)
        
        # SENIOR DEV TRIPWIRES
        if running_cost_ratio > self.tripwire_config.max_cost_ratio:
            logger.error(f"🚨 TRIPWIRE: Running cost ratio {running_cost_ratio:.4f} > {self.tripwire_config.max_cost_ratio}")
            raise RuntimeError(f"Tripwire: pathological costs over {len(self.cost_history)} updates")
        
        if bankruptcy_frac > self.tripwire_config.max_bankrupt_frac:
            logger.error(f"🚨 TRIPWIRE: Bankruptcy fraction {bankruptcy_frac:.2f} > {self.tripwire_config.max_bankrupt_frac}")
            raise RuntimeError(f"Tripwire: mass drawdowns - {bankruptcy_frac:.1%} environments bankrupt")
        
        # Log detailed diagnostics
        equities = [info.get('equity', 0.0) for info in infos]
        leverages = [info.get('leverage', 0.0) for info in infos]
        net_returns = [info.get('net_return', 0.0) for info in infos]
        gross_returns = [info.get('gross_return', 0.0) for info in infos]
        
        logger.info(f"Update {self.update_count}:")
        logger.info(f"  📊 Cost: {mean_cost_ratio:.4f} (running: {running_cost_ratio:.4f})")
        logger.info(f"  💀 Bankruptcies: {bankruptcy_frac:.1%}")
        logger.info(f"  📈 Leverage: {np.mean(leverages):.2f}±{np.std(leverages):.2f}x")
        logger.info(f"  💰 Equity: ${np.mean(equities):,.0f}±${np.std(equities):,.0f}")
        logger.info(f"  📊 Gross return: {np.mean(gross_returns):.4f}±{np.std(gross_returns):.4f}")
        logger.info(f"  📉 Net return: {np.mean(net_returns):.4f}±{np.std(net_returns):.4f}")
        logger.info(f"  🎯 Advantage: {mean_abs_advantage:.4f}")
    
    def train(self):
        """Main training loop with comprehensive tripwires"""
        logger.info("="*60)
        logger.info("PPO TRAINING - SENIOR DEV FIXED VERSION")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            for update in range(self.num_updates):
                update_start = time.time()
                
                logger.info(f"🔄 Update {update + 1}/{self.num_updates}")
                
                # Collect rollouts
                rollout_data = self.collect_rollouts()
                
                # Store experiences in agent buffer
                for i in range(len(rollout_data['observations'])):
                    self.agent.store_transition(
                        rollout_data['observations'][i],
                        rollout_data['actions'][i], 
                        rollout_data['rewards'][i],
                        rollout_data['values'][i],
                        rollout_data['log_probs'][i],
                        rollout_data['dones'][i]
                    )
                
                # PPO update using agent's built-in update method
                self.agent.update()
                
                # CRITICAL: Check tripwires after update  
                dummy_advantages = np.array(rollout_data['rewards'])  # Use rewards as proxy
                self._check_update_tripwires(rollout_data, dummy_advantages)
                
                update_time = time.time() - update_start
                logger.info(f"  ⏱️  Update time: {update_time:.1f}s")
                logger.info("")
                
        except Exception as e:
            logger.error(f"🚨 Training failed: {e}")
            logger.info(f"Completed {self.update_count}/{self.num_updates} updates before failure")
            raise
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"✅ TRAINING COMPLETED - {self.num_updates} updates in {total_time:.1f}s")
        logger.info("="*60)

def main():
    """Main training entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Trading AI with Senior Dev Fixes')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--num_updates', type=int, default=128, help='Number of training updates (start small!)')
    
    args = parser.parse_args()
    
    logger.info("🧪 SENIOR DEV FIXED PPO TRAINING")
    logger.info(f"Environments: {args.num_envs}")
    logger.info(f"Updates: {args.num_updates}")
    
    # Create and run trainer
    trainer = PPOTrainerWithTripwires(num_envs=args.num_envs, num_updates=args.num_updates)
    trainer.train()

if __name__ == "__main__":
    main()