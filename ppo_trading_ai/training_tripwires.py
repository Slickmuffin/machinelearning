"""
Training Tripwires and Economic Guards - Senior Developer Requirement
Fail-fast detection of pathological costs, mass bankruptcies, and broken economics.

Senior Dev: "Abort update if mean(info["cost"]/info["equity_pre"]) > 0.01 over last K steps."
"""
import numpy as np
from typing import List, Dict, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EconomicTripwires:
    """
    Economic tripwires to stop bad training jobs fast.
    
    Senior Dev Requirements:
    - Abort if cost_ratio > 0.01 (1% of equity per step)
    - Abort if bankrupt_envs > 0.5 (50% bankruptcy rate)
    - Alert if mean_abs_advantage collapses while high turnover
    """
    
    def __init__(self, 
                 cost_ratio_threshold: float = 0.01,
                 bankruptcy_threshold: float = 0.5,
                 turnover_threshold: float = 0.2,
                 advantage_collapse_threshold: float = 0.001,
                 lookback_steps: int = 100):
        """
        Initialize tripwires with senior dev thresholds.
        
        Args:
            cost_ratio_threshold: Max cost/equity ratio (default 1%)
            bankruptcy_threshold: Max fraction of bankrupt envs (default 50%)  
            turnover_threshold: High turnover threshold for advantage check
            advantage_collapse_threshold: Minimum mean absolute advantage
            lookback_steps: Number of steps to track for running averages
        """
        self.cost_ratio_threshold = cost_ratio_threshold
        self.bankruptcy_threshold = bankruptcy_threshold
        self.turnover_threshold = turnover_threshold
        self.advantage_collapse_threshold = advantage_collapse_threshold
        self.lookback_steps = lookback_steps
        
        # Tracking deques for running averages
        self.cost_ratios = deque(maxlen=lookback_steps)
        self.bankruptcy_rates = deque(maxlen=lookback_steps)
        self.turnover_rates = deque(maxlen=lookback_steps)
        self.advantage_magnitudes = deque(maxlen=lookback_steps)
        
        # Cumulative statistics
        self.total_updates = 0
        self.total_violations = 0
        self.violation_history = []

    def check_step_infos(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check environment step infos for economic violations.
        
        Args:
            infos: List of info dicts from vectorized environment step
            
        Returns:
            Dict with violation status and diagnostics
        """
        if not infos:
            return {"violations": [], "safe": True}
        
        # Extract metrics from infos
        cost_ratios = []
        equities = []
        turnover_rates = []
        bankruptcies = 0
        
        for info in infos:
            equity = info.get("equity", 1.0)
            cost = info.get("trading_cost", 0.0)
            turnover = info.get("traded_notional", 0.0)
            
            # Track bankruptcies (equity <= 0 or extreme drawdown)
            if equity <= 0.0 or info.get("drawdown", 0.0) >= 0.9:
                bankruptcies += 1
            
            # Cost ratio (cost / equity)
            if equity > 0:
                cost_ratio = cost / equity
                cost_ratios.append(cost_ratio)
                
                # Turnover ratio (turnover / equity)
                turnover_ratio = turnover / equity
                turnover_rates.append(turnover_ratio)
                
                equities.append(equity)
        
        # Calculate aggregated metrics
        mean_cost_ratio = np.mean(cost_ratios) if cost_ratios else 0.0
        mean_turnover_rate = np.mean(turnover_rates) if turnover_rates else 0.0
        bankruptcy_rate = bankruptcies / len(infos) if infos else 0.0
        
        # Update tracking deques
        self.cost_ratios.append(mean_cost_ratio)
        self.turnover_rates.append(mean_turnover_rate)
        self.bankruptcy_rates.append(bankruptcy_rate)
        
        # Check for violations
        violations = []
        
        # Violation 1: Excessive cost ratio
        if mean_cost_ratio > self.cost_ratio_threshold:
            violations.append({
                "type": "excessive_cost_ratio",
                "value": mean_cost_ratio,
                "threshold": self.cost_ratio_threshold,
                "message": f"Cost ratio {mean_cost_ratio:.4f} exceeds {self.cost_ratio_threshold:.4f} (pathological costs)"
            })
        
        # Violation 2: Mass bankruptcies
        if bankruptcy_rate > self.bankruptcy_threshold:
            violations.append({
                "type": "mass_bankruptcy",
                "value": bankruptcy_rate,
                "threshold": self.bankruptcy_threshold,
                "message": f"Bankruptcy rate {bankruptcy_rate:.2%} exceeds {self.bankruptcy_threshold:.2%} (mass drawdowns)"
            })
        
        # Violation 3: High turnover with low variance (advantage collapse)
        if (len(self.turnover_rates) >= 10 and 
            np.mean(list(self.turnover_rates)[-10:]) > self.turnover_threshold):
            violations.append({
                "type": "high_turnover_warning", 
                "value": mean_turnover_rate,
                "threshold": self.turnover_threshold,
                "message": f"High turnover {mean_turnover_rate:.2%} may indicate churning"
            })
        
        # Diagnostic information
        diagnostics = {
            "mean_cost_ratio": mean_cost_ratio,
            "mean_turnover_rate": mean_turnover_rate,
            "bankruptcy_rate": bankruptcy_rate,
            "mean_equity": np.mean(equities) if equities else 0.0,
            "num_bankruptcies": bankruptcies,
            "num_envs": len(infos)
        }
        
        return {
            "violations": violations,
            "safe": len(violations) == 0,
            "diagnostics": diagnostics
        }

    def check_ppo_update(self, 
                        advantages: np.ndarray,
                        infos: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check PPO update for advantage collapse and other issues.
        
        Args:
            advantages: PPO advantages array
            infos: Optional infos from recent rollout
            
        Returns:
            Dict with violation status and diagnostics
        """
        self.total_updates += 1
        
        # Calculate advantage statistics
        mean_abs_advantage = np.mean(np.abs(advantages))
        advantage_std = np.std(advantages)
        advantage_range = np.max(advantages) - np.min(advantages)
        
        self.advantage_magnitudes.append(mean_abs_advantage)
        
        violations = []
        
        # Check for advantage collapse
        if mean_abs_advantage < self.advantage_collapse_threshold:
            # Only flag if we also have high turnover (indicates churning without learning)
            recent_turnover = np.mean(list(self.turnover_rates)[-10:]) if len(self.turnover_rates) >= 10 else 0.0
            
            if recent_turnover > self.turnover_threshold:
                violations.append({
                    "type": "advantage_collapse",
                    "value": mean_abs_advantage,
                    "threshold": self.advantage_collapse_threshold,
                    "message": f"Advantage magnitude {mean_abs_advantage:.6f} collapsed while turnover {recent_turnover:.2%} high (possible churning)"
                })
        
        # Running averages for logging
        running_cost_ratio = np.mean(list(self.cost_ratios)) if self.cost_ratios else 0.0
        running_bankruptcy_rate = np.mean(list(self.bankruptcy_rates)) if self.bankruptcy_rates else 0.0
        
        diagnostics = {
            "update_number": self.total_updates,
            "mean_abs_advantage": mean_abs_advantage,
            "advantage_std": advantage_std,
            "advantage_range": advantage_range,
            "running_cost_ratio": running_cost_ratio,
            "running_bankruptcy_rate": running_bankruptcy_rate,
            "running_turnover_rate": np.mean(list(self.turnover_rates)) if self.turnover_rates else 0.0
        }
        
        return {
            "violations": violations,
            "safe": len(violations) == 0,
            "diagnostics": diagnostics
        }

    def should_abort_training(self, recent_violations: int = 3) -> bool:
        """
        Determine if training should be aborted based on recent violations.
        
        Args:
            recent_violations: Number of recent violations to trigger abort
            
        Returns:
            True if training should be aborted
        """
        if len(self.violation_history) < recent_violations:
            return False
        
        # Check last N updates for consistent violations
        recent_checks = self.violation_history[-recent_violations:]
        violation_count = sum(1 for check in recent_checks if not check.get("safe", True))
        
        return violation_count >= recent_violations

    def log_violation(self, violation_check: Dict[str, Any]):
        """Log violation check result for history tracking."""
        self.violation_history.append(violation_check)
        
        if not violation_check.get("safe", True):
            self.total_violations += 1
            
            for violation in violation_check.get("violations", []):
                logger.warning(f"🚨 TRIPWIRE: {violation['message']}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        return {
            "total_updates": self.total_updates,
            "total_violations": self.total_violations,
            "violation_rate": self.total_violations / max(self.total_updates, 1),
            "current_cost_ratio": list(self.cost_ratios)[-1] if self.cost_ratios else 0.0,
            "current_bankruptcy_rate": list(self.bankruptcy_rates)[-1] if self.bankruptcy_rates else 0.0,
            "current_turnover_rate": list(self.turnover_rates)[-1] if self.turnover_rates else 0.0,
            "mean_advantage_magnitude": list(self.advantage_magnitudes)[-1] if self.advantage_magnitudes else 0.0
        }


def create_training_guard(config_overrides: Dict[str, Any] = None) -> EconomicTripwires:
    """
    Create training guard with senior dev recommended thresholds.
    
    Args:
        config_overrides: Optional threshold overrides
        
    Returns:
        Configured EconomicTripwires instance
    """
    defaults = {
        "cost_ratio_threshold": 0.01,      # 1% cost per step (senior dev spec)
        "bankruptcy_threshold": 0.5,       # 50% bankruptcy rate (senior dev spec)
        "turnover_threshold": 0.2,         # 20% turnover per step
        "advantage_collapse_threshold": 0.001,  # Minimum advantage magnitude
        "lookback_steps": 100               # 100 steps for running averages
    }
    
    if config_overrides:
        defaults.update(config_overrides)
    
    return EconomicTripwires(**defaults)


# Example usage in training loop
def training_loop_with_tripwires():
    """
    Example of integrating tripwires into PPO training loop.
    This is a template - actual implementation would be in the training script.
    """
    tripwires = create_training_guard()
    
    for update in range(1000):
        # Collect rollout data
        # observations, actions, rewards, dones, infos = collect_rollout()
        
        # Check step-level violations
        # step_check = tripwires.check_step_infos(infos)
        # tripwires.log_violation(step_check)
        
        # Compute PPO advantages
        # advantages = compute_advantages(rewards, values, dones)
        
        # Check update-level violations  
        # update_check = tripwires.check_ppo_update(advantages, infos)
        # tripwires.log_violation(update_check)
        
        # Check if we should abort
        # if tripwires.should_abort_training():
        #     logger.error("🚨 ABORTING TRAINING: Consistent tripwire violations detected")
        #     break
        
        # Continue with PPO update if safe
        # ppo_update(advantages, ...)
        
        # Log summary every 50 updates
        # if update % 50 == 0:
        #     stats = tripwires.get_summary_stats()
        #     logger.info(f"Tripwire Stats: {stats}")
        pass  # Placeholder for actual implementation