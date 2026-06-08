#!/usr/bin/env python3
"""
Cost-Aware Policy Engine for Leverage BTC Trading
Phase 3: Tri-state decision logic with abstention and consensus

Policy Architecture:
- Break-even probability calculation from trading costs and expected moves
- Tri-state decisions: Long/Short/Flat with abstention logic
- Consensus gate requiring ensemble agreement
- Action-cost alignment with no-trade bands and hysteresis
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TradingAction(Enum):
    """Trading actions with clear semantics"""
    LONG = 1
    FLAT = 0
    SHORT = -1

@dataclass
class TradingCosts:
    """Comprehensive trading cost structure"""
    # Base costs (in basis points, i.e., 0.0001 = 1bp)
    taker_fee_bp: float = 3.0  # 3bp taker fee
    maker_fee_bp: float = 1.0  # 1bp maker fee (rarely used)
    slippage_bp: float = 0.5   # 0.5bp slippage

    # Funding costs
    funding_rate_8h_bp: float = 1.0  # 1bp per 8h
    step_hours: float = 4.0  # 4h bars

    # Friction penalties
    turnover_penalty_bp: float = 1.0  # 1bp per 10x turnover
    leverage_penalty_bp: float = 0.2  # 0.2bp per 1x leverage

    def total_execution_cost_bp(self, assume_taker: bool = True) -> float:
        """Total round-trip execution cost in basis points"""
        base_fee = self.taker_fee_bp if assume_taker else self.maker_fee_bp
        return base_fee + self.slippage_bp

    def funding_cost_per_step_bp(self) -> float:
        """Funding cost per step in basis points"""
        return self.funding_rate_8h_bp * (self.step_hours / 8.0)

@dataclass
class PolicyConfig:
    """Configuration for cost-aware policy engine"""
    # Cost structure
    costs: TradingCosts = None

    # Break-even calculation
    vol_estimator_span: int = 64  # Bars for volatility estimation
    safety_margin_bp: float = 1.0  # Additional safety margin in bp

    # Consensus requirements
    consensus_tau: float = 0.08  # Base logit margin threshold
    consensus_k: float = 0.7    # Uncertainty scaling factor
    min_ensemble_agreement: float = 0.6  # Minimum fraction of models agreeing

    # Action-cost alignment (from senior dev specs)
    no_trade_band_pct: float = 0.05  # Ignore |ΔN| < 5% of equity
    turnover_cap_pct: float = 1.0    # |ΔN| ≤ 100% of equity per step
    hysteresis_threshold_pct: float = 0.5  # Min |ΔN| for sign flips

    # Risk management
    max_leverage: float = 3.0
    volatility_cap: float = 0.1  # 10% daily vol cap

    def __post_init__(self):
        if self.costs is None:
            self.costs = TradingCosts()

@dataclass
class PolicySignal:
    """Rich policy signal with full context"""
    action: TradingAction
    confidence: float  # 0-1 confidence score
    target_leverage: float  # Signed leverage (-3 to +3)
    probability: float  # Ensemble probability
    uncertainty: float  # Ensemble disagreement
    break_even_prob: float  # Required p* for profitability
    safety_margin_met: bool  # Whether signal exceeds safety margin
    consensus_score: float  # Ensemble consensus strength
    vol_estimate: float  # Current volatility estimate
    expected_move: float  # Expected absolute move E|r|
    reasoning: str  # Human-readable reasoning

class CostAwarePolicy:
    """
    Cost-aware policy engine implementing tri-state decisions with abstention
    """

    def __init__(self, config: PolicyConfig = None):
        self.config = config or PolicyConfig()

        # State tracking for hysteresis
        self.current_position: float = 0.0  # Current leverage
        self.last_action: TradingAction = TradingAction.FLAT

        logger.info("🧠 Cost-aware policy engine initialized")
        logger.info(f"   💰 Total execution cost: {self.config.costs.total_execution_cost_bp():.1f}bp")
        logger.info(f"   📈 Safety margin: {self.config.safety_margin_bp:.1f}bp")
        logger.info(f"   🤝 Consensus threshold: τ={self.config.consensus_tau:.3f}, k={self.config.consensus_k:.1f}")

    def calculate_break_even_probability(self, vol_estimate: float) -> Tuple[float, float]:
        """
        Calculate break-even probability p* from costs and expected moves

        Returns:
            (break_even_prob, expected_abs_move)
        """
        # Expected absolute move per bar: E|r| = σ√(2/π)
        expected_abs_move = vol_estimate * np.sqrt(2 / np.pi)

        # Round-trip cost in return units (convert bp to decimal)
        cost_return_units = self.config.costs.total_execution_cost_bp() / 10000.0

        # Break-even probability: p* = 0.5 + c/(2·E|r|)
        if expected_abs_move > 0:
            break_even_prob = 0.5 + cost_return_units / (2 * expected_abs_move)
        else:
            break_even_prob = 1.0  # If no expected move, impossible to profit

        return break_even_prob, expected_abs_move

    def assess_consensus(self, ensemble_prob: float, ensemble_uncertainty: float,
                        ensemble_logits: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Assess whether ensemble has sufficient consensus for trading

        Returns:
            (consensus_met, consensus_score)
        """
        # Simple consensus check using uncertainty
        uncertainty_threshold = self.config.consensus_tau + self.config.consensus_k * ensemble_uncertainty
        consensus_met = ensemble_uncertainty <= uncertainty_threshold

        # Consensus score (higher = more consensus)
        consensus_score = max(0.0, 1.0 - (ensemble_uncertainty / uncertainty_threshold))

        return consensus_met, consensus_score

    def check_action_cost_alignment(self, target_leverage: float, current_equity: float) -> Tuple[float, str]:
        """
        Apply action-cost alignment rules (senior dev specifications)

        Returns:
            (adjusted_leverage, reasoning)
        """
        # Current notional and target notional
        current_notional = self.current_position * current_equity
        target_notional = target_leverage * current_equity

        delta_n_required = target_notional - current_notional

        # No-trade band: ignore actions that imply |ΔN| < 5% of equity
        no_trade_threshold = self.config.no_trade_band_pct * current_equity
        if abs(delta_n_required) < no_trade_threshold:
            return self.current_position, f"No-trade band: |ΔN|={abs(delta_n_required):.0f} < {no_trade_threshold:.0f}"

        # Hysteresis on sign flips
        if np.sign(target_leverage) != np.sign(self.current_position):
            hysteresis_threshold = self.config.hysteresis_threshold_pct * current_equity
            if abs(delta_n_required) < hysteresis_threshold:
                return self.current_position, f"Hysteresis: sign flip requires |ΔN| ≥ {hysteresis_threshold:.0f}"

        # Turnover cap: |ΔN| ≤ 100% of equity per step
        max_delta = self.config.turnover_cap_pct * current_equity
        if abs(delta_n_required) > max_delta:
            # Clip the change
            delta_n_capped = np.clip(delta_n_required, -max_delta, max_delta)
            adjusted_notional = current_notional + delta_n_capped
            adjusted_leverage = adjusted_notional / current_equity
            return adjusted_leverage, f"Turnover capped: |ΔN| limited to {max_delta:.0f}"

        return target_leverage, "Action-cost alignment passed"

    def generate_signal(self, ensemble_prob: float, ensemble_uncertainty: float,
                       vol_estimate: float, current_equity: float = 10000.0,
                       ensemble_logits: Optional[np.ndarray] = None) -> PolicySignal:
        """
        Generate comprehensive trading signal with cost-aware logic

        Args:
            ensemble_prob: Ensemble probability (0-1)
            ensemble_uncertainty: Ensemble disagreement/uncertainty
            vol_estimate: Current volatility estimate
            current_equity: Current account equity
            ensemble_logits: Individual model logits (for advanced consensus)

        Returns:
            PolicySignal with action, confidence, and full reasoning
        """

        # 1) Calculate break-even probability
        break_even_prob, expected_move = self.calculate_break_even_probability(vol_estimate)

        # Add safety margin
        safety_margin_decimal = self.config.safety_margin_bp / 10000.0
        safety_adjusted_threshold = break_even_prob + (safety_margin_decimal / (2 * expected_move))

        # 2) Assess consensus
        consensus_met, consensus_score = self.assess_consensus(ensemble_prob, ensemble_uncertainty, ensemble_logits)

        # 3) Make tri-state decision
        reasoning_parts = []

        # Long condition: p > p* + δ AND consensus strong
        long_threshold = safety_adjusted_threshold
        short_threshold = 1.0 - safety_adjusted_threshold

        if ensemble_prob > long_threshold and consensus_met:
            raw_action = TradingAction.LONG
            # Scale leverage by confidence (distance from threshold)
            confidence = min(1.0, (ensemble_prob - long_threshold) / (1.0 - long_threshold))
            raw_leverage = confidence * self.config.max_leverage
            reasoning_parts.append(f"Long: p={ensemble_prob:.3f} > {long_threshold:.3f}")

        elif ensemble_prob < short_threshold and consensus_met:
            raw_action = TradingAction.SHORT
            # Scale leverage by confidence (distance from threshold)
            confidence = min(1.0, (short_threshold - ensemble_prob) / short_threshold)
            raw_leverage = -confidence * self.config.max_leverage
            reasoning_parts.append(f"Short: p={ensemble_prob:.3f} < {short_threshold:.3f}")

        else:
            raw_action = TradingAction.FLAT
            confidence = 0.0
            raw_leverage = 0.0
            if not consensus_met:
                reasoning_parts.append(f"Flat: No consensus (unc={ensemble_uncertainty:.3f})")
            else:
                reasoning_parts.append(f"Flat: p={ensemble_prob:.3f} in [{short_threshold:.3f}, {long_threshold:.3f}]")

        # 4) Apply action-cost alignment
        adjusted_leverage, alignment_reason = self.check_action_cost_alignment(raw_leverage, current_equity)
        reasoning_parts.append(alignment_reason)

        # Determine final action from adjusted leverage
        if adjusted_leverage > 0.05:
            final_action = TradingAction.LONG
        elif adjusted_leverage < -0.05:
            final_action = TradingAction.SHORT
        else:
            final_action = TradingAction.FLAT

        # 5) Safety checks
        safety_margin_met = (
            (raw_action == TradingAction.LONG and ensemble_prob > safety_adjusted_threshold) or
            (raw_action == TradingAction.SHORT and ensemble_prob < (1.0 - safety_adjusted_threshold)) or
            (raw_action == TradingAction.FLAT)
        )

        # Volatility cap
        if vol_estimate > self.config.volatility_cap:
            adjusted_leverage *= 0.5  # Reduce leverage in high vol
            reasoning_parts.append(f"High vol: leverage reduced (σ={vol_estimate:.3f} > {self.config.volatility_cap:.3f})")

        # 6) Create signal
        signal = PolicySignal(
            action=final_action,
            confidence=confidence,
            target_leverage=adjusted_leverage,
            probability=ensemble_prob,
            uncertainty=ensemble_uncertainty,
            break_even_prob=break_even_prob,
            safety_margin_met=safety_margin_met,
            consensus_score=consensus_score,
            vol_estimate=vol_estimate,
            expected_move=expected_move,
            reasoning=" | ".join(reasoning_parts)
        )

        # Update state
        self.last_action = final_action
        # Note: current_position should be updated externally after trade execution

        return signal

    def update_position(self, new_leverage: float) -> None:
        """Update current position after trade execution"""
        self.current_position = new_leverage
        logger.debug(f"Position updated: leverage={new_leverage:.3f}")

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy configuration and current state statistics"""
        return {
            'current_position': self.current_position,
            'last_action': self.last_action.name,
            'execution_cost_bp': self.config.costs.total_execution_cost_bp(),
            'funding_cost_per_step_bp': self.config.costs.funding_cost_per_step_bp(),
            'safety_margin_bp': self.config.safety_margin_bp,
            'consensus_tau': self.config.consensus_tau,
            'no_trade_band_pct': self.config.no_trade_band_pct,
            'max_leverage': self.config.max_leverage
        }

# Utility functions for policy integration

def create_policy_signals_batch(policy: CostAwarePolicy,
                               ensemble_probs: np.ndarray,
                               ensemble_uncertainties: np.ndarray,
                               vol_estimates: np.ndarray,
                               equities: Optional[np.ndarray] = None) -> List[PolicySignal]:
    """
    Generate policy signals for a batch of predictions
    Useful for backtesting and bulk signal generation
    """
    if equities is None:
        equities = np.full(len(ensemble_probs), 10000.0)

    signals = []
    for i in range(len(ensemble_probs)):
        signal = policy.generate_signal(
            ensemble_prob=ensemble_probs[i],
            ensemble_uncertainty=ensemble_uncertainties[i],
            vol_estimate=vol_estimates[i],
            current_equity=equities[i]
        )
        signals.append(signal)

        # Update policy state (for hysteresis)
        if signal.action != TradingAction.FLAT:
            policy.update_position(signal.target_leverage)

    return signals

def analyze_policy_performance(signals: List[PolicySignal],
                             actual_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Analyze policy signal quality and characteristics
    """
    if not signals:
        return {}

    # Basic statistics
    actions = [s.action for s in signals]
    long_pct = (np.array([a == TradingAction.LONG for a in actions]).mean()) * 100
    short_pct = (np.array([a == TradingAction.SHORT for a in actions]).mean()) * 100
    flat_pct = (np.array([a == TradingAction.FLAT for a in actions]).mean()) * 100

    # Confidence and leverage statistics
    confidences = [s.confidence for s in signals]
    leverages = [abs(s.target_leverage) for s in signals]

    analysis = {
        'total_signals': len(signals),
        'long_pct': long_pct,
        'short_pct': short_pct,
        'flat_pct': flat_pct,
        'mean_confidence': np.mean(confidences),
        'mean_abs_leverage': np.mean(leverages),
        'max_leverage': max(leverages) if leverages else 0.0,
        'safety_margin_met_pct': (np.array([s.safety_margin_met for s in signals]).mean()) * 100,
        'consensus_met_pct': (np.array([s.consensus_score > 0.5 for s in signals]).mean()) * 100,
    }

    # If actual returns provided, calculate directional accuracy
    if actual_returns is not None and len(actual_returns) == len(signals):
        # Directional accuracy for non-flat signals
        non_flat_mask = np.array([s.action != TradingAction.FLAT for s in signals])
        if non_flat_mask.sum() > 0:
            signal_directions = np.array([s.action.value for s in signals])[non_flat_mask]
            actual_directions = np.sign(actual_returns[non_flat_mask])
            directional_accuracy = (signal_directions == actual_directions).mean() * 100
            analysis['directional_accuracy_pct'] = directional_accuracy
            analysis['total_trades'] = non_flat_mask.sum()

    return analysis

if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)

    # Test policy with sample data
    policy = CostAwarePolicy()

    # Generate sample signal
    signal = policy.generate_signal(
        ensemble_prob=0.65,  # 65% probability
        ensemble_uncertainty=0.1,  # 10% uncertainty
        vol_estimate=0.02,  # 2% volatility
        current_equity=10000.0
    )

    logger.info(f"Sample signal: {signal.action.name} @ {signal.target_leverage:.2f}x")
    logger.info(f"Reasoning: {signal.reasoning}")
    logger.info("✅ Cost-aware policy engine test complete")