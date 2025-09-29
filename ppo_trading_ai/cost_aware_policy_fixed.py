#!/usr/bin/env python3
"""
Cost-aware policy fixes from quant friend
Implements proper breakeven probability calculation and cost-aware thresholds
"""

import numpy as np
from scipy.special import logit, expit
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def break_even_prob(cost_bps: np.ndarray, sigma_hat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute break-even probability with sigma floor to prevent explosions
    Friend's fix for infinite p* values when sigma_hat = 0
    """
    sigma = np.maximum(sigma_hat, eps)
    E_abs = sigma * np.sqrt(2/np.pi)
    return 0.5 + (cost_bps/1e4) / (2*E_abs)

def ensemble_prob(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble via logit average + disagreement measure
    Friend's recommended approach vs raw averaging

    Args:
        probs: Shape [n_samples, n_models] probability matrix

    Returns:
        ensemble_prob: [n_samples] logit-averaged probabilities
        disagreement: [n_samples] logit standard deviations
    """
    # Clip to prevent logit explosion
    probs_clipped = np.clip(probs, 1e-6, 1-1e-6)

    # Convert to logits
    logits = np.array([logit(p) for p in probs_clipped.T])  # [n_models, n_samples]

    # Logit average and disagreement
    logit_mean = logits.mean(axis=0)
    logit_std = logits.std(axis=0)

    # Convert back to probabilities
    ensemble_p = expit(logit_mean)

    return ensemble_p, logit_std

def decide_cost_aware(p: np.ndarray, p_star: np.ndarray, delta_prob: float,
                     z_std: Optional[np.ndarray] = None, tau: float = 0.08, k: float = 0.7) -> np.ndarray:
    """
    Cost-aware tri-state policy decision
    Friend's implementation with proper cost gating

    Args:
        p: Model probabilities [n_samples]
        p_star: Break-even probabilities [n_samples]
        delta_prob: Additional margin (converted to probability space)
        z_std: Logit disagreement [n_samples] (optional)
        tau: Base threshold in logit space
        k: Disagreement threshold multiplier

    Returns:
        actions: {-1, 0, +1} decisions [n_samples]
    """
    # Cost-aware gates
    long_gate = p > (p_star + delta_prob)
    short_gate = p < (1 - (p_star + delta_prob))

    # Apply disagreement filter if available
    if z_std is not None:
        # Convert p to logit space for threshold comparison
        p_logit = logit(np.clip(p, 1e-6, 1-1e-6))
        p_star_logit = logit(np.clip(p_star, 1e-6, 1-1e-6))

        # Disagreement-based filtering
        low_confidence = z_std > (tau + k * np.abs(p_logit - p_star_logit))

        # Override gates if disagreement too high
        long_gate = long_gate & ~low_confidence
        short_gate = short_gate & ~low_confidence

    # Tri-state decisions
    actions = np.zeros(len(p))
    actions[long_gate] = 1
    actions[short_gate] = -1

    return actions

def pnl_bar(pos_prev: np.ndarray, pos_now: np.ndarray, r_next: np.ndarray,
           cost_bps: np.ndarray, funding: np.ndarray = None) -> np.ndarray:
    """
    PnL kernel - costs only on |Δposition|
    Friend's implementation ensuring costs applied correctly
    """
    # Position change
    dpos = np.abs(pos_now - pos_prev)

    # Trading costs (only on position changes)
    cost = (cost_bps/1e4) * dpos

    # Gross PnL from position * return
    gross_pnl = pos_now * r_next

    # Funding (default to zero if not provided)
    if funding is None:
        funding = np.zeros_like(pos_now)

    # Net PnL
    net_pnl = gross_pnl - cost + funding

    return net_pnl

def optimize_threshold_oos(df_oos: pd.DataFrame, p_col: str = 'p_ens', p_star_col: str = 'p_star',
                          cost_col: str = 'cost_bps_roundtrip', return_col: str = 'gross_ret_next',
                          delta_range: np.ndarray = None) -> pd.DataFrame:
    """
    Optimize delta threshold for maximum net expectancy on OOS data
    Friend's approach: optimize for net bps, not AUC
    """
    if delta_range is None:
        delta_range = np.linspace(0.0, 0.05, 51)  # 0..5% prob margin

    results = []

    for delta_prob in delta_range:
        # Apply cost-aware policy
        actions = decide_cost_aware(
            df_oos[p_col].values,
            df_oos[p_star_col].values,
            delta_prob
        )

        # Compute PnL
        pos_changes = np.abs(np.diff(np.r_[0, actions]))
        costs = (df_oos[cost_col].values/1e4) * np.r_[pos_changes, 0]

        net_returns = actions * df_oos[return_col].values - costs

        # Metrics
        mean_bps = net_returns.mean() * 1e4
        activity = (actions != 0).mean()

        results.append({
            'delta_prob': delta_prob,
            'mean_bps': mean_bps,
            'activity': activity,
            'total_trades': int((actions != 0).sum())
        })

    df_grid = pd.DataFrame(results)
    return df_grid.sort_values('mean_bps', ascending=False)

def calibration_diagnostics(df_oos: pd.DataFrame, p_col: str = 'p_ens',
                          return_col: str = 'gross_ret_next', n_bins: int = 10) -> pd.DataFrame:
    """
    Compute calibration diagnostics by probability bins
    Friend's diagnostic to check if probabilities are calibrated
    """
    # Probability bins (focus on >0.5 where we trade)
    bins = np.linspace(0.5, 1.0, n_bins + 1)

    results = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (df_oos[p_col] >= lo) & (df_oos[p_col] < hi)

        if mask.sum() == 0:
            continue

        subset = df_oos.loc[mask]

        # Calibration metrics
        n = int(mask.sum())
        mean_pred = subset[p_col].mean()
        emp_freq = (subset[return_col] > 0).mean()
        mean_gross_bps = subset[return_col].mean() * 1e4

        results.append({
            'bin_lo': lo,
            'bin_hi': hi,
            'n': n,
            'mean_pred': mean_pred,
            'emp_freq': emp_freq,
            'mean_gross_bps': mean_gross_bps
        })

    return pd.DataFrame(results)

def apply_friend_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all friend's critical fixes to dataset
    """
    df = df.copy()

    logger.info("🔧 Applying friend's critical fixes...")

    # 1. Fix sigma_hat zeros (already done in data_pipeline.py)
    if 'sigma_hat' in df.columns:
        df['sigma_hat'] = np.maximum(df['sigma_hat'], 1e-6)
        logger.info(f"✅ Fixed sigma_hat zeros: min={df['sigma_hat'].min():.2e}")

    # 2. Compute proper break-even probabilities
    if 'cost_bps_roundtrip' in df.columns and 'sigma_hat' in df.columns:
        df['p_star'] = break_even_prob(df['cost_bps_roundtrip'], df['sigma_hat'])
        df['E_abs_r'] = df['sigma_hat'] * np.sqrt(2/np.pi)
        logger.info(f"✅ Computed p_star: mean={df['p_star'].mean():.3f}")

    # 3. Add gross_ret_next if missing (for validation)
    if 'gross_ret_next' not in df.columns and 'close' in df.columns:
        df['gross_ret_next'] = df['close'].pct_change().shift(-1)
        logger.info("✅ Added gross_ret_next for validation")

    logger.info("🎯 Friend's fixes applied successfully")
    return df