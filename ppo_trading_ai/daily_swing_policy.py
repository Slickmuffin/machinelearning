#!/usr/bin/env python3
"""
Daily Swing Trading Policy
Friend's exact specification for "one big trade per day/week" with abstention and risk clamps

Key Features:
- Once-per-day decisions at 00:00 UTC
- Cost-aware gating with abstention as default
- Multi-clamp leverage sizing (vol-target, Kelly, liquidation, VaR)
- Positions held overnight with funding and gap costs
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DailySwingConfig:
    """Configuration for daily swing trading policy"""
    # Cost & gating parameters
    cost_bps_roundtrip: float = 7.0     # Round-trip cost in bps
    delta_bps: float = 1.5               # Safety margin (bps) -> prob
    tau: float = 0.08                    # Base logit-margin threshold
    k_disagreement: float = 0.7          # Ensemble-disagreement weight

    # Sizing (annualized targets / clamps)
    target_vol_annual: float = 0.15     # 15% annual volatility target
    kappa: float = 1.0                   # Vol target multiplier
    kelly_fraction: float = 0.20         # 20% of Kelly
    var_budget: float = 0.01             # 1-day VaR budget (1%)
    z_liq_ATR: float = 5.0              # 5x ATR liquidation distance
    zq: float = 2.33                     # 99th percentile z-score
    leverage_cap: float = 125.0          # Hard leverage limit

def ensemble_prob(calibrated_probs: List[float]) -> Tuple[float, float]:
    """
    Ensemble via logit averaging with disagreement measure
    Friend's recommended approach vs raw probability averaging

    Returns:
        ensemble_prob: Logit-averaged probability
        disagreement_std: Standard deviation in logit space
    """
    if not calibrated_probs:
        return 0.5, 0.0

    # Convert to logits with clipping
    logits = []
    for p in calibrated_probs:
        p_clipped = np.clip(p, 1e-6, 1-1e-6)
        logits.append(logit(p_clipped))

    logits = np.array(logits)

    # Logit average and disagreement
    logit_mean = logits.mean()
    logit_std = logits.std()

    # Convert back to probability
    ensemble_p = expit(logit_mean)

    return float(ensemble_p), float(logit_std)

def break_even_prob(cost_bps: float, sigma_1d: float, eps: float = 1e-6) -> float:
    """
    Compute break-even probability for daily swing trades

    For daily decisions with round-trip cost c (bps):
    E|r| = σ̂_1D * sqrt(2/π)
    p* = 0.5 + c/(2*E|r|)
    """
    sigma = max(float(sigma_1d), eps)
    E_abs = sigma * np.sqrt(2/np.pi)
    p_star = 0.5 + (cost_bps/1e4) / (2*E_abs)

    return float(np.clip(p_star, 0.5, 1.0))

def decide_and_size(p: float, sigma_1d: float, atr_1d: float,
                   config: DailySwingConfig,
                   disagreement_std: float = 0.0,
                   mu_hat: Optional[float] = None) -> Tuple[int, float]:
    """
    Daily swing trading decision with multi-clamp leverage sizing

    Returns:
        action: {-1, 0, +1} for short/flat/long
        leverage: Position size as multiple of equity
    """

    # Step 1: Cost-aware gate
    p_star = break_even_prob(config.cost_bps_roundtrip, sigma_1d)
    delta_prob = (config.delta_bps/1e4) / (2 * max(sigma_1d, 1e-8) * np.sqrt(2/np.pi))

    # Gate conditions
    long_gate = p > (p_star + delta_prob)
    short_gate = p < (1 - (p_star + delta_prob))

    if not (long_gate or short_gate):
        return 0, 0.0  # Abstain - no edge over costs

    # Step 2: Determine direction
    action = +1 if long_gate else -1

    # Step 3: Edge estimation for sizing
    if mu_hat is None:
        # Proxy: (2p-1) * E|r| - costs
        mu_hat = (2*p - 1) * (sigma_1d * np.sqrt(2/np.pi)) - (config.cost_bps_roundtrip/1e4)

    if mu_hat <= 0:
        return action, 0.0  # No positive expected edge

    # Step 4: Multi-clamp leverage sizing

    # Vol-target clamp
    L_vol = config.kappa * config.target_vol_annual / max(sigma_1d, 1e-8)

    # Fractional Kelly clamp
    L_kelly = max(0.0, config.kelly_fraction * mu_hat / max(sigma_1d**2, 1e-12))

    # Liquidation distance clamp
    L_liq = 1.0 / max(config.z_liq_ATR * atr_1d, 1e-6)

    # VaR clamp (1% daily budget)
    L_var = config.var_budget / (max(sigma_1d, 1e-8) * config.zq)

    # Take minimum of all clamps
    leverage = min(config.leverage_cap, L_vol, L_kelly, L_liq, L_var)

    # Log constraint analysis
    binding_constraint = "leverage_cap"
    if leverage == L_vol:
        binding_constraint = "vol_target"
    elif leverage == L_kelly:
        binding_constraint = "kelly"
    elif leverage == L_liq:
        binding_constraint = "liquidation"
    elif leverage == L_var:
        binding_constraint = "var"

    logger.debug(f"Leverage sizing: vol={L_vol:.2f}, kelly={L_kelly:.2f}, liq={L_liq:.2f}, "
                f"var={L_var:.2f} -> {leverage:.2f} (binding: {binding_constraint})")

    return action, float(leverage)

def run_daily_backtest(df: pd.DataFrame, models: List[Dict], feature_cols: List[str],
                      config: DailySwingConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest daily swing trading policy

    Args:
        df: Daily data with features and labels
        models: List of trained models with calibrators
        feature_cols: Feature column names
        config: Daily swing configuration

    Returns:
        ledger: Trade ledger with decisions and PnL
        metrics: Performance summary
    """
    logger.info("📊 Running daily swing backtest...")

    ledger = []
    pos_prev = 0.0

    for i in range(1, len(df)):
        # Decision made at end of previous day using t-1 features
        row_prev = df.iloc[i-1]
        row_current = df.iloc[i]

        # Get features for prediction
        X = row_prev[feature_cols].values.reshape(1, -1)

        # Get ensemble predictions
        probs = []
        for model_dict in models:
            try:
                if 'calibrator' in model_dict:
                    prob = model_dict['calibrator'].predict_proba(X)[0, 1]
                else:
                    prob = model_dict['model'].predict_proba(X)[0, 1]  # Fallback
                probs.append(prob)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue

        if not probs:
            logger.warning("No model predictions available")
            continue

        # Ensemble probability and disagreement
        p, disagreement = ensemble_prob(probs)

        # Decision and sizing
        action, leverage = decide_and_size(
            p=p,
            sigma_1d=row_prev['sigma_hat_1d'],
            atr_1d=row_prev['atr_1d'],
            config=config,
            disagreement_std=disagreement
        )

        pos_now = action * leverage

        # Realized PnL calculation
        gross_ret = row_current['gross_ret_next_k']  # Next day return
        pos_change = abs(pos_now - pos_prev)
        cost = (config.cost_bps_roundtrip/1e4) * pos_change

        # Gross PnL from position
        gross_pnl = pos_now * gross_ret

        # Funding (simplified - could be more sophisticated)
        funding = 0.0
        if 'funding_1d' in row_prev.index:
            funding = abs(pos_now) * row_prev['funding_1d']

        # Net PnL
        net_pnl = gross_pnl - cost + funding

        # Record decision
        ledger.append({
            'day': row_current['day'],
            'p': p,
            'p_star': break_even_prob(config.cost_bps_roundtrip, row_prev['sigma_hat_1d']),
            'disagreement': disagreement,
            'action': action,
            'leverage': leverage,
            'pos_prev': pos_prev,
            'pos_now': pos_now,
            'sigma_1d': row_prev['sigma_hat_1d'],
            'atr_1d': row_prev['atr_1d'],
            'gross_ret': gross_ret,
            'cost': cost,
            'funding': funding,
            'net_pnl': net_pnl
        })

        pos_prev = pos_now

    # Convert to DataFrame
    ledger_df = pd.DataFrame(ledger)

    if len(ledger_df) == 0:
        logger.warning("Empty ledger - no trades executed")
        return ledger_df, {}

    # Calculate performance metrics
    returns = ledger_df['net_pnl']

    # Basic metrics
    total_return = returns.sum()
    sharpe = (returns.mean() / max(returns.std(), 1e-8)) * np.sqrt(252)

    # Win rate and profit factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / max(len(returns), 1)
    profit_factor = winning_trades.sum() / max(-losing_trades.sum(), 1e-8)

    # Drawdown
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = -drawdown.min()

    # Trading frequency
    total_trades = (ledger_df['action'] != 0).sum()
    trade_frequency = total_trades / len(ledger_df)

    # Average leverage
    active_trades = ledger_df[ledger_df['action'] != 0]
    avg_leverage = active_trades['leverage'].mean() if len(active_trades) > 0 else 0.0

    metrics = {
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'win_rate_pct': win_rate * 100,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown * 100,
        'total_trades': int(total_trades),
        'trade_frequency_pct': trade_frequency * 100,
        'avg_leverage': avg_leverage,
        'mean_bps_per_day': returns.mean() * 1e4,
        'days_traded': len(ledger_df)
    }

    logger.info("📊 Daily swing backtest complete:")
    logger.info(f"   💰 Total Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"   📈 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"   🎯 Win Rate: {metrics['win_rate_pct']:.1f}%")
    logger.info(f"   📊 Total Trades: {metrics['total_trades']} ({metrics['trade_frequency_pct']:.1f}% frequency)")
    logger.info(f"   ⚖️ Avg Leverage: {metrics['avg_leverage']:.1f}x")

    return ledger_df, metrics

class DailySwingTrader:
    """
    Daily swing trading system with ensemble models and risk management
    """

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.models: List[Dict] = []
        self.is_fitted = False

    def add_model(self, model, calibrator=None):
        """Add a trained model to the ensemble"""
        self.models.append({
            'model': model,
            'calibrator': calibrator or model
        })

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Generate ensemble prediction for a single day"""
        if not self.is_fitted:
            raise ValueError("System not fitted yet")

        # Get predictions from all models
        probs = []
        for model_dict in self.models:
            try:
                if 'calibrator' in model_dict:
                    prob = model_dict['calibrator'].predict_proba(features.values)[0, 1]
                else:
                    prob = model_dict['model'].predict_proba(features.values)[0, 1]
                probs.append(prob)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue

        if not probs:
            return {'p': 0.5, 'disagreement': 0.0}

        p, disagreement = ensemble_prob(probs)
        return {'p': p, 'disagreement': disagreement}

    def decide(self, prediction: Dict[str, float], sigma_1d: float, atr_1d: float) -> Dict[str, Any]:
        """Make trading decision for a single day"""
        action, leverage = decide_and_size(
            p=prediction['p'],
            sigma_1d=sigma_1d,
            atr_1d=atr_1d,
            config=self.config,
            disagreement_std=prediction['disagreement']
        )

        return {
            'action': action,
            'leverage': leverage,
            'p': prediction['p'],
            'p_star': break_even_prob(self.config.cost_bps_roundtrip, sigma_1d),
            'disagreement': prediction['disagreement']
        }

    def fit_ensemble(self, models: List[Dict]):
        """Fit the ensemble with trained models"""
        self.models = models
        self.is_fitted = True
        logger.info(f"✅ Daily swing ensemble fitted with {len(models)} models")