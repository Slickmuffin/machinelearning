"""
Canonical Trading Configuration - Senior Developer Fix
Implements USD notional system to fix catastrophic cost calculation bug.
"""
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """
    Canonical trading configuration with USD notional system.
    
    Key Fix: All positions stored as signed USD notional (N), not leverage ratios.
    Fees/slippage calculated on turnover |ΔN|, preventing 29x cost overrun.
    """
    initial_balance: float = 10_000.0
    max_leverage: float = 3.0                   # L_max (safe for training)
    maker_fee: float = 0.0002                   # 2 bps
    taker_fee: float = 0.0004                   # 4 bps  
    funding_rate_8h: float = 0.0001             # 1 bp / 8h
    step_hours: float = 1.0                     # env step size
    slippage_bp: float = 1.0                    # 1 bp per turnover
    turnover_cap: float = 1.0                   # |ΔN| ≤ cap * E_pre
    cost_cap_frac: float = 0.10                 # per-trade sanity cap
    drawdown_stop: float = 0.50                 # training kill switch
    
    # Reward shaping parameters
    lambda_turnover: float = 5e-4               # turnover penalty
    lambda_leverage: float = 1e-4               # leverage penalty
    reward_clip: float = 0.1                    # clip rewards to [-0.1, 0.1]
    
    # Risk control parameters
    min_flip_fraction: float = 0.5              # minimum position flip size
    vol_lookback: int = 20                      # volatility calculation window