#!/usr/bin/env python3
"""
Experiment B Trading Environment - Lighter Penalties for Profitability

Built on the FIXED environment (no compounding leverage bug).
Additional improvements based on friend's feedback:
- Reduced penalties by 2-3x to let profitable behavior emerge
- Wider reward clipping to allow agent to feel profit signals
- Survival bonus to encourage stable trading

Key Fix (inherited): Position sizing referenced to INITIAL balance, not growing equity.
This prevents the exponential compounding effect that created fake profits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from ppo_trading_ai.data_pipeline import load_prices_or_fail
from ppo_trading_ai.features import calculate_advanced_features


@dataclass
class TradingConfigExperimentB:
    """Configuration for Experiment B trading environment with lighter penalties"""
    initial_balance: float = 10000.0  # Starting capital
    max_leverage: float = 3.0  # Maximum leverage allowed
    sequence_length: int = 50
    episode_overlap: int = 10

    # Trading costs (realistic)
    taker_fee: float = 0.0003  # 3 bps
    slippage_bp: float = 0.5  # 0.5 bp

    # Reward clipping for numerical stability
    reward_clip: float = 0.5  # Clip rewards to prevent optimizer instability

    funding_rate_8h: float = 0.0001  # 1 bp per 8h
    step_hours: float = 1.0


class ExperimentBTradingEnvironment:
    """
    Experiment B Trading Environment - HONEST P&L ONLY.

    Built on FIXED position sizing (no compounding bug):
    - Position sizing references INITIAL balance
    - No automatic position rebalancing
    - Prevents exponential compounding from growing equity

    PHASE 1.5 FIX - Removed reward hacking:
    - NO survival bonus (was causing "do nothing" exploit)
    - NO explicit turnover/leverage penalties (costs already in equity)
    - Reward = pure equity change / initial balance
    - Wider reward clipping (±0.5) for numerical stability only

    Goal: Agent optimizes ONLY for actual trading P&L, not fake bonuses.
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[TradingConfigExperimentB] = None,
        step_minutes: int = 60
    ):
        self.config = config or TradingConfigExperimentB()
        self.step_minutes = step_minutes

        # Load price data
        df = load_prices_or_fail(data_path, step_minutes=step_minutes)
        if isinstance(df, tuple):
            df, _ = df

        self.prices = df['close'].values
        self.timestamps = pd.to_datetime(df['ts']) if 'ts' in df.columns else pd.to_datetime(df.index)

        # Calculate features
        self.features = self._calculate_features(df)
        self.obs_dim = self.features.shape[1]

        # Episode management
        self.total_steps = len(self.prices) - 1
        self.episode_length = 200
        self.episode_count = 0
        self.episode_start = 0
        self.current_step = 0

        # Trading state
        self.equity = self.config.initial_balance
        self.position_btc = 0.0  # Position in BTC (NOT USD notional!)
        self.funding_accumulator = 0.0
        self.sequence_buffer = []

        # Observation space (for compatibility)
        self.observation_space = type('obj', (object,), {
            'shape': (self.config.sequence_length, self.obs_dim + 4)
        })()

        print("Experiment B Trading Environment initialized:")
        print(f"  Total timesteps: {self.total_steps}")
        print(f"  Features: {self.obs_dim}")
        print(f"  Sequence length: {self.config.sequence_length}")
        print(f"  Episode length: {self.episode_length}")
        print(f"  Episode overlap: {self.config.episode_overlap}")
        print("  FIX APPLIED: Position sizing uses INITIAL balance (no compounding)")
        print("  PHASE 1.5 FIX: Honest P&L only - no survival bonus, no fake rewards")

    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate market features using advanced 28-feature calculator.

        Upgraded from 8 features to 28 features (3.5x more signal):
        - Original 8: returns (1h, 4h, 24h), vol, MA ratios, cumulative return, vol_norm
        - New 20: Multi-timeframe returns (2h, 6h, 12h, 3d, 7d, 30d), RSI (14, 28),
                  MACD, vol intelligence, Bollinger Bands, ATR, price action, volume, time
        """
        # Use the advanced 28-feature calculator from features.py
        features = calculate_advanced_features(df)

        return features.astype(np.float32)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        # Sequential episode start with overlap
        if self.episode_count == 0:
            self.episode_start = 0
        else:
            self.episode_start = min(
                self.episode_start + self.episode_length - self.config.episode_overlap,
                self.total_steps - self.episode_length
            )

        self.current_step = 0
        self.equity = self.config.initial_balance
        self.position_btc = 0.0  # Position in BTC
        self.funding_accumulator = 0.0
        self.sequence_buffer = []

        # Build initial sequence
        current_idx = self.episode_start
        initial_obs = self.features[current_idx]

        for _ in range(self.config.sequence_length):
            account_features = self._get_account_features()
            full_obs = np.concatenate([initial_obs, account_features])
            self.sequence_buffer.append(full_obs)

        sequence = np.array(self.sequence_buffer)
        self.episode_count += 1

        info = {
            'episode_start': self.episode_start,
            'episode_count': self.episode_count
        }

        return sequence, info

    def _get_account_features(self) -> np.ndarray:
        """Get account state features"""
        current_idx = self.episode_start + self.current_step
        current_price = self.prices[min(current_idx, len(self.prices) - 1)]

        # Position in USD notional (for display)
        notional = self.position_btc * current_price

        position_pct = notional / (self.equity + 1e-8)
        leverage = abs(position_pct)

        return np.array([
            position_pct,  # Position as % of equity
            leverage,  # Leverage ratio
            0.0,  # Drawdown (placeholder)
            self.equity / self.config.initial_balance - 1.0  # Equity ratio
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading step with FIXED position sizing (no compounding bug).

        CRITICAL FIX:
        Target position calculated from INITIAL balance, not current equity.
        This prevents compounding leverage effect.

        Args:
            action: Trading action in [-1, 1]
                   -1 = max short, 0 = flat, +1 = max long

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = float(np.clip(action, -1.0, 1.0))

        # Get current price
        current_idx = self.episode_start + self.current_step
        current_price = self.prices[current_idx]
        E_pre = self.equity

        # ===================================================================
        # CRITICAL FIX: Use INITIAL balance for position sizing
        # ===================================================================
        # OLD (BUGGY): N_target = action * max_lev * E_pre  # Compounds!
        # NEW (FIXED): N_target = action * max_lev * INITIAL  # No compounding!

        target_notional = action * self.config.max_leverage * self.config.initial_balance
        target_position_btc = target_notional / current_price

        # Calculate position change in BTC
        delta_btc = target_position_btc - self.position_btc

        # Apply no-trade band (in USD terms for consistency)
        delta_notional = delta_btc * current_price
        if abs(delta_notional) < 0.05 * E_pre:
            delta_btc = 0.0  # Too small, ignore
        elif np.sign(target_position_btc) != np.sign(self.position_btc) and abs(delta_notional) < 0.5 * E_pre:
            delta_btc = 0.0  # Sign flip requires larger move

        # Execute trade and calculate costs
        traded_notional = abs(delta_btc * current_price)
        fee = traded_notional * self.config.taker_fee
        slippage = traded_notional * (self.config.slippage_bp / 1e4)
        trading_cost = fee + slippage

        self.position_btc += delta_btc
        self.equity -= trading_cost

        # Market PnL: position_btc stays constant, price changes
        if self.current_step > 0 and abs(self.position_btc) > 1e-8:
            prev_idx = current_idx - 1
            price_change = self.prices[current_idx] - self.prices[prev_idx]
            market_pnl = self.position_btc * price_change
            self.equity += market_pnl

        # Funding cost (every 8 hours)
        self.funding_accumulator += self.config.step_hours
        if self.funding_accumulator >= 8.0:
            notional = abs(self.position_btc * current_price)
            funding = notional * self.config.funding_rate_8h
            self.equity -= funding
            self.funding_accumulator -= 8.0

        # Post-step equity
        E_post = max(self.equity, 1e-12)

        # ===================================================================
        # PHASE 1.5 FIX: HONEST P&L ONLY
        # ===================================================================
        # Track actual P&L and costs for diagnostics
        step_pnl = E_post - E_pre

        # Calculate funding cost for this step (may be 0 if not at 8h mark)
        funding_cost = 0.0
        if self.funding_accumulator >= 8.0:
            notional = abs(self.position_btc * current_price)
            funding_cost = notional * self.config.funding_rate_8h

        step_cost = trading_cost + funding_cost

        # Reward = pure equity change scaled by initial capital
        # No survival bonus, no artificial penalties
        # Costs are already reflected in equity change
        step_return = step_pnl / self.config.initial_balance
        reward = step_return

        # Clip only for numerical stability (not to shape behavior)
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))

        # Update step
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Check bankruptcy
        if E_post < 0.1 * self.config.initial_balance:
            terminated = True
            reward = -1.0

        # Get next observation
        current_idx = self.episode_start + self.current_step
        if current_idx < len(self.features):
            market_features = self.features[current_idx]
        else:
            market_features = self.features[-1]

        account_features = self._get_account_features()
        full_obs = np.concatenate([market_features, account_features])

        # Update sequence buffer
        self.sequence_buffer.append(full_obs)
        if len(self.sequence_buffer) > self.config.sequence_length:
            self.sequence_buffer.pop(0)

        sequence = np.array(self.sequence_buffer)

        # Info dict with actual P&L and cost tracking
        info = {
            'equity': float(E_post),
            'position': float(self.position_btc * current_price),  # Position in USD notional
            'position_pct': float((self.position_btc * current_price) / (E_post + 1e-8)),
            'price': float(current_price),
            'pnl': float(step_pnl),  # CRITICAL: Actual P&L for backtest
            'cost': float(step_cost),  # CRITICAL: Actual costs for backtest
            'traded_notional': float(traded_notional),
            'reward': float(reward)
        }

        return sequence, reward, terminated, truncated, info
