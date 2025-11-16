#!/usr/bin/env python3
"""
FIXED Sequential Trading Environment - No Compounding Leverage Bug

Key Fix: Position sizing referenced to INITIAL balance, not growing equity.
This prevents the exponential compounding effect that created fake profits.

Following gym-trading-env best practices:
- Position as target state (not trade delta)
- Fixed reference point for sizing
- No automatic rebalancing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from ppo_trading_ai.data_pipeline import load_prices_or_fail


@dataclass
class TradingConfig:
    """Configuration for trading environment"""
    initial_balance: float = 10000.0  # Starting capital
    max_leverage: float = 3.0  # Maximum leverage allowed
    sequence_length: int = 50
    episode_overlap: int = 10

    # Trading costs (realistic)
    taker_fee: float = 0.0003  # 3 bps
    slippage_bp: float = 0.5  # 0.5 bp
    lambda_turnover: float = 1e-4
    lambda_leverage: float = 2e-5
    funding_rate_8h: float = 0.0001  # 1 bp per 8h
    step_hours: float = 1.0


class FixedTradingEnvironment:
    """
    Trading environment with FIXED position sizing (no compounding bug).

    CRITICAL FIX:
    - Position sizing references INITIAL balance
    - No automatic position rebalancing
    - Prevents exponential compounding from growing equity

    This is how real perpetual futures work:
    - You choose a position size in contracts/BTC
    - Position stays constant until YOU change it
    - Equity changes don't auto-adjust position
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[TradingConfig] = None,
        step_minutes: int = 60
    ):
        self.config = config or TradingConfig()
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

        print(f"Fixed Trading Environment initialized:")
        print(f"  Total timesteps: {self.total_steps}")
        print(f"  Features: {self.obs_dim}")
        print(f"  Sequence length: {self.config.sequence_length}")
        print(f"  Episode length: {self.episode_length}")
        print(f"  Episode overlap: {self.config.episode_overlap}")
        print(f"  FIX APPLIED: Position sizing uses INITIAL balance (no compounding)")

    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate market features"""
        prices = df['close'].values

        # Returns
        returns_1h = np.diff(np.log(prices), prepend=np.log(prices[0]))
        returns_4h = pd.Series(np.log(prices)).diff(4).fillna(0).values
        returns_24h = pd.Series(np.log(prices)).diff(24).fillna(0).values

        # Volatility (with NaN handling)
        vol_24h = pd.Series(returns_1h).rolling(24, min_periods=1).std().fillna(0).values

        # Moving averages
        ma_20 = pd.Series(prices).rolling(20, min_periods=1).mean().values
        ma_50 = pd.Series(prices).rolling(50, min_periods=1).mean().values

        ma_ratio_20 = (prices - ma_20) / (ma_20 + 1e-8)
        ma_ratio_50 = (prices - ma_50) / (ma_50 + 1e-8)

        # Normalized volatility (with NaN handling)
        vol_mean = vol_24h.mean()
        if vol_mean > 1e-8:
            vol_norm = vol_24h / vol_mean - 1.0
        else:
            vol_norm = np.zeros_like(vol_24h)

        features = np.column_stack([
            returns_1h,
            returns_4h,
            returns_24h,
            vol_24h,
            ma_ratio_20,
            ma_ratio_50,
            np.log(prices / prices[0]),  # Cumulative return
            vol_norm  # Normalized volatility (NaN-safe)
        ])

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

        # Calculate reward (log-equity with penalties)
        gross_reward = np.log(E_post / max(E_pre, 1e-12))
        turnover_penalty = self.config.lambda_turnover * (traded_notional / max(E_pre, 1.0))

        current_notional = abs(self.position_btc * current_price)
        leverage_penalty = self.config.lambda_leverage * (current_notional / max(E_post, 1.0))

        reward = gross_reward - turnover_penalty - leverage_penalty
        reward = float(np.clip(reward, -0.5, 0.5))

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

        info = {
            'equity': E_post,
            'position': self.position_btc,
            'position_pct': (self.position_btc * current_price) / (E_post + 1e-8),
            'price': current_price,
            'trading_cost': trading_cost,
            'reward': reward
        }

        return sequence, reward, terminated, truncated, info
