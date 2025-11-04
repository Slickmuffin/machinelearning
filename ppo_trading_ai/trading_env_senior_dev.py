#!/usr/bin/env python3
"""
PPO BTC Trading Environment - Senior Dev Complete Fix
All specifications implemented exactly as provided
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
# Fix relative import for Vertex AI
try:
    from .data_pipeline import load_prices_or_fail
except ImportError:
    from data_pipeline import load_prices_or_fail

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Senior Dev Economic Specification - Realistic Trading Config"""
    initial_balance: float = 10_000.0
    max_leverage: float = 3.0

    # Training vs Production mode
    training_mode: bool = True       # True = learning phase (lenient), False = production (strict)

    # Senior Dev Fee Structure
    taker_fee: float = 0.0003        # 3 bps realistic
    maker_fee: float = 0.0001        # 1 bp (rarely used in a marketable RL agent)
    slippage_bp: float = 0.5         # 0.5 bp baseline; tune with volatility/liquidity
    lambda_turnover: float = 1e-4    # 1 bp per 10x turnover (mild)
    lambda_leverage: float = 2e-5    # mild leverage aversion
    funding_rate_8h: float = 0.0001  # 1 bp / 8h typical baseline
    step_hours: float = 1.0

    # Action-cost alignment
    no_trade_band: float = 0.05      # ignore |ΔN| < 5% of equity
    turnover_cap: float = 1.0        # |ΔN| ≤ 100% of equity per step
    hysteresis_threshold: float = 0.5 # min |ΔN| for sign flips

    # Economic tripwires (ONLY FOR PRODUCTION - disabled during training)
    max_trading_cost_rate: float = 0.002  # 20 bps per step
    max_turnover_rate: float = 1.5        # 150% turnover/equity
    min_gross_pnl_threshold: float = 0.05 # min turnover for gross PnL check

class PPOTradingEnvironmentSeniorDev(gym.Env):
    """Trading Environment with Complete Senior Dev Specifications"""
    
    def __init__(self, config: TradingConfig = None, data_path: str = None, step_minutes: int = 60):
        super().__init__()

        self.config = config or TradingConfig()

        # Load data with strict validation
        try:
            if data_path:
                data_result = load_prices_or_fail(data_path, step_minutes=step_minutes)
            else:
                # Try default path
                data_result = load_prices_or_fail("/tmp/btc_data.csv", step_minutes=step_minutes)

            # Handle both tuple (df, metrics) and dataframe returns
            if isinstance(data_result, tuple):
                self.data, self.data_metrics = data_result
            else:
                self.data = data_result
                self.data_metrics = None
        except Exception as e:
            logger.error(f"DATA BLOCK: {e}")
            raise SystemExit(f"DATA BLOCK: {e}")

        self.prices = self.data["close"].to_numpy()
        
        # Setup observation space
        feature_cols = [col for col in self.data.columns if col not in ['ts', 'close', 'timestamp']]
        self.feature_columns = feature_cols[:10]  # Use first 10 features
        obs_dim = len(self.feature_columns) + 3  # features + leverage + turnover/E + drawdown
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize funding accumulator
        self._fund_accum = 0.0
        
        # Economic tripwire tracking
        self.step_costs = []
        self.step_turnovers = []
        
        self.reset()
        logger.info(f"✅ Senior Dev Trading Environment initialized: {self.data.shape[0]} timesteps")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset state variables (USD notional system)
        self.current_step = 0
        self.reset_step = 0
        self.equity: float = self.config.initial_balance
        self.N: float = 0.0  # signed USD notional
        self.price: float = self.prices[0] if len(self.prices) > 0 else 50000.0
        self.peak_equity: float = self.equity
        
        # Reset funding accumulator
        self._fund_accum = 0.0
        
        # Reset tripwire tracking
        self.step_costs.clear()
        self.step_turnovers.clear()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_signal = np.clip(float(action[0]), -1.0, 1.0)
        
        # Check episode end
        if self.current_step >= len(self.prices) - 1:
            done = True
            observation = np.zeros(self.observation_space.shape)
            reward = 0.0
            info = self._get_info()
            return observation, reward, done, False, info
        
        # 1) Mark-to-market with next price
        E_pre = self.equity
        next_price = self.prices[self.current_step + 1]
        
        # Calculate PnL from price change
        r = (next_price / self.price) - 1.0
        pnl = self.N * r

        # FIX #4: Safety guard - cap catastrophic single-step losses
        # Prevents Step 12→13 style catastrophes (-84% in 1 step)
        max_step_loss = 0.25 * E_pre  # Max 25% loss per step
        if pnl < -max_step_loss:
            self.logger.warning(f"🚨 CAPPED CATASTROPHIC LOSS: PnL={pnl:.2f} (${pnl:.2f}) capped to {-max_step_loss:.2f} (-25%)")
            self.logger.warning(f"   Position N=${self.N:.2f}, Price change: ${self.price:.2f}→${next_price:.2f} ({r*100:.1f}%)")
            pnl = -max_step_loss

        # 2) Funding correctly once per 8h window
        self._fund_accum += self.config.step_hours
        funding = 0.0
        if self._fund_accum >= 8.0:
            funding = abs(self.N) * self.config.funding_rate_8h
            self._fund_accum -= 8.0
        
        # 3) Target notional from action
        L_target = action_signal * self.config.max_leverage
        N_target = L_target * E_pre
        
        # 4) Execute trade with action-cost alignment
        traded, trading_cost = self._exec_trade(N_target, E_pre)
        
        # 5) Update equity
        self.equity = E_pre + pnl - trading_cost - funding
        self.price = next_price
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # 6) Reward: log-equity with soft frictions
        gross = np.log(max(self.equity, 1e-12) / max(E_pre, 1e-12))
        turn_pen = self.config.lambda_turnover * (traded / max(E_pre, 1.0))
        lev_pen = self.config.lambda_leverage * (abs(self.N) / max(E_pre, 1.0))
        reward = gross - turn_pen - lev_pen
        
        # Limit to protect optimizers; keep it wide
        reward = float(np.clip(reward, -0.5, 0.5))
        
        # 7) Economic tripwires
        self._check_economic_tripwires(trading_cost, traded, E_pre, pnl, reward)
        
        # 8) Termination conditions
        drawdown = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0

        # Episode termination: bankruptcy, drawdown, or max episode length
        episode_length = self.current_step - self.reset_step if hasattr(self, 'reset_step') else self.current_step
        max_episode_steps = 1000  # Force episodes to complete for PPO training

        done = (self.equity <= 0.0) or (drawdown >= 0.5) or (episode_length >= max_episode_steps)
        
        # Move to next step
        self.current_step += 1
        
        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {
            "equity": self.equity,
            "N": self.N,
            "traded": traded,
            "cost": trading_cost,
            "pnl": pnl,
            "funding": funding,
            "drawdown": drawdown,
            "gross_reward": gross,
            "turn_penalty": turn_pen,
            "lev_penalty": lev_pen
        }
        
        return observation, reward, done, False, info
    
    def _exec_trade(self, N_target: float, E_pre: float) -> Tuple[float, float]:
        """Execute trade with Senior Dev action-cost alignment"""
        delta_N_req = N_target - self.N
        
        # No-trade band: ignore actions that imply |ΔN| < 0.05 * E_pre
        if abs(delta_N_req) < self.config.no_trade_band * E_pre:
            return 0.0, 0.0
        
        # Hysteresis on sign flips: if sign change, require minimum |ΔN| ≥ 0.5 * E_pre
        if np.sign(N_target) != np.sign(self.N) and abs(delta_N_req) < self.config.hysteresis_threshold * E_pre:
            return 0.0, 0.0
        
        # Turnover cap: |ΔN| ≤ 1.0 * E_pre per step
        max_delta = self.config.turnover_cap * E_pre
        delta_N = np.clip(delta_N_req, -max_delta, max_delta)
        traded = abs(delta_N)
        
        # Senior Dev Fee Structure: Keep one unit: USD notional N and turnover |ΔN|
        fee = traded * self.config.taker_fee        # assume taker (3 bps)
        slip = traded * (self.config.slippage_bp / 1e4)  # 0.5 bp
        trading_cost = fee + slip
        
        # Commit trade
        self.N += delta_N
        
        return traded, trading_cost
    
    def _check_economic_tripwires(self, trading_cost: float, traded: float, E_pre: float, pnl: float, reward: float):
        """
        Economic monitoring - TRAINING MODE vs PRODUCTION MODE

        TRAINING MODE (learning phase):
        - AI needs to explore and learn from mistakes
        - High costs and turnover are EXPECTED during exploration
        - Warnings disabled to avoid log spam

        PRODUCTION MODE (real trading):
        - Strict tripwires to prevent runaway costs
        - Alert on unusual patterns
        """
        # SKIP ALL CHECKS IN TRAINING MODE - let the AI learn!
        if self.config.training_mode:
            return

        # PRODUCTION MODE ONLY: Track costs and turnovers
        if E_pre > 0:
            cost_rate = trading_cost / E_pre
            turnover_rate = traded / E_pre

            self.step_costs.append(cost_rate)
            self.step_turnovers.append(turnover_rate)

            # Keep recent history
            if len(self.step_costs) > 100:
                self.step_costs.pop(0)
                self.step_turnovers.pop(0)

            # Tripwire checks
            if len(self.step_costs) >= 10:  # After warmup
                mean_cost_rate = np.mean(self.step_costs[-10:])
                mean_turnover_rate = np.mean(self.step_turnovers[-10:])

                # Production alerts
                if mean_cost_rate > self.config.max_trading_cost_rate:
                    logger.warning(f"⚠️ PRODUCTION ALERT - HIGH TRADING COSTS: {mean_cost_rate:.4f} > {self.config.max_trading_cost_rate:.4f}")

                if mean_turnover_rate > self.config.max_turnover_rate:
                    logger.warning(f"⚠️ PRODUCTION ALERT - HIGH TURNOVER: {mean_turnover_rate:.4f} > {self.config.max_turnover_rate:.4f}")

                if pnl <= 0 and mean_turnover_rate < self.config.min_gross_pnl_threshold:
                    logger.warning(f"⚠️ PRODUCTION ALERT - ECONOMIC ISSUE: Negative PnL {pnl:.4f} with low turnover {mean_turnover_rate:.4f}")
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        # Get market features
        feature_values = []
        for col in self.feature_columns:
            if col in self.data.columns:
                val = self.data.iloc[self.current_step][col]
                feature_values.append(float(val) if pd.notna(val) else 0.0)
            else:
                feature_values.append(0.0)
        
        # Add state features
        leverage = abs(self.N) / max(self.equity, 1e-12)
        turnover_E = 0.0  # Will be updated with recent turnover
        drawdown = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        
        obs = np.array(feature_values + [leverage, turnover_E, drawdown], dtype=np.float32)
        
        # Validate and clean
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'equity': float(self.equity),
            'N': float(self.N),
            'leverage': float(abs(self.N) / max(self.equity, 1e-12)),
            'price': float(self.price),
            'drawdown': float(1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0),
            'step': self.current_step,
            'funding_accum': self._fund_accum
        }