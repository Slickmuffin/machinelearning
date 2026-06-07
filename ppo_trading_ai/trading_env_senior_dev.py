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
    max_leverage: float = 2.0  # Reduced from 3.0 - agent was exploiting high leverage

    # Training vs Production mode
    training_mode: bool = True       # True = learning phase (lenient), False = production (strict)

    # EPISODE FIX: Fixed-length episodes for proper RL learning
    use_fixed_episodes: bool = True  # Use fixed-length episodes (not DD termination)
    episode_length_days: int = 90  # 90-day episodes (with daily data)
    max_drawdown_training: float = 0.95  # 95% DD (only for bankruptcy check)
    max_drawdown_production: float = 0.50  # 50% DD in production
    survival_bonus: float = 0.001  # Small survival bonus (was 0.01, too strong)
    risk_scaled_leverage: bool = True  # Reduce max leverage during drawdowns

    # Senior Dev Fee Structure
    taker_fee: float = 0.0003        # 3 bps realistic
    maker_fee: float = 0.0001        # 1 bp (rarely used in a marketable RL agent)
    slippage_bp: float = 0.5         # 0.5 bp baseline; tune with volatility/liquidity
    lambda_turnover: float = 1e-4    # 1 bp per 10x turnover
    funding_rate_8h: float = 0.0001  # 1 bp / 8h typical baseline
    step_hours: float = 1.0

    # PENALTY CALIBRATION FIX: Reduced to match actual BTC return scale
    # Based on data analysis: median daily return = 1.4%, vol = 2.9%
    # Penalties now proportional to returns (not dominating them)
    lambda_lev_linear: float = 0.002     # 15% of typical return per 1x leverage
    lambda_lev_quadratic: float = 0.001  # Quadratic discouragement for high leverage

    # NEW: Volatility and drawdown penalties (Sharpe-ish reward shaping)
    sharpe_window: int = 64              # how many steps to look back for vol
    k_vol: float = 0.15                  # Vol penalty = 30% of typical return
    k_drawdown: float = 0.3              # DD penalty = 20% of typical return per 10% DD

    # NEW: Reward clipping
    reward_clip: float = 1.0

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

        # SENIOR DEV FIX: Wire step_minutes → step_hours for time consistency
        self.step_minutes = step_minutes  # Store for cadence-aware episode length
        self.config.step_hours = step_minutes / 60.0

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

        # Setup observation space - use ALL profitable features
        priority_features = [
            # Price action
            'open', 'high', 'low', 'volume_usd',
            # Multi-timeframe momentum (critical for trend following)
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
            # Volatility (critical for position sizing)
            'volatility_24h', 'volatility_168h', 'volatility_regime',
            # Trend following (most profitable strategy)
            'price_ma_50', 'price_ma_200', 'trend_alignment',
            # Sentiment
            'fear_greed_value', 'fear_greed_extreme',
            # Timing
            'volatility_rank_24h'
        ]
        self.feature_columns = [f for f in priority_features if f in self.data.columns]
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

        # SENIOR DEV FIX: Track returns, costs, and turnovers for reward shaping
        self.return_history = []   # step_ret history for volatility calculation
        self.step_costs = []
        self.step_turnovers = []

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

        # CRITICAL: Enforce max leverage BEFORE PnL calculation
        # Agent was reaching 300-700x leverage - this is the root cause
        current_leverage = abs(self.N) / max(E_pre, 1e-12)
        if current_leverage > self.config.max_leverage:
            # Force position reduction to max leverage
            max_N = self.config.max_leverage * E_pre
            if self.N > 0:
                self.N = max_N
            else:
                self.N = -max_N
            # Recalculate PnL with capped position
            pnl = self.N * r
            logger.warning(f"⚠️ LEVERAGE CAPPED PRE-MOVE: {current_leverage:.1f}x → {self.config.max_leverage}x")

        # FIX #4: Safety guard - cap catastrophic single-step losses (increased to 50%)
        # Prevents Step 12→13 style catastrophes, but agent needs to learn from big losses
        max_step_loss = 0.50 * E_pre  # Max 50% loss per step (was 25%)
        if pnl < -max_step_loss:
            logger.warning(f"🚨 CAPPED CATASTROPHIC LOSS: PnL={pnl:.2f} (${pnl:.2f}) capped to {-max_step_loss:.2f} (-50%)")
            logger.warning(f"   Position N=${self.N:.2f}, Price change: ${self.price:.2f}→${next_price:.2f} ({r*100:.1f}%)")
            pnl = -max_step_loss

        # 2) Funding correctly once per 8h window
        self._fund_accum += self.config.step_hours
        funding = 0.0
        if self._fund_accum >= 8.0:
            funding = abs(self.N) * self.config.funding_rate_8h
            self._fund_accum -= 8.0
        
        # 3) Target notional from action with RISK-SCALED LEVERAGE
        base_max_leverage = self.config.max_leverage

        # SURVIVAL FIX: Reduce allowed leverage during drawdowns
        if self.config.risk_scaled_leverage:
            current_dd = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
            # Scale down leverage as drawdown increases: 50% DD → 0.5x scale, floor at 0.5
            leverage_scale = max(0.5, 1.0 - current_dd)
            effective_max_leverage = base_max_leverage * leverage_scale
        else:
            effective_max_leverage = base_max_leverage

        L_target = action_signal * effective_max_leverage
        N_target = L_target * E_pre
        
        # 4) Execute trade with action-cost alignment
        traded, trading_cost = self._exec_trade(N_target, E_pre)
        
        # 5) Update equity
        self.equity = E_pre + pnl - trading_cost - funding
        self.price = next_price
        E_post = self.equity
        E_post = max(E_post, 1e-12)

        # 6) SENIOR DEV NEW REWARD FUNCTION: Risk-adjusted with vol and drawdown penalties

        # 6.1) Core return terms
        log_ret = np.log(E_post / max(E_pre, 1e-12))
        step_ret = (E_post / max(E_pre, 1e-12)) - 1.0

        # 6.2) Update history for volatility estimate
        self.return_history.append(step_ret)
        if len(self.return_history) > self.config.sharpe_window:
            self.return_history.pop(0)

        if len(self.return_history) >= 10:
            realized_vol = float(np.std(self.return_history))
        else:
            realized_vol = 0.0

        # 6.3) Track turnover for observation
        turnover_equity = traded / max(E_pre, 1.0)
        self.step_turnovers.append(turnover_equity)
        if len(self.step_turnovers) > 50:
            self.step_turnovers.pop(0)

        # 6.4) Turnover penalty
        turn_pen = self.config.lambda_turnover * turnover_equity

        # 6.5) Convex leverage penalty (quadratic pushes agent off max leverage)
        lev_ratio = abs(self.N) / max(E_post, 1.0)
        lev_pen = (
            self.config.lambda_lev_linear * lev_ratio +
            self.config.lambda_lev_quadratic * (lev_ratio ** 2)
        )

        # 6.6) Drawdown penalty (quadratic in drawdown)
        self.peak_equity = max(self.peak_equity, E_post)
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - E_post) / self.peak_equity
        dd_pen = self.config.k_drawdown * (drawdown ** 2)

        # 6.7) Volatility penalty (Sharpe-ish shaping - penalize high vol)
        vol_pen = self.config.k_vol * realized_vol

        # 6.8) Final reward: log-equity - all penalties + SURVIVAL BONUS
        base_reward = log_ret - turn_pen - lev_pen - vol_pen - dd_pen

        # SURVIVAL FIX: Add per-step survival bonus
        survival_bonus = self.config.survival_bonus
        reward = base_reward + survival_bonus

        # Clip to protect optimizer (wider than before: ±1.0 instead of ±0.5)
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        
        # 7) Economic tripwires
        self._check_economic_tripwires(trading_cost, traded, E_pre, pnl, reward)
        
        # 8) Termination conditions
        drawdown = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        episode_length = self.current_step - self.reset_step if hasattr(self, 'reset_step') else self.current_step

        # EPISODE FIX: Use fixed-length episodes for proper RL learning (CADENCE-AWARE)
        if self.config.use_fixed_episodes:
            # Calculate target steps based on cadence (e.g., 90 days * 24 hours = 2160 steps for hourly)
            steps_per_day = int(24 * 60 // self.step_minutes)
            target_steps = int(self.config.episode_length_days * steps_per_day)
            done = (episode_length >= target_steps) or (self.equity <= 0.0)
        else:
            # DD-based termination (old way)
            max_dd = self.config.max_drawdown_training if self.config.training_mode else self.config.max_drawdown_production
            max_episode_steps = 5000
            done = (self.equity <= 0.0) or (drawdown >= max_dd) or (episode_length >= max_episode_steps)
        
        # Move to next step
        self.current_step += 1
        
        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {
            "equity": self.equity,
            "N": self.N,
            "leverage": abs(self.N) / max(self.equity, 1e-12),
            "traded": traded,
            "cost": trading_cost,
            "pnl": pnl,
            "funding": funding,
            "drawdown": drawdown,
            "log_ret": log_ret,
            "turn_penalty": turn_pen,
            "lev_penalty": lev_pen,
            "vol_penalty": vol_pen,
            "dd_penalty": dd_pen
        }

        return observation, reward, done, False, info
    
    def _exec_trade(self, N_target: float, E_pre: float) -> Tuple[float, float]:
        """
        Execute trade with PROPER cost accounting.

        CRITICAL FIX: Charge costs based on POSITION CHANGES, not leverage changes.

        Economic Reality:
        - Maintaining constant 3x leverage during equity growth REQUIRES continuous rebalancing
        - Every rebalancing incurs real trading costs (fees + slippage)
        - Previous leverage-change approach allowed cost-free compounding exploit

        Correct Approach:
        - Charge costs for ANY position change above tiny rounding threshold (0.1% equity)
        - Agent must learn that maintaining leveraged exposure costs money
        - No free lunch from compounding at max leverage
        """
        # Apply turnover cap
        delta_N_req = N_target - self.N
        max_delta = self.config.turnover_cap * E_pre
        delta_N = np.clip(delta_N_req, -max_delta, max_delta)

        # Hysteresis: require minimum position change for sign flips
        hysteresis_threshold = self.config.hysteresis_threshold * E_pre
        if np.sign(N_target) != np.sign(self.N) and abs(delta_N) < hysteresis_threshold:
            delta_N = 0.0  # Block tiny sign flips

        # CORRECT COST GATING: Based on POSITION CHANGE, not leverage change
        # Tiny threshold (0.1% of equity) to avoid costs on rounding errors only
        no_trade_threshold = 0.001 * E_pre
        traded = abs(delta_N)

        if traded < no_trade_threshold:
            # True no-op: position change < 0.1% of equity (rounding error)
            trading_cost = 0.0
        else:
            # ANY meaningful position change incurs costs - this is economically correct
            fee = traded * self.config.taker_fee        # 3 bps
            slip = traded * (self.config.slippage_bp / 1e4)  # 0.5 bp
            trading_cost = fee + slip

        # ALWAYS update position to track target
        self.N += delta_N

        # CRITICAL: Strict leverage enforcement (NO tolerance)
        # This runs AFTER action execution to prevent runaway positions
        actual_leverage = abs(self.N) / max(self.equity, 1e-12)
        if actual_leverage > self.config.max_leverage:
            max_N = self.config.max_leverage * self.equity
            if self.N > 0:
                self.N = max_N
            else:
                self.N = -max_N
            logger.warning(f"⚠️ LEVERAGE CAP ENFORCED POST-ACTION: {actual_leverage:.2f}x → {self.config.max_leverage}x")

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

        # SENIOR DEV FIX: Actually calculate turnover/E from recent history
        if self.step_turnovers:
            turnover_E = float(np.mean(self.step_turnovers[-20:]))  # Recent 20-step avg
        else:
            turnover_E = 0.0

        drawdown = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0

        obs = np.array(feature_values + [leverage, turnover_E, drawdown], dtype=np.float32)

        # Validate and clean
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        # SENIOR DEV FIX: Enhanced info dict for proper evaluation
        position_btc = self.N / max(self.price, 1e-12)  # BTC position size
        total_pnl = self.equity - self.config.initial_balance
        total_return = (self.equity / self.config.initial_balance) - 1.0
        drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-12)

        return {
            'equity': float(self.equity),
            'N': float(self.N),
            'position': float(position_btc),  # For evaluator to see trades
            'leverage': float(abs(self.N) / max(self.equity, 1e-12)),
            'price': float(self.price),
            'drawdown': float(drawdown),
            'step': self.current_step,
            'funding_accum': self._fund_accum,
            'total_pnl': float(total_pnl),
            'total_return': float(total_return),
        }