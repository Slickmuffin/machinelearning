"""
Canonical PPO Trading Environment - Senior Developer Fix
Implements USD notional system to fix catastrophic cost calculation bug.

Key Fix: Position stored as signed USD notional (N), not leverage ratio.
Fees calculated on turnover |ΔN|, preventing 29x cost overrun.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
# Fix relative import for Vertex AI
try:
    from .canonical_config import TradingConfig
except ImportError:
    from canonical_config import TradingConfig


class CanonicalTradingEnv(gym.Env):
    """
    Corrected PPO trading environment with proper USD notional accounting.
    
    Critical Fix: All positions stored as signed USD notional (N).
    PnL = N_{t-1} * return_t
    Fees = |ΔN| * fee_rate (not position_change * equity)
    """
    
    def __init__(self, data_path: str, config: TradingConfig = None):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.data = self._load_and_validate_data(data_path)
        
        # State variables (USD notional system)
        self.equity: float = self.config.initial_balance
        self.N: float = 0.0                     # signed USD notional position
        self.price: float = 0.0                 # current price
        self.peak_equity: float = self.equity   # for drawdown calculation
        self.last_trading_cost: float = 0.0     # cost from previous trade
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
        # Funding accumulator (8-hour windows)
        self._funding_accum = 0.0
        
        # Action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation: features + state variables (leverage, equity, drawdown)
        n_features = len(self.data.columns) - 2  # exclude timestamp, close
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features + 3,), dtype=np.float32
        )

    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate price data with strict requirements."""
        df = pd.read_csv(data_path, low_memory=False)
        
        # Ensure required columns exist
        assert "close" in df.columns, "Missing 'close' price column"
        if "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("ts").drop_duplicates("ts")
        
        # Filter invalid prices
        valid_prices = np.isfinite(df["close"]) & (df["close"] > 0)
        df = df.loc[valid_prices].copy()
        
        # Data quality validation (senior dev requirement)
        nan_rate = 1.0 - len(df) / max(1, len(df))
        if len(df) < 1_000 or nan_rate > 0.02:
            raise ValueError(f"Insufficient price data: rows={len(df)}, nan_rate={nan_rate:.2%}")
        
        return df.reset_index(drop=True)

    def _target_notional_from_action(self, action: float, equity_pre: float) -> float:
        """
        Convert action [-1,1] to target USD notional position.
        
        Critical Fix: Uses pre-trade equity for sizing.
        action ∈ [-1,1] → target leverage in [-L_max, L_max]
        """
        leverage_target = float(action) * self.config.max_leverage
        return leverage_target * equity_pre

    def _exec_trade(self, notional_target: float, equity_pre: float) -> Tuple[float, float]:
        """
        Execute trade with proper USD notional accounting.
        
        Returns: (traded_notional, trading_cost)
        
        Critical Fix: Fees calculated on turnover |ΔN|, not position changes.
        """
        # Calculate required position change
        delta_N_req = notional_target - self.N
        
        # Apply turnover limit to prevent pathological costs
        max_turnover = self.config.turnover_cap * equity_pre
        delta_N = float(np.clip(delta_N_req, -max_turnover, max_turnover))
        traded_notional = abs(delta_N)
        
        # Calculate trading costs on turnover (USD notional)
        fee = traded_notional * self.config.taker_fee
        slippage = traded_notional * (self.config.slippage_bp / 1e4)
        trading_cost = fee + slippage
        
        # Apply cost sanity cap
        cost_cap = self.config.cost_cap_frac * equity_pre
        if trading_cost > cost_cap:
            trading_cost = cost_cap
        
        # Commit the trade
        self.N += delta_N
        
        return traded_notional, trading_cost

    def _step_accounting(self, new_price: float) -> Dict[str, float]:
        """
        Perform mark-to-market accounting with correct PnL calculation.
        
        Critical Fix: PnL = N_{t-1} * return_t (using prior notional)
        """
        price_return = (new_price / self.price) - 1.0 if self.price > 0 else 0.0
        
        equity_pre = self.equity
        
        # PnL from position held during price move (using prior notional)
        pnl = self.N * price_return
        
        # Funding cost (pro-rated for step size)
        self._funding_accum += self.config.step_hours
        funding_cost = 0.0
        if self._funding_accum >= 8.0:
            funding_cost = abs(self.N) * self.config.funding_rate_8h
            self._funding_accum -= 8.0
        
        # Update equity
        self.equity = equity_pre + pnl - self.last_trading_cost - funding_cost
        self.price = new_price
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Enforce leverage limits (liquidation protection)
        current_leverage = abs(self.N) / max(self.equity, 1e-12)
        if current_leverage > self.config.max_leverage:
            # Force reduce to margin-safe position
            safe_notional = np.sign(self.N) * self.config.max_leverage * self.equity
            _, emergency_cost = self._exec_trade(safe_notional, self.equity)
            self.last_trading_cost += emergency_cost
            self.equity -= emergency_cost
        
        # Calculate drawdown
        drawdown = 1.0 - (self.equity / self.peak_equity)
        
        return {
            "price_return": price_return,
            "pnl": pnl,
            "funding_cost": funding_cost,
            "drawdown": drawdown,
            "leverage": abs(self.N) / max(self.equity, 1e-12)
        }

    def _get_observation(self) -> np.ndarray:
        """Get current observation including market features and state."""
        if self.current_step >= len(self.data):
            # Return zeros if beyond data
            n_features = len(self.data.columns) - 2
            return np.zeros(n_features + 3, dtype=np.float32)
        
        # Market features (exclude timestamp and close)
        row = self.data.iloc[self.current_step]
        feature_cols = [col for col in self.data.columns if col not in ['timestamp', 'ts', 'close']]
        features = row[feature_cols].values.astype(np.float32)
        
        # State variables
        current_leverage = abs(self.N) / max(self.equity, 1e-12)
        equity_normalized = self.equity / self.config.initial_balance
        drawdown = 1.0 - (self.equity / self.peak_equity)
        
        state_vars = np.array([current_leverage, equity_normalized, drawdown], dtype=np.float32)
        
        # Combine features and state
        obs = np.concatenate([features, state_vars])
        
        # Handle NaN/inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state variables
        self.equity = self.config.initial_balance
        self.N = 0.0
        self.peak_equity = self.equity
        self.last_trading_cost = 0.0
        self.current_step = 0
        self._funding_accum = 0.0
        
        # Set initial price
        self.price = float(self.data.iloc[0]['close'])
        
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one environment step with canonical USD notional accounting.
        
        Critical Fix: All accounting done in USD notional, not leverage ratios.
        """
        if self.current_step >= self.max_steps:
            # Episode done
            return self._get_observation(), 0.0, True, False, {}
        
        # Get next price for mark-to-market
        self.current_step += 1
        new_price = float(self.data.iloc[self.current_step]['close'])
        
        # 1) Mark-to-market on price move
        accounting = self._step_accounting(new_price)
        
        # 2) Translate action to target notional (using pre-trade equity)
        equity_pre = self.equity
        notional_target = self._target_notional_from_action(action[0], equity_pre)
        
        # 3) Execute trade and record costs
        traded_notional, trading_cost = self._exec_trade(notional_target, equity_pre)
        self.last_trading_cost = trading_cost
        
        # 4) Calculate reward with turnover penalty
        equity_post = self.equity - trading_cost  # Apply trading cost
        self.equity = equity_post
        
        log_return = np.log(max(equity_post, 1e-12) / max(equity_pre, 1e-12))
        turnover_penalty = self.config.lambda_turnover * (traded_notional / max(equity_pre, 1.0))
        leverage_penalty = self.config.lambda_leverage * (abs(self.N) / max(equity_post, 1.0))
        
        reward = log_return - turnover_penalty - leverage_penalty
        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        
        # 5) Check termination conditions
        drawdown = accounting["drawdown"]
        done = (self.equity <= 0.0) or (drawdown >= self.config.drawdown_stop)
        
        # 6) Prepare info dict for monitoring
        info = {
            "equity": float(self.equity),
            "notional": float(self.N),
            "leverage": float(accounting["leverage"]),
            "trading_cost": float(trading_cost),
            "pnl": float(accounting["pnl"]),
            "funding_cost": float(accounting["funding_cost"]),
            "traded_notional": float(traded_notional),
            "drawdown": float(drawdown),
            "price_return": float(accounting["price_return"]),
            "gross_reward": float(log_return),
            "turnover_penalty": float(turnover_penalty),
            "leverage_penalty": float(leverage_penalty)
        }
        
        return self._get_observation(), reward, done, False, info


class VectorizedCanonicalEnv:
    """Vectorized wrapper for multiple canonical environments."""
    
    def __init__(self, data_path: str, num_envs: int = 16, config: TradingConfig = None):
        self.num_envs = num_envs
        self.envs = [CanonicalTradingEnv(data_path, config) for _ in range(num_envs)]
        
        # Copy spaces from first env
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, seeds=None):
        """Reset all environments."""
        if seeds is None:
            seeds = [None] * self.num_envs
        
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seeds[i])
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos

    def step(self, actions):
        """Step all environments."""
        observations = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncated.append(trunc)
            infos.append(info)
        
        return (
            np.array(observations),
            np.array(rewards),
            np.array(dones),
            np.array(truncated),
            infos
        )


# Backward compatibility alias
CanonicalPPOTradingEnv = CanonicalTradingEnv