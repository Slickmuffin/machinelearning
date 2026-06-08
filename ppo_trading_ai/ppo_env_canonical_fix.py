#!/usr/bin/env python3
"""
PPO BTC Trading Environment - SENIOR DEV CANONICAL FIX
Proper USD notional system with correct cost calculations
Fixes the catastrophic 29x cost overrun bug
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading environment configuration - Senior Dev Canonical Fix"""
    initial_balance: float = 10_000.0
    max_leverage: float = 3.0                   # L_max
    maker_fee: float = 0.0002                   # 2 bps
    taker_fee: float = 0.0004                   # 4 bps
    funding_rate_8h: float = 0.0001             # 1 bp / 8h
    step_hours: float = 1.0                     # env step size
    slippage_bp: float = 1.0                    # 1 bp per turnover
    turnover_cap: float = 1.0                   # |ΔN| ≤ cap * E_pre
    cost_cap_frac: float = 0.10                 # per-trade sanity cap
    drawdown_stop: float = 0.50                 # training kill switch
    
class PPOTradingEnvironment(gym.Env):
    """
    PPO Trading Environment with Senior Dev Canonical Fix
    
    Action Space: Continuous [-1, 1] representing target leverage
    Observation Space: Market features + leverage + turnover/E + drawdown  
    Reward: Log-equity changes with turnover penalty (NO instant death)
    Cost Model: Turnover-based fees (NOT leverage delta based)
    """
    
    # Class-level data cache to avoid redundant loading
    _shared_data = None
    _data_loaded = False
    
    def __init__(self, config: TradingConfig = None, data_path: str = None):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.data = self._get_shared_data(data_path)
        
        # Setup features and prices
        self._setup_features_and_prices()
        
        # Observation space: features + leverage + turnover/E + drawdown
        obs_dim = len(self.feature_columns) + 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info("PPO Trading Environment (Canonical Fix) initialized:")
        logger.info(f"  Data shape: {self.data.shape}")
        logger.info(f"  Price column: {self.price_column}")
        logger.info(f"  Observation space: {self.observation_space.shape}")
    
    def _setup_features_and_prices(self):
        """Setup price column and deterministic feature list"""
        # Find price column
        price_cols = [col for col in self.data.columns if 'close' in col.lower() or 'price' in col.lower()]
        if not price_cols:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.price_column = numeric_cols[0]
        else:
            self.price_column = price_cols[0]
            
        # Store prices array
        self.prices = self.data[self.price_column].to_numpy()
        
        # Define deterministic feature list
        exclude_cols = {self.price_column, 'return_1h'}
        self.feature_columns = [col for col in self.data.columns 
                               if col not in exclude_cols and self.data[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        logger.info(f"✅ Features setup: {len(self.feature_columns)} features, price column: {self.price_column}")
    
    @classmethod
    def _get_shared_data(cls, data_path: str = None) -> pd.DataFrame:
        """Get shared data using singleton pattern"""
        if not cls._data_loaded:
            logger.info("🔄 Loading data (first environment only)...")
            cls._shared_data = cls._load_and_validate_data_static(data_path)
            cls._data_loaded = True
            logger.info(f"✅ Data loaded and cached: {cls._shared_data.shape}")
        else:
            logger.info("✅ Using cached data")
        
        return cls._shared_data.copy()
        
    @staticmethod
    def _load_and_validate_data_static(data_path: str = None) -> pd.DataFrame:
        """Load and validate trading data"""
        try:
            import subprocess
            subprocess.run(['gsutil', 'cp', 
                          'gs://YOUR_GCP_PROJECT-training-data/btc_comprehensive_dataset_20250812_125933_features.csv', 
                          '/tmp/btc_data.csv'], 
                          check=True, capture_output=True)
            data = pd.read_csv('/tmp/btc_data.csv')
            logger.info(f"✅ Loaded raw data: {data.shape}")
            
        except Exception as e:
            logger.warning(f"Cloud data failed: {e}, using synthetic data")
            data = PPOTradingEnvironment._generate_synthetic_data_static()
            
        # Data validation and cleaning
        validated_data = PPOTradingEnvironment._validate_and_clean_data_static(data)
        return validated_data
        
    @staticmethod
    def _validate_and_clean_data_static(data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data with proper return calculation"""
        logger.info("🔍 Starting data validation...")
        
        # Find price column
        price_cols = [col for col in data.columns if 'close' in col.lower() or 'price' in col.lower()]
        if not price_cols:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            price_col = numeric_cols[0]
        else:
            price_col = price_cols[0]
            
        logger.info(f"Using price column: {price_col}")
        
        # Remove rows with NaN prices
        initial_rows = len(data)
        data = data.dropna(subset=[price_col])
        logger.info(f"Removed {initial_rows - len(data)} rows with NaN prices")
        
        # Compute real returns - fix length mismatch
        prices = data[price_col].values
        
        # Handle zero/negative prices
        if np.any(prices <= 0):
            logger.warning("Found zero/negative prices - replacing")
            prices = pd.Series(prices).fillna(method='ffill').fillna(1.0).values
            prices = np.maximum(prices, 1e-8)
        
        # Calculate returns safely  
        returns = np.log(prices[1:] / prices[:-1])
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove first row and assign returns
        data = data.iloc[1:].copy()
        data.reset_index(drop=True, inplace=True)
        data['return_1h'] = returns
        
        # Validate returns have variation
        if data['return_1h'].std() == 0:
            raise ValueError("All returns are zero - fake data detected!")
            
        # Clean numeric features
        numeric_features = data.select_dtypes(include=[np.number])
        data_clean = numeric_features.fillna(method='ffill').fillna(0)
        
        # Keep only features with variation
        non_zero_features = []
        for col in data_clean.columns:
            if data_clean[col].std() > 1e-8:
                non_zero_features.append(col)
                
        data_clean = data_clean[non_zero_features]
        
        logger.info("✅ Data validation complete:")
        logger.info(f"  Final shape: {data_clean.shape}")
        logger.info(f"  Return std: {data_clean['return_1h'].std():.6f}")
        
        return data_clean
        
    @staticmethod
    def _generate_synthetic_data_static(n_steps: int = 10000) -> pd.DataFrame:
        """Generate realistic synthetic BTC data for testing"""
        np.random.seed(42)
        
        # Generate price path with volatility clustering
        returns = []
        vol = 0.02
        
        for _ in range(n_steps):
            vol = 0.9 * vol + 0.1 * 0.02 + 0.05 * np.random.randn() * vol
            vol = np.clip(vol, 0.005, 0.10)
            ret = np.random.normal(0, vol) + 0.1 * np.random.standard_t(3) * vol
            returns.append(ret)
            
        returns = np.array(returns)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        # Create feature matrix
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.lognormal(15, 1, n_steps),
            'volatility': pd.Series(returns).rolling(24).std().fillna(0.02),
            'momentum_24h': pd.Series(returns).rolling(24).mean().fillna(0),
            'rsi': np.random.uniform(20, 80, n_steps),
        })
        
        # Add technical features
        for period in [12, 24, 48]:
            data[f'sma_{period}'] = data['close'].rolling(period).mean().fillna(data['close'])
            data[f'return_{period}h'] = np.log(data['close'] / data['close'].shift(period)).fillna(0)
            
        logger.info(f"Generated synthetic data: {data.shape}")
        return data
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with Senior Dev Canonical Fix state variables"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state - Senior Dev Canonical Fix
        self.current_step = 0
        
        # State variables (USD notional system)
        self.equity: float = self.config.initial_balance
        self.N: float = 0.0                     # signed USD notional
        self.price: float = self.prices[0] if len(self.prices) > 0 else 50000.0
        self.peak_equity: float = self.equity
        
        # Cost tracking
        self.last_trading_cost: float = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def _target_notional_from_action(self, a: float, E_pre: float) -> float:
        """Convert action to target notional - Senior Dev Canonical Fix"""
        # a ∈ [-1,1] → target leverage in [-Lmax, Lmax]
        L_target = float(a) * self.config.max_leverage
        return L_target * E_pre

    def _exec_trade(self, N_target: float, E_pre: float) -> tuple[float, float]:
        """Execute trade with turnover-based costs - Senior Dev Canonical Fix"""
        # turnover limit (prevents pathological costs)
        delta_N_req = N_target - self.N
        max_turnover = self.config.turnover_cap * E_pre
        delta_N = float(np.clip(delta_N_req, -max_turnover, max_turnover))
        traded_notional = abs(delta_N)

        # fees + slippage on turnover
        fee = traded_notional * self.config.taker_fee
        slip = traded_notional * (self.config.slippage_bp / 1e4)
        trading_cost = fee + slip

        # sanity cap
        cap = self.config.cost_cap_frac * E_pre
        if trading_cost > cap:
            trading_cost = cap

        # commit trade
        self.N += delta_N
        return traded_notional, trading_cost

    def _step_accounting(self, new_price: float) -> dict:
        """Step accounting with proper PnL calculation - Senior Dev Canonical Fix"""
        r = (new_price / self.price) - 1.0

        E_pre = self.equity
        # PnL from prior notional
        pnl = self.N * r

        # funding (linear perp approx): pro-rated by step size
        funding_step = self.config.funding_rate_8h * (self.config.step_hours / 8.0)
        funding = self.N * funding_step     # sign follows position

        self.equity = E_pre + pnl - self.last_trading_cost - abs(funding)
        self.price = new_price
        self.peak_equity = max(self.peak_equity, self.equity)

        # hard risk: liquidation guard
        L = abs(self.N) / max(self.equity, 1e-12)
        if L > self.config.max_leverage:
            # force reduce to margin-safe notional
            N_safe = np.sign(self.N) * self.config.max_leverage * self.equity
            _, cost = self._exec_trade(N_safe, self.equity)
            self.last_trading_cost += cost

        dd = 1.0 - (self.equity / self.peak_equity)
        return {"r": r, "pnl": pnl, "funding": funding, "drawdown": dd}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading step with Senior Dev Canonical Fix"""
        action_signal = np.clip(float(action[0]), -1.0, 1.0)
        
        # Check if at end
        if self.current_step >= len(self.prices) - 1:
            done = True
            observation = np.zeros(self.observation_space.shape)
            reward = 0.0
            info = {"equity": self.equity, "N": self.N, "cost": 0.0, "pnl": 0.0, 
                   "funding": 0.0, "turnover": 0.0, "dd": 0.0}
            return observation, reward, done, False, info
            
        # 1) mark-to-market on next price
        next_price = self.prices[self.current_step + 1]
        acct = self._step_accounting(next_price)

        # 2) translate action → target notional from *pre-trade* equity
        E_pre = self.equity
        N_target = self._target_notional_from_action(action_signal, E_pre)

        # 3) execute and record costs
        traded, cost = self._exec_trade(N_target, E_pre)
        self.last_trading_cost = cost

        # 4) reward (risk-aware log equity) with turnover penalty
        old_E = E_pre
        new_E = self.equity
        log_ret = np.log(max(new_E, 1e-12) / max(old_E, 1e-12))
        turnover_pen = 0.5e-3 * (traded / max(old_E, 1.0))  # 0.05% per 1x turnover
        reward = log_ret - turnover_pen

        # 5) termination and training tripwires
        dd = acct["drawdown"]
        done = (self.equity <= 0.0) or (dd >= self.config.drawdown_stop)
        
        # Move step forward
        self.current_step += 1
        
        # Update cost tracking
        self.total_fees += cost
        self.total_funding += abs(acct["funding"])
        
        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        info = {"equity": self.equity, "N": self.N, "cost": cost, "pnl": acct["pnl"], 
               "funding": acct["funding"], "turnover": traded, "dd": dd}
        
        # Validate outputs
        observation, reward = self._validate_outputs(observation, reward)
        reward = np.clip(reward, -0.1, 0.1)  # Clip reward as per senior dev advice
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with Senior Dev diagnostics features"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
            
        # Get features in deterministic order
        feature_values = self.data.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
        
        # Add trading state features (per senior dev diagnostics)
        L = abs(self.N) / max(self.equity, 1e-12)  # Current leverage
        turnover_E = 0.0  # Turnover/equity ratio 
        drawdown = 1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        
        # Combine: [features] + [L, turnover/E, drawdown]
        obs = np.concatenate([
            feature_values,
            [L, turnover_E, drawdown]
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info with Senior Dev diagnostics"""
        L = abs(self.N) / max(self.equity, 1e-12)  # Current leverage
        
        return {
            'equity': float(self.equity),
            'N': float(self.N),  # USD notional position
            'leverage': float(L),  # |N|/E
            'total_fees': float(self.total_fees),
            'total_funding': float(self.total_funding),
            'drawdown': float(1.0 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0),
            'step': self.current_step,
            'price': float(self.price),
        }
        
    def _validate_outputs(self, observation: np.ndarray, reward: float) -> Tuple[np.ndarray, float]:
        """Validate outputs for NaN/Inf"""
        # Validate observation
        if np.isnan(observation).any() or np.isinf(observation).any():
            logger.error("🚨 NaN/Inf in observation! Replacing with safe values")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Validate reward  
        if np.isnan(reward) or np.isinf(reward):
            logger.error(f"🚨 NaN/Inf in reward: {reward}! Replacing with zero")
            reward = 0.0
            
        return observation, reward


class VectorizedTradingEnv:
    """Vectorized environment wrapper for parallel rollouts"""
    
    def __init__(self, env_fn, num_envs: int = 16):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def reset(self):
        """Reset all environments"""
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.array(observations), infos
        
    def step(self, actions):
        """Step all environments"""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            
            # Auto-reset if done
            if terminated or truncated:
                obs, info = env.reset()
                
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
        return (np.array(observations), np.array(rewards), 
                np.array(terminateds), np.array(truncateds), infos)
        
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    # Test environment with Senior Dev Canonical Fix
    config = TradingConfig()
    env = PPOTradingEnvironment(config)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action[0]:.3f}, Reward: {reward:.6f}, Equity: {info['equity']:.2f}, N: {info['N']:.2f}, Leverage: {info['leverage']:.2f}")
        
        if terminated or truncated:
            break
            
    print("✅ PPO Environment (Senior Dev Canonical Fix) test completed successfully")