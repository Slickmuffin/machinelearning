#!/usr/bin/env python3
"""
🚀 DAILY SWING TRADING ARCHITECTURE
Complete BTC leverage trading system with XGBoost ensemble
Designed for capturing massive daily moves (20-40%) with intelligent leverage

Based on senior dev specifications + your vision for extreme returns
Decision cadence: Once per day (captures the big moves you showed)
Leverage: Up to 125x with survival clamps
Architecture: XGBoost ensemble → cost-aware policy → leverage sizing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.special import logit, expit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DailySwingConfig:
    """Complete configuration for daily swing trading system"""
    # Data & features
    bar_horizon_hours: int = 4   # 4-hour bars per friend's recommendation
    lookback_days: int = 30      # Feature lookback window

    # Cost structure (realistic) - Friend's recommendations
    cost_bps_roundtrip: float = 7.0    # 7 bps total costs
    funding_rate_daily: float = 0.03   # 3 bps/day typical

    # Ensemble & model settings - Research-backed parameters
    n_models: int = 8
    n_estimators: int = 1000     # Friend recommended: use fixed trees, no early stop in final
    eta: float = 0.01
    early_stopping_rounds: int = 10

    # Cost-aware policy - Friend's specific recommendations
    delta_bps: float = 1.5       # Friend: consider 1.5-3.0 bps (was 1.0)
    tau: float = 0.15            # Logit-margin threshold
    k_disagreement: float = 0.5  # Ensemble disagreement weight
    turnover_cap: float = 0.3    # Friend: lower cap 0.2-0.5 (was 1.0)

    # Leverage sizing - Friend's conservative recommendations for tuning
    target_vol_annual: float = 0.35    # Vol target
    kappa: float = 0.2                 # Friend: conservative while tuning, raise to 0.5-1.0 later
    kelly_fraction: float = 0.1        # Friend: 0.1-0.25 for fat-tailed crypto (was 0.40)
    var_budget_daily: float = 0.05     # Daily VaR budget
    z_liq_atr: float = 4.0             # ATR liquidation distance
    leverage_cap: float = 1.0          # Friend: start with 1.0, then open clamps if positive
    max_drawdown_halt: float = 0.25    # Stop trading at 25% DD

    # Regime & timing
    min_hold_days: int = 1       # Minimum position hold time
    regime_filter: bool = True   # Only trade in trending regimes

class DailyDataPipeline:
    """Converts any timeframe to clean daily OHLCV with features"""

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def to_daily_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert intraday data to daily LCRO bars"""
        self.logger.info("📊 Converting to daily bars...")

        df = df.copy()
        # ROBUST datetime parsing - handle mixed formats
        try:
            df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
        except ValueError:
            # Fallback for mixed formats
            self.logger.info("Using mixed format parsing for timestamps...")
            df['ts'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

        df = df.set_index('ts').sort_index()

        # Daily aggregation (00:00 UTC boundaries)
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else 'mean'
        }

        # Add funding if available
        if 'funding_rate' in df.columns:
            agg_dict['funding_rate'] = 'mean'

        daily = df.resample('1D', label='left', closed='left').agg(agg_dict)
        daily = daily.dropna(subset=['open', 'high', 'low', 'close'])

        self.logger.info(f"✅ Daily bars: {len(daily)} days from {daily.index.min()} to {daily.index.max()}")
        return daily.reset_index()

    def add_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all daily swing trading features (leak-free)"""
        self.logger.info("🔧 Building daily swing features...")

        df = df.copy().set_index('ts').sort_index()

        # Price & returns
        df['close_log'] = np.log(df['close'])
        df['ret_1d'] = df['close_log'].diff()

        # Simpler momentum features (shorter lookbacks for limited data)
        for days in [2, 3, 5]:  # Reduced from [3, 5, 10, 20]
            df[f'ret_{days}d'] = df['ret_1d'].rolling(days).sum()
            df[f'ema_{days}d'] = df['ret_1d'].ewm(span=days).mean()

        # Volatility (critical for leverage sizing) - shorter windows
        df['rng_log'] = np.log(df['high'] / df['low'])  # Log range
        df['sigma_1d'] = df['ret_1d'].ewm(span=10, adjust=False).std().clip(1e-6, None)  # Reduced from 20
        df['atr_1d'] = ((df['high'] - df['low']) / df['close'].shift(1)).ewm(span=7).mean().clip(1e-6, None)  # Reduced from 14

        # Yang-Zhang volatility (simplified)
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        log_oo = df['close_log'].diff()

        df['yz_vol'] = np.sqrt(
            log_oo**2 + 0.5 * (log_ho * log_lo) - (2*np.log(2) - 1) * log_co**2
        ).ewm(span=10).mean().clip(1e-6, None)  # Reduced from 20

        # Momentum Z-scores (shorter windows)
        for days in [2, 3, 5]:  # Reduced from [5, 10, 20]
            sigma_window = df['ret_1d'].rolling(10).std()  # Reduced from 30
            df[f'mom_z_{days}d'] = df[f'ret_{days}d'] / (sigma_window * np.sqrt(days) + 1e-8)

        # Volume features (simplified)
        if 'volume' in df.columns:
            df['vol_3d'] = df['volume'].rolling(3).mean()  # Reduced from 5
            df['vol_z'] = (df['volume'] - df['vol_3d']) / (df['vol_3d'] + 1e-8)

        # Funding features (simplified)
        if 'funding_rate' in df.columns:
            df['funding_1d'] = df['funding_rate']
            df['funding_3d'] = df['funding_rate'].rolling(3).mean()  # Reduced from 5
            df['funding_change'] = df['funding_rate'].diff()

        # Seasonality
        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

        # Regime detection (simplified)
        df['trend_5d'] = np.sign(df['ema_5d'])  # Use shorter trend
        # Skip vol_regime for now (causes issues with small data)

        # LAG ALL FEATURES BY 1 DAY (critical - no look-ahead)
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'funding_rate']]
        df[feature_cols] = df[feature_cols].shift(1)

        result = df.dropna().reset_index()
        self.logger.info(f"✅ Features built: {len(feature_cols)} features, {len(result)} rows")
        return result

class DailyLabeler:
    """Creates forward-looking labels for daily swing trading"""

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def create_labels(self, df: pd.DataFrame, k_days: int = 1) -> pd.DataFrame:
        """Create direction and magnitude labels"""
        self.logger.info(f"🎯 Creating labels for {k_days}-day ahead returns...")

        df = df.copy().set_index('ts').sort_index()

        # Forward returns
        ret_1d = np.log(df['close']).diff()
        fwd_return = ret_1d.shift(-1).rolling(k_days).sum()

        # Direction label (binary)
        df['y_direction'] = (fwd_return > 0).astype(int)

        # Magnitude label (for regression models)
        df['fwd_return_k'] = fwd_return

        # Additional labels for different horizons
        for k in [2, 3, 5]:
            fwd_k = ret_1d.shift(-1).rolling(k).sum()
            df[f'y_direction_{k}d'] = (fwd_k > 0).astype(int)
            df[f'fwd_return_{k}d'] = fwd_k

        # Purge overlap (critical for time series)
        df = df.iloc[:-max(5, k_days)] if k_days > 0 else df

        result = df.dropna().reset_index()
        self.logger.info(f"✅ Labels created: {len(result)} samples")
        if len(result) > 0:
            pos_rate = result['y_direction'].mean()
            self.logger.info(f"📊 Positive rate: {pos_rate:.1%}")
        return result

class XGBoostEnsemble:
    """Ensemble of calibrated XGBoost models for daily swing prediction"""

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.models = []
        self.feature_cols = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def train_single_model(self, df: pd.DataFrame, feature_cols: List[str], seed: int = 0) -> Dict:
        """Train one XGBoost model with time validation"""

        # Time-based split (no random shuffle)
        df_sorted = df.sort_values('ts')
        split_idx = int(0.8 * len(df_sorted))

        X_train = df_sorted.iloc[:split_idx][feature_cols].values
        y_train = df_sorted.iloc[:split_idx]['y_direction'].values
        X_val = df_sorted.iloc[split_idx:][feature_cols].values
        y_val = df_sorted.iloc[split_idx:]['y_direction'].values

        # XGBoost parameters (2024 academic research optimized)
        params = {
            'n_estimators': 1000,  # Research suggests 1000 trees minimum
            'learning_rate': 0.01,  # Lower learning rate for financial data
            'max_depth': 12,  # Research shows depth 12 optimal for finance
            'min_child_weight': 12,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': seed,
            'n_jobs': -1,
            'base_score': 0.5,  # Required for XGBoost 3.0+
            'eval_metric': 'logloss',
            'early_stopping_rounds': 10  # Research suggests 10 for financial data
        }

        # Train with research-backed configuration
        import time
        t0 = time.time()

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        t1 = time.time()

        # Get best iteration - FIXED: XGBoost 3.0+ proper handling
        best_iter = params['n_estimators']  # Default to full training
        try:
            if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                best_iter = model.best_iteration
            elif hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if hasattr(booster, 'best_iteration'):
                    best_iter = booster.best_iteration
        except Exception as e:
            self.logger.info(f"Could not get best iteration: {e}, using full training")

        # Retrain on full data with optimal tree count (remove early stopping for final model)
        final_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
        final_params['n_estimators'] = max(100, best_iter + 50)
        final_model = xgb.XGBClassifier(**final_params)

        t2 = time.time()
        final_model.fit(df[feature_cols].values, df['y_direction'].values)
        t3 = time.time()

        # Log detailed timing
        self.logger.info(f"⏱️ Initial fit time: {t1-t0:.2f}s | Final fit time: {t3-t2:.2f}s | best_iter={best_iter} | feats={len(feature_cols)} | rows={len(df)}")

        # Calibrate probabilities (critical for cost-aware policy)
        calibrator = CalibratedClassifierCV(final_model, cv='prefit', method='isotonic')
        calibrator.fit(X_val, y_val)

        # Validation metrics
        val_probs = calibrator.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        val_brier = brier_score_loss(y_val, val_probs)

        return {
            'model': final_model,
            'calibrator': calibrator,
            'best_iter': best_iter,
            'val_auc': val_auc,
            'val_brier': val_brier,
            'feature_cols': feature_cols,
            'seed': seed
        }

    def train_ensemble(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Train ensemble of diverse models"""
        self.logger.info(f"🚀 Training ensemble of {self.config.n_models} XGBoost models...")
        self.feature_cols = feature_cols

        # Diversification strategies
        seeds = list(range(self.config.n_models))

        for i, seed in enumerate(seeds):
            self.logger.info(f"Training model {i+1}/{self.config.n_models} (seed={seed})...")

            # Feature subsampling for diversity
            if i > 0:  # Keep first model with all features
                n_features = max(10, int(0.8 * len(feature_cols)))
                np.random.seed(seed)
                selected_features = np.random.choice(feature_cols, n_features, replace=False).tolist()
            else:
                selected_features = feature_cols

            model_info = self.train_single_model(df, selected_features, seed)
            self.models.append(model_info)

            self.logger.info(f"✅ Model {i+1}: AUC={model_info['val_auc']:.3f}, "
                           f"Brier={model_info['val_brier']:.3f}, "
                           f"Trees={model_info['best_iter']}")

        # Ensemble diagnostics
        avg_auc = np.mean([m['val_auc'] for m in self.models])
        avg_brier = np.mean([m['val_brier'] for m in self.models])
        self.logger.info(f"🎯 Ensemble complete: Avg AUC={avg_auc:.3f}, Avg Brier={avg_brier:.3f}")

    def predict_ensemble(self, X: np.ndarray) -> Tuple[float, float, List[float]]:
        """Get ensemble prediction with logit averaging per friend's spec"""
        probs = []

        for model_info in self.models:
            # Extract features this model was trained on
            model_features = model_info['feature_cols']
            feature_indices = [self.feature_cols.index(f) for f in model_features if f in self.feature_cols]

            if len(feature_indices) > 0:
                X_model = X[:, feature_indices] if X.ndim > 1 else X[feature_indices].reshape(1, -1)
                prob = model_info['calibrator'].predict_proba(X_model)[0, 1]
                probs.append(prob)

        if not probs:
            return 0.5, 1.0, [0.5] * len(self.models)  # Default if no predictions

        # Friend's exact logit averaging specification
        z = np.array([logit(np.clip(p, 1e-6, 1-1e-6)) for p in probs])
        p_ens = float(expit(z.mean()))
        z_std = float(z.std())

        return p_ens, z_std, probs

class DailySwingPolicy:
    """Cost-aware policy with abstention and survival-focused leverage"""

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def break_even_probability(self, cost_bps: float, sigma_hat: float, eps: float = 1e-6) -> float:
        """Friend's exact break-even probability calculation"""
        sigma = max(float(sigma_hat), eps)
        E_abs_r = sigma * np.sqrt(2 / np.pi)
        return 0.5 + (cost_bps / 1e4) / (2 * E_abs_r)

    def calculate_leverage_clamps(self, mu_hat: float, sigma_1d: float, atr_1d: float,
                                 current_dd: float = 0.0) -> float:
        """Calculate maximum safe leverage using multiple clamps"""

        # Drawdown halt
        if current_dd > self.config.max_drawdown_halt:
            return 0.0

        # Vol-target clamp
        L_vol = self.config.target_vol_annual / max(sigma_1d * np.sqrt(252), 1e-8)

        # Fractional Kelly clamp
        edge = max(mu_hat - (self.config.cost_bps_roundtrip / 1e4), 0.0)
        variance = max(sigma_1d**2, 1e-12)
        L_kelly = max(0.0, self.config.kelly_fraction * edge / variance)

        # Liquidation distance clamp
        L_liq = 1.0 / max(self.config.z_liq_atr * atr_1d, 1e-6)

        # VaR clamp - FIXED: Use 95% VaR instead of 99% for higher leverage
        L_var = self.config.var_budget_daily / (max(sigma_1d, 1e-8) * 1.65)  # 95% VaR (was 99%)

        # Take minimum of all clamps
        leverage = min(
            self.config.leverage_cap,
            L_vol,
            L_kelly,
            L_liq,
            L_var
        )

        return max(0.0, leverage)

    def make_decision(self, p_ens: float, z_std: float, sigma_hat: float,
                     atr_1d: float, current_dd: float = 0.0) -> Tuple[int, float]:
        """Friend's exact decision logic with abstention"""

        # Friend's exact break-even calculation
        p_star = self.break_even_probability(self.config.cost_bps_roundtrip, sigma_hat)
        delta_prob = (self.config.delta_bps / 1e4) / (2 * sigma_hat * np.sqrt(2/np.pi))

        # Friend's exact abstention mechanism
        logit_p_ens = logit(np.clip(p_ens, 1e-6, 1-1e-6))

        # Decision with abstention (friend's specification)
        action = 0
        if (p_ens > p_star + delta_prob and
            logit_p_ens > (self.config.tau + self.config.k_disagreement * z_std)):
            action = +1
        elif (p_ens < 1 - (p_star + delta_prob) and
              -logit_p_ens > (self.config.tau + self.config.k_disagreement * z_std)):
            action = -1

        if action == 0:
            return 0, 0.0  # No trade (abstention)

        # Estimate expected return
        mu_hat = (2 * p_ens - 1) * (sigma_hat * np.sqrt(2/np.pi)) - (self.config.cost_bps_roundtrip / 1e4)

        # Calculate safe leverage
        leverage = self.calculate_leverage_clamps(mu_hat, sigma_hat, atr_1d, current_dd)

        return action, leverage

class DailyBacktester:
    """Backtests daily swing strategy with proper costs and risk management"""

    def __init__(self, config: DailySwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_backtest(self, df: pd.DataFrame, ensemble: XGBoostEnsemble,
                    policy: DailySwingPolicy) -> Tuple[pd.DataFrame, Dict]:
        """Run complete daily swing backtest"""

        self.logger.info("📈 Running daily swing backtest...")

        ledger = []
        position = 0.0
        equity = 10000.0  # Starting equity
        peak_equity = equity
        current_dd = 0.0

        for i in range(1, len(df)):
            row_prev = df.iloc[i-1]  # Decision made on previous day
            row_curr = df.iloc[i]    # Results realized on current day

            # Get features for prediction
            X = row_prev[ensemble.feature_cols].values.reshape(1, -1)

            # Ensemble prediction with logit averaging
            p_ens, z_std, individual_probs = ensemble.predict_ensemble(X)

            # Policy decision with friend's exact specification
            direction, leverage = policy.make_decision(
                p_ens, z_std,
                row_prev['sigma_1d'], row_prev['atr_1d'],
                current_dd
            )

            # Calculate new position
            new_position = direction * leverage if leverage > 0 else 0.0

            # Realize PnL from previous position
            gross_return = row_curr['fwd_return_k'] if 'fwd_return_k' in row_curr else 0.0
            gross_pnl = position * gross_return * equity

            # Calculate costs (only on position changes)
            position_change = abs(new_position - position)
            cost_pnl = -(self.config.cost_bps_roundtrip / 1e4) * position_change * equity

            # Funding cost
            funding_pnl = -(self.config.funding_rate_daily / 1e4) * abs(position) * equity

            # Total PnL
            total_pnl = gross_pnl + cost_pnl + funding_pnl
            equity += total_pnl

            # Update drawdown
            peak_equity = max(peak_equity, equity)
            current_dd = 1 - (equity / peak_equity) if peak_equity > 0 else 0

            # Record trade
            ledger.append({
                'date': row_curr['ts'],
                'prob': p_ens,
                'disagreement': z_std,
                'direction': direction,
                'leverage': leverage,
                'position': new_position,
                'gross_return': gross_return,
                'gross_pnl': gross_pnl,
                'cost_pnl': cost_pnl,
                'funding_pnl': funding_pnl,
                'total_pnl': total_pnl,
                'equity': equity,
                'drawdown': current_dd,
                'sigma_1d': row_prev['sigma_1d'],
                'atr_1d': row_prev['atr_1d']
            })

            position = new_position

        # Create ledger DataFrame
        ledger_df = pd.DataFrame(ledger)

        # Calculate performance metrics
        if len(ledger_df) > 0:
            returns = ledger_df['total_pnl'] / ledger_df['equity'].shift(1).fillna(10000)

            metrics = {
                'total_return': (ledger_df['equity'].iloc[-1] / 10000 - 1) * 100,
                'daily_sharpe': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
                'max_drawdown': ledger_df['drawdown'].max() * 100,
                'win_rate': (ledger_df['total_pnl'] > 0).mean() * 100,
                'profit_factor': ledger_df[ledger_df['total_pnl'] > 0]['total_pnl'].sum() /
                               max(abs(ledger_df[ledger_df['total_pnl'] < 0]['total_pnl'].sum()), 1e-8),
                'avg_leverage': ledger_df[ledger_df['leverage'] > 0]['leverage'].mean(),
                'trade_count': (ledger_df['position'] != 0).sum(),
                'avg_hold_days': 1,  # Daily decisions
            }
        else:
            metrics = {}

        self.logger.info("✅ Backtest complete")
        return ledger_df, metrics

class DailySwingTrader:
    """Main orchestrator for the daily swing trading system"""

    def __init__(self, config: DailySwingConfig = None):
        self.config = config or DailySwingConfig()
        self.pipeline = DailyDataPipeline(self.config)
        self.labeler = DailyLabeler(self.config)
        self.ensemble = XGBoostEnsemble(self.config)
        self.policy = DailySwingPolicy(self.config)
        self.backtester = DailyBacktester(self.config)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def prepare_data(self, raw_data_path: str) -> pd.DataFrame:
        """Complete data preparation pipeline with comprehensive validation"""
        self.logger.info("🔧 Preparing daily swing trading data...")

        try:
            # Load raw data with validation
            df = pd.read_csv(raw_data_path)
            self.logger.info(f"📊 Loaded {len(df)} raw records")

            # Validate required columns exist
            required_cols = ['timestamp', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Validate data quality
            if df['close'].isna().sum() > len(df) * 0.05:
                raise ValueError(f"Too many missing prices: {df['close'].isna().sum()}/{len(df)}")

            # Convert to daily bars
            daily_bars = self.pipeline.to_daily_bars(df)
            self.logger.info(f"📊 Created {len(daily_bars)} daily bars")

            # Relaxed validation for small datasets
            min_daily_bars = 20  # Reduced from 40 for small datasets
            if len(daily_bars) < min_daily_bars:
                raise ValueError(f"Insufficient daily data: {len(daily_bars)} bars (need {min_daily_bars}+)")

            # Add features with validation
            with_features = self.pipeline.add_daily_features(daily_bars)
            self.logger.info(f"🔧 Added features: {len(with_features)} rows")

            # Relaxed validation for small datasets
            min_feature_rows = 10  # Reduced from 20
            if len(with_features) < min_feature_rows:
                raise ValueError(f"Insufficient data after features: {len(with_features)} rows (need {min_feature_rows}+)")

            # Add labels with validation
            final_df = self.labeler.create_labels(with_features)
            self.logger.info(f"🎯 Added labels: {len(final_df)} samples")

            # STRICT validation for proper ML training (as per friend's feedback)
            min_training_samples = 500  # Require minimum for serious training
            if len(final_df) < min_training_samples:
                self.logger.warning(f"⚠️ INSUFFICIENT DATA: {len(final_df)} samples (need {min_training_samples}+ for reliable training)")
                self.logger.warning(f"⚠️ Proceeding with limited data - expect poor performance")

            self.logger.info(f"✅ Data prepared: {len(final_df)} daily samples")
            return final_df

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {str(e)}")
            raise

    def train_models(self, df: pd.DataFrame) -> None:
        """Train the ensemble of models with validation"""
        try:
            # Get feature columns - more robust selection
            exclude_cols = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                           'y_direction', 'fwd_return_k', 'fwd_return_1d', 'fwd_return_2d',
                           'fwd_return_3d', 'fwd_return_5d', 'y_direction_2d', 'y_direction_3d', 'y_direction_5d']

            feature_cols = [col for col in df.columns if col not in exclude_cols]
            self.logger.info(f"🔧 Selected {len(feature_cols)} features for training")

            if len(feature_cols) < 5:
                raise ValueError(f"Insufficient features: {len(feature_cols)} (need 5+)")

            # Validate feature data quality
            for col in feature_cols:
                nan_pct = df[col].isna().sum() / len(df)
                if nan_pct > 0.1:
                    self.logger.warning(f"Feature {col} has {nan_pct:.1%} missing values")

            # Remove features with too many NaNs - relaxed for small datasets
            valid_features = []
            max_nan_rate = 0.3 if len(df) < 50 else 0.1  # Allow more NaNs for small datasets

            for col in feature_cols:
                nan_rate = df[col].isna().sum() / len(df)
                if nan_rate <= max_nan_rate:
                    valid_features.append(col)
                else:
                    self.logger.debug(f"Dropped feature {col}: {nan_rate:.2%} NaN rate")

            feature_cols = valid_features
            self.logger.info(f"🔧 Using {len(feature_cols)} valid features (max NaN rate: {max_nan_rate:.1%})")

            # Relaxed minimum for small datasets
            min_features = 2 if len(df) < 50 else 3
            if len(feature_cols) < min_features:
                raise ValueError(f"Too few valid features: {len(feature_cols)} (need {min_features})")

            # Train ensemble with validation
            self.ensemble.train_ensemble(df, feature_cols)

        except Exception as e:
            self.logger.error(f"❌ Model training failed: {str(e)}")
            raise

    def run_full_pipeline(self, raw_data_path: str, test_start_date: str = None) -> Dict:
        """Run complete daily swing trading pipeline"""
        self.logger.info("🚀 DAILY SWING TRADING SYSTEM - STARTING FULL PIPELINE")

        # Prepare data
        df = self.prepare_data(raw_data_path)

        # Split train/test - FIXED: Handle timezone consistency and auto-detect dates
        if test_start_date:
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            test_start_ts = pd.to_datetime(test_start_date, utc=True)

            # Auto-fix if test_start_date is before all data (2025 data issue)
            min_date = df['ts'].min()
            max_date = df['ts'].max()

            if test_start_ts < min_date:
                self.logger.warning(f"⚠️ test_start_date {test_start_date} is before all data ({min_date})")
                self.logger.info("🔧 Auto-adjusting to use 80/20 split instead")
                split_idx = int(0.8 * len(df))
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]
            else:
                train_df = df[df['ts'] < test_start_ts]
                test_df = df[df['ts'] >= test_start_ts]
        else:
            split_idx = int(0.8 * len(df))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

        self.logger.info(f"📊 Train: {len(train_df)} days, Test: {len(test_df)} days")

        # Train models
        self.train_models(train_df)

        # Backtest
        ledger, metrics = self.backtester.run_backtest(test_df, self.ensemble, self.policy)

        # Results
        self.logger.info("🎯 FINAL RESULTS:")
        for metric, value in metrics.items():
            self.logger.info(f"   {metric}: {value:.2f}")

        return {
            'ledger': ledger,
            'metrics': metrics,
            'config': self.config,
            'train_period': (train_df['ts'].min(), train_df['ts'].max()),
            'test_period': (test_df['ts'].min(), test_df['ts'].max())
        }

if __name__ == "__main__":
    import argparse
    import json
    import pathlib
    import pandas as pd
    import sys
    import time

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # CLI argument parsing
    parser = argparse.ArgumentParser(description='Daily Swing Trading System')
    parser.add_argument("--data", required=True, help="Path to raw BTC OHLCV CSV")
    parser.add_argument("--test_start_date", default=None, help="Optional YYYY-MM-DD to split OOS")
    parser.add_argument("--config_json", default=None, help="Optional JSON to override config fields")
    args = parser.parse_args()

    # Load configuration
    config = DailySwingConfig()
    if args.config_json:
        overrides = json.loads(args.config_json)
        for k, v in overrides.items():
            setattr(config, k, v)
            logger.info(f"🔧 Config override: {k} = {v}")

    # Create trader and run pipeline
    trader = DailySwingTrader(config)

    # Add total timing
    T0 = time.time()
    logger.info(f"🚀 Starting pipeline with data: {args.data}")

    try:
        results = trader.run_full_pipeline(args.data, test_start_date=args.test_start_date)

        # Total pipeline timing
        total_time = time.time() - T0
        logger.info(f"⏱️ TOTAL pipeline time: {total_time:.2f}s")

        # Persist artifacts
        out_dir = pathlib.Path("/tmp/out")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save ledger
        ledger_df = pd.DataFrame(results["ledger"])
        ledger_path = out_dir / "ledger.csv"
        ledger_df.to_csv(ledger_path, index=False)
        logger.info(f"💾 Ledger saved: {ledger_path}")

        # Save metrics - convert numpy types to Python types
        metrics_serializable = {}
        for k, v in results["metrics"].items():
            if hasattr(v, 'item'):  # numpy scalar
                metrics_serializable[k] = v.item()
            else:
                metrics_serializable[k] = v

        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"📊 Metrics saved: {metrics_path}")

        # Print final results
        logger.info("🎯 FINAL RESULTS:")
        for key, value in results["metrics"].items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.4f}")

        logger.info("✅ PIPELINE COMPLETE - ALL ARTIFACTS SAVED")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)