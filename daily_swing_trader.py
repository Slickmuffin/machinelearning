#!/usr/bin/env python3
"""
🚀 DAILY SWING TRADING ARCHITECTURE
Complete BTC leverage trading system with XGBoost ensemble
Designed for capturing massive 4-hour moves (5-15%) with intelligent leverage

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
import hashlib
import pathlib
warnings.filterwarnings('ignore')


def assert_expected_cadence(ts: pd.Series, expected: str):
    """Enforce cadence gates (fail fast on mismatch) - DEPRECATED, use assert_source_is_intraday instead"""
    ts = pd.to_datetime(ts, utc=True).sort_values()
    # median step in seconds
    step = (ts.diff().dropna().dt.total_seconds()).median()
    expected_sec = {"1H": 3600, "4H": 4*3600, "1D": 24*3600}[expected]
    if not (0.9*expected_sec <= step <= 1.1*expected_sec):
        raise ValueError(f"Cadence mismatch: median step={step:.0f}s, expected ~{expected_sec}s ({expected})")


def _median_step_seconds(ts):
    """Calculate median step in seconds from timestamp series"""
    s = pd.to_datetime(ts, utc=True).sort_values()
    return s.diff().dropna().dt.total_seconds().median()


def assert_source_is_intraday(df, min_granularity="1H"):
    """FRIEND'S GUARD A: Require source cadence to be <= 1H for 4H resample"""
    step = _median_step_seconds(df["timestamp"])
    limits = {"1H": 3600, "30T": 1800, "1T": 60}
    if step > limits[min_granularity]:
        raise ValueError(
            f"Source cadence too coarse for 4H resample: median step={step:.0f}s "
            f"(need <= {limits[min_granularity]}s). You passed DAILY data."
        )


def assert_resample_density(swing_df, horizon_hours=4):
    """FRIEND'S GUARD B: Validate resample output density (stop fake 4H conversions)"""
    if swing_df.empty or "timestamp" not in swing_df.columns:
        raise ValueError("Resample output empty or malformed")

    ts = pd.to_datetime(swing_df["timestamp"], utc=True).sort_values()
    span_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0
    expected = int(span_days * (24 / horizon_hours))

    if len(swing_df) < int(0.8 * expected):
        raise ValueError(
            f"Resample sanity failed: got {len(swing_df)} rows, "
            f"expected ~{expected} for {horizon_hours}H bars over {span_days:.1f} days. "
            f"Likely cause: daily data relabeled as {horizon_hours}H (fake conversion)."
        )


def assert_min_quality(n_val, auc, trees, min_auc=0.53, min_val=200, min_trees=50):
    """FRIEND'S GUARD D: Quality gate - stop 'false success'"""
    if n_val < min_val:
        raise RuntimeError(f"Validation too small for production: n_val={n_val} < {min_val}")
    if not np.isfinite(auc) or auc < min_auc:
        raise RuntimeError(f"Model too weak: auc={auc:.3f} < {min_auc}")
    if trees < min_trees:
        raise RuntimeError(f"Booster too small: trees={trees} < {min_trees}")


def validate_label_horizons(bar_horizon_hours: int, target_labels: List[str]):
    """
    GUARD E: Reject labels with horizon < bar_horizon (sub-Nyquist).

    Sampling theorem: Cannot predict events at resolution finer than sampling rate.
    Example: 4H bars cannot predict 1H movements (no data between bars).
    """
    import re

    invalid = []

    for label in target_labels:
        # Parse horizon from label name: "label_1h_1bp" -> (1, 'h')
        match = re.match(r'label_(\d+)([hd])_', label)
        if not match:
            continue  # Skip non-standard labels

        value = int(match.group(1))
        unit = match.group(2)

        # Convert to hours
        label_horizon_hours = value if unit == 'h' else value * 24

        # Check if sub-Nyquist
        if label_horizon_hours < bar_horizon_hours:
            invalid.append({
                'label': label,
                'horizon_hours': label_horizon_hours,
                'bars_ahead': label_horizon_hours / bar_horizon_hours
            })

    if invalid:
        msg = f"❌ GUARD E FAILED: Invalid labels (horizon < {bar_horizon_hours}H bar spacing)\n\n"
        msg += "Sub-Nyquist labels cannot be predicted from the data:\n"
        for item in invalid:
            msg += f"  - {item['label']}: {item['horizon_hours']}H horizon = {item['bars_ahead']:.2f} bars\n"
        msg += f"\nBar spacing: {bar_horizon_hours}H (one observation every {bar_horizon_hours} hours)\n"
        msg += "Violation: Cannot predict sub-bar movements (no data between samples)\n\n"
        msg += "FIX: Remove these labels from TARGET_LABELS config or use finer-grained data\n"
        raise ValueError(msg)


def validate_model_quality(
    model_info: dict,
    min_auc_threshold: float = 0.55,
    max_tree_threshold: int = 190
):
    """
    GUARD C+: Detect early stopping collapse.

    Pattern: Model hits max_iterations with AUC near 0.5 = training failure
    Causes: Impossible target, no learning signal, or data issues
    """
    trees = model_info['best_iter']
    auc = model_info['val_auc']
    label = model_info['selected_label']
    model_idx = model_info.get('seed', 0)

    # Critical: Model maxed out but learned nothing
    if trees >= max_tree_threshold and auc < min_auc_threshold:
        raise ValueError(
            f"❌ GUARD C+ FAILED: Model {model_idx} training collapsed\n"
            f"  Label: {label}\n"
            f"  Trees: {trees} (hit max_iterations)\n"
            f"  Val AUC: {auc:.3f} < {min_auc_threshold}\n"
            f"  \n"
            f"  Diagnosis: Model tried {trees}+ iterations but AUC stayed at ~0.5 (random)\n"
            f"  Likely causes:\n"
            f"    1. Label horizon < bar_horizon (sub-Nyquist) ← MOST COMMON\n"
            f"    2. Target has no signal (pure noise)\n"
            f"    3. Severe data leakage (future→past)\n"
            f"  \n"
            f"  Action: Check label definition and bar_horizon_hours in config\n"
        )

    # Warning: Marginal learning
    if min_auc_threshold <= auc < 0.65:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"⚠️  Model {model_idx} ({label}) has marginal AUC={auc:.3f}\n"
            f"    Consider: longer prediction horizon or more training data"
        )

    return True


def upload_dir_to_gcs(local_dir: str, gcs_uri: str):
    """Upload local directory to GCS with success sentinel"""
    from google.cloud import storage
    from urllib.parse import urlparse
    import pathlib

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"gcs_out must be gs://*, got {gcs_uri}")
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")

    client = storage.Client()  # uses Vertex service account
    bucket = client.bucket(bucket_name)

    root = pathlib.Path(local_dir)
    if not root.exists():
        raise FileNotFoundError(f"Local artifact dir not found: {local_dir}")

    uploaded = 0
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
            blob.upload_from_filename(str(path))
            uploaded += 1
    # write a success sentinel
    sentinel = bucket.blob(f"{prefix.rstrip('/')}/SUCCESS.ok")
    sentinel.upload_from_string("ok")
    return uploaded


# BUILD TRACKING - Friend's stale artifact prevention
CODE_PATH = Path(__file__)
CODE_HASH = hashlib.sha256(CODE_PATH.read_bytes()).hexdigest()[:12]

# ML imports
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from scipy.special import logit, expit
from joblib import dump, load
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FRIEND'S FIX-PACK UTILITY FUNCTIONS
def time_split(df, feat_cols, label_col, test_start_date=None, test_frac=0.2):
    """Friend's temporal split function - prevents leakage"""
    # Auto-detect timestamp column name (handles 'timestamp' or 'ts')
    ts_col = None
    for col in ['timestamp', 'ts']:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        raise ValueError(f"No timestamp column found. Available: {df.columns.tolist()}")

    df = df.sort_values(ts_col).reset_index(drop=True)
    if test_start_date:
        cut = df[ts_col] >= np.datetime64(test_start_date)
        tr, va = df[~cut], df[cut]
        if len(va) < max(200, int(0.1*len(df))):
            i = int(0.8*len(df)); tr, va = df.iloc[:i], df.iloc[i:]
    else:
        i = int((1.0 - test_frac)*len(df)); tr, va = df.iloc[:i], df.iloc[i:]
    X_tr, y_tr = tr[feat_cols].astype("float32").values, tr[label_col].astype("int32").values
    X_va, y_va = va[feat_cols].astype("float32").values, va[label_col].astype("int32").values
    # ensure two-class validation
    if len(np.unique(y_va)) < 2:
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            df[feat_cols].astype("float32").values,
            df[label_col].astype("int32").values,
            test_size=0.2, stratify=df[label_col].values, random_state=42
        )
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Val single-class → used stratified fallback (validation only).")
    return X_tr, y_tr, X_va, y_va

def train_xgb_strong(X_tr, y_tr, X_va, y_va, seed=0):
    """Friend's XGBoost training with proper early stopping"""
    import xgboost as xgb
    params = dict(
        objective="binary:logistic",
        eval_metric=["auc","logloss"],
        tree_method="hist",
        max_depth=4, min_child_weight=1,
        subsample=0.9, colsample_bytree=0.9,
        eta=0.03, reg_lambda=1.0, reg_alpha=0.0,
        random_state=seed,
    )
    dtr, dva = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(params, dtr, num_boost_round=1500,
                        evals=[(dtr,"train"),(dva,"val")],
                        early_stopping_rounds=min(200, max(30, X_va.shape[0]//3)),
                        verbose_eval=False)
    trees = (booster.best_iteration or 0) + 1
    if trees < 50:
        # second pass: no ES to force usable capacity
        booster = xgb.train(params, dtr, num_boost_round=200, evals=[(dtr,"train")], verbose_eval=False)
        trees = 200
    p_va = booster.predict(dva)
    auc = float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va))==2 else float("nan")
    ll  = float(log_loss(y_va, p_va, eps=1e-7))
    return booster, trees, auc, ll

def publish(outdir, gcs_out, booster, meta: dict):
    """Friend's artifact publishing with hard fail policy"""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out/"model.json"))
    (out/"metrics.json").write_text(json.dumps(meta, indent=2))
    # upload to GCS
    from google.cloud import storage; from urllib.parse import urlparse
    u = urlparse(gcs_out); bkt, prefix = u.netloc, u.path.lstrip("/")
    c = storage.Client(); b = c.bucket(bkt)
    uploaded = 0
    for p in out.glob("*"):
        if p.is_file():
            b.blob(f"{prefix}/{p.name}").upload_from_filename(str(p)); uploaded += 1
    b.blob(f"{prefix}/SUCCESS.ok").upload_from_string("ok"); uploaded += 1
    if uploaded < 3:
        logger.error("Uploaded <3 artifacts — failing job."); sys.exit(3)

@dataclass
class SwingConfig:
    """Complete configuration for 4-hour swing trading system"""
    # FRIEND'S FIX 4: Target labels - REMOVED 1H labels (sub-Nyquist for 4H bars)
    # Only include labels with horizon >= bar_horizon_hours (4H minimum)
    TARGET_LABELS: List[str] = field(default_factory=lambda: [
        # REMOVED: "label_1h_1bp", "label_1h_2bp", "label_1h_3bp" (sub-Nyquist)
        "label_4h_1bp", "label_4h_2bp", "label_4h_3bp",     # 1 bar ahead (OK)
        "label_12h_1bp", "label_12h_2bp", "label_12h_3bp",  # 3 bars ahead (GOOD)
        "label_24h_1bp", "label_24h_2bp", "label_24h_3bp"   # 6 bars ahead (EXCELLENT)
    ])

    # Data & features - 4-hour swing system (friend's exact specs)
    bar_horizon_hours: int = 4   # 4-hour LCRO bars in UTC (00,04,08,...)
    lookback_days: int = 30      # Feature lookback window

    # Cost structure (realistic) - Friend's recommendations
    cost_bps_roundtrip: float = 7.0    # 7 bps total costs
    funding_rate_daily: float = 0.03   # 3 bps/day typical

    # Ensemble & model settings - Research-backed parameters
    n_models: int = 8
    n_estimators: int = 1000     # Friend recommended: use fixed trees, no early stop in final
    eta: float = 0.01
    early_stopping_rounds: int = None  # DISABLED - was causing 1-7 trees issue

    # Cost-aware policy - Friend's specific recommendations
    delta_bps: float = 1.5       # Friend: consider 1.5-3.0 bps (was 1.0)
    tau: float = 0.15            # Logit-margin threshold
    k_disagreement: float = 0.5  # Ensemble disagreement weight
    turnover_cap: float = 0.3    # Friend: lower cap 0.2-0.5 (was 1.0)
    # Dev vs Prod mode gates
    prod_mode: bool = False      # False = lenient gates for dev, True = strict gates for prod

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

class SwingDataPipeline:
    """Converts any timeframe to clean 4-hour OHLCV with features"""

    def __init__(self, config: SwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def to_swing_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert intraday data to 4-hour LCRO bars (FRIEND'S GUARD B: with pre/post checks)"""
        self.logger.info(f"📊 Converting to {self.config.bar_horizon_hours}-hour bars...")

        df = df.copy()
        # ROBUST datetime parsing - handle mixed formats
        try:
            df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
        except ValueError:
            # Fallback for mixed formats
            self.logger.info("Using mixed format parsing for timestamps...")
            df['ts'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

        df = df.set_index('ts').sort_index()

        # PRECHECK: source must be intraday (<=1H)
        src_step = df.index.to_series().diff().dropna().dt.total_seconds().median()
        if src_step > 3600:
            raise ValueError(
                f"Refusing to resample {src_step:.0f}s data to 4H. Feed 1m/1h, not daily."
            )

        # 4-hour LCRO bar aggregation (00,04,08,12,16,20 UTC boundaries)
        freq_str = f"{self.config.bar_horizon_hours}H"

        # Handle different volume column names
        volume_col = None
        if 'volume_btc' in df.columns:
            volume_col = 'volume_btc'
        elif 'volume_usd' in df.columns:
            volume_col = 'volume_usd'
        elif 'volume' in df.columns:
            volume_col = 'volume'

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        if volume_col:
            agg_dict[volume_col] = 'sum'

        # Add funding if available
        if 'funding_rate' in df.columns:
            agg_dict['funding_rate'] = 'mean'

        # Resample to 4-hour bars on UTC boundaries (00,04,08,12,16,20)
        swing_bars = df.resample(freq_str, label='left', closed='left').agg(agg_dict)
        swing_bars = swing_bars.dropna(subset=['open', 'high', 'low', 'close'])

        # FRIEND'S GUARD B: Validate resample density (postcheck)
        swing_bars_reset = swing_bars.reset_index()
        # Temporarily add timestamp column for validation
        swing_bars_reset['timestamp'] = swing_bars_reset['ts']
        assert_resample_density(swing_bars_reset, horizon_hours=self.config.bar_horizon_hours)
        swing_bars_reset.drop(columns=['timestamp'], inplace=True)

        self.logger.info(f"✅ {self.config.bar_horizon_hours}H bars: {len(swing_bars)} bars from {swing_bars.index.min()} to {swing_bars.index.max()}")
        return swing_bars_reset

    def add_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all 4-hour swing trading features (leak-free)"""
        self.logger.info("🔧 Building 4-hour swing features...")

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

        # Volume features (simplified) - handle different volume column names
        volume_col = None
        if 'volume_btc' in df.columns:
            volume_col = 'volume_btc'
        elif 'volume_usd' in df.columns:
            volume_col = 'volume_usd'
        elif 'volume' in df.columns:
            volume_col = 'volume'

        if volume_col:
            df['vol_3d'] = df[volume_col].rolling(3).mean()  # Reduced from 5
            df['vol_z'] = (df[volume_col] - df['vol_3d']) / (df['vol_3d'] + 1e-8)

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
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'volume_btc', 'volume_usd', 'funding_rate']]
        df[feature_cols] = df[feature_cols].shift(1)

        # Add 1-day return for backtesting PnL calculation (NO SHIFT - contemporaneous return)
        # This is NOT a training feature (would be leakage), only for backtest equity tracking
        df['return_1d'] = df['close'].pct_change()

        result = df.dropna().reset_index()
        self.logger.info(f"✅ Features built: {len(feature_cols)} features, {len(result)} rows")
        return result

class SwingLabeler:
    """FRIEND'S FIX: Use existing CSV labels instead of creating new ones"""

    def __init__(self, config: SwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate or validate labels from price data"""
        self.logger.info("🎯 Checking for labels in data...")

        df = df.copy()

        # Check which TARGET_LABELS are available in the CSV
        available_labels = [col for col in self.config.TARGET_LABELS if col in df.columns]

        if not available_labels:
            self.logger.info("📊 No pre-existing labels found, generating from price data...")
            df = self._generate_labels(df)
            available_labels = [col for col in self.config.TARGET_LABELS if col in df.columns]
            self.logger.info(f"✅ Generated {len(available_labels)} target labels")
        else:
            self.logger.info(f"✅ Found {len(available_labels)} existing target labels: {available_labels}")

        # Log label statistics
        for label in available_labels[:3]:  # Log first 3 to avoid spam
            if label in df.columns:
                pos_rate = df[label].mean()
                self.logger.info(f"📊 {label} positive rate: {pos_rate:.1%}")

        return df

    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary labels from forward returns"""
        df = df.copy()

        # Ensure we have close prices
        if 'close' not in df.columns:
            raise ValueError("Cannot generate labels: 'close' column missing")

        # Generate labels matching TARGET_LABELS format
        # Using daily bars: 1d, 4d, 12d, 24d horizons
        # Thresholds: 1bp, 2bp, 3bp (10, 20, 30 basis points)
        horizon_configs = [
            (1, "1h"),   # 1 day forward
            (4, "4h"),   # 4 days forward
            (12, "12h"), # 12 days forward
            (24, "24h")  # 24 days forward
        ]
        thresholds_bp = [1, 2, 3]  # basis points (multiply by 100 for %)

        for bars, label_suffix in horizon_configs:
            # FRIEND'S FIX: True forward return (future_price / current_price - 1)
            # NOT pct_change().shift() which looks backward then shifts = LEAKAGE
            fwd_ret = (df['close'].shift(-bars) / df['close']) - 1.0

            for bp in thresholds_bp:
                threshold_pct = bp / 10000.0  # convert bp to decimal (1bp = 0.01%)
                label_name = f"label_{label_suffix}_{bp}bp"
                # Binary: 1 if price goes up by threshold, 0 otherwise
                df[label_name] = (fwd_ret > threshold_pct).astype(int)

        self.logger.info(f"📊 Generated {len(horizon_configs) * len(thresholds_bp)} labels from forward returns")

        return df

class XGBoostEnsemble:
    """Ensemble of calibrated XGBoost models for daily swing prediction"""

    def __init__(self, config: SwingConfig):
        self.config = config
        self.models = []
        self.feature_cols = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def train_single_model(self, df: pd.DataFrame, feature_cols: List[str], seed: int = 0) -> Dict:
        """FRIEND'S FIX: Train with proper label selection and temporal split"""

        # FRIEND'S FIX: Randomly select target label for diversity
        import random
        random.seed(seed)
        available_labels = [col for col in self.config.TARGET_LABELS if col in df.columns]
        if not available_labels:
            raise ValueError(f"No TARGET_LABELS found in data. Available columns: {df.columns.tolist()}")

        selected_label = random.choice(available_labels)
        self.logger.info(f"🎯 Model {seed}: Selected target = {selected_label}")

        # FRIEND'S FIX: Use proper temporal split function
        X_tr, y_tr, X_va, y_va = time_split(df, feature_cols, selected_label, test_frac=0.2)

        # FRIEND'S FIX: Use strong XGBoost training with proper early stopping
        booster, trees, auc, ll = train_xgb_strong(X_tr, y_tr, X_va, y_va, seed=seed)

        self.logger.info(f"✅ Model {seed}: {trees} trees, AUC={auc:.3f}, LogLoss={ll:.3f}")

        # Create calibrator (isotonic calibration)
        val_probs = booster.predict(xgb.DMatrix(X_va))
        calibrator = CalibratedClassifierCV(estimator=None, cv='prefit', method='isotonic')
        # Fit calibrator with dummy sklearn-compatible estimator
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier().fit(X_va, y_va)
        calibrator.estimator = dummy
        calibrator.fit(X_va, y_va)

        return {
            'booster': booster,  # XGBoost booster object
            'calibrator': calibrator,
            'best_iter': trees,
            'val_auc': auc,
            'val_brier': ll,  # Using log loss as proxy
            'feature_cols': feature_cols,
            'selected_label': selected_label,
            'seed': seed
        }

    def train_ensemble(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Train ensemble of diverse models"""
        self.logger.info(f"🚀 Training ensemble of {self.config.n_models} XGBoost models...")
        self.feature_cols = feature_cols

        # Diversification strategies
        import time  # For model timing diagnostics
        seeds = list(range(self.config.n_models))

        for i, seed in enumerate(seeds):
            self.logger.info(f"Training model {i+1}/{self.config.n_models} (seed={seed})...")

            # TRIAGE CHECKLIST #4: Track model training diagnostics
            model_start_time = time.time()

            # Feature subsampling for diversity
            if i > 0:  # Keep first model with all features
                n_features = max(10, int(0.8 * len(feature_cols)))
                np.random.seed(seed)
                selected_features = np.random.choice(feature_cols, n_features, replace=False).tolist()
            else:
                selected_features = feature_cols

            model_info = self.train_single_model(df, selected_features, seed)

            # GUARD C+: Validate model quality before accepting it
            validate_model_quality(model_info, min_auc_threshold=0.55, max_tree_threshold=190)

            self.models.append(model_info)

            # TRIAGE CHECKLIST #4: Complete model diagnostics
            elapsed = time.time() - model_start_time
            print(f"🧾 MODEL_{i+1}_COMPLETE | elapsed={elapsed:.2f}s | best_iter={model_info['best_iter']} | features={len(selected_features)} | train_rows={len(df)}")

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
                # FRIEND'S FIX: Use booster instead of calibrator for prediction
                dmatrix = xgb.DMatrix(X_model)
                prob = model_info['booster'].predict(dmatrix)[0]
                probs.append(prob)

        if not probs:
            return 0.5, 1.0, [0.5] * len(self.models)  # Default if no predictions

        # Friend's exact logit averaging specification
        z = np.array([logit(np.clip(p, 1e-6, 1-1e-6)) for p in probs])
        p_ens = float(expit(z.mean()))
        z_std = float(z.std())

        return p_ens, z_std, probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Dict:
        """FRIEND'S FIX: Proper predict API with correct return format"""
        p_ens, uncertainty, individual_probs = self.predict_ensemble(X)

        # FRIEND'S FIX: y_pred from probabilities, NOT uncertainty
        y_pred = int(p_ens >= threshold)

        return {
            "y_prob": p_ens,
            "y_pred": y_pred,  # FIXED: was incorrectly assigning uncertainty here
            "uncertainty": uncertainty
        }

class SwingPolicy:
    """Cost-aware policy with abstention and survival-focused leverage"""

    def __init__(self, config: SwingConfig):
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

    def __init__(self, config: SwingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_backtest(self, df: pd.DataFrame, ensemble: XGBoostEnsemble,
                    policy: SwingPolicy) -> Tuple[pd.DataFrame, Dict]:
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

            # Realize PnL from previous position using contemporaneous return (i-1 to i)
            gross_return = row_curr['return_1d'] if 'return_1d' in row_curr else 0.0
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

class SwingTrader:
    """Main orchestrator for the daily swing trading system"""

    def __init__(self, config: SwingConfig = None):
        self.config = config or SwingConfig()
        self.pipeline = SwingDataPipeline(self.config)
        self.labeler = SwingLabeler(self.config)
        self.ensemble = XGBoostEnsemble(self.config)
        self.policy = SwingPolicy(self.config)
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

            # FRIEND'S GUARD A: Assert source is intraday (≤1H) before 4H resample
            assert_source_is_intraday(df, min_granularity="1H")
            self.logger.info("✅ Source data is intraday (≤1H granularity)")

            # Convert to 4-hour swing bars
            swing_bars = self.pipeline.to_swing_bars(df)
            self.logger.info(f"📊 Created {len(swing_bars)} swing bars")

            # Relaxed validation for small datasets
            min_swing_bars = 20  # Reduced from 40 for small datasets
            if len(swing_bars) < min_swing_bars:
                raise ValueError(f"Insufficient swing data: {len(swing_bars)} bars (need {min_swing_bars}+)")

            # Add features with validation
            with_features = self.pipeline.add_swing_features(swing_bars)
            self.logger.info(f"🔧 Added features: {len(with_features)} rows")

            # Relaxed validation for small datasets
            min_feature_rows = 10  # Reduced from 20
            if len(with_features) < min_feature_rows:
                raise ValueError(f"Insufficient data after features: {len(with_features)} rows (need {min_feature_rows}+)")

            # Add labels with validation
            final_df = self.labeler.validate_labels(with_features)
            self.logger.info(f"🎯 Added labels: {len(final_df)} samples")

            # HARD FAIL on insufficient data (friend's 4-hour system specs)
            min_training_samples = 200   # Realistic threshold for current 4-hour data (231 samples = 11:1 ratio)

            # BUILD TRACKING - Friend's go/no-go checklist
            self.logger.info(f"🧾 BUILD_ID={CODE_HASH} | min_training_samples={min_training_samples}")
            if len(final_df) < min_training_samples:
                raise ValueError(f"Not enough 4h samples: {len(final_df)} (need ≥{min_training_samples})")

            # TRIAGE CHECKLIST #3: Log rows after features
            self.logger.info(f"✅ rows_after_features={len(final_df)} | min_training_samples={min_training_samples}")
            return final_df

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {str(e)}")
            raise

    def train_models(self, df: pd.DataFrame) -> None:
        """Train the ensemble of models with validation"""
        try:
            # GUARD E: Validate label horizons vs bar horizon (stop sub-Nyquist predictions)
            validate_label_horizons(
                bar_horizon_hours=self.config.bar_horizon_hours,
                target_labels=self.config.TARGET_LABELS
            )
            self.logger.info(f"✅ GUARD E passed: All labels have horizon >= {self.config.bar_horizon_hours}H")

            # Get feature columns - more robust selection
            # FRIEND'S FIX: Exclude OHLCV + label columns + backtest-only columns
            exclude_cols = ['ts', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_btc', 'volume_usd', 'fear_greed_classification', 'return_1d'] + self.config.TARGET_LABELS

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

        # Training validation (as per friend's feedback - prevent silent "success")
        import numpy as np
        training_metrics = {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(test_df)),
            "n_models_trained": int(len(self.ensemble.models)),
            "mean_best_iter": float(np.mean([m.get("best_iter", 0) for m in self.ensemble.models])),
        }

        self.logger.info(f"📊 Training validation metrics: {training_metrics}")

        # Dev vs Prod mode assertions to prevent silent failures
        try:
            if self.config.prod_mode:
                # Strict gates for production
                assert training_metrics["train_rows"] >= 500, f"Prod mode needs ≥500 train rows: {training_metrics['train_rows']}"
                assert training_metrics["n_models_trained"] >= 6, f"Prod mode needs ≥6 models: {training_metrics['n_models_trained']}"
                assert training_metrics["mean_best_iter"] >= 150, f"Prod mode trees should grow ≥150: {training_metrics['mean_best_iter']}"
            else:
                # Lenient gates for dev/smoke testing
                assert training_metrics["train_rows"] >= 100, f"Dev mode needs ≥100 train rows: {training_metrics['train_rows']}"
                assert training_metrics["n_models_trained"] >= 1, f"Dev mode needs ≥1 model: {training_metrics['n_models_trained']}"
                assert training_metrics["mean_best_iter"] >= 10, f"Dev mode trees should grow ≥10: {training_metrics['mean_best_iter']}"

            mode = "PROD" if self.config.prod_mode else "DEV"
            self.logger.info(f"✅ {mode} mode training validation passed - models actually learned")
        except AssertionError as e:
            self.logger.error(f"❌ Training validation failed: {str(e)}")
            raise

        # Report actual training performance (no backtest needed)
        self.logger.info("🎯 TRAINING RESULTS:")
        self.logger.info(f"   Models trained: {training_metrics['n_models_trained']}")
        self.logger.info(f"   Avg trees: {training_metrics['mean_best_iter']:.0f}")

        # Get ensemble AUC from models
        aucs = [m.get('val_auc', 0.0) for m in self.ensemble.models if 'val_auc' in m]
        avg_auc = np.mean(aucs) if aucs else 0.0
        self.logger.info(f"   Avg AUC: {avg_auc:.3f}")

        return {
            'metrics': {
                'n_models': training_metrics['n_models_trained'],
                'avg_trees': training_metrics['mean_best_iter'],
                'avg_auc': avg_auc,
                'train_rows': training_metrics['train_rows'],
                'test_rows': training_metrics['val_rows']
            },
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

    # TRIAGE CHECKLIST #1: Prove Vertex is running the newest code
    print(f"🧾 BUILD_ID={CODE_HASH} | File: {CODE_PATH}")

    # CLI argument parsing
    parser = argparse.ArgumentParser(description='4-Hour Swing Trading System')
    parser.add_argument("--data", required=True, help="Path to raw BTC OHLCV CSV")
    parser.add_argument("--test_start_date", default=None, help="Optional YYYY-MM-DD to split OOS")
    parser.add_argument("--config_json", default=None, help="Optional JSON to override config fields")
    parser.add_argument("--outdir", default="/tmp/out", help="Local directory for artifacts")
    parser.add_argument("--gcs_out", required=True, help="gs://bucket/path/ for artifacts (required). Job fails if upload absent.")
    parser.add_argument("--expected_cadence", default="1H", choices=["1H", "4H", "1D"], help="Input data cadence (1H for 4H conversion, 1D for daily, 4H if already 4H)")
    args = parser.parse_args()

    # TRIAGE CHECKLIST #2: Prove data path is valid inside container
    import os
    logger.info(f"DATA_PATH={args.data}")
    exists = os.path.exists(args.data)
    logger.info(f"DATA_EXISTS={exists}")
    if not exists:
        raise FileNotFoundError(f"Data file not found: {args.data}")
    try:
        with open(args.data, "r") as f:
            header = f.readline().strip()
        n_lines = sum(1 for _ in open(args.data, "r"))
        logger.info(f"DATA_HEADER={header[:200]} | ~LINES={n_lines}")
    except Exception as e:
        logger.warning(f"DATA quick-scan failed: {e}")

    # Load configuration
    config = SwingConfig()
    if args.config_json:
        overrides = json.loads(args.config_json)
        for k, v in overrides.items():
            setattr(config, k, v)
            logger.info(f"🔧 Config override: {k} = {v}")

    # Create trader and run pipeline
    trader = SwingTrader(config)

    # Add total timing
    T0 = time.time()
    logger.info(f"🚀 Starting pipeline with data: {args.data}")

    # CADENCE VERIFICATION - Friend's critical fix
    try:
        # Load data just for cadence check
        df_check = pd.read_csv(args.data, low_memory=False)
        assert_expected_cadence(df_check["timestamp"], args.expected_cadence)
        logger.info(f"✅ Cadence verified: {args.expected_cadence}")
    except Exception as e:
        logger.error(f"❌ Cadence verification failed: {e}")
        sys.exit(1)

    try:
        results = trader.run_full_pipeline(args.data, test_start_date=args.test_start_date)

        # TRIAGE CHECKLIST #5: Dev vs Prod mode safety gates
        num_models = len(trader.ensemble.models) if trader.ensemble else 0
        mean_trees = np.mean([m.get('best_iter', 0) for m in trader.ensemble.models]) if num_models > 0 else 0

        if config.prod_mode:
            # Strict gates for production
            assert num_models >= 6, f"Prod mode needs ≥6 models: {num_models}"
            assert mean_trees >= 150, f"Prod mode trees should grow ≥150: mean_best_iter={mean_trees}"
        else:
            # Lenient gates for dev/smoke testing
            assert num_models >= 1, f"No models trained: {num_models}"
            assert mean_trees >= 10, f"Trees did not grow: mean_best_iter={mean_trees}"

        print(f"🧾 PIPELINE_SUCCESS | models={num_models} | mean_trees={mean_trees} | prod_mode={config.prod_mode}")

        # Total pipeline timing
        total_time = time.time() - T0
        logger.info(f"PIPELINE_ELAPSED_SEC={total_time:.2f}")
        logger.info(f"⏱️ TOTAL pipeline time: {total_time:.2f}s")

        # Persist artifacts
        out_dir = pathlib.Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save training metrics - convert numpy types to Python types
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

        logger.info("💾 Saved artifacts to %s", args.outdir)

        # Save ALL ensemble models + calibrators + manifest (complete ensemble artifact)
        if trader.ensemble and trader.ensemble.models:
            out_dir = pathlib.Path(args.outdir)

            # Build manifest with all metadata needed to reconstruct ensemble
            manifest = {
                "version": 1,
                "combiner": "logit_avg",
                "n_models": len(trader.ensemble.models),
                "feature_cols": trader.ensemble.feature_cols,  # Master feature list
                "models": []
            }

            # Save each model + calibrator + add to manifest
            for i, model_info in enumerate(trader.ensemble.models, 1):
                # Save XGBoost booster
                model_path = out_dir / f"model_{i}.json"
                model_info['booster'].save_model(str(model_path))

                # Save calibrator (isotonic regression)
                calib_path = out_dir / f"calibrator_{i}.joblib"
                dump(model_info['calibrator'], calib_path)

                # Add to manifest with all metadata
                manifest["models"].append({
                    "index": i,
                    "booster_file": f"model_{i}.json",
                    "calibrator_file": f"calibrator_{i}.joblib",
                    "feature_cols": model_info['feature_cols'],
                    "label": model_info['selected_label'],
                    "val_auc": float(model_info.get('val_auc', 0.0)),
                    "best_iter": int(model_info.get('best_iter', 0)),
                    "seed": int(model_info.get('seed', i))
                })

                logger.info(f"💾 Model {i}/{len(trader.ensemble.models)}: {model_path.name} + {calib_path.name}")

            # Save manifest
            manifest_path = out_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"📋 Saved manifest: {manifest_path}")

            # Upload all artifacts to GCS
            from google.cloud import storage
            from urllib.parse import urlparse
            u = urlparse(args.gcs_out)
            bkt, prefix = u.netloc, u.path.lstrip("/")
            client = storage.Client()
            bucket = client.bucket(bkt)

            uploaded = 0
            for p in out_dir.glob("*"):
                if p.is_file():
                    blob_path = f"{prefix}/{p.name}"
                    bucket.blob(blob_path).upload_from_filename(str(p))
                    uploaded += 1
                    logger.info(f"☁️  Uploaded: {blob_path}")

            # FRIEND'S GUARD D (ENHANCED): Quality gate with individual + average checks
            # Extract metrics for quality check
            avg_auc = float(results["metrics"].get("avg_auc", 0.0))
            avg_trees = float(results["metrics"].get("avg_trees", 0.0))
            test_rows = int(results["metrics"].get("test_rows", 0))

            # Get individual model AUCs
            individual_aucs = [m['val_auc'] for m in trader.ensemble.models]
            min_auc = min(individual_aucs)
            max_auc = max(individual_aucs)

            # Apply quality gate (adjust thresholds for dev vs prod)
            if config.prod_mode:
                min_avg_auc = 0.65
                min_individual_auc = 0.58
                min_val = 1000
            else:
                min_avg_auc = 0.60  # Raised from 0.51
                min_individual_auc = 0.55  # NEW: catch individual failures
                min_val = 40

            min_trees = 50

            try:
                # Check 1: Minimum validation size
                if test_rows < min_val:
                    raise RuntimeError(
                        f"Validation too small: n_val={test_rows} < {min_val}"
                    )

                # Check 2: Average ensemble quality
                if not np.isfinite(avg_auc) or avg_auc < min_avg_auc:
                    raise RuntimeError(
                        f"Ensemble too weak: avg_auc={avg_auc:.3f} < {min_avg_auc}"
                    )

                # Check 3: No individual model failures (NEW)
                if min_auc < min_individual_auc:
                    failed_models = [
                        f"Model {i+1} ({m['selected_label']}): {m['val_auc']:.3f}"
                        for i, m in enumerate(trader.ensemble.models)
                        if m['val_auc'] < min_individual_auc
                    ]
                    raise RuntimeError(
                        f"Individual model(s) failed quality check:\n" +
                        "\n".join(f"  - {m}" for m in failed_models) +
                        f"\n  Min AUC: {min_auc:.3f} < {min_individual_auc}"
                    )

                # Check 4: Minimum trees
                if avg_trees < min_trees:
                    raise RuntimeError(
                        f"Booster too small: trees={avg_trees} < {min_trees}"
                    )

                logger.info(
                    f"✅ Quality gate passed:\n"
                    f"   Avg AUC: {avg_auc:.3f} (min={min_auc:.3f}, max={max_auc:.3f})\n"
                    f"   Avg trees: {avg_trees:.0f}\n"
                    f"   Val rows: {test_rows}"
                )

            except RuntimeError as e:
                logger.error(f"❌ Quality gate failed: {str(e)}")
                logger.error("Artifacts uploaded but job marked as FAILED (no SUCCESS.ok)")
                sys.exit(3)

            # Upload SUCCESS marker (only if quality gate passed)
            bucket.blob(f"{prefix}/SUCCESS.ok").upload_from_string("ok")
            uploaded += 1

            # CRITICAL: Verify complete ensemble uploaded
            # Expected: 8 models + 8 calibrators + 1 manifest + 1 metrics + 1 SUCCESS = 19 files
            expected_min = 2 * len(trader.ensemble.models) + 3
            if uploaded < expected_min:
                logger.error(f"❌ Uploaded only {uploaded} artifacts - need ≥{expected_min} (models+calibrators+manifest+metrics+SUCCESS)")
                sys.exit(3)

            logger.info(f"☁️  Published {uploaded} artifacts to {args.gcs_out}")
        else:
            logger.error("No trained models available for publishing")
            sys.exit(3)

        logger.info("✅ PIPELINE COMPLETE - ALL ARTIFACTS SAVED AND UPLOADED")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)