#!/usr/bin/env python3
"""
XGBoost Ensemble System for Leverage BTC Trading
Phase 2: M=7-10 diversified models with probability calibration and logit averaging

Architecture:
- Base learners: XGBoost with production defaults
- Diversification: seed, train window, feature subsets, label horizons
- Calibration: isotonic/Platt scaling per model
- Ensemble: logit averaging with uncertainty estimation
- Cross-validation: purged & embargoed for time series
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import warnings

# XGBoost and calibration
try:
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    from scipy.special import logit, expit
except ImportError as e:
    logging.error(f"Missing required ML libraries: {e}")
    raise

logger = logging.getLogger(__name__)

@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters optimized for financial time series"""
    # Core XGBoost parameters - optimized for proper training
    objective: str = 'binary:logistic'
    tree_method: str = 'hist'
    max_depth: int = 5
    min_child_weight: int = 12  # Eased regularization for better learning
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0  # Reduced L2 regularization
    reg_alpha: float = 0.0   # L1 regularization
    eta: float = 0.01        # Lower learning rate for more trees
    n_estimators: int = 5000 # Large cap - let early stopping decide

    # Early stopping and validation
    early_stopping_rounds: int = 200
    eval_metric: str = 'logloss'

    # Reproducibility and performance
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0

@dataclass
class EnsembleConfig:
    """Ensemble configuration for diversification and robustness"""
    n_models: int = 8  # M=8 base models

    # Diversification strategies
    seed_range: Tuple[int, int] = (42, 1000)
    train_window_ratios: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 1.0])
    feature_sample_ratios: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0])
    label_horizons: List[int] = field(default_factory=lambda: [2, 4])

    # Calibration
    calibration_method: str = 'isotonic'  # 'isotonic' or 'platt'
    calibration_cv: int = 3

    # Ensemble method
    ensemble_method: str = 'logit_avg'  # 'logit_avg', 'mean', 'median'

@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics"""
    model_id: str
    train_score: float
    val_score: float
    test_score: float
    log_loss_train: float
    log_loss_val: float
    log_loss_test: float
    brier_score: float
    auc_score: float
    calibration_error: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo
    Implementation of purged & embargoed CV for financial time series
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.2,
                 purge_length: int = 10, embargo_length: int = 4):
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged and embargoed train/test splits"""
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)

        splits = []

        for i in range(self.n_splits):
            # Test set: evenly spaced blocks
            test_start = i * (n_samples // self.n_splits)
            test_end = min(test_start + test_size_samples, n_samples)
            test_indices = np.arange(test_start, test_end)

            # Train set: everything except test + purge + embargo
            train_indices = []

            # Pre-test training data (with purge)
            pre_test_end = max(0, test_start - self.purge_length)
            if pre_test_end > 0:
                train_indices.extend(range(pre_test_end))

            # Post-test training data (with embargo)
            post_test_start = min(n_samples, test_end + self.embargo_length)
            if post_test_start < n_samples:
                train_indices.extend(range(post_test_start, n_samples))

            train_indices = np.array(train_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"Generated {len(splits)} purged & embargoed CV splits")
        return splits

class XGBoostBaseModel:
    """Single XGBoost model with calibration"""

    def __init__(self, config: XGBoostConfig, model_id: str,
                 feature_cols: List[str], calibration_method: str = 'isotonic'):
        self.config = config
        self.model_id = model_id
        self.feature_cols = feature_cols
        self.calibration_method = calibration_method

        # Initialize model - remove early_stopping_rounds for XGBoost 3.0 compatibility
        model_params = config.__dict__.copy()
        model_params.pop('early_stopping_rounds', None)  # Handle separately
        self.model = xgb.XGBClassifier(**model_params)
        self.calibrator = None
        self.is_fitted = False

        # Performance tracking
        self.metrics = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            sample_weight: np.ndarray = None) -> 'XGBoostBaseModel':
        """Fit XGBoost model with proper time-ordered validation and early stopping"""

        # Prepare data
        X_train_features = X_train[self.feature_cols].values

        # Always use time-ordered validation for proper early stopping
        # Split last 20% as time validation (chronological order preserved)
        cut = int(0.8 * len(X_train_features))
        X_tr, y_tr = X_train_features[:cut], y_train.iloc[:cut]
        X_va, y_va = X_train_features[cut:], y_train.iloc[cut:]

        logger.info(f"Training {self.model_id}...")
        logger.info(f"  Train shape: {X_tr.shape}, positives: {int(y_tr.sum())}")
        logger.info(f"  Val shape: {X_va.shape}, positives: {int(y_va.sum())}")

        # Phase 1: Train with early stopping to find optimal iteration count
        # CRITICAL FIX: Pass early_stopping_rounds correctly to fit method
        # XGBoost 3.0 API fix: early stopping is set in constructor, not fit()
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            sample_weight=sample_weight[:len(y_tr)] if sample_weight is not None else None,
            verbose=False
        )

        # Sanity checks (as recommended by quant friend)
        bst = self.model.get_booster()
        trees_built = bst.num_boosted_rounds()
        logger.info(f"  Phase 1 - Trees built: {trees_built}/{self.config.n_estimators}")
        logger.info(f"  Best iteration: {getattr(self.model, 'best_iteration_', 'N/A')}")

        # Phase 2: Refit on full data with optimal tree count + buffer
        if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ is not None:
            optimal_trees = self.model.best_iteration_ + 50  # Small buffer as recommended
            logger.info(f"  Phase 2 - Refitting with {optimal_trees} trees on full data")

            # Create new model with optimal tree count
            refit_params = self.config.__dict__.copy()
            refit_params['n_estimators'] = optimal_trees
            # Remove early_stopping_rounds from config since it's not a model parameter
            refit_params.pop('early_stopping_rounds', None)

            final_model = xgb.XGBClassifier(**refit_params)
            final_model.fit(
                X_train_features, y_train,
                sample_weight=sample_weight,
                verbose=False
            )

            self.model = final_model

            # Final sanity check
            final_bst = self.model.get_booster()
            final_trees = final_bst.num_boosted_rounds()
            logger.info(f"  Final trees built: {final_trees}")
        else:
            logger.warning(f"  No best_iteration found, using {trees_built} trees")

        # Calibrate probabilities using held-out portion of training data
        if len(X_train) > 1000:  # Only calibrate if enough data
            # Use last 20% for calibration
            cal_split = int(0.8 * len(X_train))

            X_cal = X_train.iloc[cal_split:][self.feature_cols].values
            y_cal = y_train.iloc[cal_split:].values

            # Get uncalibrated predictions
            pred_cal = self.model.predict_proba(X_cal)[:, 1]

            # Fit calibrator
            if self.calibration_method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
            else:  # 'platt'
                self.calibrator = LogisticRegression()
                pred_cal = pred_cal.reshape(-1, 1)

            self.calibrator.fit(pred_cal, y_cal)

        self.is_fitted = True
        logger.info(f"Model {self.model_id} fitted with {len(self.feature_cols)} features")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.model_id} not fitted yet")

        X_features = X[self.feature_cols].values
        pred_uncal = self.model.predict_proba(X_features)[:, 1]

        # Apply calibration if available
        if self.calibrator is not None:
            if self.calibration_method == 'platt':
                pred_cal = self.calibrator.predict_proba(pred_uncal.reshape(-1, 1))[:, 1]
            else:  # isotonic
                pred_cal = self.calibrator.predict(pred_uncal)
        else:
            pred_cal = pred_uncal

        return pred_cal

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_fitted:
            return {}

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_cols, importance_scores))

class XGBoostEnsemble:
    """
    XGBoost ensemble with diversification, calibration, and uncertainty estimation
    """

    def __init__(self, ensemble_config: EnsembleConfig, xgb_config: XGBoostConfig):
        self.ensemble_config = ensemble_config
        self.xgb_config = xgb_config

        # Model storage
        self.models: List[XGBoostBaseModel] = []
        self.ensemble_metrics: List[ModelMetrics] = []

        # Dataset information
        self.feature_cols: List[str] = []
        self.label_cols: List[str] = []

        self.is_fitted = False

    def _generate_model_configs(self, feature_cols: List[str]) -> List[Dict[str, Any]]:
        """Generate diversified model configurations"""
        configs = []

        for i in range(self.ensemble_config.n_models):
            # Random seed for diversity
            seed = np.random.RandomState(42 + i).randint(*self.ensemble_config.seed_range)

            # Feature subset
            feature_ratio = np.random.choice(self.ensemble_config.feature_sample_ratios)
            n_features = max(1, int(len(feature_cols) * feature_ratio))
            selected_features = np.random.RandomState(seed).choice(
                feature_cols, size=n_features, replace=False
            ).tolist()

            # Label horizon
            label_horizon = np.random.choice(self.ensemble_config.label_horizons)

            # Training window ratio
            train_ratio = np.random.choice(self.ensemble_config.train_window_ratios)

            config = {
                'model_id': f'xgb_model_{i:02d}',
                'seed': seed,
                'feature_cols': selected_features,
                'label_col': f'y_direction_{label_horizon}',
                'train_window_ratio': train_ratio
            }
            configs.append(config)

        logger.info(f"Generated {len(configs)} diversified model configurations")
        return configs

    def fit(self, dataset: pd.DataFrame, feature_cols: List[str], label_cols: List[str],
            cv_splits: Optional[PurgedTimeSeriesSplit] = None) -> 'XGBoostEnsemble':
        """Fit ensemble of XGBoost models with diversification"""

        self.feature_cols = feature_cols
        self.label_cols = label_cols

        # Generate model configurations
        model_configs = self._generate_model_configs(feature_cols)

        # Cross-validation setup
        if cv_splits is None:
            cv_splits = PurgedTimeSeriesSplit(n_splits=3, purge_length=10, embargo_length=4)

        logger.info(f"Training ensemble of {len(model_configs)} XGBoost models...")

        for config in model_configs:
            logger.info(f"Training {config['model_id']}...")

            # Select label and apply training window
            if config['label_col'] not in dataset.columns:
                logger.warning(f"Label {config['label_col']} not found, using {label_cols[0]}")
                label_col = label_cols[0]
            else:
                label_col = config['label_col']

            # Apply training window
            n_samples = len(dataset)
            train_size = int(n_samples * config['train_window_ratio'])
            train_data = dataset.iloc[:train_size].copy()

            # Prepare data
            X = train_data[config['feature_cols']]
            y = train_data[label_col]

            # Remove NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                logger.warning(f"Insufficient data for {config['model_id']}: {len(X)} samples")
                continue

            # Update XGBoost config with model-specific seed
            model_xgb_config = XGBoostConfig()
            model_xgb_config.__dict__.update(self.xgb_config.__dict__)
            model_xgb_config.random_state = config['seed']

            # Create and fit model
            model = XGBoostBaseModel(
                config=model_xgb_config,
                model_id=config['model_id'],
                feature_cols=config['feature_cols'],
                calibration_method=self.ensemble_config.calibration_method
            )

            # Cross-validation training
            cv_scores = []
            for train_idx, val_idx in cv_splits.split(X):
                X_train_cv = X.iloc[train_idx]
                y_train_cv = y.iloc[train_idx]
                X_val_cv = X.iloc[val_idx]
                y_val_cv = y.iloc[val_idx]

                # Fit on CV fold
                model_cv = XGBoostBaseModel(
                    config=model_xgb_config,
                    model_id=f"{config['model_id']}_cv",
                    feature_cols=config['feature_cols'],
                    calibration_method=self.ensemble_config.calibration_method
                )

                model_cv.fit(X_train_cv, y_train_cv, X_val_cv, y_val_cv)

                # Evaluate
                pred_val = model_cv.predict_proba(X_val_cv)
                score = log_loss(y_val_cv, pred_val)
                cv_scores.append(score)

            # Fit final model on all data
            model.fit(X, y)

            # Calculate metrics
            train_pred = model.predict_proba(X)
            metrics = ModelMetrics(
                model_id=config['model_id'],
                train_score=np.mean(cv_scores),
                val_score=np.std(cv_scores),  # Using std as validation uncertainty
                test_score=0.0,  # Will be updated during testing
                log_loss_train=log_loss(y, train_pred),
                log_loss_val=np.mean(cv_scores),
                log_loss_test=0.0,
                brier_score=brier_score_loss(y, train_pred),
                auc_score=roc_auc_score(y, train_pred) if len(np.unique(y)) > 1 else 0.5,
                calibration_error=self._calculate_calibration_error(y, train_pred),
                feature_importance=model.get_feature_importance()
            )

            self.models.append(model)
            self.ensemble_metrics.append(metrics)

            logger.info(f"✅ {config['model_id']}: CV score={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}, AUC={metrics.auc_score:.4f}")

        self.is_fitted = True
        logger.info(f"🎉 Ensemble training complete: {len(self.models)} models fitted")

        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Unified prediction method for ensemble"""
        if not self.models:
            raise ValueError("Ensemble not fitted yet")

        probs, predictions = self.predict_ensemble(X)

        # Handle both 1D and 2D probability arrays
        if probs.ndim == 1:
            # If 1D, assume it's already class 1 probabilities
            class_1_probs = probs
        else:
            # If 2D, extract class 1 probabilities
            class_1_probs = probs[:, 1]

        # Return both probabilities and predictions in a dict
        return {
            'y_prob': class_1_probs,  # Probability of class 1
            'y_pred': predictions,    # Discrete predictions
            'y': class_1_probs        # Alias for compatibility
        }

    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with ensemble using logit averaging
        Returns: (ensemble_probabilities, uncertainty_scores)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        # Collect predictions from all models
        model_predictions = []

        for model in self.models:
            try:
                pred = model.predict_proba(X)
                model_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {model.model_id} prediction failed: {e}")
                continue

        if not model_predictions:
            raise ValueError("No models available for prediction")

        # Convert to logit space for averaging
        logit_predictions = []
        for pred in model_predictions:
            # Clip probabilities to prevent logit overflow
            pred_clipped = np.clip(pred, 1e-6, 1-1e-6)
            logit_pred = logit(pred_clipped)
            logit_predictions.append(logit_pred)

        # Average in logit space
        logit_avg = np.mean(logit_predictions, axis=0)

        # Convert back to probability space
        ensemble_prob = expit(logit_avg)

        # Calculate uncertainty as standard deviation in logit space
        uncertainty = np.std(logit_predictions, axis=0)

        return ensemble_prob, uncertainty

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        CRITICAL FIX: Wrapper method for compatibility with leverage_trading_system.py
        Maps to predict_ensemble and returns just probabilities (not uncertainty)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")

        ensemble_prob, _ = self.predict_ensemble(X)
        return ensemble_prob

    def _calculate_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate calibration error (reliability diagram)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def get_ensemble_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance metrics"""
        if not self.ensemble_metrics:
            return {}

        metrics_df = pd.DataFrame([m.__dict__ for m in self.ensemble_metrics])

        ensemble_summary = {
            'n_models': len(self.models),
            'mean_cv_score': metrics_df['val_score'].mean(),
            'std_cv_score': metrics_df['val_score'].std(),
            'mean_auc': metrics_df['auc_score'].mean(),
            'mean_calibration_error': metrics_df['calibration_error'].mean(),
            'best_model_id': metrics_df.loc[metrics_df['auc_score'].idxmax(), 'model_id'],
            'worst_model_id': metrics_df.loc[metrics_df['auc_score'].idxmin(), 'model_id']
        }

        return ensemble_summary

    def save_ensemble(self, filepath: str) -> None:
        """Save the entire ensemble to disk"""
        ensemble_data = {
            'models': self.models,
            'metrics': self.ensemble_metrics,
            'feature_cols': self.feature_cols,
            'label_cols': self.label_cols,
            'ensemble_config': self.ensemble_config,
            'xgb_config': self.xgb_config
        }

        joblib.dump(ensemble_data, filepath)
        logger.info(f"💾 Ensemble saved to {filepath}")

    @classmethod
    def load_ensemble(cls, filepath: str) -> 'XGBoostEnsemble':
        """Load ensemble from disk"""
        ensemble_data = joblib.load(filepath)

        ensemble = cls(ensemble_data['ensemble_config'], ensemble_data['xgb_config'])
        ensemble.models = ensemble_data['models']
        ensemble.ensemble_metrics = ensemble_data['metrics']
        ensemble.feature_cols = ensemble_data['feature_cols']
        ensemble.label_cols = ensemble_data['label_cols']
        ensemble.is_fitted = True

        logger.info(f"📂 Ensemble loaded from {filepath}")
        return ensemble

# Utility functions for ensemble training pipeline
def train_xgboost_ensemble(dataset: pd.DataFrame, feature_cols: List[str], label_cols: List[str],
                          ensemble_config: EnsembleConfig = None,
                          xgb_config: XGBoostConfig = None) -> XGBoostEnsemble:
    """
    High-level function to train XGBoost ensemble with sensible defaults
    """
    if ensemble_config is None:
        ensemble_config = EnsembleConfig()

    if xgb_config is None:
        xgb_config = XGBoostConfig()

    # Initialize ensemble
    ensemble = XGBoostEnsemble(ensemble_config, xgb_config)

    # Fit ensemble
    ensemble.fit(dataset, feature_cols, label_cols)

    # Print summary
    metrics = ensemble.get_ensemble_metrics()
    logger.info(f"🎯 Ensemble Summary:")
    logger.info(f"   Models trained: {metrics.get('n_models', 0)}")
    logger.info(f"   Mean CV score: {metrics.get('mean_cv_score', 0):.4f}±{metrics.get('std_cv_score', 0):.4f}")
    logger.info(f"   Mean AUC: {metrics.get('mean_auc', 0):.4f}")
    logger.info(f"   Best model: {metrics.get('best_model_id', 'N/A')}")

    return ensemble

if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)
    logger.info("XGBoost Ensemble System initialized")