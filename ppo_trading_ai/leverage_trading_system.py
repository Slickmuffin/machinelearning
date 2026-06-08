#!/usr/bin/env python3
"""
Comprehensive Leverage BTC Trading System
Integration of all components: Data Pipeline + XGBoost Ensemble + Policy + Leverage Sizing

Complete pipeline: Ingest → Features → XGBoost → Policy → Sizing → Backtest → Deploy
Targeting 2-10% daily returns with up to 125x leverage through systematic risk management
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Import our components
try:
    from .data_pipeline import (
        load_prices_or_fail, extract_comprehensive_features,
        prepare_ml_dataset, DataQualityMetrics,
        resample_to_daily_bars, create_daily_features, create_daily_directional_labels
    )
    from .xgboost_ensemble import (
        XGBoostEnsemble, EnsembleConfig, XGBoostConfig, train_xgboost_ensemble
    )
    from .cost_aware_policy import (
        CostAwarePolicy, PolicyConfig, PolicySignal, TradingAction,
        create_policy_signals_batch, analyze_policy_performance
    )
    from .cost_aware_policy_fixed import (
        apply_friend_fixes, break_even_prob, ensemble_prob, decide_cost_aware
    )
    from .daily_swing_policy import (
        DailySwingTrader, DailySwingConfig, run_daily_backtest
    )
    from .leverage_sizing import (
        FourClampLeverageSizer, LeverageSizingConfig, LeverageComponents,
        create_leverage_signals_batch
    )
except ImportError:
    # Direct imports for standalone execution
    from data_pipeline import (
        load_prices_or_fail, DataQualityMetrics,
        resample_to_daily_bars, create_daily_features, create_daily_directional_labels
    )
    from xgboost_ensemble import (
        XGBoostEnsemble, EnsembleConfig, XGBoostConfig, train_xgboost_ensemble
    )
    from cost_aware_policy import (
        CostAwarePolicy, PolicyConfig, PolicySignal, TradingAction,
        create_policy_signals_batch
    )
    from leverage_sizing import (
        FourClampLeverageSizer, LeverageSizingConfig, LeverageComponents,
        create_leverage_signals_batch
    )

logger = logging.getLogger(__name__)

@dataclass
class TradingSystemConfig:
    """Master configuration for the complete trading system"""
    # Data pipeline
    data_path: str = "/tmp/btc_data.csv"
    reference_venues: Optional[Dict[str, str]] = None
    step_minutes: int = 240  # 4H bars

    # Feature engineering
    lookback_bars: int = 64
    label_horizons: List[int] = None

    # XGBoost ensemble
    ensemble_config: EnsembleConfig = None
    xgb_config: XGBoostConfig = None

    # Policy engine
    policy_config: PolicyConfig = None

    # Leverage sizing
    leverage_config: LeverageSizingConfig = None

    # Backtesting
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    initial_equity: float = 10000.0

    # System settings
    model_save_path: str = "/tmp/leverage_trading_model.joblib"
    results_save_path: str = "/tmp/leverage_trading_results.json"

    def __post_init__(self):
        if self.label_horizons is None:
            self.label_horizons = [2, 4]
        if self.ensemble_config is None:
            self.ensemble_config = EnsembleConfig()
        if self.xgb_config is None:
            self.xgb_config = XGBoostConfig()
        if self.policy_config is None:
            self.policy_config = PolicyConfig()
        if self.leverage_config is None:
            self.leverage_config = LeverageSizingConfig()

@dataclass
class TradingResults:
    """Comprehensive trading system results"""
    # Data quality
    data_quality: DataQualityMetrics

    # Model performance
    ensemble_metrics: Dict[str, Any]

    # Policy performance
    policy_analysis: Dict[str, Any]

    # Leverage statistics
    leverage_stats: Dict[str, Any]

    # Backtest results
    backtest_summary: Dict[str, Any]

    # System metadata
    config_summary: Dict[str, Any]
    training_time: float
    total_signals: int

class LeverageTradingSystem:
    """
    Complete leverage BTC trading system integrating all components
    """

    def __init__(self, config: TradingSystemConfig = None):
        self.config = config or TradingSystemConfig()

        # System components (initialized during fit)
        self.ensemble: Optional[XGBoostEnsemble] = None
        self.policy: Optional[CostAwarePolicy] = None
        self.leverage_sizer: Optional[FourClampLeverageSizer] = None

        # Data and features
        self.dataset: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None
        self.label_cols: Optional[List[str]] = None
        self.data_quality: Optional[DataQualityMetrics] = None

        # Results
        self.results: Optional[TradingResults] = None

        self.is_fitted = False

        logger.info("🚀 Leverage BTC Trading System initialized")
        logger.info(f"   📊 Data: {self.config.data_path}")
        logger.info("   🎯 Target: 2-10% daily returns with up to 125x leverage")
        logger.info(f"   ⏱️ Timeframe: {self.config.step_minutes}min bars")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, List[str], List[str], DataQualityMetrics]:
        """
        SWING TRADING: Daily data loading and preparation pipeline
        Friend's specification for swing trading on daily horizons

        Returns:
            (daily_dataset, feature_cols, label_cols, quality_metrics)
        """
        logger.info("📅 Loading and preparing DAILY swing trading data...")

        # 1) Load raw intraday data with comprehensive washing
        df, quality_metrics = load_prices_or_fail(
            path=self.config.data_path,
            step_minutes=self.config.step_minutes,
            reference_venues=self.config.reference_venues
        )

        logger.info(f"✅ Raw intraday data loaded: {len(df)} bars, quality score: {100-quality_metrics.nan_rate*100:.1f}%")

        # 2) SWING TRADING: Resample to daily bars
        daily_df = resample_to_daily_bars(df)

        logger.info(f"📅 Daily bars created: {len(daily_df)} days")

        # 3) Create daily features (momentum, volatility, seasonality)
        df_features = create_daily_features(daily_df)

        logger.info(f"🔧 Daily features created: {df_features.shape}")

        # 4) Create daily directional labels (next day direction)
        dataset = create_daily_directional_labels(df_features, k_days=1)

        logger.info(f"🎯 Daily labels created: {dataset.shape}")

        # 5) Define feature columns for daily swing trading
        feature_cols = [
            'ret_3', 'ret_5', 'ret_10', 'ret_20',        # Momentum
            'sigma_hat_1d', 'atr_1d',                    # Volatility
            'funding_1d', 'funding_5d',                  # Carry
            'dow', 'month',                              # Seasonality
            'vol_q_roll5', 'vol_q_z'                     # Volume
        ]

        # Filter to available columns
        available_features = [col for col in feature_cols if col in dataset.columns]
        label_cols = ['y']  # Single binary label for daily direction

        logger.info(f"🏗️ Daily swing dataset prepared: {dataset.shape} with {len(available_features)} features")
        logger.info(f"   📈 Features: {available_features}")
        logger.info(f"   🎯 Labels: {label_cols}")

        return dataset, available_features, label_cols, quality_metrics

    def train_ensemble_model(self, dataset: pd.DataFrame, feature_cols: List[str],
                           label_cols: List[str]) -> XGBoostEnsemble:
        """
        Train XGBoost ensemble with diversification and calibration
        """
        logger.info("🤖 Training XGBoost ensemble...")

        # Train ensemble with configured parameters
        ensemble = train_xgboost_ensemble(
            dataset=dataset,
            feature_cols=feature_cols,
            label_cols=label_cols,
            ensemble_config=self.config.ensemble_config,
            xgb_config=self.config.xgb_config
        )

        ensemble_metrics = ensemble.get_ensemble_metrics()
        logger.info(f"✅ Ensemble trained: {ensemble_metrics.get('n_models', 0)} models")
        logger.info(f"   📈 Mean AUC: {ensemble_metrics.get('mean_auc', 0):.3f}")
        logger.info(f"   🎯 Best model: {ensemble_metrics.get('best_model_id', 'N/A')}")

        return ensemble

    def create_policy_engine(self) -> CostAwarePolicy:
        """
        Initialize cost-aware policy engine
        """
        logger.info("🧠 Creating cost-aware policy engine...")

        policy = CostAwarePolicy(self.config.policy_config)

        policy_stats = policy.get_policy_stats()
        logger.info(f"✅ Policy initialized: {policy_stats['execution_cost_bp']:.1f}bp execution cost")

        return policy

    def create_leverage_sizer(self) -> FourClampLeverageSizer:
        """
        Initialize four-clamp leverage sizing system
        """
        logger.info("⚡ Creating leverage sizing system...")

        sizer = FourClampLeverageSizer(self.config.leverage_config)
        sizer.reset_equity_tracking(self.config.initial_equity)

        logger.info(f"✅ Leverage sizer initialized with {self.config.initial_equity:,.0f} initial equity")

        return sizer

    def evaluate_predictions_only(self, test_dataset: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """
        LAYER A: Evaluate pure prediction quality WITHOUT any trading decisions
        Friend's approach - separate learning from acting
        """
        logger.info("🔍 Layer A: Evaluating pure predictive power (no trading pressure)...")

        if not self.is_fitted or self.ensemble is None:
            logger.error("System not fitted - cannot evaluate predictions")
            return {'mean_prediction': 0.5, 'prediction_std': 0.0, 'calibration_score': 0.0, 'oos_auc': 0.5}

        # Get features and labels for evaluation
        X_test = test_dataset[feature_cols].values
        y_test = test_dataset['y_direction_2'].values  # Binary labels

        # Get ensemble predictions (pure prediction, no trading logic)
        try:
            predictions = self.ensemble.predict_proba(X_test)

            # Calibration check
            from sklearn.metrics import roc_auc_score, brier_score_loss

            # AUC (discriminative power)
            oos_auc = roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0.5

            # Calibration (reliability)
            calibration_score = brier_score_loss(y_test, predictions)

            # Prediction statistics
            mean_prediction = float(np.mean(predictions))
            prediction_std = float(np.std(predictions))

            logger.info("📊 Pure prediction evaluation:")
            logger.info(f"   🎯 OOS AUC: {oos_auc:.3f} (discriminative power)")
            logger.info(f"   📐 Calibration: {calibration_score:.3f} (lower = better)")
            logger.info(f"   📈 Mean prediction: {mean_prediction:.3f}")
            logger.info(f"   📊 Prediction std: {prediction_std:.3f}")

            return {
                'oos_auc': oos_auc,
                'calibration_score': calibration_score,
                'mean_prediction': mean_prediction,
                'prediction_std': prediction_std
            }

        except Exception as e:
            logger.error(f"Prediction evaluation failed: {e}")
            return {'mean_prediction': 0.5, 'prediction_std': 0.0, 'calibration_score': 1.0, 'oos_auc': 0.5}

    def evaluate_layer_b_policy(self, test_dataset: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        LAYER B: Apply cost-aware policy to FROZEN model predictions
        Friend's approach - only after model is completely trained and calibrated
        """
        logger.info("🎯 Layer B: Applying cost-aware policy to frozen predictions...")

        if not self.is_fitted:
            raise ValueError("System must be fitted before policy evaluation")

        # Generate trading signals using frozen model
        policy_signals, leverage_signals = self.generate_trading_signals(test_dataset)

        # Run full backtest with costs
        backtest_results = self.run_backtest(test_dataset, policy_signals, leverage_signals)

        logger.info("📊 Layer B Policy Results:")
        logger.info(f"   💰 Total Return: {backtest_results['total_return_pct']:.2f}%")
        logger.info(f"   📈 Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        logger.info(f"   🎯 Win Rate: {backtest_results['win_rate_pct']:.1f}%")
        logger.info(f"   📊 Total Trades: {backtest_results['total_trades']}")
        logger.info(f"   ⚖️ Avg Leverage: {backtest_results['avg_leverage']:.1f}x")

        return {
            'policy_signals': policy_signals,
            'leverage_signals': leverage_signals,
            'backtest_results': backtest_results
        }

    def generate_trading_signals(self, test_dataset: pd.DataFrame) -> Tuple[List[PolicySignal], List[LeverageComponents]]:
        """
        Generate complete trading signals combining ensemble predictions, policy decisions, and leverage sizing
        """
        if not self.is_fitted:
            raise ValueError("System must be fitted before generating signals")

        logger.info("📡 Generating trading signals...")

        # 1) Get ensemble predictions
        X_test = test_dataset[self.feature_cols]
        ensemble_probs, ensemble_uncertainties = self.ensemble.predict_ensemble(X_test)

        logger.info(f"🤖 Ensemble predictions: {len(ensemble_probs)} signals")

        # 2) Calculate volatility estimates for policy
        vol_estimates = []
        for i in range(len(test_dataset)):
            # Use recent volatility (simplified)
            recent_vol = test_dataset.iloc[max(0, i-20):i+1]['vol_ewma_lag1'].mean()
            if pd.isna(recent_vol):
                recent_vol = 0.02  # 2% default
            vol_estimates.append(recent_vol)

        vol_estimates = np.array(vol_estimates)

        # 3) Generate policy signals
        policy_signals = create_policy_signals_batch(
            policy=self.policy,
            ensemble_probs=ensemble_probs,
            ensemble_uncertainties=ensemble_uncertainties,
            vol_estimates=vol_estimates
        )

        logger.info(f"🧠 Policy signals: {len(policy_signals)} decisions")

        # 4) Calculate leverage sizing
        probabilities = ensemble_probs

        # Create price dataframe for leverage sizing (need OHLC)
        price_df = test_dataset[['ts', 'close']].copy()
        if 'open' not in test_dataset.columns:
            price_df['open'] = price_df['close'].shift(1).fillna(price_df['close'])
        if 'high' not in test_dataset.columns:
            price_df['high'] = price_df['close'] * 1.01  # Approximate
        if 'low' not in test_dataset.columns:
            price_df['low'] = price_df['close'] * 0.99   # Approximate

        leverage_signals = create_leverage_signals_batch(
            sizer=self.leverage_sizer,
            df=price_df,
            probabilities=probabilities,
            window_size=64
        )

        logger.info(f"⚡ Leverage signals: {len(leverage_signals)} sizing decisions")

        return policy_signals, leverage_signals

    def run_backtest(self, test_dataset: pd.DataFrame,
                    policy_signals: List[PolicySignal],
                    leverage_signals: List[LeverageComponents]) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic trading costs and risk management
        """
        logger.info("📊 Running backtest...")

        if len(policy_signals) != len(leverage_signals):
            raise ValueError("Policy and leverage signals must have same length")

        # Initialize backtest state
        equity = self.config.initial_equity
        position_leverage = 0.0  # Current leverage
        total_trades = 0
        total_costs = 0.0

        # Performance tracking
        equity_curve = [equity]
        leverage_curve = [position_leverage]
        returns_curve = []
        trade_log = []

        # Process each signal
        for i, (policy_signal, leverage_signal) in enumerate(zip(policy_signals, leverage_signals)):
            if i >= len(test_dataset) - 1:
                break

            # Current and next prices
            current_price = test_dataset.iloc[i]['close']
            next_price = test_dataset.iloc[i + 1]['close'] if i + 1 < len(test_dataset) else current_price

            # Price return
            price_return = (next_price / current_price) - 1.0 if current_price > 0 else 0.0

            # Determine target position from both signals
            if policy_signal.action == TradingAction.FLAT:
                target_leverage = 0.0
            else:
                # Use leverage sizing from four-clamp system
                target_leverage = leverage_signal.final_leverage
                if policy_signal.action == TradingAction.SHORT:
                    target_leverage = -target_leverage

            # Calculate position change
            leverage_change = target_leverage - position_leverage

            # Trading costs (only on position changes)
            if abs(leverage_change) > 0.001:  # Minimum change threshold
                notional_traded = abs(leverage_change * equity)
                execution_cost = notional_traded * (self.config.policy_config.costs.total_execution_cost_bp() / 10000.0)
                total_costs += execution_cost
                total_trades += 1

                trade_log.append({
                    'timestamp': test_dataset.iloc[i]['ts'],
                    'action': policy_signal.action.name,
                    'old_leverage': position_leverage,
                    'new_leverage': target_leverage,
                    'cost': execution_cost,
                    'equity_before': equity,
                    'price': current_price,
                    'ensemble_prob': policy_signal.probability,
                    'confidence': policy_signal.confidence
                })
            else:
                execution_cost = 0.0

            # Update position
            position_leverage = target_leverage

            # Calculate PnL
            position_pnl = position_leverage * equity * price_return

            # Funding costs (simplified - charged on leveraged position)
            funding_cost = abs(position_leverage * equity) * (self.config.policy_config.costs.funding_cost_per_step_bp() / 10000.0)

            # Update equity
            equity = equity + position_pnl - execution_cost - funding_cost

            # Track curves
            equity_curve.append(equity)
            leverage_curve.append(position_leverage)
            returns_curve.append((equity / equity_curve[-2]) - 1.0 if len(equity_curve) > 1 else 0.0)

            # Bankruptcy check
            if equity <= 0:
                logger.warning(f"💀 Bankruptcy at step {i}: equity=${equity:.2f}")
                break

        # Calculate performance metrics
        if len(returns_curve) > 0:
            total_return = (equity / self.config.initial_equity) - 1.0
            returns_array = np.array(returns_curve)

            # Sharpe ratio (annualized for 4H bars)
            mean_return = np.mean(returns_array)
            return_vol = np.std(returns_array)
            bars_per_year = 365 * 24 / (self.config.step_minutes / 60)
            sharpe_ratio = (mean_return / return_vol) * np.sqrt(bars_per_year) if return_vol > 0 else 0.0

            # Drawdown
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = 1.0 - (np.array(equity_curve) / peak_equity)
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

            # Win rate
            positive_returns = returns_array > 0
            win_rate = np.mean(positive_returns) if len(returns_array) > 0 else 0.0

            # Trading metrics
            cost_ratio = total_costs / self.config.initial_equity
            avg_leverage = np.mean(np.abs(leverage_curve))

        else:
            total_return = sharpe_ratio = max_drawdown = win_rate = cost_ratio = avg_leverage = 0.0

        backtest_summary = {
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            'total_trades': total_trades,
            'total_costs': total_costs,
            'cost_ratio_pct': cost_ratio * 100,
            'avg_leverage': avg_leverage,
            'final_equity': equity,
            'bars_traded': len(returns_curve),
            'bankruptcy': equity <= 0
        }

        logger.info("📈 Backtest complete:")
        logger.info(f"   💰 Total return: {total_return*100:.2f}%")
        logger.info(f"   📊 Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"   📉 Max drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"   🎯 Win rate: {win_rate*100:.1f}%")
        logger.info(f"   💸 Total costs: ${total_costs:.2f} ({cost_ratio*100:.2f}%)")
        logger.info(f"   ⚖️ Avg leverage: {avg_leverage:.1f}x")

        # Add detailed data to summary
        backtest_summary.update({
            'equity_curve': equity_curve,
            'leverage_curve': leverage_curve,
            'returns_curve': returns_curve,
            'trade_log': trade_log
        })

        return backtest_summary

    def fit(self) -> 'LeverageTradingSystem':
        """
        Train the complete trading system
        """
        start_time = datetime.now()

        logger.info("🎯 Training complete leverage BTC trading system...")

        # 1) Load and prepare data
        dataset, feature_cols, label_cols, data_quality = self.load_and_prepare_data()

        self.dataset = dataset
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.data_quality = data_quality

        # 2) Split data for training and testing
        if self.config.train_end:
            train_mask = dataset['ts'] <= pd.to_datetime(self.config.train_end)
            test_mask = dataset['ts'] > pd.to_datetime(self.config.train_end)
        else:
            # Use 80-20 split
            split_idx = int(0.8 * len(dataset))
            train_mask = dataset.index < split_idx
            test_mask = dataset.index >= split_idx

        train_dataset = dataset[train_mask].copy()
        test_dataset = dataset[test_mask].copy()

        logger.info(f"📊 Data split: {len(train_dataset)} train, {len(test_dataset)} test")

        # 3) Train ensemble model
        self.ensemble = self.train_ensemble_model(train_dataset, feature_cols, label_cols)

        # 4) Create policy and leverage sizing components
        self.policy = self.create_policy_engine()
        self.leverage_sizer = self.create_leverage_sizer()

        # Mark system as fitted after all components are ready
        self.is_fitted = True

        # CRITICAL FIX: Two-layer architecture - SEPARATE learning from trading
        # Layer A: Pure prediction (COMPLETE) - model is now trained and calibrated
        # Layer B: Trading evaluation (SEPARATE) - only evaluate predictive power, no forced trading

        logger.info("🎯 LAYER A COMPLETE: Pure prediction model trained and calibrated")
        logger.info("📊 Evaluating predictive power on OOS data (NO TRADING YET)...")

        # Evaluate model predictions WITHOUT trading pressure
        if len(test_dataset) > 0:
            # Get pure predictions (no trading decisions)
            predictions = self.evaluate_predictions_only(test_dataset, feature_cols)

            # Create diagnostic backtest summary (prediction quality only)
            backtest_summary = {
                'total_return_pct': 0.0,  # No trading during training
                'sharpe_ratio': 0.0,     # No trading during training
                'max_drawdown_pct': 0.0, # No trading during training
                'win_rate_pct': 0.0,     # No trading during training
                'total_trades': 0,       # No trading during training
                'total_costs': 0.0,      # No trading during training
                'cost_ratio_pct': 0.0,   # No trading during training
                'avg_leverage': 0.0,     # No trading during training
                'final_equity': self.config.initial_equity,
                'bars_traded': 0,
                'bankruptcy': False,
                # NEW: Prediction quality metrics
                'mean_prediction': predictions['mean_prediction'],
                'prediction_std': predictions['prediction_std'],
                'calibration_score': predictions['calibration_score'],
                'oos_auc': predictions['oos_auc']
            }
            policy_signals = []
            leverage_signals = []
        else:
            logger.warning("No OOS data for prediction evaluation")
            backtest_summary = {
                'total_return_pct': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown_pct': 0.0,
                'win_rate_pct': 0.0, 'total_trades': 0, 'total_costs': 0.0, 'cost_ratio_pct': 0.0,
                'avg_leverage': 0.0, 'final_equity': self.config.initial_equity, 'bars_traded': 0,
                'bankruptcy': False, 'mean_prediction': 0.5, 'prediction_std': 0.0,
                'calibration_score': 0.0, 'oos_auc': 0.5
            }
            policy_signals = []
            leverage_signals = []

        logger.info("✅ TRAINING PHASE COMPLETE - Model ready for Layer B policy evaluation")

        # 7) Analyze performance (prediction quality only during training)
        ensemble_metrics = self.ensemble.get_ensemble_metrics()
        policy_analysis = {'total_signals': 0, 'execution_rate': 0.0, 'avg_confidence': 0.5}  # No policy analysis during training
        leverage_stats = self.leverage_sizer.get_leverage_statistics()

        # 8) Compile results
        training_time = (datetime.now() - start_time).total_seconds()

        self.results = TradingResults(
            data_quality=data_quality,
            ensemble_metrics=ensemble_metrics,
            policy_analysis=policy_analysis,
            leverage_stats=leverage_stats,
            backtest_summary=backtest_summary,
            config_summary=asdict(self.config),
            training_time=training_time,
            total_signals=len(policy_signals)
        )

        logger.info(f"🎉 Training complete in {training_time:.1f}s")
        logger.info(f"   🤖 {ensemble_metrics.get('n_models', 0)} ensemble models")
        logger.info(f"   📡 {len(policy_signals)} trading signals generated")
        logger.info(f"   💰 {backtest_summary['total_return_pct']:.2f}% total return")
        logger.info(f"   📊 {backtest_summary['sharpe_ratio']:.2f} Sharpe ratio")

        return self

    def save_system(self, model_path: str = None, results_path: str = None) -> None:
        """Save the complete trained system and results"""
        if not self.is_fitted:
            raise ValueError("System must be fitted before saving")

        # Save model
        model_path = model_path or self.config.model_save_path
        self.ensemble.save_ensemble(model_path)

        # Save results
        results_path = results_path or self.config.results_save_path

        # Convert results to serializable format
        results_dict = asdict(self.results)
        # Remove numpy arrays and other non-serializable objects
        if 'equity_curve' in results_dict['backtest_summary']:
            results_dict['backtest_summary']['equity_curve'] = None
        if 'leverage_curve' in results_dict['backtest_summary']:
            results_dict['backtest_summary']['leverage_curve'] = None
        if 'returns_curve' in results_dict['backtest_summary']:
            results_dict['backtest_summary']['returns_curve'] = None
        if 'trade_log' in results_dict['backtest_summary']:
            results_dict['backtest_summary']['trade_log'] = None

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info("💾 System saved:")
        logger.info(f"   🤖 Model: {model_path}")
        logger.info(f"   📊 Results: {results_path}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get human-readable performance summary"""
        if not self.is_fitted or not self.results:
            return {"status": "not_fitted"}

        return {
            "system_status": "fitted",
            "training_time_seconds": self.results.training_time,
            "data_quality_score": 100 - self.results.data_quality.nan_rate * 100,
            "ensemble_models": self.results.ensemble_metrics.get('n_models', 0),
            "ensemble_auc": self.results.ensemble_metrics.get('mean_auc', 0),
            "total_return_pct": self.results.backtest_summary['total_return_pct'],
            "sharpe_ratio": self.results.backtest_summary['sharpe_ratio'],
            "max_drawdown_pct": self.results.backtest_summary['max_drawdown_pct'],
            "win_rate_pct": self.results.backtest_summary['win_rate_pct'],
            "avg_leverage": self.results.backtest_summary['avg_leverage'],
            "total_trades": self.results.backtest_summary['total_trades'],
            "cost_ratio_pct": self.results.backtest_summary['cost_ratio_pct'],
            "most_binding_constraint": self.results.leverage_stats.get('most_binding_constraint', 'unknown')
        }

# High-level training function
def train_leverage_trading_system(config: TradingSystemConfig = None) -> LeverageTradingSystem:
    """
    High-level function to train the complete leverage BTC trading system
    """
    if config is None:
        config = TradingSystemConfig()

    system = LeverageTradingSystem(config)
    system.fit()

    return system

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Basic test
    logger.info("🚀 Testing Leverage BTC Trading System...")

    # Create test configuration
    config = TradingSystemConfig(
        data_path="/home/user/projects/machinelearning.1/data/processed/btc_comprehensive_dataset_20250812_125933_features.csv"
    )

    try:
        # Train system
        system = train_leverage_trading_system(config)

        # Get performance summary
        summary = system.get_performance_summary()

        logger.info("🎯 System Performance Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")

        logger.info("✅ Leverage BTC Trading System test complete")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise