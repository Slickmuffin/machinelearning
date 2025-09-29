#!/usr/bin/env python3
"""
Four-Clamp Leverage Sizing System for 125x Leverage BTC Trading
Phase 4: Dynamic leverage control with volatility targeting, Kelly criterion, liquidation distance, and VaR

Leverage Architecture:
- Vol-target clamp: L_vol = κ·V*/(σ̂_bar·√B)
- Edge-based clamp: L_kelly = α·e/v (fractional Kelly)
- Liquidation-distance clamp: L_liq ≤ 1/(z·ATR)
- Drawdown & VaR clamp: Dynamic scaling based on risk metrics

Final leverage = min(125, L_vol, L_kelly, L_liq, L_var)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

@dataclass
class VolatilityTargetConfig:
    """Volatility targeting configuration"""
    target_annual_vol: float = 0.15  # 15% annual volatility target
    vol_estimator_span: int = 24     # Bars for volatility estimation
    vol_adjustment_factor: float = 1.0  # κ factor (0.8-1.2 range)
    bars_per_year: int = 2190        # 4H bars per year (365*24/4 = 2190)
    min_vol_estimate: float = 0.001  # Minimum vol to prevent division by zero

@dataclass
class KellyConfig:
    """Kelly criterion configuration"""
    kelly_fraction: float = 0.2      # Fractional Kelly (α ∈ [0.1, 0.25])
    min_edge: float = -0.1           # Minimum edge to consider (negative allows some risk)
    max_kelly_leverage: float = 10.0 # Cap Kelly leverage for safety
    edge_smoothing_span: int = 12    # Bars to smooth edge estimates

@dataclass
class LiquidationConfig:
    """Liquidation distance configuration"""
    min_liq_distance_atr: float = 5.0  # Minimum z×ATR distance (z ≈ 4-6)
    atr_span: int = 20                 # Bars for ATR calculation
    exchange_margin_buffer: float = 1.2 # Buffer factor for exchange differences
    max_theoretical_leverage: float = 125.0  # Hard cap

@dataclass
class RiskConfig:
    """Drawdown and VaR configuration"""
    # Drawdown thresholds
    dd_warning_threshold: float = 0.10   # 10% DD warning
    dd_danger_threshold: float = 0.20    # 20% DD danger
    dd_warning_scale: float = 0.5        # Scale leverage by 50% at warning
    dd_danger_scale: float = 0.0         # Flat at danger level

    # VaR settings
    var_confidence: float = 0.01         # 1% VaR (99th percentile)
    var_z_score: float = 2.33           # Z-score for 99% confidence
    var_lookback: int = 64              # Bars for VaR calculation
    daily_var_budget: float = 0.01      # 1% daily VaR budget

@dataclass
class LeverageSizingConfig:
    """Master configuration for four-clamp leverage sizing"""
    vol_config: VolatilityTargetConfig = None
    kelly_config: KellyConfig = None
    liq_config: LiquidationConfig = None
    risk_config: RiskConfig = None

    # Global constraints
    absolute_max_leverage: float = 125.0
    min_position_size: float = 0.01     # Minimum leverage to avoid dust
    leverage_smoothing: bool = True     # Smooth leverage changes
    smoothing_factor: float = 0.8       # Exponential smoothing factor

    def __post_init__(self):
        if self.vol_config is None:
            self.vol_config = VolatilityTargetConfig()
        if self.kelly_config is None:
            self.kelly_config = KellyConfig()
        if self.liq_config is None:
            self.liq_config = LiquidationConfig()
        if self.risk_config is None:
            self.risk_config = RiskConfig()

@dataclass
class LeverageComponents:
    """Detailed breakdown of leverage calculation"""
    vol_target_leverage: float
    kelly_leverage: float
    liquidation_leverage: float
    var_leverage: float
    dd_scale_factor: float
    final_leverage: float
    active_constraint: str  # Which clamp is binding
    reasoning: str

class AdvancedVolatilityEstimator:
    """Advanced volatility estimation with multiple methods"""

    def __init__(self, method: str = 'yang_zhang'):
        self.method = method

    def estimate_volatility(self, df: pd.DataFrame, span: int = 24) -> pd.Series:
        """
        Estimate volatility using specified method

        Methods:
        - 'returns': Simple return-based EWMA
        - 'yang_zhang': Yang-Zhang estimator (requires OHLC)
        - 'garman_klass': Garman-Klass estimator
        """
        if self.method == 'yang_zhang' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return self._yang_zhang_volatility(df, span)
        elif self.method == 'garman_klass' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return self._garman_klass_volatility(df, span)
        else:
            # Fallback to simple returns
            return self._returns_volatility(df, span)

    def _returns_volatility(self, df: pd.DataFrame, span: int) -> pd.Series:
        """Simple log return volatility"""
        returns = np.log(df['close'] / df['close'].shift(1))
        return returns.ewm(span=span, adjust=False).std()

    def _yang_zhang_volatility(self, df: pd.DataFrame, span: int) -> pd.Series:
        """Yang-Zhang volatility estimator"""
        # Overnight returns
        overnight = np.log(df['open'] / df['close'].shift(1))

        # Open-to-close returns
        open_close = np.log(df['close'] / df['open'])

        # High/low components
        high_close = np.log(df['high'] / df['close'])
        low_close = np.log(df['low'] / df['close'])
        high_open = np.log(df['high'] / df['open'])
        low_open = np.log(df['low'] / df['open'])

        # Yang-Zhang formula
        yz_component = high_open * low_open + high_close * low_close
        yz_variance = (overnight**2 + open_close**2 + yz_component).ewm(span=span, adjust=False).mean()

        return np.sqrt(yz_variance)

    def _garman_klass_volatility(self, df: pd.DataFrame, span: int) -> pd.Series:
        """Garman-Klass volatility estimator"""
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])

        gk_variance = (0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2).ewm(span=span, adjust=False).mean()
        return np.sqrt(gk_variance)

class FourClampLeverageSizer:
    """
    Advanced leverage sizing with four independent clamps
    Implements the complete mathematical framework for 125x leverage trading
    """

    def __init__(self, config: LeverageSizingConfig = None):
        self.config = config or LeverageSizingConfig()
        self.vol_estimator = AdvancedVolatilityEstimator()

        # State tracking
        self.current_leverage: float = 0.0
        self.peak_equity: float = 10000.0
        self.current_equity: float = 10000.0

        # Performance tracking
        self.leverage_history: List[float] = []
        self.component_history: List[LeverageComponents] = []

        logger.info("⚡ Four-clamp leverage sizing system initialized")
        logger.info(f"   🎯 Vol target: {self.config.vol_config.target_annual_vol:.1%}")
        logger.info(f"   📊 Kelly fraction: {self.config.kelly_config.kelly_fraction:.1f}")
        logger.info(f"   🛡️ Min liq distance: {self.config.liq_config.min_liq_distance_atr:.1f}×ATR")
        logger.info(f"   🔒 Max leverage: {self.config.absolute_max_leverage:.0f}x")

    def calculate_vol_target_leverage(self, vol_estimate: float) -> float:
        """
        Volatility targeting leverage: L_vol = κ·V*/(σ̂_bar·√B)

        Args:
            vol_estimate: Per-bar volatility estimate

        Returns:
            Volume-targeted leverage
        """
        # Annualize the bar volatility
        vol_annual = vol_estimate * np.sqrt(self.config.vol_config.bars_per_year)

        # Avoid division by zero
        vol_annual = max(vol_annual, self.config.vol_config.min_vol_estimate)

        # Vol-target leverage
        vol_leverage = (
            self.config.vol_config.vol_adjustment_factor *
            self.config.vol_config.target_annual_vol /
            vol_annual
        )

        return max(0.0, vol_leverage)

    def calculate_kelly_leverage(self, probability: float, vol_estimate: float,
                               expected_abs_move: float, cost_bp: float = 3.5) -> float:
        """
        Edge-based Kelly leverage: L_kelly = α·e/v

        Args:
            probability: Directional probability from ensemble
            vol_estimate: Per-bar volatility
            expected_abs_move: Expected absolute move E|r|
            cost_bp: Trading costs in basis points

        Returns:
            Kelly-optimal leverage (fractional)
        """
        # Calculate edge: e = (2p-1)·E|r| - c
        directional_edge = (2 * probability - 1) * expected_abs_move
        cost_decimal = cost_bp / 10000.0
        edge = directional_edge - cost_decimal

        # Risk proxy: v = σ²
        risk_proxy = vol_estimate ** 2
        risk_proxy = max(risk_proxy, 1e-12)  # Prevent division by zero

        # Kelly leverage with fractional scaling
        kelly_raw = edge / risk_proxy if edge > self.config.kelly_config.min_edge else 0.0
        kelly_leverage = self.config.kelly_config.kelly_fraction * max(0.0, kelly_raw)

        # Cap Kelly leverage for safety
        kelly_leverage = min(kelly_leverage, self.config.kelly_config.max_kelly_leverage)

        return kelly_leverage

    def calculate_liquidation_leverage(self, atr_estimate: float, current_price: float = None) -> float:
        """
        Liquidation distance leverage: L_liq ≤ 1/(z·ATR)

        Args:
            atr_estimate: Average True Range in return units
            current_price: Current price (optional, for logging)

        Returns:
            Maximum leverage to maintain liquidation distance
        """
        # Required liquidation distance in return units
        min_distance = self.config.liq_config.min_liq_distance_atr * atr_estimate

        # Add exchange margin buffer
        min_distance *= self.config.liq_config.exchange_margin_buffer

        # Liquidation leverage: approximately 1/distance
        if min_distance > 0:
            liq_leverage = 1.0 / min_distance
        else:
            liq_leverage = self.config.liq_config.max_theoretical_leverage

        # Cap at theoretical maximum
        liq_leverage = min(liq_leverage, self.config.liq_config.max_theoretical_leverage)

        return liq_leverage

    def calculate_var_leverage(self, vol_1d: float) -> float:
        """
        VaR-based leverage: L_var ≤ R/(σ_1d·z_q)

        Args:
            vol_1d: 1-day volatility estimate

        Returns:
            VaR-constrained leverage
        """
        # Daily VaR constraint
        var_denominator = vol_1d * self.config.risk_config.var_z_score
        var_denominator = max(var_denominator, 1e-8)  # Prevent division by zero

        var_leverage = self.config.risk_config.daily_var_budget / var_denominator

        return max(0.0, var_leverage)

    def calculate_drawdown_scale_factor(self) -> float:
        """
        Calculate leverage scaling factor based on current drawdown

        Returns:
            Scale factor (0.0 to 1.0)
        """
        if self.peak_equity <= 0:
            return 0.0

        current_drawdown = 1.0 - (self.current_equity / self.peak_equity)

        if current_drawdown >= self.config.risk_config.dd_danger_threshold:
            return self.config.risk_config.dd_danger_scale  # Flat position
        elif current_drawdown >= self.config.risk_config.dd_warning_threshold:
            return self.config.risk_config.dd_warning_scale  # Half leverage
        else:
            return 1.0  # Full leverage

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            # Fallback: use simple range based on close prices
            returns = np.log(df['close'] / df['close'].shift(1))
            return returns.abs().rolling(self.config.liq_config.atr_span).mean()

        # True range calculation
        high_low = df['high'] / df['low']
        high_close_prev = df['high'] / df['close'].shift(1)
        low_close_prev = df['low'] / df['close'].shift(1)

        true_ranges = np.maximum.reduce([
            np.log(high_low),
            np.log(high_close_prev),
            np.log(low_close_prev)
        ])

        # Convert to pandas Series for rolling calculation
        true_ranges_series = pd.Series(true_ranges, index=df.index)
        return true_ranges_series.rolling(self.config.liq_config.atr_span).mean()

    def size_leverage(self, df: pd.DataFrame, probability: float,
                     current_equity: float = None, current_price: float = None) -> LeverageComponents:
        """
        Master leverage sizing function applying all four clamps

        Args:
            df: Price dataframe with OHLC data
            probability: Ensemble directional probability
            current_equity: Current account equity
            current_price: Current BTC price

        Returns:
            LeverageComponents with full breakdown
        """
        # Update equity tracking
        if current_equity is not None:
            self.current_equity = current_equity
            self.peak_equity = max(self.peak_equity, current_equity)

        # Get latest data point
        if len(df) == 0:
            raise ValueError("Empty dataframe provided")

        latest_idx = len(df) - 1

        # 1) Volatility estimates
        vol_series = self.vol_estimator.estimate_volatility(df, self.config.vol_config.vol_estimator_span)
        vol_estimate = vol_series.iloc[-1] if len(vol_series) > 0 else 0.02

        # 2) ATR estimate
        atr_series = self.calculate_atr(df)
        atr_estimate = atr_series.iloc[-1] if len(atr_series) > 0 else vol_estimate

        # 3) Expected absolute move
        expected_abs_move = vol_estimate * np.sqrt(2 / np.pi)

        # 4) Daily volatility (for VaR)
        bars_per_day = self.config.vol_config.bars_per_year / 365.25
        vol_1d = vol_estimate * np.sqrt(bars_per_day)

        # 5) Calculate all four leverage components
        vol_leverage = self.calculate_vol_target_leverage(vol_estimate)
        kelly_leverage = self.calculate_kelly_leverage(probability, vol_estimate, expected_abs_move)
        liq_leverage = self.calculate_liquidation_leverage(atr_estimate, current_price)
        var_leverage = self.calculate_var_leverage(vol_1d)

        # 6) Drawdown scaling
        dd_scale = self.calculate_drawdown_scale_factor()

        # 7) Find minimum (most restrictive) constraint
        leverage_candidates = {
            'vol_target': vol_leverage,
            'kelly': kelly_leverage,
            'liquidation': liq_leverage,
            'var': var_leverage,
            'absolute_max': self.config.absolute_max_leverage
        }

        # Find the binding constraint
        binding_leverage = min(leverage_candidates.values())
        active_constraint = min(leverage_candidates.keys(), key=lambda k: leverage_candidates[k])

        # Apply drawdown scaling
        final_leverage = binding_leverage * dd_scale

        # Apply minimum position filter
        if abs(final_leverage) < self.config.min_position_size:
            final_leverage = 0.0
            active_constraint = 'min_position'

        # Leverage smoothing
        if self.config.leverage_smoothing and len(self.leverage_history) > 0:
            prev_leverage = self.leverage_history[-1]
            smoothed_leverage = (
                self.config.smoothing_factor * prev_leverage +
                (1 - self.config.smoothing_factor) * final_leverage
            )
            final_leverage = smoothed_leverage

        # Create detailed breakdown
        reasoning_parts = [
            f"Vol: {vol_leverage:.2f}x (σ={vol_estimate:.3f})",
            f"Kelly: {kelly_leverage:.2f}x (p={probability:.3f})",
            f"Liq: {liq_leverage:.2f}x (ATR={atr_estimate:.3f})",
            f"VaR: {var_leverage:.2f}x",
            f"DD scale: {dd_scale:.2f}",
            f"Active: {active_constraint}"
        ]

        components = LeverageComponents(
            vol_target_leverage=vol_leverage,
            kelly_leverage=kelly_leverage,
            liquidation_leverage=liq_leverage,
            var_leverage=var_leverage,
            dd_scale_factor=dd_scale,
            final_leverage=final_leverage,
            active_constraint=active_constraint,
            reasoning=" | ".join(reasoning_parts)
        )

        # Update history
        self.leverage_history.append(final_leverage)
        self.component_history.append(components)
        self.current_leverage = final_leverage

        return components

    def get_leverage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive leverage sizing statistics"""
        if not self.component_history:
            return {}

        leverages = [c.final_leverage for c in self.component_history]
        constraints = [c.active_constraint for c in self.component_history]

        # Constraint frequency
        constraint_counts = pd.Series(constraints).value_counts()
        constraint_pct = (constraint_counts / len(constraints) * 100).round(1)

        stats = {
            'total_signals': len(leverages),
            'mean_leverage': np.mean(leverages),
            'max_leverage': max(leverages),
            'min_leverage': min(leverages),
            'leverage_std': np.std(leverages),
            'zero_leverage_pct': (np.array(leverages) == 0).mean() * 100,
            'constraint_frequency': constraint_pct.to_dict(),
            'most_binding_constraint': constraint_pct.index[0] if len(constraint_pct) > 0 else 'none',
            'current_drawdown': 1.0 - (self.current_equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        }

        return stats

    def reset_equity_tracking(self, initial_equity: float = 10000.0):
        """Reset equity and drawdown tracking"""
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        logger.info(f"🔄 Equity tracking reset to ${initial_equity:,.0f}")

# Utility functions for leverage sizing integration

def create_leverage_signals_batch(sizer: FourClampLeverageSizer,
                                df: pd.DataFrame,
                                probabilities: np.ndarray,
                                equities: Optional[np.ndarray] = None,
                                window_size: int = 100) -> List[LeverageComponents]:
    """
    Generate leverage signals for a batch of predictions using rolling windows

    Args:
        sizer: FourClampLeverageSizer instance
        df: Full price dataframe
        probabilities: Array of ensemble probabilities
        equities: Array of account equities (optional)
        window_size: Rolling window size for calculations

    Returns:
        List of LeverageComponents for each prediction
    """
    if equities is None:
        equities = np.full(len(probabilities), 10000.0)

    if len(probabilities) != len(equities):
        raise ValueError("Probabilities and equities arrays must have same length")

    leverage_signals = []

    # Process in rolling windows
    for i in range(len(probabilities)):
        # Define window boundaries
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1

        # Extract window data
        window_df = df.iloc[start_idx:end_idx].copy()

        if len(window_df) < 10:  # Need minimum data
            # Create dummy components for insufficient data
            components = LeverageComponents(
                vol_target_leverage=0.0,
                kelly_leverage=0.0,
                liquidation_leverage=0.0,
                var_leverage=0.0,
                dd_scale_factor=1.0,
                final_leverage=0.0,
                active_constraint='insufficient_data',
                reasoning="Insufficient historical data for sizing"
            )
        else:
            # Calculate leverage with full four-clamp system
            components = sizer.size_leverage(
                df=window_df,
                probability=probabilities[i],
                current_equity=equities[i]
            )

        leverage_signals.append(components)

    logger.info(f"✅ Generated {len(leverage_signals)} leverage signals using four-clamp sizing")

    return leverage_signals

if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)

    # Create test data
    np.random.seed(42)
    n_bars = 100

    test_data = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=n_bars, freq='4H'),
        'close': 50000 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_bars))),
        'open': np.nan,
        'high': np.nan,
        'low': np.nan
    })

    # Fill OHLC
    test_data['open'] = test_data['close'].shift(1).fillna(test_data['close'])
    test_data['high'] = test_data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    test_data['low'] = test_data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))

    # Test leverage sizer
    sizer = FourClampLeverageSizer()

    # Test single calculation
    components = sizer.size_leverage(
        df=test_data,
        probability=0.65,  # 65% directional probability
        current_equity=10000.0
    )

    logger.info(f"🎯 Test leverage calculation:")
    logger.info(f"   Final leverage: {components.final_leverage:.2f}x")
    logger.info(f"   Active constraint: {components.active_constraint}")
    logger.info(f"   Reasoning: {components.reasoning}")

    # Test batch processing
    probabilities = np.random.uniform(0.4, 0.8, 20)  # Sample probabilities
    batch_signals = create_leverage_signals_batch(sizer, test_data, probabilities)

    stats = sizer.get_leverage_statistics()
    logger.info(f"📊 Batch statistics: Mean leverage {stats['mean_leverage']:.2f}x, Most binding: {stats['most_binding_constraint']}")
    logger.info("✅ Four-clamp leverage sizing system test complete")