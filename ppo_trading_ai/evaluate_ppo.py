#!/usr/bin/env python3
"""
PPO Trading AI Evaluation and Backtesting Framework
Comprehensive evaluation with risk metrics and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ppo_env import PPOTradingEnvironment, TradingConfig
from ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float
    total_trades: int
    avg_trade_duration: float
    total_fees: float
    total_funding: float
    total_slippage: float
    final_balance: float

class TradingEvaluator:
    """Comprehensive trading strategy evaluator"""
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.results_history = []
        
    def backtest_strategy(self, agent: PPOAgent, data: pd.DataFrame, 
                         start_idx: int = 0, end_idx: int = None) -> BacktestResults:
        """Run comprehensive backtest with realistic costs"""
        
        if end_idx is None:
            end_idx = len(data)
            
        logger.info(f"Running backtest from {start_idx} to {end_idx} ({end_idx - start_idx} steps)")
        
        # Create environment for backtesting
        env = PPOTradingEnvironment(self.config)
        env.data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        env.prices = env.prices[start_idx:end_idx]
        
        # Tracking variables
        equity_curve = []
        positions = []
        returns = []
        trades = []
        drawdowns = []
        
        # Run backtest
        obs, info = env.reset()
        done = False
        step = 0
        
        agent.hidden_state = None  # Reset recurrent state
        
        while not done:
            # Get action from agent (deterministic)
            action, _, _ = agent.get_action(obs, deterministic=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record metrics
            equity_curve.append(info['equity'])
            positions.append(info['position'])
            returns.append(reward)
            
            # Calculate drawdown
            if equity_curve:
                peak = max(equity_curve)
                drawdown = (peak - info['equity']) / peak
                drawdowns.append(drawdown)
            else:
                drawdowns.append(0.0)
                
            # Track trades (position changes)
            if len(positions) > 1 and abs(positions[-1] - positions[-2]) > 1e-6:
                trades.append({
                    'step': step,
                    'position_change': positions[-1] - positions[-2],
                    'equity': info['equity'],
                })
                
            obs = next_obs
            step += 1
            
        # Calculate comprehensive metrics
        results = self._calculate_metrics(
            equity_curve, returns, positions, trades, drawdowns, info
        )
        
        logger.info(f"Backtest completed: {results.total_return:.2%} return, "
                   f"{results.sharpe_ratio:.2f} Sharpe, {results.max_drawdown:.2%} max DD")
        
        return results
        
    def _calculate_metrics(self, equity_curve: List[float], returns: List[float],
                          positions: List[float], trades: List[Dict], 
                          drawdowns: List[float], final_info: Dict) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        if not equity_curve:
            # Return zeros if no data
            return BacktestResults(**{k: 0.0 for k in BacktestResults.__annotations__})
            
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)
        positions = np.array(positions)
        drawdowns = np.array(drawdowns)
        
        # Basic returns
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Assume each step is 1 hour for annualization
        periods_per_year = 365 * 24  # 8760 hours per year
        n_periods = len(equity_curve)
        years = n_periods / periods_per_year
        
        annualized_return = (1 + total_return) ** (1 / max(years, 1/252)) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(periods_per_year)
        else:
            volatility = 0.0
            
        # Risk-adjusted returns
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0
            
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_volatility = np.std(negative_returns) * np.sqrt(periods_per_year)
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        else:
            sortino_ratio = float('inf') if annualized_return > 0 else 0
            
        # Drawdown metrics
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Trade analysis
        if len(trades) > 0:
            # Win rate
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades)
            
            # Average trade duration (simplified)
            avg_trade_duration = n_periods / len(trades)
            
            # Profit factor (simplified)
            total_wins = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
            total_losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            win_rate = 0.0
            avg_trade_duration = 0.0
            profit_factor = 0.0
            
        # Value at Risk (VaR) and Conditional VaR
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)  # 5th percentile (95% VaR)
            cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0
            
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            total_fees=final_info.get('total_fees', 0),
            total_funding=final_info.get('total_funding', 0),
            total_slippage=final_info.get('total_slippage', 0),
            final_balance=final_info.get('equity', equity_curve[-1])
        )
        
    def benchmark_comparison(self, agent: PPOAgent, data: pd.DataFrame) -> Dict[str, BacktestResults]:
        """Compare against standard benchmarks"""
        results = {}
        
        # PPO strategy
        logger.info("Evaluating PPO strategy...")
        results['PPO'] = self.backtest_strategy(agent, data)
        
        # Buy and Hold
        logger.info("Evaluating Buy & Hold benchmark...")
        results['Buy_Hold'] = self._backtest_buy_hold(data)
        
        # Simple momentum
        logger.info("Evaluating Momentum benchmark...")
        results['Momentum'] = self._backtest_momentum(data)
        
        return results
        
    def _backtest_buy_hold(self, data: pd.DataFrame) -> BacktestResults:
        """Simple buy and hold benchmark"""
        if 'close' in data.columns:
            prices = data['close'].values
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            prices = data[numeric_cols[0]].values
            
        initial_price = prices[0]
        final_price = prices[-1]
        total_return = (final_price - initial_price) / initial_price
        
        # Create simple equity curve
        equity_curve = self.config.initial_balance * (prices / initial_price)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=(1 + total_return) ** (365 * 24 / len(prices)) - 1,
            volatility=np.std(returns) * np.sqrt(365 * 24),
            sharpe_ratio=(total_return) / (np.std(returns) * np.sqrt(365 * 24)) if np.std(returns) > 0 else 0,
            sortino_ratio=0.0,  # Simplified
            max_drawdown=np.max(drawdowns),
            calmar_ratio=0.0,  # Simplified
            win_rate=0.0,
            profit_factor=0.0,
            var_95=np.percentile(returns, 5) if len(returns) > 0 else 0,
            cvar_95=0.0,
            total_trades=1,  # Single buy trade
            avg_trade_duration=len(prices),
            total_fees=0.0,  # Ignore fees for benchmark
            total_funding=0.0,
            total_slippage=0.0,
            final_balance=equity_curve[-1]
        )
        
    def _backtest_momentum(self, data: pd.DataFrame) -> BacktestResults:
        """Simple momentum benchmark"""
        # Simplified momentum strategy - go long when 24h return > 0
        if 'return_1h' in data.columns:
            returns_1h = data['return_1h'].values
        else:
            # Calculate simple returns
            prices = data.iloc[:, 0].values  # First column as price
            returns_1h = np.diff(np.log(prices))
            returns_1h = np.concatenate([[0], returns_1h])  # Pad first value
            
        # Momentum signal: 24h rolling average return
        momentum_signal = pd.Series(returns_1h).rolling(24).mean().fillna(0).values
        
        # Simple position: +1 if momentum > 0, -1 if < 0, 0 otherwise
        positions = np.where(momentum_signal > 0.001, 0.5, 
                           np.where(momentum_signal < -0.001, -0.5, 0))
        
        # Calculate strategy returns
        strategy_returns = positions[:-1] * returns_1h[1:]  # Lag positions by 1
        equity_curve = self.config.initial_balance * np.cumprod(1 + strategy_returns)
        
        # Calculate metrics (simplified)
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=(1 + total_return) ** (365 * 24 / len(strategy_returns)) - 1,
            volatility=np.std(strategy_returns) * np.sqrt(365 * 24),
            sharpe_ratio=total_return / (np.std(strategy_returns) * np.sqrt(365 * 24)) if np.std(strategy_returns) > 0 else 0,
            sortino_ratio=0.0,
            max_drawdown=0.0,  # Simplified
            calmar_ratio=0.0,
            win_rate=np.mean(strategy_returns > 0),
            profit_factor=0.0,
            var_95=np.percentile(strategy_returns, 5),
            cvar_95=0.0,
            total_trades=np.sum(np.abs(np.diff(positions)) > 0),
            avg_trade_duration=0.0,
            total_fees=0.0,
            total_funding=0.0,
            total_slippage=0.0,
            final_balance=equity_curve[-1]
        )
        
    def stress_test(self, agent: PPOAgent, data: pd.DataFrame) -> Dict[str, BacktestResults]:
        """Stress test on different market regimes"""
        results = {}
        
        # Define stress test periods (simplified - use different data slices)
        n_samples = len(data)
        
        # Crash period (first 20% - simulate crash)
        crash_start = 0
        crash_end = n_samples // 5
        logger.info(f"Stress test - Crash period: {crash_start} to {crash_end}")
        results['Crash'] = self.backtest_strategy(agent, data, crash_start, crash_end)
        
        # Pump period (middle 20% - simulate bull run)
        pump_start = n_samples // 5 * 2
        pump_end = n_samples // 5 * 3
        logger.info(f"Stress test - Pump period: {pump_start} to {pump_end}")
        results['Pump'] = self.backtest_strategy(agent, data, pump_start, pump_end)
        
        # Chop period (last 20% - simulate sideways)
        chop_start = n_samples // 5 * 4
        chop_end = n_samples
        logger.info(f"Stress test - Chop period: {chop_start} to {chop_end}")
        results['Chop'] = self.backtest_strategy(agent, data, chop_start, chop_end)
        
        return results
        
    def generate_report(self, results: Dict[str, BacktestResults], save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("PPO TRADING AI - COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Results table
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}")
        report.append("-" * 40)
        
        for name, result in results.items():
            report.append(f"{name:<15} {result.total_return:>7.2%} {result.sharpe_ratio:>7.2f} "
                         f"{result.max_drawdown:>7.2%} {result.total_trades:>7d}")
                         
        report.append("")
        
        # Detailed metrics for main strategy (PPO)
        if 'PPO' in results:
            ppo_results = results['PPO']
            report.append("PPO STRATEGY - DETAILED METRICS")
            report.append("-" * 40)
            report.append(f"Total Return:          {ppo_results.total_return:>10.2%}")
            report.append(f"Annualized Return:     {ppo_results.annualized_return:>10.2%}")
            report.append(f"Volatility:            {ppo_results.volatility:>10.2%}")
            report.append(f"Sharpe Ratio:          {ppo_results.sharpe_ratio:>10.2f}")
            report.append(f"Sortino Ratio:         {ppo_results.sortino_ratio:>10.2f}")
            report.append(f"Max Drawdown:          {ppo_results.max_drawdown:>10.2%}")
            report.append(f"Calmar Ratio:          {ppo_results.calmar_ratio:>10.2f}")
            report.append(f"Win Rate:              {ppo_results.win_rate:>10.2%}")
            report.append(f"Profit Factor:         {ppo_results.profit_factor:>10.2f}")
            report.append(f"VaR (95%):             {ppo_results.var_95:>10.4f}")
            report.append(f"CVaR (95%):            {ppo_results.cvar_95:>10.4f}")
            report.append(f"Total Trades:          {ppo_results.total_trades:>10d}")
            report.append(f"Total Fees:            ${ppo_results.total_fees:>9.2f}")
            report.append(f"Total Funding:         ${ppo_results.total_funding:>9.2f}")
            report.append(f"Final Balance:         ${ppo_results.final_balance:>9.2f}")
            report.append("")
            
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
            
        return report_text

def main():
    """Main evaluation function"""
    logger.info("🔍 Starting PPO Trading AI Evaluation...")
    
    # Load trained model
    model_path = '/tmp/ppo_final_model.pth'  # or from cloud storage
    
    try:
        # Try to download from cloud storage
        import subprocess
        subprocess.run(['gsutil', 'cp', 
                       'gs://YOUR_GCP_PROJECT-ml-trading-central/ppo_final_model.pth',
                       model_path], check=True)
        logger.info("✅ Downloaded model from cloud storage")
    except:
        logger.warning("⚠️ Could not download model from cloud - using local if available")
        
    # Load model
    if os.path.exists(model_path):
        # Create dummy agent to get dimensions
        env = PPOTradingEnvironment()
        obs_dim = env.observation_space.shape[0]
        
        agent = PPOAgent(obs_dim)
        agent.load(model_path)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ No trained model found - skipping evaluation")
        return
        
    # Load data
    logger.info("Loading evaluation data...")
    env = PPOTradingEnvironment()
    data = env.data
    
    # Create evaluator
    evaluator = TradingEvaluator()
    
    # Run comprehensive evaluation
    logger.info("Running benchmark comparison...")
    benchmark_results = evaluator.benchmark_comparison(agent, data)
    
    logger.info("Running stress tests...")
    stress_results = evaluator.stress_test(agent, data)
    
    # Combine results
    all_results = {**benchmark_results, **stress_results}
    
    # Generate report
    report = evaluator.generate_report(all_results, '/tmp/ppo_evaluation_report.txt')
    print(report)
    
    logger.info("✅ Evaluation completed!")

if __name__ == "__main__":
    import os
    main()