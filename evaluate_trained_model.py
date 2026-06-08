#!/usr/bin/env python3
"""
Comprehensive PPO Trading Model Evaluation
Evaluates trained model against strict production criteria:
- Sharpe Ratio ≥1.5
- Win Rate ≥65%
"""

import numpy as np
import torch
import logging
from dataclasses import dataclass
import sys
from pathlib import Path

# Import from PPO trading system
sys.path.insert(0, str(Path(__file__).parent / 'ppo_trading_ai'))
from data_pipeline import load_prices_or_fail
from trading_env_fixed import SlidingWindowTradingEnv
from ppo_agent import PPOAgent, PPOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    final_balance: float
    total_pnl: float

    def meets_production_criteria(self) -> bool:
        """Check if metrics meet strict production requirements"""
        return self.sharpe_ratio >= 1.5 and self.win_rate >= 0.65

    def format_report(self) -> str:
        """Format metrics as readable report"""
        report = []
        report.append("=" * 80)
        report.append("PPO TRADING MODEL - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("PRODUCTION CRITERIA:")
        report.append(f"  Sharpe Ratio: {self.sharpe_ratio:>8.3f} (Target: ≥1.5) {'✓ PASS' if self.sharpe_ratio >= 1.5 else '✗ FAIL'}")
        report.append(f"  Win Rate:     {self.win_rate:>8.1%} (Target: ≥65%) {'✓ PASS' if self.win_rate >= 0.65 else '✗ FAIL'}")
        report.append("")
        report.append(f"PRODUCTION READINESS: {'✓ APPROVED FOR DEPLOYMENT' if self.meets_production_criteria() else '✗ NOT READY - NEEDS IMPROVEMENT'}")
        report.append("")
        report.append("COMPREHENSIVE METRICS:")
        report.append(f"  Total Return:          {self.total_return:>10.2%}")
        report.append(f"  Annualized Return:     {self.annualized_return:>10.2%}")
        report.append(f"  Volatility (Annual):   {self.volatility:>10.2%}")
        report.append(f"  Sharpe Ratio:          {self.sharpe_ratio:>10.3f}")
        report.append(f"  Sortino Ratio:         {self.sortino_ratio:>10.3f}")
        report.append(f"  Calmar Ratio:          {self.calmar_ratio:>10.3f}")
        report.append(f"  Max Drawdown:          {self.max_drawdown:>10.2%}")
        report.append("")
        report.append("TRADING STATISTICS:")
        report.append(f"  Total Trades:          {self.total_trades:>10d}")
        report.append(f"  Winning Trades:        {self.winning_trades:>10d}")
        report.append(f"  Losing Trades:         {self.losing_trades:>10d}")
        report.append(f"  Win Rate:              {self.win_rate:>10.1%}")
        report.append(f"  Average Win:           ${self.avg_win:>9.2f}")
        report.append(f"  Average Loss:          ${self.avg_loss:>9.2f}")
        report.append(f"  Profit Factor:         {self.profit_factor:>10.3f}")
        report.append("")
        report.append("PORTFOLIO PERFORMANCE:")
        report.append(f"  Final Balance:         ${self.final_balance:>9.2f}")
        report.append(f"  Total PnL:             ${self.total_pnl:>9.2f}")
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)


def evaluate_model(model_path: str, data_path: str, step_minutes: int = 1440) -> EvaluationMetrics:
    """
    Evaluate trained PPO model on full dataset

    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to evaluation data
        step_minutes: Data frequency in minutes (1440 = daily)
    """
    logger.info(f"Loading model from {model_path}")

    # Load data
    logger.info(f"Loading data from {data_path}")
    data_result = load_prices_or_fail(data_path, step_minutes=step_minutes)

    # Handle tuple return (df, metrics)
    if isinstance(data_result, tuple):
        data_df, data_metrics = data_result
    else:
        data_df = data_result
        data_metrics = None

    logger.info(f"Data loaded: {data_df.shape[0]} timesteps, {data_df.shape[1]} features")

    # Create environment (use SlidingWindowTradingEnv to match BC training setup)
    env = SlidingWindowTradingEnv(
        data_path=data_path,
        episode_length=200,
        step_minutes=step_minutes
    )
    logger.info(f"Environment created: {len(env.prices)} timesteps")

    # Load trained model
    obs_dim = env.observation_space.shape[0]
    ppo_config = PPOConfig()

    agent = PPOAgent(obs_dim=obs_dim, config=ppo_config)
    agent.load(model_path)  # Use agent's load method which handles checkpoint format

    # Move to CPU for evaluation
    agent.device = torch.device('cpu')
    agent.network = agent.network.to('cpu')
    agent.network.eval()
    logger.info(f"Model loaded: {obs_dim} features, {sum(p.numel() for p in agent.network.parameters())} parameters")

    # Run evaluation
    logger.info("Running evaluation...")
    obs, info = env.reset()

    episode_returns = []
    episode_balances = []
    all_trades = []
    equity_curve = []

    done = False
    episode_count = 0
    step_count = 0

    current_position = 0.0
    trade_entry_price = None

    while not done:
        # Get action from model (deterministic)
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track equity
        equity = info.get('equity', 100000.0)  # Default initial balance
        equity_curve.append(equity)

        # Detect trades (position changes)
        new_position = info.get('position', 0.0)
        if abs(new_position - current_position) > 0.01:  # Threshold for position change
            # Position changed - close old trade if exists
            if trade_entry_price is not None:
                current_price = info.get('price', 0)
                pnl = (current_price - trade_entry_price) * current_position
                all_trades.append(pnl)

            # Open new trade if position non-zero
            if abs(new_position) > 0.01:
                trade_entry_price = info.get('price', 0)
            else:
                trade_entry_price = None

            current_position = new_position

        # Track episode completion
        if terminated:
            episode_return = info.get('total_return', 0.0)
            episode_returns.append(episode_return)
            episode_balances.append(equity)
            episode_count += 1

        obs = next_obs
        step_count += 1

        if step_count % 100 == 0:
            logger.info(f"Progress: {step_count} steps, {episode_count} episodes")

    logger.info(f"Evaluation complete: {step_count} steps, {episode_count} episodes, {len(all_trades)} trades")

    # Calculate metrics
    return calculate_metrics(
        equity_curve=equity_curve,
        all_trades=all_trades,
        initial_balance=100000.0,
        periods_per_year=8760 if step_minutes == 60 else 365  # Hourly or daily
    )


def calculate_metrics(equity_curve: list, all_trades: list, initial_balance: float,
                     periods_per_year: int = 365) -> EvaluationMetrics:
    """Calculate comprehensive performance metrics"""

    if len(equity_curve) == 0:
        logger.error("No data to calculate metrics")
        return EvaluationMetrics(**{k: 0.0 for k in EvaluationMetrics.__annotations__ if k != 'meets_production_criteria'})

    equity_curve = np.array(equity_curve)

    # Basic returns
    final_balance = equity_curve[-1]
    total_pnl = final_balance - initial_balance
    total_return = total_pnl / initial_balance

    # Annualized return
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / max(years, 1/252)) - 1 if years > 0 else 0

    # Calculate period returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]  # Remove inf/nan

    # Volatility (annualized)
    if len(returns) > 1:
        volatility = np.std(returns) * np.sqrt(periods_per_year)
    else:
        volatility = 0.0

    # Sharpe Ratio
    if volatility > 0 and len(returns) > 0:
        sharpe_ratio = (annualized_return - 0.0) / volatility  # Assuming 0% risk-free rate
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (downside deviation only)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 1:
        downside_std = np.std(negative_returns) * np.sqrt(periods_per_year)
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0.0
    else:
        sortino_ratio = 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    # Trade statistics
    if len(all_trades) > 0:
        all_trades = np.array(all_trades)
        winning_trades = all_trades[all_trades > 0]
        losing_trades = all_trades[all_trades < 0]

        total_trades = len(all_trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = num_wins / total_trades if total_trades > 0 else 0.0

        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0

        # Profit factor
        total_wins = np.sum(winning_trades)
        total_losses = abs(np.sum(losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    else:
        total_trades = 0
        num_wins = 0
        num_losses = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

    return EvaluationMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        winning_trades=num_wins,
        losing_trades=num_losses,
        avg_win=avg_win,
        avg_loss=avg_loss,
        final_balance=final_balance,
        total_pnl=total_pnl
    )


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained PPO trading model')
    parser.add_argument('--model', type=str,
                       default='checkpoints_daily_enriched/final_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data', type=str,
                       default='data/4h_processed/btc_daily_REAL_FIXED.csv',
                       help='Path to evaluation data')
    parser.add_argument('--step-minutes', type=int, default=1440,
                       help='Data frequency in minutes (1440=daily)')
    parser.add_argument('--output', type=str, default='evaluation_report.txt',
                       help='Output file for report')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PPO TRADING MODEL EVALUATION")
    logger.info("=" * 80)

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        step_minutes=args.step_minutes
    )

    # Generate report
    report = metrics.format_report()
    print("\n" + report)

    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {args.output}")

    # Exit code based on production readiness
    if metrics.meets_production_criteria():
        logger.info("✓ MODEL APPROVED FOR PRODUCTION DEPLOYMENT")
        sys.exit(0)
    else:
        logger.warning("✗ MODEL NOT READY FOR PRODUCTION")
        logger.warning(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f} (need ≥1.5)")
        logger.warning(f"  Win Rate: {metrics.win_rate:.1%} (need ≥65%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
