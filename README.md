# Bitcoin Trading Research: Reinforcement Learning vs Simple Rules

A research codebase for Bitcoin trading strategies. It contains a PPO reinforcement
learning agent with a realistic cost and execution model, a set of rule based
baseline strategies, the data pipeline that feeds them, and the training and
evaluation tooling used to compare the two approaches on the same data.

The headline result of the project: on properly validated walk-forward tests, the
simple rule based strategies outperformed the trained reinforcement learning agent.
This repository keeps both so the comparison is reproducible and the failure modes
of the RL approach are documented rather than hidden.

## Key result

The reinforcement learning agent was evaluated on the first four walk-forward splits.
On those same four splits, every simple rule based strategy beat it. The averages below
are an apples-to-apples comparison on identical data and splits.

| Strategy | Average return (splits 1 to 4, same as the RL agent) |
|---|---|
| Trend following (ADX > 20) | +161.11% |
| Trend following (ADX > 25) | +147.72% |
| MA crossover (20 / 50) | +1.05% |
| PPO / SAC reinforcement learning agent | -36.75% |

Trend following (ADX > 20) beat the reinforcement learning agent by about 198 percentage
points on the same data. The lesson the project records is that added model complexity
did not translate into performance once both approaches were measured fairly.

### Walk-forward detail (all seven splits)

Per-split returns for trend following (ADX > 20). The RL agent was only run on splits 1
to 4, which is why the head-to-head average above uses those four splits.

| Split | Period | Trend following (ADX > 20) |
|---|---|---|
| 1 | Mar to Apr 2024 | +194.52% |
| 2 | Jun to Jul 2024 | +168.48% |
| 3 | Sep to Oct 2024 | +135.34% |
| 4 | Dec 2024 to Jan 2025 | +146.09% |
| 5 | Mar to Apr 2025 | +218.15% |
| 6 | Jun to Jul 2025 | +80.57% |
| 7 | Sep to Oct 2025 | +96.61% |
| | Average (all 7) | +148.54% |

The full numbers for every strategy and split are in `walkforward_comparison.csv`,
reproduced by `test_baselines_walkforward.py`.

An initial paper-trading run of the trend following bot returned +0.78% over its first
15 hours. That sample is far too short to draw any conclusion from and is included only
for transparency.

## The strategy that worked

Trend following on Bitcoin, expressed in three rules:

1. Compute the ADX (Average Directional Index), a 0 to 100 measure of trend strength.
2. If ADX is above 20 (a strong trend), take a 2x position in the trend direction:
   long when price is above the 20 period SMA, short when it is below.
3. If ADX is below 20 (a weak or ranging market), stay flat.

The first version live bot that implements this rule is `live_trend_following_bot.py`.
Related rule based logic is in `daily_swing_trader.py` and
`ppo_trading_ai/daily_swing_policy.py`.

## Repository structure

```
live_trend_following_bot.py   First version live trend-following bot (ADX + SMA)
test_baselines_walkforward.py Walk-forward comparison of rules vs the RL agent
walkforward_comparison.csv    Per-split results behind the key result above
bot_config.example.json       Template config for the live bot
ppo_trading_ai/        PPO reinforcement learning package
  ppo_agent.py           Actor-critic PPO implementation
  trading_env_senior_dev.py   Trading environment with realistic costs
  data_pipeline.py       Strict price-data loader with hard validation
  daily_swing_policy.py  Rule based swing strategy
  train_senior_dev.py    Training entry point
  evaluate_ppo.py        Evaluation
  leverage_sizing.py     Position sizing helpers
daily_swing_trader.py  Rule based trend / swing trader
ppo_simple.py          Minimal PPO reference
evaluate_trained_model.py   Model evaluation utility
config.yaml            Strategy and data configuration
.env.example           Environment variable template
requirements.txt       Python dependencies
```

## Realistic cost and execution model

A core part of the work was making the trading environment economically honest so
that backtests could not profit from unrealistic assumptions. The environment in
`ppo_trading_ai/trading_env_senior_dev.py` models:

- Taker fees and slippage in basis points (3 bps fee plus 0.5 bps slippage per
  execution by default), tracked in USD notional.
- Funding charged once per 8 hour window rather than every step.
- Log-equity rewards with mild turnover and leverage penalties, so the agent is not
  rewarded for churn.
- A no-trade band, a turnover cap, and sign-flip hysteresis to stop the agent paying
  costs for tiny or oscillating positions.
- Economic tripwires that fail fast if per-step costs or turnover exceed sane limits.

The data pipeline hard-fails the run if there are fewer than 1,000 usable price rows
or more than a 2% NaN rate, with no synthetic fallback.

## Setup

```bash
git clone https://github.com/Slickmuffin/machinelearning.git
cd machinelearning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in only if you connect a live data or trading account
```

No API keys are required to run the backtests or training once the public data is in
place (see Data and models for how to regenerate it). Keys are only needed for live or
exchange-connected use and are read from environment variables, never hard-coded.

## Usage

- Reproduce the baselines vs RL comparison: `python test_baselines_walkforward.py` (needs
  the 1 hour OHLCV data in `data/sac_training/`, see the Data section)
- Run the first version trend-following bot: `cp bot_config.example.json bot_config.json`,
  add your Binance keys, then `python live_trend_following_bot.py` (test on a paper or
  testnet account first)
- Train the PPO agent: `python ppo_trading_ai/train_senior_dev.py`
- Evaluate a trained model: `python evaluate_trained_model.py`
- Run the rule based swing strategy: `python daily_swing_trader.py`

Cloud training was run on Google Cloud Vertex AI; the build and job definitions are in
`cloudbuild.yaml` and `ppo_trading_ai/`.

## Data and models

Neither the datasets nor the trained model weights are committed to this repository, to keep
it light. Both regenerate from public Bitcoin market data.

The datasets are public Bitcoin market data (OHLCV candles plus engineered features such as
technical indicators, funding rate, open interest, long/short ratios, and the Fear and Greed
index). The `data/` directory is gitignored.

Trained model checkpoints (the `.pt` and `.pth` files) are gitignored as well. They
regenerate by running the training pipeline (`python ppo_trading_ai/train_senior_dev.py`) on
the data described below.

Regenerate locally from public sources:
- `build_4h_data.py` fetches Bitcoin klines from the public Binance API and builds the
  feature datasets.
- `ppo_trading_ai/data_pipeline.py` loads and strictly validates price data for training.
- The live bot pulls OHLCV directly from Binance through ccxt at runtime.

The walk-forward validation script `test_baselines_walkforward.py` expects 1 hour BTC/USDT
candles at `data/sac_training/btc_1h_2years.csv` (columns: timestamp, open, high, low,
close, volume) and the split definitions at `data/sac_training/walk_forward_splits.csv`.
Pull the 1 hour candles from Binance to recreate them. The per-split results those scripts
produced are preserved in `walkforward_comparison.csv`.

## Lessons learned

1. Validate on real data. A funding-rate strategy that looked profitable was traced to
   a threshold that never actually occurred in 2,000 real Binance funding records. Only
   results validated against real data were trusted.
2. Compare fairly. An early result that looked spectacular came from testing the two
   approaches on different periods. Once both were measured on identical data and
   splits, the picture changed completely (+161% vs -36%).
3. Simple can beat complex. A 2.5M parameter agent with 47 features and many hours of
   training lost to a three-rule strategy that takes minutes to evaluate.
4. Why the RL agent underperformed: a reward function that penalized trend following,
   policy collapse with no KL constraint, too many low-signal features, randomized
   episodes that broke temporal structure, and an over-broad continuous action space.

## Risk warning

This is research code, not financial advice. The strategies use leverage, which
amplifies losses as well as gains. Backtest and paper-trading results do not predict
future performance. If you ever connect real funds, start small, monitor closely, and
never risk money you cannot afford to lose.

## License

Released under the MIT License. See `LICENSE`.
