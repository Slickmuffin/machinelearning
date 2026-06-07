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

All numbers below come from walk-forward validation on the same historical data with
the same splits. They are the project's documented findings.

| Strategy | Average return (7 walk-forward splits) |
|---|---|
| Trend following (ADX > 20) | +161.11% |
| Trend following (ADX > 25) | +147.72% |
| MA crossover (20 / 50) | +1.05% |
| PPO / SAC reinforcement learning agent | -36.75% |

The trend following rule beat the reinforcement learning agent by roughly 198
percentage points. The lesson the project records is that added model complexity did
not translate into performance once both approaches were measured fairly.

Walk-forward detail for trend following (ADX > 20):

| Period | Return |
|---|---|
| Mar to Apr 2024 | +194.52% |
| Jun to Jul 2024 | +168.48% |
| Sep to Oct 2024 | +135.34% |
| Dec 2024 to Jan 2025 | +146.09% |
| Mar to Apr 2025 | +218.15% |
| Jun to Jul 2025 | +80.57% |
| Sep to Oct 2025 | +96.61% |
| Average | +161.11% |

An initial paper-trading run of the trend following rule returned +0.78% over its
first 15 hours. That sample is far too short to draw any conclusion from and is
included only for transparency.

## The strategy that worked

Trend following on Bitcoin, expressed in three rules:

1. Compute the ADX (Average Directional Index), a 0 to 100 measure of trend strength.
2. If ADX is above 20 (a strong trend), take a 2x position in the trend direction:
   long when price is above the 20 period SMA, short when it is below.
3. If ADX is below 20 (a weak or ranging market), stay flat.

The rule based logic lives in `daily_swing_trader.py` and
`ppo_trading_ai/daily_swing_policy.py`.

## Repository structure

```
ppo_trading_ai/        PPO reinforcement learning package
  ppo_agent.py           Actor-critic PPO implementation
  trading_env_senior_dev.py   Trading environment with realistic costs
  data_pipeline.py       Strict price-data loader with hard validation
  daily_swing_policy.py  Rule based swing strategy
  train_senior_dev.py    Training entry point
  evaluate_ppo.py        Evaluation
  leverage_sizing.py     Position sizing helpers
data/                  Public Bitcoin OHLCV and feature datasets
checkpoints*/          Trained model weights
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

No API keys are required to run the included backtests and training on the bundled
public data. Keys are only needed for live or exchange-connected use and are read
from environment variables, never hard-coded.

## Usage

- Train the PPO agent: `python ppo_trading_ai/train_senior_dev.py`
- Evaluate a trained model: `python evaluate_trained_model.py`
- Run the rule based swing strategy: `python daily_swing_trader.py`

Cloud training was run on Google Cloud Vertex AI; the build and job definitions are in
`cloudbuild.yaml` and `ppo_trading_ai/`.

## Data

The `data/` directory holds historical Bitcoin OHLCV candles and engineered features
(technical indicators, funding rate, open interest, long/short ratios, Fear and Greed
index) sourced from public exchange and market APIs. It contains no account, personal,
or private information and can be regenerated from public sources.

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
