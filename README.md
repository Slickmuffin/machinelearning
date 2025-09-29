# AI-Powered Trading System with Vertex AI Integration

Advanced cryptocurrency trading system using XGBoost with isotonic calibration, integrated with Google Cloud Vertex AI for scalable ML operations.

## Features

- **Model**: XGBoost multi-class classification with isotonic probability calibration
- **Data**: Binance USDT-M BTCUSDT OHLCV with extensible enrichers for derivatives/sentiment/whale data
- **Training**: Walk-forward validation with strict temporal splits
- **Execution**: Paper/live trading via ccxt with confidence-based position sizing
- **Risk**: Drawdown limits, abstention on weak signals, post-only orders
- **Vertex AI**: Custom training jobs, HPO with Vizier, Model Registry, Pipelines
- **Monitoring**: Model drift detection, Discord alerts, Prometheus metrics

## Architecture

```
├── src/
│   ├── data/           # Data loading and feature storage
│   ├── features/       # Technical indicators and feature engineering
│   ├── labels/         # Forward return labeling with meta-labeling
│   ├── model/          # XGBoost training, calibration, persistence
│   ├── train/          # Walk-forward training and HPO
│   ├── execution/      # Risk management and trade execution
│   └── service/        # FastAPI inference service
├── scripts/
│   ├── gcp/           # Vertex AI integration scripts
│   └── *.py           # Training, backtesting, data fetching
└── tests/             # Unit and integration tests
```

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp .env.example .env
   cp config.example.yaml config.yaml
   # Edit .env and config.yaml with your API keys and settings
   ```

3. **Fetch Data & Train**:
   ```bash
   python -m scripts.fetch_history
   python -m scripts.train_walkforward
   ```

4. **Run Service**:
   ```bash
   uvicorn src.service.main:app --host 0.0.0.0 --port 8010
   ```

## API Usage

```bash
# Get prediction probabilities
curl -X POST http://localhost:8010/predict

# Get trading signal
curl -X POST http://localhost:8010/signal

# Execute trade
curl -X POST http://localhost:8010/trade \
  -H "Content-Type: application/json" \
  -d '{"equity_usd": 1000, "allow_live": false}'
```

## Vertex AI Integration

Deploy to Google Cloud for production-scale ML operations:

```bash
# Submit training job
python -m scripts.gcp.vertex_train_job --project YOUR_PROJECT

# Deploy to endpoint
python -m scripts.gcp.vertex_deploy_endpoint --project YOUR_PROJECT

# Run full ML pipeline
python -m scripts.gcp.vertex_pipeline --project YOUR_PROJECT
```

## Safety Features

- **Paper Trading**: Default mode, no real money at risk
- **Confidence Thresholds**: Abstain on weak predictions
- **Position Limits**: Single position, confidence-based sizing
- **Drawdown Protection**: Automatic trading halt on losses
- **Kill Switches**: Anomaly detection and automatic shutdown