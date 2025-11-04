FROM gcr.io/deeplearning-platform-release/base-cpu.py310:latest

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    xgboost==1.7.6 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    google-cloud-storage==2.10.0 \
    joblib==1.3.2

# Copy files
COPY daily_swing_trader.py /app/
# FRIEND'S FIX: Use 1H data (not daily) for proper 4H resample
COPY data/vertex_processed/btc_v2_binance_1h_20250819_093151.csv /app/btc_data.csv

# FRIEND'S FIX: Default CMD must fail without proper args (prevents silent success)
CMD ["python3", "-c", "import sys; print('ERROR: container-args not provided'); sys.exit(2)"]
