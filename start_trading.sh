#!/bin/bash
# Robust startup script for BTC Trading AI

# Set working directory
cd "$(dirname "$0")"

# Activate virtual environment
source training_venv/bin/activate

# Create logs directory
mkdir -p logs

# Function to start trading
start_trading() {
    echo "🚀 Starting BTC Trading AI Daemon..."
    echo "📅 $(date)"
    echo "📂 Working directory: $(pwd)"
    echo "🐍 Python: $(which python3)"
    echo "========================================"
    
    # Start with output logging
    python3 robust_trading_daemon.py 2>&1 | tee logs/trading_$(date +%Y%m%d_%H%M%S).log
}

# Function to check if trading is running
is_trading_running() {
    pgrep -f "robust_trading_daemon.py" > /dev/null
}

# Auto-restart loop
while true; do
    if ! is_trading_running; then
        echo "🔄 Trading daemon not running. Starting..."
        start_trading
        echo "⚠️  Trading daemon stopped. Waiting 30 seconds before restart..."
        sleep 30
    else
        echo "✅ Trading daemon is running. Checking again in 60 seconds..."
        sleep 60
    fi
done