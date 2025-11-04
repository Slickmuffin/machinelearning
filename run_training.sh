#!/bin/bash

# Quick start script for local PPO training with progress monitoring

set -e

echo "Starting PPO Training with Real-Time Progress Monitoring"
echo "=========================================================="

# Check if data file exists
DATA_FILE="${1:-/tmp/btc_data.csv}"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Usage: ./run_training.sh [path_to_data.csv]"
    exit 1
fi

echo "Data file: $DATA_FILE"

# Check if virtual environment should be created
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements_local.txt
fi

# Activate virtual environment
source venv/bin/activate

# Run test first
echo "Running system test..."
python3 test_progress_system.py

echo ""
echo "Test complete! Starting training..."
echo ""

# Run training with default parameters
# Adjust these as needed:
# --updates: Number of PPO updates (epochs)
# --steps: Maximum total environment steps
# --rollout-steps: Steps collected per update
# --save-interval: Save checkpoint every N updates

python3 train_local_with_progress.py \
    --data "$DATA_FILE" \
    --updates 1000 \
    --steps 1000000 \
    --rollout-steps 256 \
    --save-interval 100 \
    --checkpoint-dir ./checkpoints

echo ""
echo "Training complete! Checkpoints saved in ./checkpoints/"
