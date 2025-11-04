#!/usr/bin/env python3
"""
Quick test script to verify the progress monitoring system works
"""

import sys
import os
sys.path.insert(0, 'ppo_trading_ai')

print("Testing progress monitoring system...")
print("=" * 60)

# Test 1: Import checks
print("\n1. Testing imports...")
try:
    import torch
    print(f"   PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"   ERROR: PyTorch not installed: {e}")
    sys.exit(1)

try:
    import tqdm
    print(f"   tqdm: {tqdm.__version__}")
except ImportError as e:
    print(f"   ERROR: tqdm not installed: {e}")
    sys.exit(1)

try:
    import colorama
    print(f"   colorama: {colorama.__version__}")
except ImportError as e:
    print(f"   ERROR: colorama not installed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"   NumPy: {np.__version__}")
except ImportError:
    print("   ERROR: NumPy not installed")
    sys.exit(1)

try:
    import pandas as pd
    print(f"   Pandas: {pd.__version__}")
except ImportError:
    print("   ERROR: Pandas not installed")
    sys.exit(1)

try:
    import gymnasium as gym
    print(f"   Gymnasium: {gym.__version__}")
except ImportError:
    print("   ERROR: Gymnasium not installed")
    sys.exit(1)

# Test 2: GPU check
print("\n2. Checking GPU availability...")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU: {device_name}")
    print(f"   Memory: {memory_total:.1f}GB")
else:
    print("   No GPU detected - will use CPU (slower)")

# Test 3: Data file check
print("\n3. Checking for data files...")
data_files = [
    "data/vertex_processed/btc_v2_binance_1h_20250819_093151.csv",
    "data/processed/btc_combined_training_data.csv",
    "/tmp/btc_data.csv"
]

found_data = False
for data_file in data_files:
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / 1e6
        print(f"   Found: {data_file} ({size_mb:.1f}MB)")
        found_data = True

if not found_data:
    print("   WARNING: No data files found!")
    print("   Available data files:")
    os.system("find data -name '*.csv' -type f 2>/dev/null | head -5")

# Test 4: Module imports
print("\n4. Testing PPO module imports...")
try:
    from ppo_trading_ai.trading_env_senior_dev import PPOTradingEnvironmentSeniorDev, TradingConfig
    print("   Environment module: OK")
except ImportError as e:
    print(f"   ERROR: Cannot import environment: {e}")
    sys.exit(1)

try:
    from ppo_trading_ai.ppo_agent import PPOAgent, PPOConfig
    print("   Agent module: OK")
except ImportError as e:
    print(f"   ERROR: Cannot import agent: {e}")
    sys.exit(1)

# Test 5: Progress bar demo
print("\n5. Testing progress bar system...")
from tqdm import tqdm
import time
from colorama import Fore, Style

print(f"{Fore.CYAN}   Running progress bar demo...{Style.RESET_ALL}")
for i in tqdm(range(10), desc="Demo Progress", unit="iter"):
    time.sleep(0.1)

print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
print("=" * 60)
print(f"\nYou're ready to train! Run:")
print(f"{Fore.YELLOW}python train_local_with_progress.py --data <your_data.csv>{Style.RESET_ALL}")
print(f"\nOr use the quick start script:")
print(f"{Fore.YELLOW}./run_training.sh data/vertex_processed/btc_v2_binance_1h_20250819_093151.csv{Style.RESET_ALL}")
