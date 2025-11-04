#!/usr/bin/env python3
"""
True 4H Data Builder - Path A Implementation
Builds real 4-hour bars with proper feature computation
Based on friend's specifications for semantic correctness
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import requests
import json
from datetime import datetime, timezone
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=1000, start_time=None):
    """Fetch hourly data from Binance"""
    base_url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                  'taker_buy_quote_volume', 'ignore']

        df = pd.DataFrame(data, columns=columns)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Convert price columns to float
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()

        return df

    except Exception as e:
        logger.error(f"Failed to fetch Binance data: {e}")
        return None


def fetch_fear_greed_index():
    """Fetch Fear & Greed Index data"""
    try:
        # Alternative Fear & Greed API
        url = "https://api.alternative.me/fng/?limit=2000&format=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'data' in data:
            fg_data = []
            for item in data['data']:
                fg_data.append({
                    'timestamp': pd.to_datetime(item['timestamp'], unit='s', utc=True),
                    'fear_greed_value': float(item['value']),
                    'fear_greed_classification': item['value_classification']
                })

            fg_df = pd.DataFrame(fg_data)
            return fg_df

    except Exception as e:
        logger.warning(f"Could not fetch Fear & Greed data: {e}")

    return None


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Convert hourly data to true 4-hour bars"""
    logger.info("📊 Resampling hourly data to 4H bars...")

    # Set timestamp as index for resampling
    df_1h = df_1h.set_index('timestamp').sort_index()

    # Resample to 4H bars starting at midnight UTC (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum'
    }

    # Use 4H frequency with anchor at midnight
    df_4h = df_1h.resample('4H', label='left', closed='left').agg(agg_dict)

    # Remove any incomplete bars
    df_4h = df_4h.dropna(subset=['open', 'high', 'low', 'close'])

    # Reset index to get timestamp as column
    df_4h = df_4h.reset_index()

    logger.info(f"✅ Created {len(df_4h)} 4H bars from {len(df_1h)} hourly bars")

    return df_4h


def compute_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features with proper 4H semantics"""
    logger.info("🔧 Computing 4H features...")

    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Price returns (in 4H bar units)
    df['close_log'] = np.log(df['close'].clip(1e-9, None))

    for bars in [1, 2, 3, 6, 12, 24]:  # 4H, 8H, 12H, 1D, 2D, 4D
        df[f'return_{bars}bars'] = df['close_log'] - df['close_log'].shift(bars)

    # Volatility features (in 4H bar units)
    df['volatility_6bars'] = df['return_1bars'].rolling(6).std()    # 24H volatility
    df['volatility_12bars'] = df['return_1bars'].rolling(12).std()  # 48H volatility
    df['volatility_42bars'] = df['return_1bars'].rolling(42).std()  # 1 week volatility

    # Volatility ranks
    for window in [6, 12, 42]:
        vol_col = f'volatility_{window}bars'
        df[f'volatility_rank_{window}bars'] = df[vol_col].rolling(100).rank(pct=True)

    # Moving averages (in 4H bar units)
    df['price_ma_12bars'] = df['close'].rolling(12).mean()   # 48H MA
    df['price_ma_50bars'] = df['close'].rolling(50).mean()   # ~8 day MA
    df['price_ma_200bars'] = df['close'].rolling(200).mean() # ~33 day MA

    # Trend alignment
    df['trend_alignment'] = np.where(
        (df['close'] > df['price_ma_12bars']) &
        (df['price_ma_12bars'] > df['price_ma_50bars']) &
        (df['price_ma_50bars'] > df['price_ma_200bars']), 1,
        np.where(
            (df['close'] < df['price_ma_12bars']) &
            (df['price_ma_12bars'] < df['price_ma_50bars']) &
            (df['price_ma_50bars'] < df['price_ma_200bars']), -1, 0
        )
    )

    # Volume features
    df['volume_ma_6bars'] = df['volume'].rolling(6).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_6bars'] + 1e-8)

    # Volatility regime
    vol_median = df['volatility_12bars'].rolling(100).median()
    df['volatility_regime'] = np.where(df['volatility_12bars'] > vol_median, 1, 0)

    logger.info(f"✅ Computed 4H features: {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']])} features")

    return df


def compute_4h_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute labels with proper 4H semantics"""
    logger.info("🎯 Computing 4H labels...")

    df = df.copy()

    # Forward returns for different horizons (in 4H bar units)
    horizons = [1, 6, 12, 18]  # 4H, 1D, 2D, 3D
    thresholds_bp = [0, 1, 2, 3, 5]  # basis points

    for horizon in horizons:
        # Forward return
        fwd_return = df['close_log'].shift(-horizon) - df['close_log']

        for bp in thresholds_bp:
            threshold = bp / 10000.0  # Convert bp to decimal

            # Ternary labels: 1 (up), 0 (neutral), -1 (down)
            label = np.where(
                fwd_return > threshold, 1,
                np.where(fwd_return < -threshold, -1, 0)
            )

            df[f'label_{horizon}bars_{bp}bp'] = label

    # Volume labels
    df['vol_label_6bars'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    df['vol_label_12bars'] = np.where(df['volume'].rolling(12).mean() > df['volume'].rolling(50).mean(), 1, 0)

    logger.info(f"✅ Computed 4H labels for {len(horizons)} horizons × {len(thresholds_bp)} thresholds")

    return df


def merge_fear_greed(df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Fear & Greed data with proper alignment"""
    if fg_df is None:
        logger.warning("⚠️ No Fear & Greed data available, filling with neutral values")
        df['fear_greed_value'] = 50.0
        df['fear_greed_classification'] = 'Neutral'
        return df

    logger.info("🧠 Merging Fear & Greed data...")

    # Convert to daily for F&G alignment (F&G is daily data)
    fg_df['date'] = fg_df['timestamp'].dt.date
    df['date'] = df['timestamp'].dt.date

    # Merge on date
    df = df.merge(
        fg_df[['date', 'fear_greed_value', 'fear_greed_classification']],
        on='date',
        how='left'
    )

    # Forward fill missing values
    df['fear_greed_value'] = df['fear_greed_value'].ffill().fillna(50.0)
    df['fear_greed_classification'] = df['fear_greed_classification'].ffill().fillna('Neutral')

    # Fear & Greed features
    df['fear_greed_ma_7d'] = df['fear_greed_value'].rolling(42).mean()  # 7 days * 6 bars/day
    df['fear_greed_momentum'] = df['fear_greed_value'] - df['fear_greed_ma_7d']
    df['fear_greed_extreme'] = np.where(
        (df['fear_greed_value'] <= 20) | (df['fear_greed_value'] >= 80), 1, 0
    )

    # Clean up
    df = df.drop('date', axis=1)

    logger.info("✅ Fear & Greed data merged")

    return df


def build_4h_dataset(start_date="2020-01-01", end_date="2025-08-19", output_path="btc_4h_complete.csv"):
    """Build complete 4H dataset with proper semantics"""
    logger.info("🚀 Building true 4H dataset...")

    # Step 1: Fetch hourly data from Binance
    logger.info("📡 Fetching hourly Bitcoin data...")

    # Use REAL Bitcoin data instead of synthetic
    print("🔄 Loading REAL Bitcoin data from vertex_processed...")
    real_data_path = "data/vertex_processed/btc_v2_full_20250819_094411.csv"

    try:
        df_real = pd.read_csv(real_data_path, low_memory=False)
        print(f"✅ Loaded REAL data: {len(df_real)} records")

        # Convert timestamp and sort
        df_real['timestamp'] = pd.to_datetime(df_real['timestamp'], utc=True)
        df_real = df_real.sort_values('timestamp').reset_index(drop=True)

        # Filter by date range if specified
        if start_date and end_date:
            start_ts = pd.to_datetime(start_date, utc=True)
            end_ts = pd.to_datetime(end_date, utc=True)
            df_real = df_real[(df_real['timestamp'] >= start_ts) & (df_real['timestamp'] <= end_ts)]
            print(f"✅ Filtered to date range: {len(df_real)} records")

    except Exception as e:
        raise ValueError(f"Failed to load real Bitcoin data: {e}")

    # Use real OHLCV data directly
    hourly_data = []
    for _, row in df_real.iterrows():
        # Ensure we have required columns with fallbacks
        volume = row.get('volume_btc', row.get('volume', 1000))

        hourly_data.append({
            'timestamp': row['timestamp'],
            'open': row.get('open', row['close']),
            'high': row.get('high', row['close']),
            'low': row.get('low', row['close']),
            'close': row['close'],
            'volume': volume,
            'quote_volume': volume * row['close']
        })

    df_1h = pd.DataFrame(hourly_data)
    logger.info(f"📊 Generated {len(df_1h)} hourly bars")

    # Step 2: Resample to 4H
    df_4h = resample_to_4h(df_1h)

    # Step 3: Fetch Fear & Greed data
    fg_df = fetch_fear_greed_index()

    # Step 4: Compute features
    df_4h = compute_4h_features(df_4h)

    # Step 5: Merge Fear & Greed
    df_4h = merge_fear_greed(df_4h, fg_df)

    # Step 6: Compute labels
    df_4h = compute_4h_labels(df_4h)

    # Step 7: Clean and validate
    logger.info("🧹 Cleaning and validating dataset...")

    # Remove rows with excessive NaN
    initial_len = len(df_4h)
    df_4h = df_4h.dropna(subset=['close', 'volume', 'label_6bars_1bp'])  # Keep core requirements
    logger.info(f"📊 Removed {initial_len - len(df_4h)} rows with excessive NaN")

    # Ensure proper data types
    categorical_cols = ['fear_greed_classification']
    for col in categorical_cols:
        if col in df_4h.columns:
            df_4h[col] = df_4h[col].astype('category')

    # Step 8: Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_4h.to_csv(output_path, index=False)

    logger.info(f"✅ 4H dataset saved: {output_path}")
    logger.info(f"📊 Final dataset: {len(df_4h)} rows × {len(df_4h.columns)} columns")
    logger.info(f"📅 Date range: {df_4h['timestamp'].min()} to {df_4h['timestamp'].max()}")

    return df_4h


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build true 4H Bitcoin dataset')
    parser.add_argument('--start_date', default='2020-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', default='2025-08-19', help='End date YYYY-MM-DD')
    parser.add_argument('--output', default='data/4h_processed/btc_4h_complete.csv', help='Output file path')

    args = parser.parse_args()

    df = build_4h_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output
    )

    print(f"🎉 True 4H dataset ready: {args.output}")
    print(f"📊 Shape: {df.shape}")
    print(f"🔧 Features: {len([c for c in df.columns if c.startswith(('return_', 'volatility_', 'label_'))])} computed")