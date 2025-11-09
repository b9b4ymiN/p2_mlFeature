"""
Example Usage: Complete ML Feature Engineering Pipeline

This script demonstrates how to use all components of the feature engineering system:
1. Load sample data
2. Engineer features
3. Create targets
4. Select best features
5. Split data
6. Store features
7. Analyze results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from features import FeatureEngineer, TargetEngineer, FeatureStore
from utils import (
    time_series_split,
    select_features_combined,
    analyze_feature_importance,
    generate_feature_report
)


def generate_sample_data(n_samples: int = 10000) -> dict:
    """
    Generate sample OHLCV, OI, and funding data for demonstration

    Args:
        n_samples: Number of 5-minute bars to generate

    Returns:
        Dictionary with DataFrames
    """
    print("Generating sample data...")

    # Generate timestamps (5-minute intervals)
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]

    # Generate realistic price data (random walk with trend)
    np.random.seed(42)
    price_base = 30000  # BTC starting price
    returns = np.random.normal(0.0001, 0.01, n_samples)
    prices = price_base * np.exp(np.cumsum(returns))

    # OHLCV data
    ohlcv = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_samples)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n_samples)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_samples)
    })

    # Open Interest data (correlated with price)
    oi = pd.DataFrame({
        'timestamp': timestamps,
        'open_interest': np.random.uniform(10000, 50000, n_samples) * (prices / price_base)
    })

    # Funding rate data (mean-reverting)
    funding_rate = np.random.normal(0.0001, 0.0005, n_samples)
    funding = pd.DataFrame({
        'timestamp': timestamps,
        'funding_rate': funding_rate
    })

    # Liquidation data (sparse events)
    n_liquidations = int(n_samples * 0.1)  # 10% of periods have liquidations
    liq_timestamps = np.random.choice(timestamps, n_liquidations, replace=False)
    liquidations = pd.DataFrame({
        'timestamp': liq_timestamps,
        'quantity': np.random.uniform(0.1, 10, n_liquidations),
        'side': np.random.choice(['BUY', 'SELL'], n_liquidations),
        'order_id': range(n_liquidations)
    })

    # Long/Short ratio
    ls_ratio = pd.DataFrame({
        'timestamp': timestamps,
        'longShortRatio': np.random.uniform(0.5, 2.0, n_samples)
    })

    print(f"✓ Generated {n_samples} samples of synthetic data")

    return {
        'ohlcv': ohlcv,
        'oi': oi,
        'funding': funding,
        'liquidations': liquidations,
        'ls_ratio': ls_ratio
    }


def main():
    """
    Main pipeline demonstration
    """
    print("\n" + "=" * 70)
    print("ML FEATURE ENGINEERING PIPELINE - EXAMPLE USAGE")
    print("=" * 70 + "\n")

    # ========== 1. GENERATE/LOAD DATA ==========
    print("\n[STEP 1] Loading Data...")
    data = generate_sample_data(n_samples=5000)

    # ========== 2. ENGINEER FEATURES ==========
    print("\n[STEP 2] Engineering Features...")
    engineer = FeatureEngineer()

    features_df = engineer.engineer_all_features(
        ohlcv=data['ohlcv'],
        oi=data['oi'],
        funding=data['funding'],
        liquidations=data['liquidations'],
        ls_ratio=data['ls_ratio']
    )

    print(f"\n✓ Engineered {len(features_df.columns)} features")
    print(f"  Shape: {features_df.shape}")

    # Get feature names (exclude base OHLCV columns)
    feature_cols = engineer.get_feature_names(features_df)
    print(f"  Feature columns: {len(feature_cols)}")

    # ========== 3. CREATE TARGET VARIABLES ==========
    print("\n[STEP 3] Creating Target Variables...")
    target_engineer = TargetEngineer()

    # Create classification target (predict 4-hour ahead moves)
    features_with_target = target_engineer.create_classification_target(
        features_df.reset_index(),
        horizon=48,  # 4 hours = 48 5-min bars
        threshold=0.01,  # 1% move
        n_classes=3  # LONG, NEUTRAL, SHORT
    )

    print(f"✓ Created classification target")
    print(f"  Target distribution:")
    print(target_engineer.get_target_distribution(features_with_target))

    # ========== 4. SPLIT DATA ==========
    print("\n[STEP 4] Splitting Data (Time-Series Aware)...")

    # Separate features and target
    X = features_with_target[feature_cols]
    y = features_with_target['target']

    # Time-series split (no shuffling!)
    train_indices = int(len(X) * 0.6)
    val_indices = int(len(X) * 0.8)

    X_train = X.iloc[:train_indices]
    y_train = y.iloc[:train_indices]

    X_val = X.iloc[train_indices:val_indices]
    y_val = y.iloc[train_indices:val_indices]

    X_test = X.iloc[val_indices:]
    y_test = y.iloc[val_indices:]

    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ========== 5. FEATURE SELECTION ==========
    print("\n[STEP 5] Selecting Best Features...")

    X_train_selected, selection_report = select_features_combined(
        X_train,
        y_train,
        n_features=50,
        task_type='classification',
        correlation_threshold=0.9,
        variance_threshold=0.001
    )

    # Apply same selection to val and test
    selected_features = X_train_selected.columns.tolist()
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"\n✓ Selected {len(selected_features)} features")

    # ========== 6. FEATURE ANALYSIS ==========
    print("\n[STEP 6] Analyzing Features...")

    importance_df = analyze_feature_importance(
        X_train_selected,
        y_train,
        task_type='classification',
        top_n=20
    )

    # ========== 7. SAVE TO FEATURE STORE ==========
    print("\n[STEP 7] Saving to Feature Store...")

    # Use mock store (in-memory) for demo
    feature_store = FeatureStore(use_mock=True)

    # Save feature metadata
    feature_store.save_feature_metadata(
        feature_names=selected_features,
        selected_features=selected_features
    )

    # Save batch of features
    features_to_save = features_with_target[selected_features].iloc[:100]
    features_to_save.index = features_with_target['timestamp'].iloc[:100]
    count = feature_store.save_batch_features('BTCUSDT', features_to_save)

    print(f"✓ Saved {count} feature rows to store")

    # Retrieve latest features
    latest = feature_store.get_latest_features('BTCUSDT')
    print(f"✓ Retrieved latest features: {len(latest)} features")

    # ========== 8. SUMMARY ==========
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total samples:              {len(features_df)}")
    print(f"Features engineered:        {len(feature_cols)}")
    print(f"Features selected:          {len(selected_features)}")
    print(f"Feature reduction:          {(1 - len(selected_features)/len(feature_cols))*100:.1f}%")
    print(f"\nData splits:")
    print(f"  Train:                    {len(X_train)} samples")
    print(f"  Validation:               {len(X_val)} samples")
    print(f"  Test:                     {len(X_test)} samples")
    print(f"\nTarget distribution (train):")
    for class_val in sorted(y_train.unique()):
        count = (y_train == class_val).sum()
        pct = count / len(y_train) * 100
        class_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}.get(class_val, str(class_val))
        print(f"  {class_name:10s}              {count:5d} ({pct:5.1f}%)")

    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    print("=" * 70)
    print("\n✓ Pipeline completed successfully!")
    print("\nNext steps:")
    print("  1. Train ML models (XGBoost, LightGBM, Neural Networks)")
    print("  2. Evaluate model performance on test set")
    print("  3. Deploy for live trading")
    print("=" * 70 + "\n")

    return {
        'X_train': X_train_selected,
        'y_train': y_train,
        'X_val': X_val_selected,
        'y_val': y_val,
        'X_test': X_test_selected,
        'y_test': y_test,
        'selected_features': selected_features,
        'importance': importance_df
    }


if __name__ == '__main__':
    results = main()
