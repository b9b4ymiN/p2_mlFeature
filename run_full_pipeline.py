"""
Complete End-to-End Pipeline
Phase 1 (Data Collection) → Phase 2 (Feature Engineering) → Phase 3 (ML Training)
"""

from data_integration.phase1_connector import Phase1DataConnector
from features import FeatureEngineer, TargetEngineer
from utils import time_series_split, select_features_combined
from models.training_pipeline import MLTrainingPipeline
from datetime import datetime, timedelta
import os


def run_complete_pipeline(
    db_config=None,
    symbol='BTCUSDT',
    days_back=60,
    target_horizon=48,  # 4 hours in 5-min bars
    n_features=50,
    use_mock_data=False
):
    """
    Complete end-to-end pipeline from Phase 1 to Phase 3

    Args:
        db_config: Database configuration dict
        symbol: Trading symbol
        days_back: Number of days of historical data
        target_horizon: Prediction horizon (in 5-min bars)
        n_features: Number of features to select
        use_mock_data: If True, use generated mock data instead of DB

    Returns:
        Trained pipeline and results
    """

    print("="*70)
    print("COMPLETE ML TRADING PIPELINE")
    print("Phase 1 (Data) → Phase 2 (Features) → Phase 3 (Models)")
    print("="*70)

    # ========== PHASE 1: FETCH DATA ==========
    print("\n[PHASE 1] Fetching data...")

    connector = Phase1DataConnector(**(db_config or {}))
    connector.connect()

    if use_mock_data or connector.conn is None:
        print("Using mock data generation...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

    data = connector.fetch_all_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    if connector.conn:
        connector.disconnect()

    # ========== PHASE 2: FEATURE ENGINEERING ==========
    print("\n[PHASE 2] Engineering features...")

    engineer = FeatureEngineer()
    features_df = engineer.engineer_all_features(
        ohlcv=data['ohlcv'],
        oi=data['oi'],
        funding=data['funding'],
        liquidations=data['liquidations'],
        ls_ratio=data['ls_ratio']
    )

    print(f"✓ Engineered {len(features_df.columns)} total features")

    # Create targets
    target_engineer = TargetEngineer()

    # Classification target (3-class: LONG, NEUTRAL, SHORT)
    df_with_target = target_engineer.create_classification_target(
        features_df.reset_index(),
        horizon=target_horizon,
        threshold=0.005,  # 0.5% move
        n_classes=3
    )

    # Add regression target
    df_with_target['target_return'] = df_with_target['future_return']

    # Get feature columns (exclude OHLCV and targets)
    feature_cols = engineer.get_feature_names(features_df)

    # Remove any NaN rows
    df_final = df_with_target.dropna()

    print(f"✓ Final dataset: {len(df_final)} samples, {len(feature_cols)} features")

    # ========== DATA SPLITTING ==========
    print("\n[PHASE 2] Splitting data (time-series aware)...")

    train_idx = int(len(df_final) * 0.6)
    val_idx = int(len(df_final) * 0.8)

    X_train = df_final[feature_cols].iloc[:train_idx]
    y_train_class = df_final['target'].iloc[:train_idx]
    y_train_reg = df_final['target_return'].iloc[:train_idx]

    X_val = df_final[feature_cols].iloc[train_idx:val_idx]
    y_val_class = df_final['target'].iloc[train_idx:val_idx]
    y_val_reg = df_final['target_return'].iloc[train_idx:val_idx]

    X_test = df_final[feature_cols].iloc[val_idx:]
    y_test_class = df_final['target'].iloc[val_idx:]
    y_test_reg = df_final['target_return'].iloc[val_idx:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ========== FEATURE SELECTION ==========
    print("\n[PHASE 2] Selecting best features...")

    X_train_selected, report = select_features_combined(
        X_train, y_train_class,
        n_features=n_features,
        correlation_threshold=0.9,
        variance_threshold=0.001,
        task_type='classification'
    )

    selected_features = X_train_selected.columns.tolist()
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    print(f"✓ Selected {len(selected_features)} features")

    # ========== PHASE 3: MODEL TRAINING ==========
    print("\n[PHASE 3] Training ML models...")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    pipeline = MLTrainingPipeline()
    results = pipeline.run_full_pipeline(
        X_train_selected, y_train_class, y_train_reg,
        X_val_selected, y_val_class, y_val_reg,
        X_test_selected, y_test_class, y_test_reg,
        skip_lstm=False,  # Set to True if torch not installed
        skip_catboost=False  # Set to True if catboost not installed
    )

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70)
    print(f"\nData: {len(df_final)} samples, {days_back} days")
    print(f"Features: {len(feature_cols)} → {len(selected_features)} (after selection)")
    print(f"Models: {len(pipeline.models)} trained")
    print(f"\nModels saved in ./models/")
    print("="*70 + "\n")

    return {
        'pipeline': pipeline,
        'features': selected_features,
        'results': results,
        'data': {
            'X_test': X_test_selected,
            'y_test_class': y_test_class,
            'y_test_reg': y_test_reg
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run complete ML trading pipeline')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data')
    parser.add_argument('--features', type=int, default=50, help='Number of features to select')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of database')
    parser.add_argument('--db-host', type=str, default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--db-name', type=str, default='trading_db', help='Database name')
    parser.add_argument('--db-user', type=str, default='postgres', help='Database user')
    parser.add_argument('--db-password', type=str, default='postgres', help='Database password')

    args = parser.parse_args()

    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }

    # Run pipeline
    output = run_complete_pipeline(
        db_config=db_config,
        symbol=args.symbol,
        days_back=args.days,
        n_features=args.features,
        use_mock_data=args.mock
    )

    print("Pipeline complete! You can now use the trained models for predictions.")
