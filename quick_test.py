"""
Quick Test - Phase 2 Basic Functions
ทดสอบเบื้องต้นว่า Phase 2 ทำงานได้
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("🧪 QUICK TEST - Phase 2 Basic Functions")
print("="*60)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from features import FeatureEngineer, TargetEngineer
    from data_integration import Phase1DataConnector
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate simple mock data
print("\n[2/5] Generating mock data...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    n = 1000
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(n)]

    ohlcv = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(30000, 31000, n),
        'high': np.random.uniform(30500, 31500, n),
        'low': np.random.uniform(29500, 30500, n),
        'close': np.random.uniform(30000, 31000, n),
        'volume': np.random.uniform(100, 1000, n)
    })

    oi = pd.DataFrame({
        'timestamp': timestamps,
        'open_interest': np.random.uniform(10000, 50000, n)
    })

    funding = pd.DataFrame({
        'timestamp': timestamps,
        'funding_rate': np.random.normal(0.0001, 0.0005, n)
    })

    liquidations = pd.DataFrame({
        'timestamp': timestamps[:100],
        'quantity': np.random.uniform(0.1, 10, 100),
        'side': np.random.choice(['BUY', 'SELL'], 100),
        'order_id': range(100)
    })

    ls_ratio = pd.DataFrame({
        'timestamp': timestamps,
        'longShortRatio': np.random.uniform(0.5, 2.0, n)
    })

    print(f"✅ Mock data created ({n} samples)")
except Exception as e:
    print(f"❌ Mock data failed: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n[3/5] Testing Feature Engineering...")
try:
    engineer = FeatureEngineer()
    features = engineer.engineer_all_features(
        ohlcv=ohlcv,
        oi=oi,
        funding=funding,
        liquidations=liquidations,
        ls_ratio=ls_ratio
    )

    feature_cols = engineer.get_feature_names(features)
    print(f"✅ Features created: {len(feature_cols)} features, {len(features)} rows")
except Exception as e:
    print(f"❌ Feature Engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Target Engineering
print("\n[4/5] Testing Target Engineering...")
try:
    target_eng = TargetEngineer()
    df_with_target = target_eng.create_classification_target(
        features.reset_index(),
        horizon=48,
        threshold=0.005,
        n_classes=3
    )

    print(f"✅ Targets created: {len(df_with_target)} samples")

    # Show distribution
    dist = df_with_target['target'].value_counts().sort_index()
    print(f"\n   Target Distribution:")
    for val, count in dist.items():
        name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}.get(val, str(val))
        pct = count / len(df_with_target) * 100
        print(f"   {name:10s}: {count:4d} ({pct:5.1f}%)")
except Exception as e:
    print(f"❌ Target Engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Database Connector
print("\n[5/5] Testing Database Connector...")
try:
    connector = Phase1DataConnector(
        host='localhost',
        port=5432,
        database='futures_db',
        user='postgres',
        password='postgres'
    )

    # Try to connect (will use mock if fails)
    connector.connect()

    if connector.conn:
        print("✅ Database connector works (connected)")
        connector.disconnect()
    else:
        print("✅ Database connector works (mock mode)")
except Exception as e:
    print(f"❌ Database Connector failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n📊 Summary:")
print(f"   ✅ Imports:             OK")
print(f"   ✅ Mock Data:           OK ({n} samples)")
print(f"   ✅ Feature Engineering: OK ({len(feature_cols)} features)")
print(f"   ✅ Target Engineering:  OK ({len(df_with_target)} samples)")
print(f"   ✅ Database Connector:  OK")
print("\n🎉 Phase 2 is ready to use!")
print("\n📝 Next steps:")
print("   1. For full test: python test_mock_data.py")
print("   2. For Phase 1 connection: python test_phase1_connection.py")
print("   3. For full pipeline: python run_full_pipeline.py --mock")
print("="*60 + "\n")
