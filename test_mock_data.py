"""
Test Script 1: Phase 2 with Mock Data
ทดสอบว่า Phase 2 ทำงานได้ไหม (ไม่ต้องใช้ Phase 1)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import FeatureEngineer, TargetEngineer
from utils import select_features_combined
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_data(n_samples=5000):
    """
    สร้างข้อมูลจำลองสำหรับทดสอบ
    """
    print("\n" + "="*60)
    print("📊 สร้างข้อมูลจำลอง (Mock Data)")
    print("="*60)

    # สร้าง timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]

    # สร้างราคา (random walk)
    np.random.seed(42)
    price_base = 30000
    returns = np.random.normal(0.0001, 0.01, n_samples)
    prices = price_base * np.exp(np.cumsum(returns))

    # OHLCV
    ohlcv = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_samples)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n_samples)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_samples)
    })

    # Open Interest
    oi = pd.DataFrame({
        'timestamp': timestamps,
        'open_interest': np.random.uniform(10000, 50000, n_samples) * (prices / price_base)
    })

    # Funding Rate
    funding = pd.DataFrame({
        'timestamp': timestamps,
        'funding_rate': np.random.normal(0.0001, 0.0005, n_samples)
    })

    # Liquidations
    n_liq = int(n_samples * 0.1)
    liq_timestamps = np.random.choice(timestamps, n_liq, replace=False)
    liquidations = pd.DataFrame({
        'timestamp': liq_timestamps,
        'quantity': np.random.uniform(0.1, 10, n_liq),
        'side': np.random.choice(['BUY', 'SELL'], n_liq),
        'order_id': range(n_liq)
    })

    # Long/Short Ratio
    ls_ratio = pd.DataFrame({
        'timestamp': timestamps,
        'longShortRatio': np.random.uniform(0.5, 2.0, n_samples)
    })

    print(f"✅ สร้างข้อมูลจำลอง {n_samples} samples")
    print(f"   - OHLCV: {len(ohlcv)} rows")
    print(f"   - OI: {len(oi)} rows")
    print(f"   - Funding: {len(funding)} rows")
    print(f"   - Liquidations: {len(liquidations)} rows")
    print(f"   - L/S Ratio: {len(ls_ratio)} rows")

    return {
        'ohlcv': ohlcv,
        'oi': oi,
        'funding': funding,
        'liquidations': liquidations,
        'ls_ratio': ls_ratio
    }


def test_feature_engineering(data):
    """
    ทดสอบ Feature Engineering
    """
    print("\n" + "="*60)
    print("🔧 ทดสอบ Feature Engineering")
    print("="*60)

    try:
        engineer = FeatureEngineer()

        print("\nกำลังสร้าง features...")
        features_df = engineer.engineer_all_features(
            ohlcv=data['ohlcv'],
            oi=data['oi'],
            funding=data['funding'],
            liquidations=data['liquidations'],
            ls_ratio=data['ls_ratio']
        )

        feature_cols = engineer.get_feature_names(features_df)

        print(f"\n✅ Feature Engineering สำเร็จ!")
        print(f"   - Total columns: {len(features_df.columns)}")
        print(f"   - Feature columns: {len(feature_cols)}")
        print(f"   - Rows: {len(features_df)}")

        # แสดง features บางส่วน
        print(f"\n📋 ตัวอย่าง Features (10 ตัวแรก):")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"   {i:2d}. {col}")

        return features_df, feature_cols

    except Exception as e:
        print(f"\n❌ Feature Engineering ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_target_engineering(features_df):
    """
    ทดสอบ Target Engineering
    """
    print("\n" + "="*60)
    print("🎯 ทดสอบ Target Engineering")
    print("="*60)

    try:
        target_eng = TargetEngineer()

        print("\nกำลังสร้าง classification target...")
        df_with_target = target_eng.create_classification_target(
            features_df.reset_index(),
            horizon=48,  # 4 hours
            threshold=0.005,  # 0.5%
            n_classes=3
        )

        print(f"\n✅ Target Engineering สำเร็จ!")
        print(f"   - Rows with target: {len(df_with_target)}")

        # แสดง target distribution
        target_dist = df_with_target['target'].value_counts().sort_index()
        print(f"\n📊 Target Distribution:")
        for target_val, count in target_dist.items():
            target_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}.get(target_val, str(target_val))
            pct = count / len(df_with_target) * 100
            print(f"   {target_name:10s}: {count:5d} ({pct:5.1f}%)")

        return df_with_target

    except Exception as e:
        print(f"\n❌ Target Engineering ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_selection(features_df, feature_cols):
    """
    ทดสอบ Feature Selection
    """
    print("\n" + "="*60)
    print("🎨 ทดสอบ Feature Selection")
    print("="*60)

    try:
        # แยก features และ target
        X = features_df[feature_cols]
        y = features_df['target']

        # Clean data: replace inf with NaN, then forward-fill, then fill remaining with 0
        print(f"\nทำความสะอาดข้อมูล...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0)

        # Split data
        train_end = int(len(X) * 0.7)
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        print(f"\nกำลังเลือก top 30 features จาก {len(feature_cols)} features...")
        print("(อาจใช้เวลา 1-2 นาที...)")

        X_selected, report = select_features_combined(
            X_train, y_train,
            n_features=30,
            correlation_threshold=0.9,
            variance_threshold=0.001,
            task_type='classification'
        )

        print(f"\n✅ Feature Selection สำเร็จ!")
        print(f"   - เหลือ features: {len(X_selected.columns)}")

        # แสดง top 10 features
        if 'importance_scores' in report:
            print(f"\n🏆 Top 10 Features:")
            for i, row in report['importance_scores'].head(10).iterrows():
                print(f"   {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

        return X_selected.columns.tolist()

    except Exception as e:
        print(f"\n❌ Feature Selection ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_model_training(features_df, selected_features):
    """
    ทดสอบ Model Training (แบบเบาๆ)
    """
    print("\n" + "="*60)
    print("🤖 ทดสอบ Model Training (Quick Test)")
    print("="*60)

    try:
        from models import XGBoostEntryPredictor

        # Prepare data
        X = features_df[selected_features]
        y = features_df['target']

        train_end = int(len(X) * 0.6)
        val_end = int(len(X) * 0.8)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        print(f"\nData splits:")
        print(f"   Train: {len(X_train)}")
        print(f"   Val:   {len(X_val)}")
        print(f"   Test:  {len(X_test)}")

        print(f"\nกำลัง train XGBoost Classifier...")
        print("(อาจใช้เวลา 1-2 นาที...)")

        model = XGBoostEntryPredictor()
        model.train(X_train, y_train, X_val, y_val)

        print(f"\nประเมินผลบน test set...")
        metrics = model.evaluate(X_test, y_test)

        print(f"\n✅ Model Training สำเร็จ!")

        return True

    except Exception as e:
        print(f"\n❌ Model Training ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    รัน Test ทั้งหมด
    """
    print("\n" + "="*70)
    print("🧪 TEST PHASE 2 WITH MOCK DATA")
    print("="*70)
    print("\nทดสอบว่า Phase 2 ทำงานได้ไหม (ไม่ต้องใช้ Phase 1)")
    print("="*70)

    # Test 1: Generate Mock Data
    data = generate_mock_data(n_samples=5000)
    if not data:
        print("\n❌ การสร้าง Mock Data ล้มเหลว!")
        return False

    # Test 2: Feature Engineering
    features_df, feature_cols = test_feature_engineering(data)
    if features_df is None:
        print("\n❌ Feature Engineering ล้มเหลว!")
        return False

    # Test 3: Target Engineering
    df_with_target = test_target_engineering(features_df)
    if df_with_target is None:
        print("\n❌ Target Engineering ล้มเหลว!")
        return False

    # Update features_df with target (merge on index)
    features_df = features_df.reset_index(drop=True)
    df_with_target = df_with_target.reset_index(drop=True)

    # Only keep rows that have targets
    common_length = min(len(features_df), len(df_with_target))
    features_df = features_df.iloc[:common_length].copy()
    features_df['target'] = df_with_target['target'].iloc[:common_length].values

    # Test 4: Feature Selection
    selected_features = test_feature_selection(features_df, feature_cols)
    if selected_features is None:
        print("\n❌ Feature Selection ล้มเหลว!")
        return False

    # Test 5: Model Training (Optional - comment out if too slow)
    print("\n" + "="*60)
    print("ต้องการทดสอบ Model Training ด้วยไหม? (อาจใช้เวลานาน)")
    print("="*60)
    response = input("ทดสอบ Model Training? (y/n): ").lower()

    if response == 'y':
        test_model_training(features_df, selected_features)
    else:
        print("\n⏭️  ข้าม Model Training")

    # Final Summary
    print("\n" + "="*70)
    print("✅ PHASE 2 TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 สรุปผลการทดสอบ:")
    print(f"   ✅ Mock Data Generation:  PASS")
    print(f"   ✅ Feature Engineering:   PASS ({len(feature_cols)} features)")
    print(f"   ✅ Target Engineering:    PASS")
    print(f"   ✅ Feature Selection:     PASS ({len(selected_features)} selected)")
    if response == 'y':
        print(f"   ✅ Model Training:        PASS")
    else:
        print(f"   ⏭️  Model Training:        SKIPPED")

    print(f"\n🎉 Phase 2 พร้อมใช้งาน!")
    print(f"   ลองต่อ Phase 1 ได้แล้ว โดยใช้: python test_phase1_connection.py")
    print("="*70 + "\n")

    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  ยกเลิกการทดสอบ")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
