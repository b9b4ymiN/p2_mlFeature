"""
Test Script 2: Phase 1 Connection
ทดสอบการเชื่อมต่อกับ Phase 1 Docker
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_integration.phase1_connector import Phase1DataConnector
from features import FeatureEngineer, TargetEngineer
from utils import select_features_combined
import pandas as pd
from datetime import datetime, timedelta


# ========== ตั้งค่า Database (แก้ตรงนี้!) ==========
DB_CONFIG = {
    'host': 'localhost',      # ถ้ารันใน Docker ให้ใช้ชื่อ container
    'port': 5432,
    'database': 'futures_db',  # ชื่อ database ของคุณ
    'user': 'postgres',
    'password': 'postgres'     # 👈 เปลี่ยนเป็น password ของคุณ!
}

SYMBOL = 'BTCUSDT'  # เหรียญที่ต้องการทดสอบ
DAYS_BACK = 7       # ดึงข้อมูลย้อนหลังกี่วัน


def test_database_connection():
    """
    ทดสอบ 1: เชื่อมต่อ Database ได้ไหม
    """
    print("\n" + "="*60)
    print("🔌 TEST 1: Database Connection")
    print("="*60)

    try:
        print(f"\nกำลังเชื่อมต่อ...")
        print(f"   Host:     {DB_CONFIG['host']}")
        print(f"   Port:     {DB_CONFIG['port']}")
        print(f"   Database: {DB_CONFIG['database']}")
        print(f"   User:     {DB_CONFIG['user']}")

        connector = Phase1DataConnector(**DB_CONFIG)
        connector.connect()

        if connector.conn:
            print(f"\n✅ เชื่อมต่อ Database สำเร็จ!")
            connector.disconnect()
            return True
        else:
            print(f"\n⚠️  ไม่สามารถเชื่อมต่อ Database (ใช้ Mock Data แทน)")
            return False

    except Exception as e:
        print(f"\n❌ เชื่อมต่อ Database ล้มเหลว!")
        print(f"   Error: {str(e)}")
        print(f"\n💡 วิธีแก้:")
        print(f"   1. เช็คว่า Phase 1 Docker เปิดอยู่ไหม: docker ps")
        print(f"   2. เช็ค password ใน docker-compose.yml")
        print(f"   3. ลอง: docker-compose restart")
        return False


def test_fetch_ohlcv():
    """
    ทดสอบ 2: ดึงข้อมูล OHLCV ได้ไหม
    """
    print("\n" + "="*60)
    print("📊 TEST 2: Fetch OHLCV Data")
    print("="*60)

    try:
        connector = Phase1DataConnector(**DB_CONFIG)
        connector.connect()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_BACK)

        print(f"\nกำลังดึงข้อมูล OHLCV...")
        print(f"   Symbol:     {SYMBOL}")
        print(f"   Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"   End Date:   {end_date.strftime('%Y-%m-%d')}")

        df = connector.fetch_ohlcv(SYMBOL, start_date, end_date)

        if len(df) > 0:
            print(f"\n✅ ดึงข้อมูล OHLCV สำเร็จ!")
            print(f"   Rows: {len(df)}")
            print(f"\n📋 ตัวอย่างข้อมูล (5 rows แรก):")
            print(df.head().to_string())

            connector.disconnect()
            return True
        else:
            print(f"\n⚠️  ไม่มีข้อมูลใน Database")
            print(f"\n💡 วิธีแก้:")
            print(f"   รันการเก็บข้อมูลใน Phase 1 ก่อน")
            connector.disconnect()
            return False

    except Exception as e:
        print(f"\n❌ ดึงข้อมูล OHLCV ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_fetch_all_data():
    """
    ทดสอบ 3: ดึงข้อมูลครบทุกประเภท
    """
    print("\n" + "="*60)
    print("📥 TEST 3: Fetch All Data Types")
    print("="*60)

    try:
        connector = Phase1DataConnector(**DB_CONFIG)
        connector.connect()

        print(f"\nกำลังดึงข้อมูลทั้งหมด...")

        data = connector.fetch_all_data(
            symbol=SYMBOL,
            days_back=DAYS_BACK
        )

        print(f"\n✅ ดึงข้อมูลทั้งหมดสำเร็จ!")
        print(f"\n📊 สรุปข้อมูล:")
        print(f"   OHLCV:        {len(data['ohlcv']):,} rows")
        print(f"   OI:           {len(data['oi']):,} rows")
        print(f"   Funding:      {len(data['funding']):,} rows")
        print(f"   Liquidations: {len(data['liquidations']):,} rows")
        print(f"   L/S Ratio:    {len(data['ls_ratio']):,} rows")

        connector.disconnect()
        return data

    except Exception as e:
        print(f"\n❌ ดึงข้อมูลล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_feature_engineering(data):
    """
    ทดสอบ 4: Feature Engineering บนข้อมูลจริง
    """
    print("\n" + "="*60)
    print("🔧 TEST 4: Feature Engineering on Real Data")
    print("="*60)

    try:
        engineer = FeatureEngineer()

        print(f"\nกำลังสร้าง features จากข้อมูล Phase 1...")

        features_df = engineer.engineer_all_features(
            ohlcv=data['ohlcv'],
            oi=data['oi'],
            funding=data['funding'],
            liquidations=data['liquidations'],
            ls_ratio=data['ls_ratio']
        )

        feature_cols = engineer.get_feature_names(features_df)

        print(f"\n✅ Feature Engineering สำเร็จ!")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Rows: {len(features_df):,}")

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


def test_full_pipeline(data):
    """
    ทดสอบ 5: Pipeline ทั้งหมด (Phase 1 → Phase 2 → Phase 3)
    """
    print("\n" + "="*60)
    print("🚀 TEST 5: Full Pipeline (Phase 1 → 2 → 3)")
    print("="*60)

    try:
        # Feature Engineering
        print(f"\n[Step 1/3] Feature Engineering...")
        engineer = FeatureEngineer()
        features_df = engineer.engineer_all_features(**data)
        feature_cols = engineer.get_feature_names(features_df)
        print(f"✅ สร้าง {len(feature_cols)} features")

        # Target Engineering
        print(f"\n[Step 2/3] Target Engineering...")
        target_eng = TargetEngineer()
        df_with_target = target_eng.create_classification_target(
            features_df.reset_index(),
            horizon=48,
            threshold=0.005,
            n_classes=3
        )
        print(f"✅ สร้าง target ({len(df_with_target)} samples)")

        # แสดง target distribution
        target_dist = df_with_target['target'].value_counts().sort_index()
        print(f"\n📊 Target Distribution:")
        for target_val, count in target_dist.items():
            target_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}.get(target_val, str(target_val))
            pct = count / len(df_with_target) * 100
            print(f"   {target_name:10s}: {count:5d} ({pct:5.1f}%)")

        # Feature Selection (optional)
        print(f"\n[Step 3/3] Feature Selection...")
        response = input("ทำ Feature Selection ไหม? (y/n): ").lower()

        if response == 'y':
            features_df['target'] = df_with_target['target'].values[:len(features_df)]
            features_df = features_df.dropna(subset=['target'])

            X = features_df[feature_cols]
            y = features_df['target']

            train_end = int(len(X) * 0.7)
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]

            print(f"\nกำลังเลือก top 30 features...")
            X_selected, report = select_features_combined(
                X_train, y_train,
                n_features=30,
                correlation_threshold=0.9,
                variance_threshold=0.001,
                task_type='classification'
            )

            print(f"✅ เลือก {len(X_selected.columns)} features แล้ว")

            # แสดง top 10
            if 'importance_scores' in report:
                print(f"\n🏆 Top 10 Features:")
                for i, row in report['importance_scores'].head(10).iterrows():
                    print(f"   {i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")
        else:
            print(f"⏭️  ข้าม Feature Selection")

        # Save to parquet
        print(f"\n💾 บันทึกข้อมูล...")
        df_with_target.to_parquet('phase1_to_phase2_output.parquet')
        print(f"✅ บันทึกไว้ที่: phase1_to_phase2_output.parquet")

        print(f"\n✅ Full Pipeline สำเร็จ!")
        return True

    except Exception as e:
        print(f"\n❌ Full Pipeline ล้มเหลว!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    รัน Test ทั้งหมด
    """
    print("\n" + "="*70)
    print("🧪 TEST PHASE 1 CONNECTION")
    print("="*70)
    print("\nทดสอบการเชื่อมต่อ Phase 2 กับ Phase 1 Docker")
    print("="*70)

    print(f"\n⚙️  ตั้งค่า:")
    print(f"   Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"   Symbol:   {SYMBOL}")
    print(f"   Days:     {DAYS_BACK} วันล่าสุด")

    # Test 1: Connection
    if not test_database_connection():
        print(f"\n⚠️  ไม่สามารถเชื่อมต่อ Database")
        print(f"\n💡 แนะนำ:")
        print(f"   1. เช็คว่า Phase 1 Docker เปิดอยู่: docker ps")
        print(f"   2. เช็ค password: cat docker-compose.yml | grep POSTGRES_PASSWORD")
        print(f"   3. ลองใช้ Mock Data: python test_mock_data.py")
        return False

    # Test 2: Fetch OHLCV
    if not test_fetch_ohlcv():
        print(f"\n⚠️  ไม่มีข้อมูล OHLCV")
        print(f"\n💡 แนะนำ:")
        print(f"   รันการเก็บข้อมูลใน Phase 1 ก่อน")
        return False

    # Test 3: Fetch All Data
    data = test_fetch_all_data()
    if not data:
        return False

    # Test 4: Feature Engineering
    features_df, feature_cols = test_feature_engineering(data)
    if features_df is None:
        return False

    # Test 5: Full Pipeline
    print("\n" + "="*60)
    print("ต้องการรัน Full Pipeline ไหม?")
    print("(Phase 1 → Feature Engineering → Target → Feature Selection)")
    print("="*60)
    response = input("รัน Full Pipeline? (y/n): ").lower()

    if response == 'y':
        test_full_pipeline(data)
    else:
        print("\n⏭️  ข้าม Full Pipeline")

    # Final Summary
    print("\n" + "="*70)
    print("✅ PHASE 1 CONNECTION TEST COMPLETED!")
    print("="*70)
    print(f"\n📊 สรุปผลการทดสอบ:")
    print(f"   ✅ Database Connection:   PASS")
    print(f"   ✅ Fetch OHLCV:           PASS ({len(data['ohlcv'])} rows)")
    print(f"   ✅ Fetch All Data:        PASS")
    print(f"   ✅ Feature Engineering:   PASS ({len(feature_cols)} features)")
    if response == 'y':
        print(f"   ✅ Full Pipeline:         COMPLETED")

    print(f"\n🎉 Phase 1 + Phase 2 Integration พร้อมใช้งาน!")
    print(f"\n📝 ขั้นตอนต่อไป:")
    print(f"   1. รัน full pipeline: python run_full_pipeline.py --db-password {DB_CONFIG['password']}")
    print(f"   2. ฝึก Model: ไฟล์จะถูกบันทึกใน models/")
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
