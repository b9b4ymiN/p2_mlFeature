# 🔍 Gap Analysis: Phase 2/3 vs Best Practices

**วันที่:** 2025-11-10
**อ้างอิง:** "The Small Gaps to Close Before You Start Model Training"

---

## 📊 สรุปผลการตรวจสอบ

| ข้อ | หัวข้อ | สถานะ | ความสำคัญ | การดำเนินการ |
|-----|--------|-------|-----------|-------------|
| 1 | Data Contracts & Time Alignment | ❌ ไม่มี | 🔴 CRITICAL | ต้องสร้าง |
| 2 | Leakage Guards | ✅ ดี | 🔴 CRITICAL | ผ่าน |
| 3 | Feature Stability & Drift Monitoring | ❌ ไม่มี | 🟡 NICE-TO-HAVE | ข้ามได้ |
| 4 | Adaptive Extremes | ⚠️ ครึ่งหนึ่ง | 🟡 NICE-TO-HAVE | ข้ามได้ |
| 5 | Feature List & Versioning | ❌ ไม่มี | 🔴 CRITICAL | ต้องสร้าง |
| 6 | Scaling Done Right | ❌ ไม่มี | 🔴 CRITICAL | ต้องสร้าง |
| 7 | Walk-Forward CV | ✅ มีแล้ว | 🟢 COMPLETE | ผ่าน |
| 8 | Class Imbalance Handling | ⚠️ พื้นฐาน | 🟢 OK | พอใช้ได้ |
| 9 | Reproducible Artifacts | ❌ ไม่มี | 🔴 CRITICAL | ต้องสร้าง |
| 10 | CI, Tests, Docs | ⚠️ บางส่วน | 🟡 NICE-TO-HAVE | ข้ามได้ |

---

## 📋 รายละเอียดแต่ละข้อ

### ✅ 1. Data Contracts & Time Alignment
**สถานะ:** ❌ **ไม่มี**
**ความสำคัญ:** 🔴 **CRITICAL**

**ปัญหาที่พบ:**
- ❌ ไม่มี `schemas.py` สำหรับกำหนด data contract
- ❌ ไม่มี `align_and_resample()` utility สำหรับจัดการ timestamp alignment
- ❌ ไม่มีการตรวจสอบ monotonic timestamps
- ❌ ไม่มีการตรวจสอบ missing intervals

**สิ่งที่ต้องสร้าง:**
```python
# schemas.py
OHLCV_SCHEMA = {
    'columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    'dtypes': {...},
    'frequency': '5min',
    'timezone': 'UTC'
}

# utils/data_alignment.py
def align_and_resample(ohlcv, oi, funding, ...):
    # Align all dataframes to same timestamp index
    # Handle missing intervals with explicit rules
    # Return aligned data + missing report
```

**ผลกระทบ:**
- Silent data drift เมื่อ schema เปลี่ยน
- Misaligned timestamps ระหว่าง feeds ต่างๆ
- ข้อมูลผิดพลาดโดยไม่รู้ตัว

---

### ✅ 2. Leakage Guards (Targets & Features)
**สถานะ:** ✅ **ดี**
**ความสำคัญ:** 🔴 **CRITICAL**

**ตรวจสอบแล้ว:**
```python
# features/target_engineer.py (line 49-50)
future_close = df['close'].shift(-horizon)  # ✅ ใช้ shift(-horizon) ถูกต้อง
future_return = (future_close - df['close']) / df['close']

# Line 71: Remove last horizon rows
df = df.iloc[:-horizon]  # ✅ ป้องกัน leakage ถูกต้อง
```

**คำแนะนำเพิ่มเติม (Optional):**
- สร้าง `test_leakage.py` สำหรับ unit test
- สร้าง "live-safe" variant ของ features (shifted by 1 bar)

**สถานะ:** ✅ **ผ่าน - ไม่มี data leakage**

---

### ⚠️ 3. Feature Stability & Drift Monitoring
**สถานะ:** ❌ **ไม่มี**
**ความสำคัญ:** 🟡 **NICE-TO-HAVE**

**ปัญหา:**
- ไม่มีการคำนวณ PSI (Population Stability Index)
- ไม่มีการทำ KS tests
- ไม่มี train-val-test drift comparison

**คำแนะนำ:**
- สร้าง `utils/feature_stability.py`
- คำนวณ PSI, KS, Wasserstein distance
- สร้าง HTML report ใน `reports/feature_stability/`

**สถานะ:** ⏭️ **ข้ามได้ - Not blocking for MVP**

---

### ⚠️ 4. Adaptive Extremes for Funding & Liquidations
**สถานะ:** ⚠️ **ครึ่งหนึ่ง**
**ความสำคัญ:** 🟡 **NICE-TO-HAVE**

**ที่มีอยู่:**
- มีการคำนวณ features จาก funding และ liquidations
- มี moving averages และ rolling statistics

**ที่ยังขาด:**
- ไม่มีการใช้ rolling z-scores/percentiles สำหรับ "extreme" states
- ใช้ static thresholds แทน adaptive ones

**คำแนะนำ:**
```python
# แทนที่จะใช้ static threshold
# Before: funding > 0.05
# After:  funding_zscore > 2.0  (rolling 90d window)
```

**สถานะ:** ⏭️ **ข้ามได้ - Static thresholds ใช้ได้ก่อน**

---

### ❌ 5. Final Feature List & Versioning
**สถานะ:** ❌ **ไม่มี**
**ความสำคัญ:** 🔴 **CRITICAL**

**ปัญหา:**
- ไม่มี `artifacts/feature_list_v{hash}.json`
- ไม่มี `feature_set_id` สำหรับ tracking
- ไม่สามารถ reproduce feature set ได้

**สิ่งที่ต้องสร้าง:**
```python
# utils/feature_versioning.py
def save_feature_list(feature_names, config, output_dir='artifacts/'):
    hash_id = compute_hash(feature_names + config)
    feature_set = {
        'feature_set_id': hash_id,
        'features': feature_names,
        'config': config,
        'created_at': timestamp,
        'git_commit': get_git_commit()
    }
    save_json(f'artifacts/feature_list_v{hash_id}.json')
```

**ผลกระท่:**
- ไม่สามารถ reproduce models ได้
- Feature list เปลี่ยนโดยไม่รู้ตัว

---

### ❌ 6. Scaling Done Right (Train-Only)
**สถานะ:** ❌ **ไม่มี**
**ความสำคัญ:** 🔴 **CRITICAL**

**ปัญหาร้ายแรง:**
- ไม่พบการทำ scaling/normalization ใน training pipeline
- ถ้า fit scaler บน all data = data leakage!
- Neural Network และ LSTM ต้องการ scaled data

**สิ่งที่ต้องเพิ่ม:**
```python
# models/preprocessing.py
from sklearn.preprocessing import StandardScaler

def fit_scaler(X_train, feature_set_id):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, f'artifacts/scaler_{feature_set_id}.pkl')
    return scaler

def apply_scaler(X, scaler):
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
```

**ใน training_pipeline.py ต้องเพิ่ม:**
```python
# FIT on train only!
scaler = fit_scaler(X_train, feature_set_id)

# APPLY to train/val/test
X_train_scaled = apply_scaler(X_train, scaler)
X_val_scaled = apply_scaler(X_val, scaler)
X_test_scaled = apply_scaler(X_test, scaler)
```

**สถานะ:** 🚨 **URGENT - ต้องแก้ก่อน train models**

---

### ✅ 7. Walk-Forward Cross-Validation
**สถานะ:** ✅ **มีแล้ว**
**ความสำคัญ:** 🟢 **COMPLETE**

**ตรวจสอบแล้ว:**
```python
# utils/data_split.py (line 157-200)
def walk_forward_split(...):  # ✅ มีแล้ว
def purge_and_embargo(...):   # ✅ มีแล้ว

# models/validation.py
class WalkForwardValidator:  # ✅ มีแล้ว
```

**สถานะ:** ✅ **ผ่าน - Complete implementation**

---

### ⚠️ 8. Class Imbalance & Regime Labels
**สถานะ:** ⚠️ **พื้นฐาน**
**ความสำคัญ:** 🟢 **OK**

**ที่มีอยู่:**
- XGBoost/LightGBM มี `scale_pos_weight` support
- มี time-based features (hour, day of week)

**ที่ยังขาด:**
- ไม่มี class balance report by month
- ไม่มี explicit regime features (ADX-based, BB width)
- ไม่มี block-wise undersampling

**คำแนะนำ:**
```python
# เพิ่ม regime features ใน feature_engineer.py
df['regime_trend'] = df['adx'] > 25  # Trending
df['regime_range'] = df['adx'] <= 25  # Ranging
df['regime_highvol'] = df['atr'] > df['atr'].rolling(50).mean()
```

**สถานะ:** ⚠️ **พอใช้ได้ - มี basic support แล้ว**

---

### ❌ 9. Reproducible Data Artifacts
**สถานะ:** ❌ **ไม่มี**
**ความสำคัญ:** 🔴 **CRITICAL**

**ปัญหา:**
- ไม่มี prepared datasets (X_train.parquet, etc.)
- ไม่มี meta.json สำหรับ tracking versions/seeds
- ต้อง recompute features ทุกครั้ง

**สิ่งที่ต้องสร้าง:**
```python
# utils/artifact_manager.py
def export_prepared_datasets(X_train, y_train, X_val, y_val, X_test, y_test,
                            feature_set_id, scaler_path):

    output_dir = f'artifacts/datasets_{feature_set_id}/'

    # Export data
    X_train.to_parquet(f'{output_dir}/X_train.parquet')
    y_train.to_parquet(f'{output_dir}/y_train.parquet')
    ...

    # Export metadata
    meta = {
        'feature_set_id': feature_set_id,
        'scaler_path': scaler_path,
        'created_at': timestamp,
        'seeds': {'numpy': 42, 'torch': 42, ...},
        'versions': {'pandas': pd.__version__, ...}
    }
    save_json(f'{output_dir}/meta.json')
```

**ผลกระทบ:**
- ไม่สามารถ reproduce results ได้
- เสียเวลาในการ recompute features

---

### ⚠️ 10. CI, Tests, and Lightweight Docs
**สถานะ:** ⚠️ **บางส่วน**
**ความสำคัญ:** 🟡 **NICE-TO-HAVE**

**ที่มีอยู่:**
- ✅ `quick_test.py` - Basic tests
- ✅ `test_mock_data.py` - Comprehensive tests
- ✅ `TEST_GUIDE.md` - Documentation

**ที่ยังขาด:**
- ❌ Pre-commit hooks (ruff, black, mypy)
- ❌ GitHub Actions CI
- ❌ Auto-generated reports
- ❌ README "Data Contract" section

**คำแนะนำ:**
```yaml
# .github/workflows/ci.yml
- Lint with ruff
- Format check with black
- Type check with mypy
- Run tests
- Upload coverage reports
```

**สถานะ:** ⏭️ **ข้ามได้ - Tests มีพอสำหรับ MVP**

---

## 🎯 สรุปสิ่งที่ต้องทำ (Priority Order)

### 🔴 CRITICAL (ต้องทำก่อน train models)

1. **Scaling Pipeline** (ข้อ 6)
   - สร้าง `models/preprocessing.py`
   - เพิ่ม scaler fitting ใน training pipeline
   - FIT บน train เท่านั้น, APPLY ทั้งหมด

2. **Feature Versioning** (ข้อ 5)
   - สร้าง `utils/feature_versioning.py`
   - Export `feature_list_v{hash}.json`
   - Propagate `feature_set_id` ทุกที่

3. **Reproducible Artifacts** (ข้อ 9)
   - สร้าง `utils/artifact_manager.py`
   - Export prepared datasets (parquet)
   - Save meta.json with versions/seeds

4. **Data Contracts** (ข้อ 1)
   - สร้าง `schemas.py`
   - สร้าง `utils/data_alignment.py`
   - เพิ่ม timestamp validation

### 🟡 NICE-TO-HAVE (ทำหลังจาก MVP)

5. **Feature Stability Report** (ข้อ 3)
6. **Adaptive Extremes** (ข้อ 4)
7. **CI/CD Pipeline** (ข้อ 10)
8. **Enhanced Regime Features** (ข้อ 8)

### ✅ COMPLETE (ผ่านแล้ว)

- ✅ Leakage Guards (ข้อ 2)
- ✅ Walk-Forward CV (ข้อ 7)

---

## 💡 คำแนะนำการดำเนินการ

### Option A: แก้ไขทั้งหมด (Recommended)
ทำทั้ง 4 ข้อ CRITICAL เพื่อให้ pipeline production-ready:
- เวลาประมาณ: 3-4 ชั่วโมง
- ผลลัพธ์: Production-grade pipeline ที่ reproducible

### Option B: แก้เฉพาะ Scaling (Minimum)
ทำเฉพาะข้อ 6 (Scaling) ก่อน เพราะจำเป็นสำหรับ NN/LSTM:
- เวลาประมาณ: 1 ชั่วโมง
- ผลลัพธ์: Models สามารถ train ได้ถูกต้อง

### Option C: ข้ามทั้งหมด (Not Recommended)
ข้ามการแก้ไขและ train models เลย:
- ⚠️ NN/LSTM อาจ train ไม่ดีเพราะไม่มี scaling
- ⚠️ ไม่สามารถ reproduce results ได้
- ⚠️ ไม่มี version control สำหรับ features

---

## 📝 สรุปท้ายสุด

**โปรเจกต์ปัจจุบัน:**
- ✅ Phase 2 ทำงานได้ดี (160+ features)
- ✅ ไม่มี data leakage
- ✅ มี walk-forward validation
- ⚠️ **แต่ขาด 4 จุดสำคัญ:** Scaling, Feature Versioning, Artifacts, Data Contracts

**คำแนะนำ:**
1. ถ้าต้องการ **production-ready**: ทำทั้ง 4 ข้อ CRITICAL
2. ถ้าต้องการ **quick prototype**: ทำแค่ Scaling (ข้อ 6)
3. ข้อ NICE-TO-HAVE สามารถทำทีหลังได้

**คุณต้องการให้ผมช่วยแก้ไขหรือไม่? เลือกได้:**
- `Option A`: ทำทั้ง 4 ข้อ CRITICAL (แนะนำ)
- `Option B`: ทำแค่ Scaling
- `Option C`: ข้ามไปและดูก่อนว่าโมเดลทำงานได้ไหม

บอกมาได้เลยครับ!
