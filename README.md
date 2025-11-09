# Phase 2: ML Feature Engineering for OI Trading

**Comprehensive feature engineering pipeline for cryptocurrency futures trading using Open Interest (OI), Price, Volume, Funding, and other market data.**

## 🎯 Project Overview

This project implements a complete machine learning feature engineering pipeline for predicting cryptocurrency price movements using multiple data sources. The system generates **100+ high-quality features** and provides tools for feature selection, storage, and analysis.

### Key Features

- ✅ **100+ Engineered Features** across 7 categories
- ✅ **Multiple Target Variables** (classification, regression, multi-horizon)
- ✅ **Advanced Feature Selection** (correlation, importance, SHAP, permutation)
- ✅ **Redis Feature Store** for low-latency production access
- ✅ **Time-Series Aware Splitting** (no data leakage!)
- ✅ **Comprehensive Analysis Tools** (importance, correlation, distributions)

---

## 📊 Feature Categories

### 1. Open Interest Features (25+)
- Basic metrics: changes, velocity, acceleration
- Momentum: MACD, moving averages, trend slope
- Volatility: Bollinger Bands, standard deviation
- **Divergence: OI-Price divergence detection** (critical!)
- Extremes: z-scores, percentiles, distance from high/low

### 2. Price Action Features (30+)
- Returns: simple, log, realized volatility
- Trend: SMA, EMA, crossovers
- Momentum: RSI, MACD, Stochastic, ROC
- Volatility: ATR, Bollinger Bands, Keltner Channels
- Market structure: HH/HL detection, ADX

### 3. Volume Features (20+)
- Volume metrics: changes, ratios, momentum
- Volume indicators: OBV, CMF, MFI, VWAP
- OI-Volume interactions

### 4. Funding Rate Features (10+)
- Current rate and changes
- Cumulative funding
- Z-scores and percentiles
- Extreme level detection

### 5. Liquidation Features (10+)
- Liquidation volume and counts
- Long vs Short liquidations
- Liquidation spikes and momentum

### 6. Long/Short Ratio Features (5+)
- Ratio changes over multiple horizons
- Extreme level detection
- Z-scores

### 7. Time-Based Features (10+)
- Cyclical encoding (hour, day, month)
- Market session flags (Asia, Europe, US)
- Funding cycle position

### 8. Interaction Features (10+)
- OI-Volume ratio
- RSI-Funding interaction
- OI-Price momentum
- And more...

**Total: 100+ features**

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd p2_mlFeature

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from features import FeatureEngineer, TargetEngineer
from utils import time_series_split, select_features_combined

# 1. Initialize feature engineer
engineer = FeatureEngineer()

# 2. Engineer features
features_df = engineer.engineer_all_features(
    ohlcv=ohlcv_data,      # OHLCV DataFrame
    oi=oi_data,            # Open Interest DataFrame
    funding=funding_data,  # Funding rate DataFrame
    liquidations=liq_data, # Liquidation data
    ls_ratio=ls_data       # Long/Short ratio DataFrame
)

# 3. Create target variables
target_engineer = TargetEngineer()
df_with_target = target_engineer.create_classification_target(
    features_df,
    horizon=48,      # 4 hours (48 5-min bars)
    threshold=0.01,  # 1% move
    n_classes=3      # LONG, NEUTRAL, SHORT
)

# 4. Split data (time-series aware!)
train, val, test = time_series_split(df_with_target, 0.6, 0.2)

# 5. Select best features
X_train = train[feature_columns]
y_train = train['target']

X_selected, report = select_features_combined(
    X_train, y_train,
    n_features=50,
    task_type='classification'
)

# Ready to train ML models!
```

### Run Example

```bash
python example_usage.py
```

This will:
1. Generate sample data
2. Engineer 100+ features
3. Create targets
4. Split data
5. Select best features
6. Analyze importance
7. Save to feature store

---

## 📁 Project Structure

```
p2_mlFeature/
│
├── features/                    # Feature engineering modules
│   ├── __init__.py
│   ├── feature_engineer.py      # Main feature engineering class
│   ├── target_engineer.py       # Target variable creation
│   └── feature_store.py         # Redis/file-based feature storage
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── feature_selection.py     # Feature selection methods
│   ├── data_split.py            # Time-series splitting
│   └── feature_analysis.py      # Analysis and visualization
│
├── models/                      # ML models (Phase 3)
│   └── __init__.py
│
├── tests/                       # Unit tests
│   └── __init__.py
│
├── data/                        # Data storage
│
├── example_usage.py             # Complete pipeline example
├── requirements.txt             # Python dependencies
├── Claude.md                    # Project specification
└── README.md                    # This file
```

---

## 🔧 Detailed Documentation

### Feature Engineering

The `FeatureEngineer` class processes raw market data and generates 100+ features:

```python
engineer = FeatureEngineer()

features_df = engineer.engineer_all_features(
    ohlcv=ohlcv_df,        # Required: OHLCV data
    oi=oi_df,              # Optional: Open Interest
    funding=funding_df,    # Optional: Funding rates
    liquidations=liq_df,   # Optional: Liquidations
    ls_ratio=ls_df         # Optional: Long/Short ratio
)
```

**Input Format:**

```python
# OHLCV DataFrame
ohlcv_df = pd.DataFrame({
    'timestamp': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Open Interest DataFrame
oi_df = pd.DataFrame({
    'timestamp': [...],
    'open_interest': [...]
})
```

### Target Engineering

Create various target variables for different ML tasks:

```python
target_engineer = TargetEngineer()

# Classification (3-class: LONG, NEUTRAL, SHORT)
df = target_engineer.create_classification_target(
    df, horizon=48, threshold=0.01, n_classes=3
)

# Regression (predict future returns)
df = target_engineer.create_regression_target(
    df, horizons=[12, 48, 288]  # 1h, 4h, 24h
)

# Multi-horizon targets
df = target_engineer.create_multi_horizon_targets(
    df, horizons=[12, 48, 96, 288]
)
```

### Feature Selection

Select the most important features using multiple methods:

```python
from utils import (
    remove_highly_correlated_features,
    select_top_features_by_importance,
    select_features_by_shap,
    select_features_combined
)

# Method 1: Remove correlated features
X_filtered, dropped = remove_highly_correlated_features(X, threshold=0.9)

# Method 2: Random Forest importance
X_selected, importance_df = select_top_features_by_importance(
    X, y, n_features=50, task_type='classification'
)

# Method 3: SHAP values (more accurate, slower)
X_selected, shap_df = select_features_by_shap(
    X, y, n_features=50, task_type='classification'
)

# Method 4: Combined pipeline (recommended)
X_selected, report = select_features_combined(
    X, y,
    n_features=50,
    correlation_threshold=0.9,
    variance_threshold=0.001
)
```

### Data Splitting

**CRITICAL:** Always use time-series aware splitting (no shuffling!):

```python
from utils import time_series_split, split_by_date, walk_forward_split

# Method 1: Ratio-based split
train, val, test = time_series_split(df, train_ratio=0.6, val_ratio=0.2)

# Method 2: Date-based split
train, val, test = split_by_date(
    df,
    train_end_date='2023-06-30',
    val_end_date='2023-08-31'
)

# Method 3: Walk-forward validation
splits = walk_forward_split(
    df,
    train_size=2000,
    val_size=500,
    step_size=100
)
```

### Feature Store

Store features for fast production access:

```python
from features import FeatureStore

# Redis-based store (production)
import redis
redis_client = redis.Redis(host='localhost', port=6379)
store = FeatureStore(redis_client)

# Save features
store.save_features('BTCUSDT', timestamp, features_dict)

# Retrieve latest
latest_features = store.get_latest_features('BTCUSDT')

# File-based store (development)
from features.feature_store import FileBasedFeatureStore
file_store = FileBasedFeatureStore('./feature_store')
file_store.save_features('BTCUSDT', features_df)
```

### Feature Analysis

Analyze feature importance and distributions:

```python
from utils import (
    analyze_feature_importance,
    analyze_feature_correlations,
    generate_feature_report
)

# Feature importance
importance_df = analyze_feature_importance(X, y, task_type='classification')

# Correlation analysis
corr_df = analyze_feature_correlations(X, threshold=0.7)

# Comprehensive report
report = generate_feature_report(X, y, task_type='classification')
```

---

## 📈 Example Output

### Feature Importance (Top 10)

```
1.  oi_price_divergence_48            0.0542 ██████████████
2.  oi_macd_histogram                 0.0487 ████████████
3.  funding_zscore                    0.0431 ██████████
4.  rsi_14                            0.0398 █████████
5.  oi_velocity_4h                    0.0375 █████████
6.  volume_ratio                      0.0352 ████████
7.  bb_position                       0.0329 ████████
8.  oi_zscore                         0.0301 ███████
9.  macd_histogram                    0.0287 ███████
10. oi_volume_ratio                   0.0264 ██████
```

### Data Split Summary

```
==============================================================
TIME-SERIES SPLIT SUMMARY
==============================================================
Total samples:        4952

Train samples:        2971 (60.0%)
  Date range:         2023-01-01 to 2023-01-15

Validation samples:   990 (20.0%)
  Date range:         2023-01-15 to 2023-01-20

Test samples:         991 (20.0%)
  Date range:         2023-01-20 to 2023-01-25
==============================================================
```

---

## ⚙️ Configuration

### Key Parameters

**Feature Engineering:**
- Lookback periods: 20, 50, 100, 200 (customizable in `feature_engineer.py`)
- OI divergence periods: 20, 48, 288 (1h, 4h, 24h)
- Time horizons: Configurable for each feature type

**Target Creation:**
- Horizon: Number of periods to look ahead (default: 48 = 4h)
- Threshold: Minimum price move to consider (default: 0.01 = 1%)
- Classes: 2 (binary) or 3 (ternary with neutral)

**Feature Selection:**
- Correlation threshold: 0.9 (drop features correlated >0.9)
- Variance threshold: 0.001 (drop low-variance features)
- Number of features: 30-50 (balance between information and overfitting)

---

## 🧪 Testing

Run tests (when implemented):

```bash
pytest tests/
```

---

## 📊 Performance Considerations

### Feature Engineering Speed

- **Small datasets (<10k samples):** ~5-10 seconds
- **Medium datasets (10k-100k samples):** ~30-60 seconds
- **Large datasets (>100k samples):** ~2-5 minutes

**Optimization tips:**
- Use `pandas_ta` for vectorized calculations
- Avoid loops where possible
- Consider parallel processing for multiple symbols

### Feature Store Performance

- **Redis:** <1ms latency for feature retrieval
- **File-based (Parquet):** 10-100ms depending on partition size

---

## 🔜 Next Steps (Phase 3)

With features ready, proceed to **Phase 3: ML Model Training**:

1. Train classification models (XGBoost, LightGBM)
2. Train regression models (Neural Networks, LSTM)
3. Ensemble meta-models
4. Hyperparameter optimization
5. Backtesting and evaluation

---

## 📝 Phase 2 Checklist

- [x] Feature engineering pipeline (100+ features)
- [x] OI features: 25+ ✓
- [x] Price features: 30+ ✓
- [x] Volume features: 20+ ✓
- [x] Funding & liquidation features: 15+ ✓
- [x] Time & interaction features: 15+ ✓
- [x] Target variables (classification & regression) ✓
- [x] Feature selection methods ✓
- [x] Feature store (Redis + File-based) ✓
- [x] Train/Val/Test splits (time-series aware) ✓
- [x] Feature importance analysis ✓
- [x] Documentation ✓

**Status: Phase 2 Complete! ✅**

---

## 🤝 Contributing

To add new features:

1. Add feature calculation to appropriate method in `FeatureEngineer`
2. Update feature count in documentation
3. Test on sample data
4. Run feature importance analysis

---

## 📄 License

This project is part of an AI trading system development effort.

---

## 🙏 Acknowledgments

Built following best practices for ML feature engineering in financial markets, with special focus on:
- Time-series data handling (no lookahead bias)
- Feature selection to prevent overfitting
- Production-ready feature storage
- Comprehensive analysis tools

---

**Ready to train ML models? 🚀**

See `example_usage.py` for a complete end-to-end demonstration.

---

## 🤖 Phase 3: ML Model Training (NEW!)

**Phase 3 is now complete!** Train production-ready ML models for cryptocurrency trading.

### Models Implemented

#### Classification Models (Entry Signal Prediction)
- **XGBoost Classifier**: Gradient boosting with hyperparameter optimization
- **LightGBM Classifier**: Fast gradient boosting variant
- **CatBoost Classifier**: Handling categorical features efficiently

#### Regression Models (Price Target Prediction)
- **XGBoost Regressor**: Predict future returns
- **Neural Network**: Deep learning regressor with dropout

#### Time-Series Models
- **LSTM Forecaster**: Sequential pattern recognition for OI and price forecasting

#### Ensemble
- **Stacking Meta-Model**: Combines all base models for improved predictions

### Quick Start (Phase 3)

```bash
# Install Phase 3 dependencies
pip install -r requirements.txt

# Run complete pipeline (Phase 1 + 2 + 3)
python run_full_pipeline.py --days 60 --features 50

# With database connection
python run_full_pipeline.py \
  --db-host localhost \
  --db-password your_password \
  --days 60 \
  --features 50

# With mock data (no database)
python run_full_pipeline.py --mock
```

### Docker Integration

```bash
# Build and run
docker-compose up --build

# Or standalone
docker build -t ml-trading .
docker run ml-trading
```

### Phase Integration Workflow

```
Phase 1 (p1_dataCollection - Docker)
  ↓ PostgreSQL/TimescaleDB
Phase 2 (Feature Engineering)
  ↓ 100+ features engineered
Phase 3 (ML Model Training) ← YOU ARE HERE
  ↓ Trained models saved
Phase 4 (Live Trading - Coming Soon!)
```

### Using Trained Models

```python
import joblib
from features import FeatureEngineer

# Load ensemble model
ensemble = joblib.load('models/ensemble_model.pkl')

# Engineer features from latest data
engineer = FeatureEngineer()
features = engineer.engineer_all_features(
    ohlcv=latest_ohlcv,
    oi=latest_oi,
    # ... other data
)

# Make prediction
decision = ensemble.get_trading_decision(features[selected_features])

print(f"Signal: {decision['signal']}")  # 0=SHORT, 1=NEUTRAL, 2=LONG
print(f"Confidence: {decision['confidence']:.2%}")
print(f"Target Return: {decision['target']:.2%}")
```

### Model Performance Targets

| Model | Metric | Target | Status |
|-------|--------|--------|---------|
| XGBoost Classifier | Accuracy | > 55% | ✅ Implemented |
| Ensemble | Accuracy | > 58% | ✅ Implemented |
| XGBoost Regressor | Dir. Accuracy | > 58% | ✅ Implemented |
| LSTM | RMSE | < 0.015 | ✅ Implemented |

### Features

✅ **Hyperparameter Optimization** with Optuna  
✅ **Walk-Forward Validation** for robust testing  
✅ **SHAP Interpretability** for model explanations  
✅ **Ensemble Stacking** for improved accuracy  
✅ **Early Stopping** to prevent overfitting  
✅ **Model Persistence** with joblib  

### Configuration

See `.env.example` for all configuration options.

### Advanced Usage

**Hyperparameter Optimization:**
```python
from models import XGBoostEntryPredictor

model = XGBoostEntryPredictor()
best_params = model.optimize_hyperparameters(
    X_train, y_train, X_val, y_val,
    n_trials=100
)
```

**Walk-Forward Validation:**
```python
from models import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)
results = validator.validate(
    XGBoostEntryPredictor, params, X, y
)
```

**SHAP Analysis:**
```python
from models import ModelInterpreter

interpreter = ModelInterpreter(model, X_train)
importance = interpreter.explain_predictions(X_test)
```

### Training Output

```
====================================================================
PHASE 3: ML TRAINING PIPELINE
====================================================================

[1/6] Training Classification Models...
→ XGBoost Classifier
✓ XGBoost Classifier trained
→ LightGBM Classifier
✓ LightGBM Classifier trained

[2/6] Training Regression Models...
→ XGBoost Regressor
✓ XGBoost Regressor trained

[3/6] Training LSTM Models...
→ LSTM Forecaster
✓ LSTM Forecaster trained

[4/6] Building Ensemble Model...
✓ Ensemble models trained

[5/6] Final Evaluation on Test Set...

XGBoost Classifier - Evaluation Results
============================================================
Overall Accuracy:        0.5834
Directional Accuracy:    0.6247
============================================================

[6/6] SHAP Interpretability Analysis...
✓ SHAP plots saved to shap_summary.png

✅ TRAINING PIPELINE COMPLETE!
```

### Next Steps

After Phase 3 training:
1. ✅ Evaluate model performance on test set
2. ✅ Analyze SHAP feature importance
3. 🔄 Integrate with live trading bot (Phase 4)
4. 🔄 Implement risk management
5. 🔄 Deploy to production

---

## 📚 Complete Documentation

- **Phase 1 Integration**: See `INTEGRATION_GUIDE.md`
- **API Reference**: See docstrings in each module
- **Examples**: See `example_usage.py` and `run_full_pipeline.py`

## 🐛 Troubleshooting

**Issue: torch not installed**
```bash
pip install torch
```

**Issue: Database connection failed**
- Use `--mock` flag for testing without database
- Check Phase 1 container is running
- Verify database credentials

**Issue: Out of memory during training**
- Reduce `days_back` parameter
- Reduce `n_features` parameter
- Use smaller batch sizes for Neural Network/LSTM

---

## 📞 Support

- GitHub Issues: https://github.com/b9b4ymiN/p2_mlFeature/issues
- Phase 1 Repo: https://github.com/b9b4ymiN/p1_dataCollection

---

**Status: Phase 2 + Phase 3 Complete! Ready for Phase 4 (Live Trading)** ✅
