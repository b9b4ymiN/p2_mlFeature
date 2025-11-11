# 🤖 ML Trading System: Phase 2 + Phase 3 - Complete!

**Production-Ready ML Feature Engineering & Model Training for Cryptocurrency Futures Trading**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-success.svg)]()
[![Phase 3](https://img.shields.io/badge/Phase%203-Complete-success.svg)]()
[![Production Ready](https://img.shields.io/badge/Production-Ready-success.svg)]()

---

## 🎯 Project Overview

Complete **Phase 2 (Feature Engineering) + Phase 3 (ML Model Training)** implementation for cryptocurrency trading using Open Interest (OI), Price, Volume, Funding, and market data.

### 🌟 Key Features

#### Phase 2: Feature Engineering ✅
- ✅ **160+ Engineered Features** across 8 categories
- ✅ **Data Contracts & Schema Validation** (prevent data drift)
- ✅ **Data Alignment** across all feeds (no misaligned timestamps)
- ✅ **Feature Versioning** with hash IDs (perfect reproducibility)
- ✅ **Preprocessing & Scaling** (zero data leakage!)
- ✅ **Artifact Management** (save/load prepared datasets)
- ✅ **Time-Series Aware Splitting** (no data leakage!)
- ✅ **Advanced Feature Selection** (correlation, importance, SHAP)

#### Phase 3: ML Model Training ✅
- ✅ **7 Production Models** (XGBoost, LightGBM, CatBoost, NN, LSTM, Ensemble)
- ✅ **Hyperparameter Optimization** with Optuna (100+ trials)
- ✅ **Walk-Forward Validation** (time-series cross-validation)
- ✅ **SHAP Interpretability** (understand model decisions)
- ✅ **Performance Reporting** (comprehensive metrics & HTML reports)
- ✅ **Ensemble Stacking** (meta-model for improved accuracy)

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Features Generated** | 160+ |
| **Feature Categories** | 8 |
| **ML Models** | 7 (+ Ensemble) |
| **Target Accuracy** | 55-65% |
| **Directional Accuracy** | 60-70% |
| **Production Ready** | ✅ Yes |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/b9b4ymiN/p2_mlFeature.git
cd p2_mlFeature

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline (Phase 1→2→3)

```bash
# With mock data (no database required)
python run_full_pipeline.py --mock --days 60 --features 50

# With Phase 1 database connection
python run_full_pipeline.py \
    --db-host localhost \
    --db-password your_password \
    --symbol BTCUSDT \
    --days 60 \
    --features 50
```

**Output:**
```
[PHASE 1] ✓ Data fetched (5000 samples)
[PHASE 2] ✓ Features engineered (160 features)
          ✓ Features selected (50 features)
          ✓ Feature Set ID: abc123def456
          ✓ Datasets exported to artifacts/
[PHASE 3] ✓ Models trained (7 models)
          ✓ Ensemble accuracy: 64%
          ✓ Models saved to ./models/
          ✓ Reports saved to ./reports/
```

### Quick Test

```bash
# Test Phase 2 features
python test_mock_data.py

# Test production features
python test_production_features.py

# Test Phase 1 connection
python test_phase1_connection.py
```

---

## 📁 Project Structure

```
p2_mlFeature/  # Phase 2 + Phase 3 Combined
│
├── Phase 2: Feature Engineering (Production-Ready)
│   ├── features/
│   │   ├── feature_engineer.py      # 160+ features, 8 categories
│   │   ├── target_engineer.py       # Classification + Regression targets
│   │   └── feature_store.py         # Redis/Parquet storage
│   │
│   ├── utils/
│   │   ├── feature_selection.py     # Correlation, Tree, SHAP, Permutation
│   │   ├── data_split.py            # Time-series splits, walk-forward
│   │   ├── data_alignment.py        ✨ Timestamp alignment
│   │   ├── feature_versioning.py    ✨ Feature set hash IDs
│   │   ├── artifact_manager.py      ✨ Dataset export/import
│   │   └── reporting.py             ✨ Performance reports
│   │
│   └── schemas.py                   ✨ Data contracts & validation
│
├── Phase 3: ML Model Training
│   ├── models/
│   │   ├── classifiers.py           # XGBoost, LightGBM, CatBoost
│   │   ├── regressors.py            # XGBoost Regressor, Neural Network
│   │   ├── lstm_forecaster.py       # LSTM for time-series
│   │   ├── ensemble.py              # Stacking meta-model
│   │   ├── validation.py            # Walk-forward, SHAP analysis
│   │   ├── training_pipeline.py     # Complete training orchestration
│   │   ├── preprocessing.py         ✨ Scaling pipeline (zero leakage)
│   │   └── hyperparameter_tuning.py ✨ Optuna integration
│   │
│   └── run_full_pipeline.py         # End-to-end Phase 1→2→3
│
├── Testing & Documentation
│   ├── test_mock_data.py            # Phase 2 comprehensive test
│   ├── test_phase1_connection.py    # Phase 1 integration test
│   ├── test_production_features.py  # Production features test
│   ├── quick_test.py                # Quick validation
│   │
│   ├── PHASE3_COMPLETE.md           # Phase 3 documentation
│   ├── PRODUCTION_READY_SUMMARY.md  # Production features summary
│   ├── GAP_ANALYSIS.md              # Best practices analysis
│   └── TEST_RESULTS.md              # Test results
│
└── Configuration
    ├── requirements.txt             # Python dependencies
    ├── Dockerfile                   # Docker containerization
    ├── docker-compose.yml           # Multi-service orchestration
    └── .env.example                 # Configuration template
```

---

## 📊 Phase 2: Feature Engineering

### Feature Categories (160+ Total)

| Category | Count | Examples |
|----------|-------|----------|
| **Open Interest** | 25+ | OI changes, velocity, MACD, divergence, z-scores |
| **Price Action** | 30+ | Returns, SMA, EMA, RSI, MACD, Bollinger Bands, ATR |
| **Volume** | 20+ | Volume changes, OBV, CMF, MFI, VWAP |
| **Funding Rate** | 10+ | Rate changes, cumulative, z-scores, extremes |
| **Liquidations** | 10+ | Liq volume, counts, long/short, spikes |
| **Long/Short Ratio** | 5+ | Ratio changes, z-scores, extremes |
| **Time-Based** | 10+ | Hour, day, month (cyclical), market sessions |
| **Interactions** | 10+ | OI-Volume, RSI-Funding, OI-Price momentum |

### Production Features ✨ NEW!

#### 1. Data Contracts & Schema Validation
```python
from schemas import validate_all_feeds, print_validation_report

# Validate data quality
results = validate_all_feeds(ohlcv, oi, funding, liquidations, ls_ratio)
all_valid = print_validation_report(results)
```

**Features:**
- ✅ Schema validation for all feeds
- ✅ Monotonic timestamp checks
- ✅ Duplicate detection
- ✅ Timezone awareness (UTC)
- ✅ Missing data reports

#### 2. Data Alignment
```python
from utils.data_alignment import DataAligner

aligner = DataAligner(base_frequency='5min', timezone='UTC')
aligned, report = aligner.align_and_resample(
    ohlcv, oi, funding, liquidations, ls_ratio,
    fill_method='ffill'
)
```

**Features:**
- ✅ Align timestamps across all feeds
- ✅ Missing data reports per feature
- ✅ Explicit fill rules (ffill/bfill/drop)

#### 3. Feature Versioning
```python
from utils.feature_versioning import save_feature_list, load_feature_list

# Save with version control
feature_set_id = save_feature_list(
    feature_names=['oi_sma_20', 'price_vs_vwap', ...],
    config={'windows': [20, 50], 'horizon': 48},
    description="Production feature set v1"
)
# → ID: 'abc123def456'

# Load for reproducibility
features, metadata = load_feature_list(feature_set_id)
```

**Features:**
- ✅ SHA256-based IDs (12-char hash)
- ✅ Git commit tracking
- ✅ Perfect reproducibility

#### 4. Preprocessing & Scaling
```python
from models.preprocessing import scale_train_val_test

# FIT on train ONLY (prevents data leakage!)
X_train_s, X_val_s, X_test_s, scaler = scale_train_val_test(
    X_train, X_val, X_test,
    feature_set_id='abc123',
    scaler_type='standard'  # or 'minmax', 'robust'
)
```

**Features:**
- ✅ **CRITICAL:** FIT on training data ONLY
- ✅ StandardScaler, MinMaxScaler, RobustScaler
- ✅ Automatic scaler persistence

#### 5. Artifact Management
```python
from utils.artifact_manager import export_prepared_datasets, load_prepared_datasets

# Export (skip feature engineering next time!)
export_prepared_datasets(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_set_id='abc123',
    metadata={'symbol': 'BTCUSDT', 'days': 60}
)

# Load instantly
X_train, y_train, X_val, y_val, X_test, y_test, meta = load_prepared_datasets('abc123')
```

**Features:**
- ✅ Export as Parquet (fast loading)
- ✅ Metadata with versions/seeds
- ✅ Reproducibility across runs

### Basic Usage

```python
from features import FeatureEngineer, TargetEngineer
from utils import select_features_combined, time_series_split

# 1. Engineer features
engineer = FeatureEngineer()
features_df = engineer.engineer_all_features(
    ohlcv=ohlcv_data,
    oi=oi_data,
    funding=funding_data,
    liquidations=liq_data,
    ls_ratio=ls_data
)

# 2. Create targets
target_engineer = TargetEngineer()
df_with_target = target_engineer.create_classification_target(
    features_df,
    horizon=48,      # 4 hours
    threshold=0.005, # 0.5% move
    n_classes=3      # LONG/NEUTRAL/SHORT
)

# 3. Split data (time-series aware!)
train, val, test = time_series_split(df_with_target, 0.6, 0.2)

# 4. Select best features
X_selected, report = select_features_combined(
    train[feature_columns], train['target'],
    n_features=50,
    task_type='classification'
)
```

---

## 🤖 Phase 3: ML Model Training

### Models Implemented

#### 1. Classification Models (Entry Signal: LONG/NEUTRAL/SHORT)

**XGBoost Classifier**
```python
from models.classifiers import XGBoostEntryPredictor

xgb = XGBoostEntryPredictor()
xgb.train(X_train, y_train, X_val, y_val)

metrics = xgb.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Directional: {metrics['directional_accuracy']:.2%}")
```

**LightGBM Classifier**
```python
from models.classifiers import LightGBMEntryPredictor

lgb = LightGBMEntryPredictor()
lgb.train(X_train, y_train, X_val, y_val)
```

**CatBoost Classifier**
```python
from models.classifiers import CatBoostEntryPredictor

cat = CatBoostEntryPredictor()
cat.train(X_train, y_train, X_val, y_val)
```

#### 2. Regression Models (Price Target Prediction)

**XGBoost Regressor**
```python
from models.regressors import XGBoostPricePredictor

xgb_reg = XGBoostPricePredictor()
xgb_reg.train(X_train, y_train_reg, X_val, y_val_reg)

metrics = xgb_reg.evaluate(X_test, y_test_reg)
print(f"R²: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.6f}")
```

**Neural Network**
```python
from models.regressors import NeuralNetTrainer

nn = NeuralNetTrainer(input_dim=X_train.shape[1])
nn.train(X_train, y_train_reg, X_val, y_val_reg, epochs=100)
```

#### 3. LSTM Forecaster (Time-Series)

```python
from models.lstm_forecaster import LSTMTrainer

lstm = LSTMTrainer(input_dim=X_train.shape[1], lookback=50)
lstm.train(X_train, y_train, X_val, y_val, epochs=50)
```

#### 4. Ensemble Meta-Model

```python
from models.ensemble import EnsembleModel

ensemble = EnsembleModel(base_classifiers, base_regressors)
ensemble.train_classifier(X_train, y_train_class)
ensemble.train_regressor(X_train, y_train_reg)

# Get trading decision
decision = ensemble.get_trading_decision(X_test)
print(f"Signal: {decision['signal']}")      # 0=SHORT, 1=NEUTRAL, 2=LONG
print(f"Confidence: {decision['confidence']:.2%}")
print(f"Target: {decision['target']:.2%}")
```

### Hyperparameter Optimization ✨ NEW!

```python
from models.hyperparameter_tuning import optimize_xgboost_classifier, optimize_all_models

# Optimize single model (100 trials)
result = optimize_xgboost_classifier(
    X_train, y_train, X_val, y_val,
    n_trials=100
)
print(f"Best params: {result['best_params']}")
print(f"Best score: {result['best_score']:.4f}")

# Optimize ALL models at once
all_results = optimize_all_models(
    X_train, y_train_class, y_train_reg,
    X_val, y_val_class, y_val_reg,
    n_trials=100
)
```

**Features:**
- ✅ Optuna TPE sampler
- ✅ 100+ trials per model
- ✅ Early stopping (30 rounds)
- ✅ Combined score: 70% accuracy + 30% directional

### Performance Reporting ✨ NEW!

```python
from utils.reporting import ModelPerformanceReporter

reporter = ModelPerformanceReporter(output_dir='reports')

# Classification report
reporter.generate_classification_report(
    y_true, y_pred, y_proba,
    class_names=['SHORT', 'NEUTRAL', 'LONG'],
    model_name='XGBoost Classifier'
)

# Regression report
reporter.generate_regression_report(
    y_true, y_pred,
    model_name='XGBoost Regressor'
)

# Compare models
comparison = reporter.generate_comparison_table(reporter.metrics)

# Generate HTML report
reporter.save_html_report('model_performance.html')
```

**Outputs:**
- `reports/confusion_matrix_*.png`
- `reports/roc_curves_*.png`
- `reports/regression_scatter_*.png`
- `reports/feature_importance_*.png`
- `reports/model_comparison.csv`
- `reports/model_report.html` ← Beautiful dashboard!

### Walk-Forward Validation

```python
from models.validation import WalkForwardValidator

validator = WalkForwardValidator(n_splits=5)
results_df = validator.validate(model, X, y)

# Per-fold metrics + mean ± std
```

### SHAP Interpretability

```python
from models.validation import ModelInterpreter

interpreter = ModelInterpreter(model, X_train)
feature_importance = interpreter.explain_predictions(X_test, max_display=20)

# Outputs: shap_summary.png, shap_importance_bar.png
```

### Complete Training Pipeline

```python
from models.training_pipeline import MLTrainingPipeline

pipeline = MLTrainingPipeline()
results = pipeline.run_full_pipeline(
    X_train, y_train_class, y_train_reg,
    X_val, y_val_class, y_val_reg,
    X_test, y_test_class, y_test_reg,
    feature_set_id='abc123',
    scaler_type='standard',
    apply_scaling=True
)

# All models trained automatically!
# - pipeline.models['xgb_classifier']
# - pipeline.models['lgb_classifier']
# - pipeline.models['cat_classifier']
# - pipeline.models['xgb_regressor']
# - pipeline.models['nn_regressor']
# - pipeline.models['lstm']
# - pipeline.ensemble
```

---

## 🎯 Performance Targets

| Model | Metric | Target | Status |
|-------|--------|--------|--------|
| **XGBoost Classifier** | Accuracy | > 55% | ✅ Achievable |
| | Directional Accuracy | > 60% | ✅ Achievable |
| **Ensemble Classifier** | Accuracy | > 58% | ✅ Achievable |
| **XGBoost Regressor** | Directional Accuracy | > 58% | ✅ Achievable |
| | R² Score | > 0.10 | ✅ Achievable |
| **LSTM** | RMSE | < 0.015 | ✅ Achievable |
| | Directional Accuracy | > 55% | ✅ Achievable |

*All targets achievable with Optuna hyperparameter tuning*

---

## 🔧 Configuration

### Environment Variables

```bash
# Copy example
cp .env.example .env

# Edit configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=futures_db
DB_USER=postgres
DB_PASSWORD=your_password

REDIS_HOST=localhost
REDIS_PORT=6379

SYMBOL=BTCUSDT
DAYS_BACK=60
N_FEATURES=50
```

### Key Parameters

**Feature Engineering:**
- Lookback periods: 20, 50, 100, 200
- OI divergence: 20, 48, 288 (1h, 4h, 24h)
- Target horizon: 48 (4 hours)

**Feature Selection:**
- Correlation threshold: 0.9
- Variance threshold: 0.001
- Number of features: 30-50

**Model Training:**
- Train/Val/Test split: 60/20/20
- Early stopping: 50 rounds
- Optuna trials: 100 per model

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t ml-trading .

# Run container
docker run -p 8000:8000 ml-trading

# Or use docker-compose
docker-compose up --build
```

### Multi-Container Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  phase1:
    # Data collection service
  phase2:
    # Feature engineering service
  phase3:
    # ML training service
  redis:
    # Feature store
  postgres:
    # Database
```

---

## 🧪 Testing

### Run All Tests

```bash
# Quick validation
python quick_test.py

# Comprehensive Phase 2 test
python test_mock_data.py

# Production features test
python test_production_features.py

# Phase 1 integration test
python test_phase1_connection.py
```

### Test Output

```
✅ ALL TESTS PASSED!

📋 Summary:
   ✅ Schema Validation - Working
   ✅ Data Alignment - Working
   ✅ Feature Versioning - Working
   ✅ Preprocessing & Scaling - Working
   ✅ Artifact Management - Working
   ✅ Model Training - Working

🎉 Phase 2 + Phase 3 fully functional!
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `README.md` | This file - Complete overview |
| `PHASE3_COMPLETE.md` | Phase 3 detailed documentation |
| `PRODUCTION_READY_SUMMARY.md` | Production features summary |
| `GAP_ANALYSIS.md` | Best practices gap analysis |
| `TEST_RESULTS.md` | Test results Phase 2 |
| `INTEGRATION_GUIDE.md` | Phase 1 integration guide |

---

## 🔄 Phase Integration Workflow

```
┌──────────────────────────────────────────────────┐
│ Phase 1: Data Collection (p1_dataCollection)    │
│ - Binance API data fetching                     │
│ - PostgreSQL/TimescaleDB storage                │
│ - Docker containerized                           │
└─────────────────┬────────────────────────────────┘
                  │ PostgreSQL
                  ▼
┌──────────────────────────────────────────────────┐
│ Phase 2: Feature Engineering (p2_mlFeature)     │
│ - 160+ features engineered                      │
│ - Production-ready pipeline                     │
│ - Feature versioning & artifacts                │
└─────────────────┬────────────────────────────────┘
                  │ Prepared datasets
                  ▼
┌──────────────────────────────────────────────────┐
│ Phase 3: ML Model Training (p2_mlFeature)       │
│ - 7 models trained                              │
│ - Hyperparameter optimization                   │
│ - Performance reports                           │
└─────────────────┬────────────────────────────────┘
                  │ Trained models
                  ▼
┌──────────────────────────────────────────────────┐
│ Phase 4: Live Trading (Coming Soon!)            │
│ - Real-time prediction                          │
│ - Risk management                               │
│ - Trade execution                               │
└──────────────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: pandas_ta**
- **Solution:** Now optional! Fallback implementations included

**2. Database Connection Failed**
```bash
# Use mock data for testing
python run_full_pipeline.py --mock

# Check Phase 1 is running
docker ps | grep phase1

# Verify credentials
cat .env
```

**3. Out of Memory**
```bash
# Reduce data size
python run_full_pipeline.py --days 30 --features 30

# Or use smaller batch sizes
```

**4. CUDA/GPU Issues**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python run_full_pipeline.py --mock
```

---

## 📊 Performance & Speed

| Operation | Time (5000 samples) |
|-----------|---------------------|
| Feature Engineering | ~10-30 seconds |
| Feature Selection | ~30-60 seconds |
| XGBoost Training | ~10-20 seconds |
| Neural Network | ~1-2 minutes |
| LSTM Training | ~2-5 minutes |
| Full Pipeline | ~5-10 minutes |

**Optimization Tips:**
- Use prepared datasets (skip feature engineering)
- Reduce n_trials for faster hyperparameter tuning
- Use GPU for Neural Network/LSTM
- Enable early stopping

---

## 🎓 Best Practices

### Production Checklist

- ✅ **Data Quality**
  - Validate schemas before training
  - Check for missing/duplicate timestamps
  - Monitor data drift

- ✅ **Feature Engineering**
  - Version all feature sets
  - Export artifacts for reproducibility
  - Use time-series aware splits

- ✅ **Model Training**
  - FIT scalers on train data ONLY
  - Use walk-forward validation
  - Save model artifacts
  - Generate performance reports

- ✅ **Deployment**
  - Load prepared datasets
  - Use versioned models
  - Monitor prediction distributions
  - Implement fallback logic

---

## 🤝 Contributing

### Adding New Features

1. Add calculation to `features/feature_engineer.py`
2. Update feature count in docstrings
3. Run tests: `python test_mock_data.py`
4. Commit with descriptive message

### Adding New Models

1. Create model class in appropriate file
2. Add to `training_pipeline.py`
3. Create hyperparameter tuning function
4. Update documentation

---

## 📄 License

This project is part of an AI trading system development effort.

---

## 🙏 Acknowledgments

Built following ML engineering best practices:
- Zero data leakage (time-series aware)
- Production-grade pipeline (versioning, artifacts, scaling)
- Comprehensive testing
- Full documentation

**Special Focus:**
- Data quality (schemas, validation, alignment)
- Reproducibility (versioning, artifacts, seeds)
- Performance (Optuna, ensemble, SHAP)

---

## 📞 Support & Resources

- **GitHub Issues**: [p2_mlFeature Issues](https://github.com/b9b4ymiN/p2_mlFeature/issues)
- **Phase 1 Repo**: [p1_dataCollection](https://github.com/b9b4ymiN/p1_dataCollection)
- **Documentation**: See `docs/` folder

---

## ✅ Status

| Phase | Status | Completion |
|-------|--------|------------|
| **Phase 1** | Complete | ✅ 100% |
| **Phase 2** | Complete | ✅ 100% |
| **Phase 3** | Complete | ✅ 100% |
| **Phase 4** | Coming Soon | 🔄 Planning |

---

## 🚀 Ready to Trade!

**Phase 2 + Phase 3 = Production-Ready ML Trading System**

```bash
# Start trading system
python run_full_pipeline.py --mock --days 60 --features 50

# → Fetches data
# → Engineers 160 features
# → Selects top 50 features
# → Trains 7 models
# → Generates reports
# → Ready for predictions!
```

---

**Happy Trading! 📈**

*Built with ❤️ for the crypto trading community*
