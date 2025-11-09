# Integration Guide: Phase 1 + Phase 2 + Phase 3

This guide shows how to integrate Phase 1 (data collection), Phase 2 (feature engineering), and Phase 3 (ML model training) for a complete end-to-end trading system.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA COLLECTION                  │
│                 (Docker - Binance API)                       │
│                                                              │
│  • OHLCV Data (5-min candles)                               │
│  • Open Interest Data                                        │
│  • Funding Rate Data                                         │
│  • Liquidation Data                                          │
│  • Long/Short Ratio Data                                     │
│                                                              │
│  Storage: PostgreSQL/TimescaleDB                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASE 2: FEATURE ENGINEERING                │
│                 (This Project - p2_mlFeature)                │
│                                                              │
│  • FeatureEngineer: 100+ features                           │
│  • TargetEngineer: Classification & Regression targets      │
│  • Feature Selection: Top 50 features                       │
│  • Feature Store: Redis/Parquet                             │
│                                                              │
│  Output: Engineered features + targets ready for ML         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PHASE 3: ML MODEL TRAINING                 │
│                                                              │
│  Classification Models:                                      │
│  • XGBoost, LightGBM, CatBoost                              │
│                                                              │
│  Regression Models:                                          │
│  • XGBoost Regressor, Neural Network                        │
│                                                              │
│  Time-Series Models:                                         │
│  • LSTM Forecaster                                           │
│                                                              │
│  Ensemble:                                                   │
│  • Stacking Meta-Model                                       │
│                                                              │
│  Output: Trained models ready for live trading              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### Phase 1 Setup (Already Running)
- Docker with Phase 1 data collection from: https://github.com/b9b4ymiN/p1_dataCollection
- PostgreSQL/TimescaleDB with collected data
- Data tables: ohlcv, open_interest, funding_rate, liquidations, long_short_ratio

### Phase 2 Setup (Current Project)
```bash
cd p2_mlFeature
pip install -r requirements.txt
```

---

## 🔌 Step 1: Connect to Phase 1 Database

Create a database connector to fetch data from Phase 1:

```python
# data_integration/phase1_connector.py

import psycopg2
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

class Phase1DataConnector:
    """
    Connect to Phase 1 PostgreSQL database and fetch market data
    """

    def __init__(self, host='localhost', port=5432, database='trading_db',
                 user='postgres', password='your_password'):
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.conn_params)
        print("✓ Connected to Phase 1 database")

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Disconnected from database")

    def fetch_ohlcv(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """
        Fetch OHLCV data

        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} OHLCV records for {symbol}")
        return df

    def fetch_open_interest(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Open Interest data"""
        query = f"""
            SELECT timestamp, open_interest
            FROM open_interest
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} OI records")
        return df

    def fetch_funding_rate(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Funding Rate data"""
        query = f"""
            SELECT timestamp, funding_rate
            FROM funding_rate
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} funding rate records")
        return df

    def fetch_liquidations(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Liquidation data"""
        query = f"""
            SELECT timestamp, quantity, side, order_id
            FROM liquidations
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} liquidation records")
        return df

    def fetch_long_short_ratio(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Long/Short Ratio data"""
        query = f"""
            SELECT timestamp, long_short_ratio as "longShortRatio"
            FROM long_short_ratio
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} L/S ratio records")
        return df

    def fetch_all_data(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """
        Fetch all data types at once

        Returns dictionary with all DataFrames
        """
        print(f"\nFetching all data for {symbol}...")
        print(f"Date range: {start_date} to {end_date}")

        data = {
            'ohlcv': self.fetch_ohlcv(symbol, start_date, end_date),
            'oi': self.fetch_open_interest(symbol, start_date, end_date),
            'funding': self.fetch_funding_rate(symbol, start_date, end_date),
            'liquidations': self.fetch_liquidations(symbol, start_date, end_date),
            'ls_ratio': self.fetch_long_short_ratio(symbol, start_date, end_date)
        }

        print(f"\n✓ All data fetched successfully!")
        return data
```

---

## 🔄 Step 2: Create End-to-End Pipeline

Create a script that connects everything:

```python
# run_full_pipeline.py

from data_integration.phase1_connector import Phase1DataConnector
from features import FeatureEngineer, TargetEngineer
from utils import time_series_split, select_features_combined
from models.training_pipeline import MLTrainingPipeline
from datetime import datetime, timedelta

def run_complete_pipeline(
    db_config,
    symbol='BTCUSDT',
    days_back=30,
    target_horizon=48,  # 4 hours
    n_features=50
):
    """
    Complete end-to-end pipeline from Phase 1 to Phase 3
    """

    print("="*70)
    print("COMPLETE ML TRADING PIPELINE")
    print("Phase 1 → Phase 2 → Phase 3")
    print("="*70)

    # ========== PHASE 1: FETCH DATA ==========
    print("\n[PHASE 1] Fetching data from Phase 1 database...")

    connector = Phase1DataConnector(**db_config)
    connector.connect()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    data = connector.fetch_all_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

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

    print(f"✓ Engineered {len(features_df.columns)} features")

    # Create targets
    target_engineer = TargetEngineer()

    # Classification target
    df_with_class = target_engineer.create_classification_target(
        features_df.reset_index(),
        horizon=target_horizon,
        threshold=0.005,  # 0.5%
        n_classes=3
    )

    # Regression target
    df_with_reg = target_engineer.create_regression_target(
        features_df.reset_index(),
        horizons=[target_horizon]
    )

    # Combine
    df_final = df_with_class.copy()
    df_final['target_return'] = df_with_reg[f'target_return_{target_horizon*5/60:.0f}h']

    # Get feature columns
    feature_cols = engineer.get_feature_names(features_df)

    # Split data
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

    # Feature selection
    print("\n[PHASE 2] Selecting best features...")
    X_train_selected, report = select_features_combined(
        X_train, y_train_class,
        n_features=n_features,
        correlation_threshold=0.9,
        variance_threshold=0.001
    )

    selected_features = X_train_selected.columns.tolist()
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    # ========== PHASE 3: MODEL TRAINING ==========
    print("\n[PHASE 3] Training ML models...")

    pipeline = MLTrainingPipeline()
    pipeline.run_full_pipeline(
        X_train_selected, y_train_class, y_train_reg,
        X_val_selected, y_val_class, y_val_reg,
        X_test_selected, y_test_class, y_test_reg
    )

    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70)

    return pipeline, selected_features


if __name__ == '__main__':
    # Database configuration (update with your Phase 1 credentials)
    db_config = {
        'host': 'localhost',  # or Docker container name
        'port': 5432,
        'database': 'trading_db',
        'user': 'postgres',
        'password': 'your_password'
    }

    # Run complete pipeline
    pipeline, features = run_complete_pipeline(
        db_config=db_config,
        symbol='BTCUSDT',
        days_back=60,  # Use 60 days of data
        target_horizon=48,  # Predict 4 hours ahead
        n_features=50  # Select top 50 features
    )
```

---

## 🐳 Step 3: Docker Integration

### Option 1: Add to Existing Phase 1 Docker Compose

Add Phase 2 & 3 service to your Phase 1 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Existing Phase 1 services...
  postgres:
    image: timescale/timescaledb:latest-pg14
    # ... existing config ...

  data_collector:
    # ... existing Phase 1 collector ...

  # NEW: Phase 2 & 3 ML Service
  ml_service:
    build:
      context: ./p2_mlFeature
      dockerfile: Dockerfile
    depends_on:
      - postgres
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=trading_db
      - DB_USER=postgres
      - DB_PASSWORD=your_password
    volumes:
      - ./p2_mlFeature:/app
      - ./models:/app/models  # Persist trained models
    command: python run_full_pipeline.py
```

### Option 2: Standalone Docker Container

Create `Dockerfile` in p2_mlFeature:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Phase 3
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Run pipeline
CMD ["python", "run_full_pipeline.py"]
```

Build and run:

```bash
cd p2_mlFeature
docker build -t ml-trading:phase2-3 .
docker run --network phase1_default ml-trading:phase2-3
```

---

## 📊 Step 4: Usage Examples

### Example 1: Train models with last 30 days of data

```python
from data_integration.phase1_connector import Phase1DataConnector
from features import FeatureEngineer, TargetEngineer
from models.training_pipeline import MLTrainingPipeline

# Connect to Phase 1 DB
connector = Phase1DataConnector(host='localhost', password='your_password')
connector.connect()

# Fetch data
data = connector.fetch_all_data(symbol='BTCUSDT', days_back=30)

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_all_features(**data)

# Create targets and train models
# ... (see run_full_pipeline.py for complete example)
```

### Example 2: Use trained models for live prediction

```python
import joblib
from features import FeatureEngineer

# Load trained ensemble model
ensemble = joblib.load('ensemble_model.pkl')

# Fetch latest data from Phase 1
connector = Phase1DataConnector(...)
connector.connect()
latest_data = connector.fetch_all_data(symbol='BTCUSDT', hours_back=24)

# Engineer features
engineer = FeatureEngineer()
features = engineer.engineer_all_features(**latest_data)

# Get latest features (last row)
X_latest = features[selected_features].iloc[[-1]]

# Make prediction
prediction = ensemble.get_trading_decision(X_latest)

print(f"Signal: {prediction['signal'][0]}")  # 0=SHORT, 1=NEUTRAL, 2=LONG
print(f"Confidence: {prediction['confidence'][0]:.2%}")
print(f"Target Return: {prediction['target'][0]:.2%}")
```

---

## 🔧 Configuration

### Database Connection

Update `.env` file:

```bash
# Phase 1 Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=postgres
DB_PASSWORD=your_password

# Redis (for feature store)
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Training
SYMBOL=BTCUSDT
TRAINING_DAYS=60
TARGET_HORIZON=48
N_FEATURES=50
```

---

## 📈 Expected Workflow

1. **Phase 1 (Running)**: Continuously collects market data → PostgreSQL
2. **Phase 2 (Manual/Scheduled)**:
   - Fetch data from Phase 1 DB
   - Engineer features
   - Store in feature store
3. **Phase 3 (Manual/Scheduled)**:
   - Train models on engineered features
   - Save models to disk
   - Evaluate performance
4. **Phase 4 (Live Trading)**:
   - Fetch latest data
   - Engineer features
   - Load trained models
   - Make predictions
   - Execute trades

---

## 🚀 Quick Start

```bash
# 1. Ensure Phase 1 is running
cd p1_dataCollection
docker-compose up -d

# 2. Install Phase 2 dependencies
cd ../p2_mlFeature
pip install -r requirements.txt

# 3. Create database connector
# Edit data_integration/phase1_connector.py with your DB credentials

# 4. Run complete pipeline
python run_full_pipeline.py

# 5. Models will be saved to ./models/
```

---

## 📝 Next Steps

After completing this integration:
1. ✅ Automated daily retraining
2. ✅ Model performance monitoring
3. ✅ Live trading integration
4. ✅ Risk management system
5. ✅ Backtesting framework

---

This integration guide connects all three phases into a complete, production-ready ML trading system!
