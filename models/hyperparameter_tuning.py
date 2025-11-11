"""
Hyperparameter Tuning with Optuna

Automated hyperparameter optimization for all models:
- XGBoost Classifier & Regressor
- LightGBM Classifier
- CatBoost Classifier
- Neural Network
- LSTM

Usage:
    >>> from models.hyperparameter_tuning import optimize_xgboost_classifier
    >>> best_params = optimize_xgboost_classifier(X_train, y_train, X_val, y_val, n_trials=100)
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ========================================
# XGBoost Classifier Optimization
# ========================================

def optimize_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize XGBoost classifier hyperparameters using Optuna

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
        timeout: Timeout in seconds

    Returns:
        Dict with best_params and best_score
    """

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Also compute directional accuracy (excluding NEUTRAL)
        mask = (y_val != 1) & (y_pred != 1)
        if mask.sum() > 0:
            directional_acc = accuracy_score(y_val[mask], y_pred[mask])
        else:
            directional_acc = 0

        # Weighted score (70% accuracy, 30% directional)
        score = 0.7 * accuracy + 0.3 * directional_acc

        return score

    print("="*70)
    print("OPTIMIZING XGBOOST CLASSIFIER")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


# ========================================
# XGBoost Regressor Optimization
# ========================================

def optimize_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize XGBoost regressor hyperparameters
    """

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'random_state': 42,
            'tree_method': 'hist'
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )

        y_pred = model.predict(X_val)

        # MSE
        mse = mean_squared_error(y_val, y_pred)

        # Directional accuracy
        direction_correct = (np.sign(y_pred) == np.sign(y_val)).mean()

        # R2 score
        r2 = r2_score(y_val, y_pred)

        # Combined score (minimize MSE, maximize directional accuracy)
        # Lower MSE is better, higher directional is better
        score = direction_correct - (mse * 10)  # Scale MSE to balance

        return score

    print("="*70)
    print("OPTIMIZING XGBOOST REGRESSOR")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


# ========================================
# LightGBM Classifier Optimization
# ========================================

def optimize_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize LightGBM classifier hyperparameters
    """

    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 1.0),
            'verbose': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500)
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0)
            ]
        )

        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_val, y_pred)

        # Directional accuracy
        mask = (y_val != 1) & (y_pred != 1)
        if mask.sum() > 0:
            directional_acc = accuracy_score(y_val[mask], y_pred[mask])
        else:
            directional_acc = 0

        score = 0.7 * accuracy + 0.3 * directional_acc

        return score

    print("="*70)
    print("OPTIMIZING LIGHTGBM CLASSIFIER")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


# ========================================
# CatBoost Classifier Optimization
# ========================================

def optimize_catboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize CatBoost classifier hyperparameters
    """

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("⚠️  CatBoost not installed. Skipping optimization.")
        return {}

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 30
        }

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Directional accuracy
        mask = (y_val != 1) & (y_pred != 1)
        if mask.sum() > 0:
            directional_acc = accuracy_score(y_val[mask], y_pred[mask])
        else:
            directional_acc = 0

        score = 0.7 * accuracy + 0.3 * directional_acc

        return score

    print("="*70)
    print("OPTIMIZING CATBOOST CLASSIFIER")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


# ========================================
# Neural Network Optimization
# ========================================

def optimize_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = None
) -> Dict[str, Any]:
    """
    Optimize Neural Network architecture hyperparameters
    """

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("⚠️  PyTorch not installed. Skipping NN optimization.")
        return {}

    def objective(trial):
        # Suggest architecture
        n_layers = trial.suggest_int('n_layers', 2, 4)
        hidden_dims = []
        for i in range(n_layers):
            dim = trial.suggest_int(f'hidden_dim_{i}', 32, 256)
            hidden_dims.append(dim)

        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

        # Build model
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super(SimpleNN, self).__init__()

                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim

                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNN(X_train.shape[1]).to(device)

        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.values).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train for limited epochs
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = 50

        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        return -best_val_loss  # Negative because we want to minimize loss

    print("="*70)
    print("OPTIMIZING NEURAL NETWORK")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',  # Maximize negative loss
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


# ========================================
# Convenience Function: Optimize All Models
# ========================================

def optimize_all_models(
    X_train: pd.DataFrame,
    y_train_class: pd.Series,
    y_train_reg: pd.Series,
    X_val: pd.DataFrame,
    y_val_class: pd.Series,
    y_val_reg: pd.Series,
    n_trials: int = 100,
    timeout_per_model: int = None
) -> Dict[str, Dict]:
    """
    Optimize hyperparameters for all models

    Args:
        X_train, y_train_class, y_train_reg: Training data
        X_val, y_val_class, y_val_reg: Validation data
        n_trials: Number of trials per model
        timeout_per_model: Timeout per model in seconds

    Returns:
        Dict of {model_name: optimization_results}
    """

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION FOR ALL MODELS")
    print("="*70)

    results = {}

    # Classification models
    print("\n[1/5] XGBoost Classifier...")
    results['xgb_classifier'] = optimize_xgboost_classifier(
        X_train, y_train_class, X_val, y_val_class,
        n_trials=n_trials, timeout=timeout_per_model
    )

    print("\n[2/5] LightGBM Classifier...")
    results['lgb_classifier'] = optimize_lightgbm_classifier(
        X_train, y_train_class, X_val, y_val_class,
        n_trials=n_trials, timeout=timeout_per_model
    )

    print("\n[3/5] CatBoost Classifier...")
    results['cat_classifier'] = optimize_catboost_classifier(
        X_train, y_train_class, X_val, y_val_class,
        n_trials=n_trials, timeout=timeout_per_model
    )

    # Regression models
    print("\n[4/5] XGBoost Regressor...")
    results['xgb_regressor'] = optimize_xgboost_regressor(
        X_train, y_train_reg, X_val, y_val_reg,
        n_trials=n_trials, timeout=timeout_per_model
    )

    print("\n[5/5] Neural Network...")
    results['neural_network'] = optimize_neural_network(
        X_train, y_train_reg, X_val, y_val_reg,
        n_trials=min(n_trials, 50),  # NN takes longer
        timeout=timeout_per_model
    )

    print("\n" + "="*70)
    print("✅ ALL MODELS OPTIMIZED!")
    print("="*70)

    # Print summary
    print("\n📊 Optimization Summary:")
    for model_name, result in results.items():
        if result:
            print(f"\n{model_name}:")
            print(f"  Best score: {result.get('best_score', 'N/A')}")

    return results


# ========================================
# Usage Example
# ========================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("Example: Hyperparameter Optimization\n")

    # Create mock data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(1000, 50))
    y_train_class = pd.Series(np.random.randint(0, 3, 1000))
    y_train_reg = pd.Series(np.random.randn(1000))

    X_val = pd.DataFrame(np.random.randn(200, 50))
    y_val_class = pd.Series(np.random.randint(0, 3, 200))
    y_val_reg = pd.Series(np.random.randn(200))

    # Optimize single model
    result = optimize_xgboost_classifier(
        X_train, y_train_class,
        X_val, y_val_class,
        n_trials=10  # Small number for demo
    )

    print(f"\n✓ Best params: {result['best_params']}")
    print(f"✓ Best score: {result['best_score']:.4f}")
