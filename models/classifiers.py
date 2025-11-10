"""
Classification Models for Entry Signal Prediction

Models:
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class XGBoostEntryPredictor:
    """
    XGBoost model for entry signal classification

    Predicts: 0=SHORT, 1=NEUTRAL, 2=LONG
    """

    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.model = None
        self.feature_importance = None

    def _default_params(self):
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }

    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Install with: pip install xgboost")

        print("Training XGBoost Classifier...")

        self.model = xgb.XGBClassifier(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"✓ XGBoost Classifier trained")

        return self.model

    def predict(self, X):
        """Predict class (0, 1, 2)"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Classification metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Trading-specific metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Directional accuracy (ignore NEUTRAL)
        mask = (y_test != 1) & (y_pred != 1)
        directional_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        print("\n" + "="*60)
        print("XGBoost Classifier - Evaluation Results")
        print("="*60)
        print(f"Overall Accuracy:        {accuracy:.4f}")
        print(f"Directional Accuracy:    {directional_acc:.4f}")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        print("="*60)

        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_acc,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'feature_importance': self.feature_importance
        }

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Hyperparameter optimization with Optuna"""
        try:
            import optuna
            import xgboost as xgb
        except ImportError:
            print("⚠ optuna not installed. Skipping hyperparameter optimization.")
            return self.params

        print(f"Optimizing hyperparameters ({n_trials} trials)...")

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
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                'random_state': 42,
                'tree_method': 'hist',
                'eval_metric': 'mlogloss'
            }

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n✓ Best accuracy: {study.best_value:.4f}")
        print(f"✓ Best params: {study.best_params}")

        self.params.update(study.best_params)

        return study.best_params


class LightGBMEntryPredictor:
    """
    LightGBM model for entry signal classification
    """

    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.model = None

    def _default_params(self):
        return {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 300,
            'random_state': 42
        }

    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Install with: pip install lightgbm")

        print("Training LightGBM Classifier...")

        self.model = lgb.LGBMClassifier(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        print(f"✓ LightGBM Classifier trained")

        return self.model

    def predict(self, X):
        """Predict class"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        mask = (y_test != 1) & (y_pred != 1)
        directional_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        print("\n" + "="*60)
        print("LightGBM Classifier - Evaluation Results")
        print("="*60)
        print(f"Overall Accuracy:        {accuracy:.4f}")
        print(f"Directional Accuracy:    {directional_acc:.4f}")
        print("="*60)

        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_acc
        }


class CatBoostEntryPredictor:
    """
    CatBoost model for entry signal classification
    """

    def __init__(self, params=None):
        self.params = params or self._default_params()
        self.model = None

    def _default_params(self):
        return {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50
        }

    def train(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost not installed. Install with: pip install catboost")

        print("Training CatBoost Classifier...")

        self.model = CatBoostClassifier(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        print(f"✓ CatBoost Classifier trained")

        return self.model

    def predict(self, X):
        return self.model.predict(X).flatten()

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        mask = (y_test != 1) & (y_pred != 1)
        directional_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        print("\n" + "="*60)
        print("CatBoost Classifier - Evaluation Results")
        print("="*60)
        print(f"Overall Accuracy:        {accuracy:.4f}")
        print(f"Directional Accuracy:    {directional_acc:.4f}")
        print("="*60)

        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_acc
        }
