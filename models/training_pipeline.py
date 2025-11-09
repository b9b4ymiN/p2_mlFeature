"""
Complete ML Training Pipeline

Integrates all Phase 3 models into a single training workflow
"""

import joblib
import pandas as pd
import numpy as np
from .classifiers import XGBoostEntryPredictor, LightGBMEntryPredictor, CatBoostEntryPredictor
from .regressors import XGBoostPricePredictor, NeuralNetRegressor
from .lstm_forecaster import LSTMForecaster
from .ensemble import EnsembleModel
from .validation import WalkForwardValidator, ModelInterpreter
import warnings
warnings.filterwarnings('ignore')


class MLTrainingPipeline:
    """
    Complete ML training pipeline for Phase 3
    """

    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.results = {}

    def run_full_pipeline(
        self,
        X_train, y_train_class, y_train_reg,
        X_val, y_val_class, y_val_reg,
        X_test, y_test_class, y_test_reg,
        skip_lstm=False,
        skip_catboost=False
    ):
        """
        Run complete training pipeline

        Args:
            X_train, y_train_class, y_train_reg: Training data
            X_val, y_val_class, y_val_reg: Validation data
            X_test, y_test_class, y_test_reg: Test data
            skip_lstm: Skip LSTM training (requires torch)
            skip_catboost: Skip CatBoost training
        """

        print("="*70)
        print("PHASE 3: ML TRAINING PIPELINE")
        print("="*70)

        # Step 1: Train classifiers
        print("\n[1/6] Training Classification Models...")
        self.train_classifiers(X_train, y_train_class, X_val, y_val_class, skip_catboost)

        # Step 2: Train regressors
        print("\n[2/6] Training Regression Models...")
        self.train_regressors(X_train, y_train_reg, X_val, y_val_reg)

        # Step 3: Train LSTM (optional)
        if not skip_lstm:
            print("\n[3/6] Training LSTM Models...")
            try:
                self.train_lstm(X_train, y_train_reg, X_val, y_val_reg)
            except ImportError as e:
                print(f"⚠ Skipping LSTM: {e}")
        else:
            print("\n[3/6] Skipping LSTM (skip_lstm=True)")

        # Step 4: Build ensemble
        print("\n[4/6] Building Ensemble Model...")
        self.build_ensemble(X_train, y_train_class, y_train_reg)

        # Step 5: Evaluate on test set
        print("\n[5/6] Final Evaluation on Test Set...")
        self.evaluate_all_models(X_test, y_test_class, y_test_reg)

        # Step 6: SHAP Analysis
        print("\n[6/6] SHAP Interpretability Analysis...")
        try:
            self.analyze_interpretability(X_test)
        except Exception as e:
            print(f"⚠ Skipping SHAP analysis: {e}")

        # Save models
        print("\nSaving Models...")
        self.save_models()

        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)

        self.print_summary()

        return self.results

    def train_classifiers(self, X_train, y_train, X_val, y_val, skip_catboost=False):
        """Train all classification models"""

        # XGBoost
        print("\n→ XGBoost Classifier")
        xgb_clf = XGBoostEntryPredictor()
        xgb_clf.train(X_train, y_train, X_val, y_val)
        self.models['xgb_classifier'] = xgb_clf

        # LightGBM
        print("\n→ LightGBM Classifier")
        lgb_clf = LightGBMEntryPredictor()
        lgb_clf.train(X_train, y_train, X_val, y_val)
        self.models['lgb_classifier'] = lgb_clf

        # CatBoost (optional)
        if not skip_catboost:
            try:
                print("\n→ CatBoost Classifier")
                cat_clf = CatBoostEntryPredictor()
                cat_clf.train(X_train, y_train, X_val, y_val)
                self.models['cat_classifier'] = cat_clf
            except ImportError:
                print("⚠ CatBoost not installed, skipping")

    def train_regressors(self, X_train, y_train, X_val, y_val):
        """Train all regression models"""

        # XGBoost Regressor
        print("\n→ XGBoost Regressor")
        xgb_reg = XGBoostPricePredictor()
        xgb_reg.train(X_train, y_train, X_val, y_val)
        self.models['xgb_regressor'] = xgb_reg

        # Neural Network (optional)
        try:
            print("\n→ Neural Network Regressor")
            nn_reg = NeuralNetRegressor(input_dim=X_train.shape[1])
            nn_reg.train(X_train, y_train, X_val, y_val, epochs=50)
            self.models['nn_regressor'] = nn_reg
        except ImportError:
            print("⚠ PyTorch not installed, skipping Neural Network")

    def train_lstm(self, X_train, y_train, X_val, y_val):
        """Train LSTM models"""

        print("\n→ LSTM Forecaster")
        lstm = LSTMForecaster(input_dim=X_train.shape[1], lookback=50)
        lstm.train(X_train, y_train, X_val, y_val, epochs=30)
        self.models['lstm'] = lstm

    def build_ensemble(self, X_train, y_train_class, y_train_reg):
        """Build ensemble model"""

        # Prepare base classifiers
        base_classifiers = []
        if 'xgb_classifier' in self.models:
            base_classifiers.append(('xgb', self.models['xgb_classifier'].model))
        if 'lgb_classifier' in self.models:
            base_classifiers.append(('lgb', self.models['lgb_classifier'].model))
        if 'cat_classifier' in self.models:
            base_classifiers.append(('cat', self.models['cat_classifier'].model))

        # Prepare base regressors
        base_regressors = []
        if 'xgb_regressor' in self.models:
            base_regressors.append(('xgb', self.models['xgb_regressor'].model))

        # Build and train ensemble
        if base_classifiers and base_regressors:
            self.ensemble = EnsembleModel()
            self.ensemble.build_classifier(base_classifiers)
            self.ensemble.build_regressor(base_regressors)

            print("\nTraining ensemble models...")
            self.ensemble.train_classifier(X_train, y_train_class)
            self.ensemble.train_regressor(X_train, y_train_reg)

            print("✓ Ensemble models trained")

    def evaluate_all_models(self, X_test, y_test_class, y_test_reg):
        """Evaluate all models on test set"""

        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)

        results = {}

        # Classification models
        for name in ['xgb_classifier', 'lgb_classifier', 'cat_classifier']:
            if name in self.models:
                metrics = self.models[name].evaluate(X_test, y_test_class)
                results[name] = metrics

        # Regression models
        for name in ['xgb_regressor', 'nn_regressor']:
            if name in self.models:
                metrics = self.models[name].evaluate(X_test, y_test_reg)
                results[name] = metrics

        # LSTM
        if 'lstm' in self.models:
            metrics = self.models['lstm'].evaluate(X_test, y_test_reg)
            results['lstm'] = metrics

        # Ensemble
        if self.ensemble:
            clf_metrics = self.ensemble.evaluate_classifier(X_test, y_test_class)
            results['ensemble_classifier'] = clf_metrics

        self.results = results

        return results

    def analyze_interpretability(self, X_test):
        """Perform SHAP interpretability analysis"""

        if 'xgb_classifier' in self.models:
            print("\nPerforming SHAP analysis on XGBoost Classifier...")

            interpreter = ModelInterpreter(
                self.models['xgb_classifier'],
                X_test.sample(min(1000, len(X_test)))
            )

            importance = interpreter.explain_predictions(
                X_test.sample(min(500, len(X_test))),
                max_display=20
            )

            self.results['shap_importance'] = importance

    def save_models(self):
        """Save all trained models"""

        # Save sklearn-compatible models
        if 'xgb_classifier' in self.models:
            joblib.dump(self.models['xgb_classifier'].model, 'models/xgb_classifier.pkl')
            print("✓ Saved xgb_classifier.pkl")

        if 'lgb_classifier' in self.models:
            joblib.dump(self.models['lgb_classifier'].model, 'models/lgb_classifier.pkl')
            print("✓ Saved lgb_classifier.pkl")

        if 'xgb_regressor' in self.models:
            joblib.dump(self.models['xgb_regressor'].model, 'models/xgb_regressor.pkl')
            print("✓ Saved xgb_regressor.pkl")

        if self.ensemble:
            joblib.dump(self.ensemble, 'models/ensemble_model.pkl')
            print("✓ Saved ensemble_model.pkl")

        # PyTorch models are saved during training
        print("✓ PyTorch models saved during training")

    def print_summary(self):
        """Print training summary"""

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)

        print("\nModels Trained:")
        for name in self.models.keys():
            print(f"  ✓ {name}")

        if self.ensemble:
            print(f"  ✓ ensemble (stacking)")

        print("\nTest Set Performance:")
        for name, metrics in self.results.items():
            if 'accuracy' in metrics:
                print(f"  {name:30s} Accuracy: {metrics['accuracy']:.4f}")
                if 'directional_accuracy' in metrics:
                    print(f"  {name:30s} Directional: {metrics['directional_accuracy']:.4f}")

        print("="*70)
