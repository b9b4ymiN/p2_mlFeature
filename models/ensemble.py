"""
Ensemble Meta-Model

Combines predictions from multiple base models using stacking
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models
    """

    def __init__(self, base_classifiers=None, base_regressors=None):
        """
        Initialize ensemble model

        Args:
            base_classifiers: List of (name, model) tuples for classification
            base_regressors: List of (name, model) tuples for regression
        """
        self.base_classifiers = base_classifiers or []
        self.base_regressors = base_regressors or []
        self.stacking_classifier = None
        self.stacking_regressor = None

    def build_classifier(self, base_classifiers):
        """Build stacking classifier"""
        print("Building ensemble classifier...")

        self.stacking_classifier = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )

        print("✓ Ensemble classifier built")

    def build_regressor(self, base_regressors):
        """Build stacking regressor"""
        print("Building ensemble regressor...")

        self.stacking_regressor = StackingRegressor(
            estimators=base_regressors,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=5,
            n_jobs=-1
        )

        print("✓ Ensemble regressor built")

    def train_classifier(self, X_train, y_train):
        """Train ensemble classifier"""
        print("Training ensemble classifier...")

        self.stacking_classifier.fit(X_train, y_train)

        print("✓ Ensemble classifier trained")

    def train_regressor(self, X_train, y_train):
        """Train ensemble regressor"""
        print("Training ensemble regressor...")

        self.stacking_regressor.fit(X_train, y_train)

        print("✓ Ensemble regressor trained")

    def predict_signal(self, X):
        """Predict entry signal (classification)"""
        return self.stacking_classifier.predict(X)

    def predict_signal_proba(self, X):
        """Predict signal probabilities"""
        return self.stacking_classifier.predict_proba(X)

    def predict_target(self, X):
        """Predict price target (regression)"""
        return self.stacking_regressor.predict(X)

    def get_trading_decision(self, X):
        """
        Get complete trading decision with confidence

        Returns:
            Dictionary with signal, confidence, target, and probabilities
        """
        signal_proba = self.predict_signal_proba(X)
        signal = np.argmax(signal_proba, axis=1)
        confidence = np.max(signal_proba, axis=1)
        target = self.predict_target(X)

        return {
            'signal': signal,
            'confidence': confidence,
            'target': target,
            'signal_proba': signal_proba
        }

    def evaluate_classifier(self, X_test, y_test):
        """Evaluate ensemble classifier"""
        y_pred = self.predict_signal(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        mask = (y_test != 1) & (y_pred != 1)
        directional_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else 0

        print("\n" + "="*60)
        print("Ensemble Classifier - Evaluation Results")
        print("="*60)
        print(f"Overall Accuracy:        {accuracy:.4f}")
        print(f"Directional Accuracy:    {directional_acc:.4f}")
        print("="*60)

        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_acc
        }
