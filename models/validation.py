"""
Walk-Forward Validation and SHAP Interpretability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def validate(self, model_class, model_params, X, y, task_type='classification'):
        """
        Perform walk-forward validation

        Returns performance metrics for each fold
        """
        fold_metrics = []

        print("\n" + "="*60)
        print(f"WALK-FORWARD VALIDATION ({self.n_splits} folds)")
        print("="*60)

        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            print(f"\nFold {fold + 1}/{self.n_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Split train into train/val
            val_split = int(len(X_train) * 0.8)
            X_train_fold = X_train.iloc[:val_split]
            y_train_fold = y_train.iloc[:val_split]
            X_val_fold = X_train.iloc[val_split:]
            y_val_fold = y_train.iloc[val_split:]

            # Initialize and train model
            model = model_class(params=model_params)
            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            metrics['fold'] = fold + 1
            metrics['train_size'] = len(train_idx)
            metrics['test_size'] = len(test_idx)

            fold_metrics.append(metrics)

        # Aggregate results
        results_df = pd.DataFrame(fold_metrics)

        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*60)

        if 'accuracy' in results_df.columns:
            print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
            if 'directional_accuracy' in results_df.columns:
                print(f"Mean Directional Acc: {results_df['directional_accuracy'].mean():.4f} ± {results_df['directional_accuracy'].std():.4f}")

        print("="*60)

        return results_df


class ModelInterpreter:
    """
    Analyze and visualize model decisions using SHAP
    """

    def __init__(self, model, X_train):
        """
        Initialize SHAP explainer

        Args:
            model: Trained model
            X_train: Training data for SHAP background
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap not installed. Install with: pip install shap")

        # Get the actual model object
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        print("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(actual_model)
        self.X_train = X_train

        print("✓ SHAP explainer ready")

    def explain_predictions(self, X_test, max_display=20, save_plots=True):
        """
        Generate SHAP values and feature importance

        Args:
            X_test: Test data to explain
            max_display: Number of features to display
            save_plots: Whether to save plots

        Returns:
            DataFrame with feature importance
        """
        import shap

        print("Calculating SHAP values...")

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_test)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values_avg = np.abs(shap_values).mean(axis=0)
        else:
            shap_values_avg = shap_values

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values_avg).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\nTop Features by SHAP Importance:")
        print(feature_importance.head(max_display))

        # Save plots if requested
        if save_plots:
            try:
                import matplotlib.pyplot as plt

                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values,
                    X_test,
                    max_display=max_display,
                    show=False
                )
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300)
                plt.close()

                print("✓ SHAP plots saved to shap_summary.png")

            except Exception as e:
                print(f"⚠ Could not save plots: {e}")

        return feature_importance
