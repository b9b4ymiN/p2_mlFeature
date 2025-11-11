"""
Performance Reporting & Visualization

Generate comprehensive reports and visualizations for model performance:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves and AUC
- Regression metrics (MSE, RMSE, MAE, R²)
- Feature importance plots
- SHAP summary plots
- Performance comparison tables
- HTML reports

Usage:
    >>> from utils.reporting import ModelPerformanceReporter
    >>> reporter = ModelPerformanceReporter()
    >>> reporter.generate_classification_report(y_true, y_pred, y_proba)
    >>> reporter.save_html_report('model_report.html')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelPerformanceReporter:
    """
    Comprehensive model performance reporter
    """

    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize reporter

        Args:
            output_dir: Directory to save reports and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.plots = {}

    # ========================================
    # Classification Reports
    # ========================================

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: List[str] = None,
        model_name: str = 'Model'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: Names of classes
            model_name: Name of the model

        Returns:
            Dict with all metrics
        """
        if class_names is None:
            class_names = ['SHORT', 'NEUTRAL', 'LONG']

        print(f"\n{'='*70}")
        print(f"CLASSIFICATION REPORT: {model_name}")
        print('='*70)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Directional accuracy (excluding NEUTRAL class 1)
        mask = (y_true != 1) & (y_pred != 1)
        if mask.sum() > 0:
            directional_acc = accuracy_score(y_true[mask], y_pred[mask])
        else:
            directional_acc = 0.0

        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # ROC AUC (if probabilities provided)
        if y_proba is not None and len(class_names) == 3:
            try:
                roc_auc = roc_auc_score(
                    y_true, y_proba,
                    multi_class='ovr',
                    average='weighted'
                )
            except:
                roc_auc = None
        else:
            roc_auc = None

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'directional_accuracy': directional_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'class_report': class_report
        }

        # Print summary
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:              {accuracy:.4f}")
        print(f"   Directional Accuracy:  {directional_acc:.4f}")
        print(f"   Weighted Precision:    {precision:.4f}")
        print(f"   Weighted Recall:       {recall:.4f}")
        print(f"   Weighted F1:           {f1:.4f}")
        if roc_auc:
            print(f"   ROC AUC:               {roc_auc:.4f}")

        print(f"\n📋 Per-Class Metrics:")
        print(pd.DataFrame(class_report).transpose())

        print(f"\n🔢 Confusion Matrix:")
        conf_df = pd.DataFrame(
            conf_matrix,
            index=class_names,
            columns=class_names
        )
        print(conf_df)

        # Store metrics
        self.metrics[model_name] = metrics

        # Generate plots
        self._plot_confusion_matrix(conf_matrix, class_names, model_name)

        if y_proba is not None:
            self._plot_roc_curves(y_true, y_proba, class_names, model_name)

        return metrics

    def _plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: List[str],
        model_name: str
    ):
        """Plot confusion matrix"""

        plt.figure(figsize=(10, 8))

        # Normalize
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            conf_matrix_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}
        )

        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'{model_name}_confusion'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        model_name: str
    ):
        """Plot ROC curves for each class"""

        from sklearn.preprocessing import label_binarize

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})',
                linewidth=2
            )

        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {model_name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f'roc_curves_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'{model_name}_roc'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    # ========================================
    # Regression Reports
    # ========================================

    def generate_regression_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regression report
        """

        print(f"\n{'='*70}")
        print(f"REGRESSION REPORT: {model_name}")
        print('='*70)

        # Metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Directional accuracy
        directional_acc = (np.sign(y_pred) == np.sign(y_true)).mean()

        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_acc,
            'correlation': correlation
        }

        # Print summary
        print(f"\n📊 Regression Metrics:")
        print(f"   MSE:                   {mse:.6f}")
        print(f"   RMSE:                  {rmse:.6f}")
        print(f"   MAE:                   {mae:.6f}")
        print(f"   R²:                    {r2:.4f}")
        print(f"   Directional Accuracy:  {directional_acc:.4f}")
        print(f"   Correlation:           {correlation:.4f}")

        # Store metrics
        self.metrics[model_name] = metrics

        # Generate plots
        self._plot_regression_scatter(y_true, y_pred, model_name)
        self._plot_residuals(y_true, y_pred, model_name)

        return metrics

    def _plot_regression_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot actual vs predicted scatter plot"""

        plt.figure(figsize=(10, 8))

        plt.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted: {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f'regression_scatter_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'{model_name}_scatter'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    def _plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot residuals"""

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        axes[0].grid(alpha=0.3)

        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(alpha=0.3)

        plt.suptitle(f'Residual Analysis: {model_name}')
        plt.tight_layout()

        # Save
        filename = f'residuals_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'{model_name}_residuals'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    # ========================================
    # Feature Importance
    # ========================================

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        model_name: str = 'Model',
        top_n: int = 20
    ):
        """
        Plot feature importance

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            model_name: Model name
            top_n: Number of top features to show
        """

        # Sort and select top N
        top_features = feature_importance.nlargest(top_n, 'importance')

        plt.figure(figsize=(12, 8))

        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features: {model_name}')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'{model_name}_features'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    # ========================================
    # Model Comparison
    # ========================================

    def compare_models(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        metric_name: str = 'accuracy'
    ):
        """
        Compare multiple models on a specific metric

        Args:
            model_metrics: Dict of {model_name: metrics_dict}
            metric_name: Metric to compare
        """

        # Extract metrics
        models = []
        values = []

        for model_name, metrics in model_metrics.items():
            if metric_name in metrics:
                models.append(model_name)
                values.append(metrics[metric_name])

        if not models:
            print(f"⚠️  No models have metric '{metric_name}'")
            return

        # Create bar plot
        plt.figure(figsize=(12, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = plt.bar(models, values, color=colors)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{value:.4f}',
                ha='center',
                va='bottom'
            )

        plt.xlabel('Model')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric_name.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save
        filename = f'model_comparison_{metric_name}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.plots[f'comparison_{metric_name}'] = str(filepath)
        print(f"✓ Saved: {filepath}")

    def generate_comparison_table(
        self,
        model_metrics: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Generate comparison table for all models

        Returns:
            DataFrame with model comparison
        """

        comparison_df = pd.DataFrame(model_metrics).transpose()

        # Sort by accuracy (if available)
        if 'accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)

        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        print(comparison_df.to_string())
        print("="*100)

        # Save to CSV
        csv_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(csv_path)
        print(f"\n✓ Saved comparison table: {csv_path}")

        return comparison_df

    # ========================================
    # HTML Report Generation
    # ========================================

    def save_html_report(self, filename: str = 'model_report.html'):
        """
        Generate HTML report with all metrics and plots

        Args:
            filename: Output filename
        """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Model Performance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #666;
                    margin-top: 30px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-left: 4px solid #4CAF50;
                }}
                .metric-name {{
                    font-size: 14px;
                    color: #666;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 12px;
                    text-align: right;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 ML Model Performance Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>📊 Model Metrics</h2>
        """

        # Add metrics for each model
        for model_name, metrics in self.metrics.items():
            html_content += f"""
                <h3>{model_name}</h3>
                <div>
            """

            # Display key metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    html_content += f"""
                        <div class="metric">
                            <div class="metric-name">{key.replace('_', ' ').title()}</div>
                            <div class="metric-value">{value:.4f}</div>
                        </div>
                    """

            html_content += "</div>"

        # Add plots
        html_content += "<h2>📈 Visualizations</h2>"

        for plot_name, plot_path in self.plots.items():
            html_content += f"""
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{plot_path}" alt="{plot_name}">
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Save HTML
        html_path = self.output_dir / filename
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"\n✓ HTML report saved: {html_path}")

        return str(html_path)


# ========================================
# Usage Example
# ========================================

if __name__ == "__main__":
    print("Example: Performance Reporting\n")

    # Create mock data
    np.random.seed(42)
    y_true_class = np.random.randint(0, 3, 1000)
    y_pred_class = np.random.randint(0, 3, 1000)
    y_proba = np.random.rand(1000, 3)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_true_reg = np.random.randn(1000)
    y_pred_reg = y_true_reg + np.random.randn(1000) * 0.5

    # Create reporter
    reporter = ModelPerformanceReporter()

    # Classification report
    reporter.generate_classification_report(
        y_true_class, y_pred_class, y_proba,
        model_name='XGBoost Classifier'
    )

    # Regression report
    reporter.generate_regression_report(
        y_true_reg, y_pred_reg,
        model_name='XGBoost Regressor'
    )

    # Generate HTML report
    reporter.save_html_report()

    print("\n✓ Example complete!")
