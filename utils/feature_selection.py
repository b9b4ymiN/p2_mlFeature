"""
Feature Selection Methods

Implements multiple feature selection strategies:
1. Correlation-based filtering
2. Tree-based feature importance
3. SHAP values
4. Permutation importance
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = 'pearson'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features that are highly correlated with each other

    Args:
        df: DataFrame with features
        threshold: Correlation threshold (0-1)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Tuple of (filtered DataFrame, list of dropped columns)
    """
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method).abs()

    # Get upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    print(f"Dropping {len(to_drop)} highly correlated features (threshold={threshold})")

    return df.drop(columns=to_drop), to_drop


def select_top_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    task_type: str = 'classification',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top N features by Random Forest feature importance

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of top features to select
        task_type: 'classification' or 'regression'
        random_state: Random seed

    Returns:
        Tuple of (selected features DataFrame, importance DataFrame)
    """
    print(f"Training Random Forest to compute feature importances...")

    # Train Random Forest model
    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(X, y)

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Select top N features
    top_features = importances.head(n_features)['feature'].tolist()

    print(f"Selected top {n_features} features by importance")
    print(f"Top 10 features:\n{importances.head(10)}")

    return X[top_features], importances


def select_features_by_shap(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    task_type: str = 'classification',
    random_state: int = 42,
    max_samples: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select features by SHAP (SHapley Additive exPlanations) values

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of top features to select
        task_type: 'classification' or 'regression'
        random_state: Random seed
        max_samples: Maximum samples to use for SHAP calculation (for speed)

    Returns:
        Tuple of (selected features DataFrame, SHAP importance DataFrame)
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP not installed. Install with: pip install shap")
        print("Falling back to standard feature importance...")
        return select_top_features_by_importance(X, y, n_features, task_type, random_state)

    print(f"Computing SHAP values (this may take a while)...")

    # Sample data if too large
    if len(X) > max_samples:
        sample_idx = np.random.RandomState(random_state).choice(
            len(X), max_samples, replace=False
        )
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X
        y_sample = y

    # Train model
    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(X_sample, y_sample)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For classification, shap_values is a list (one per class)
    if isinstance(shap_values, list):
        # Use absolute mean across all classes
        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # For regression
        shap_importance = np.abs(shap_values).mean(axis=0)

    # Create importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)

    # Select top N features
    top_features = feature_importance.head(n_features)['feature'].tolist()

    print(f"Selected top {n_features} features by SHAP values")
    print(f"Top 10 features:\n{feature_importance.head(10)}")

    return X[top_features], feature_importance


def select_features_by_permutation(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    task_type: str = 'classification',
    random_state: int = 42,
    n_repeats: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select features by permutation importance

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of top features to select
        task_type: 'classification' or 'regression'
        random_state: Random seed
        n_repeats: Number of times to permute each feature

    Returns:
        Tuple of (selected features DataFrame, permutation importance DataFrame)
    """
    print(f"Computing permutation importance (n_repeats={n_repeats})...")

    # Train model
    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(X, y)

    # Calculate permutation importance
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Create importance DataFrame
    perm_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    # Select top N features
    top_features = perm_importance.head(n_features)['feature'].tolist()

    print(f"Selected top {n_features} features by permutation importance")
    print(f"Top 10 features:\n{perm_importance.head(10)}")

    return X[top_features], perm_importance


def select_features_by_variance(
    df: pd.DataFrame,
    threshold: float = 0.01
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove low-variance features

    Args:
        df: DataFrame with features
        threshold: Minimum variance threshold

    Returns:
        Tuple of (filtered DataFrame, list of dropped columns)
    """
    variances = df.var()
    to_drop = variances[variances < threshold].index.tolist()

    print(f"Dropping {len(to_drop)} low-variance features (threshold={threshold})")

    return df.drop(columns=to_drop), to_drop


def select_features_combined(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 50,
    task_type: str = 'classification',
    correlation_threshold: float = 0.9,
    variance_threshold: float = 0.01,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Combined feature selection pipeline

    Steps:
    1. Remove low-variance features
    2. Remove highly correlated features
    3. Select top N by importance

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features: Number of final features to select
        task_type: 'classification' or 'regression'
        correlation_threshold: Correlation threshold for removal
        variance_threshold: Variance threshold for removal
        random_state: Random seed

    Returns:
        Tuple of (selected features DataFrame, selection report dict)
    """
    report = {}
    report['initial_features'] = len(X.columns)

    print("=" * 60)
    print("COMBINED FEATURE SELECTION PIPELINE")
    print("=" * 60)

    # Step 1: Remove low-variance features
    print("\nStep 1: Removing low-variance features...")
    X_filtered, low_var_dropped = select_features_by_variance(X, variance_threshold)
    report['low_variance_dropped'] = len(low_var_dropped)
    report['after_variance_filter'] = len(X_filtered.columns)

    # Step 2: Remove highly correlated features
    print("\nStep 2: Removing highly correlated features...")
    X_filtered, corr_dropped = remove_highly_correlated_features(
        X_filtered, correlation_threshold
    )
    report['correlation_dropped'] = len(corr_dropped)
    report['after_correlation_filter'] = len(X_filtered.columns)

    # Step 3: Select top N by importance
    print("\nStep 3: Selecting top features by importance...")
    X_selected, importance_df = select_top_features_by_importance(
        X_filtered, y, n_features, task_type, random_state
    )
    report['final_features'] = len(X_selected.columns)
    report['importance_scores'] = importance_df

    print("\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)
    print(f"Initial features:             {report['initial_features']}")
    print(f"After variance filter:        {report['after_variance_filter']} ({report['low_variance_dropped']} dropped)")
    print(f"After correlation filter:     {report['after_correlation_filter']} ({report['correlation_dropped']} dropped)")
    print(f"Final selected features:      {report['final_features']}")
    print("=" * 60)

    return X_selected, report


def get_feature_importance_summary(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Get comprehensive feature importance from multiple methods

    Args:
        X: Feature DataFrame
        y: Target variable
        task_type: 'classification' or 'regression'
        random_state: Random seed

    Returns:
        DataFrame with importance scores from multiple methods
    """
    print("Computing feature importance from multiple methods...")

    # Method 1: Tree-based importance
    _, tree_importance = select_top_features_by_importance(
        X, y, len(X.columns), task_type, random_state
    )
    tree_importance = tree_importance.set_index('feature')['importance']

    # Method 2: Permutation importance
    _, perm_importance = select_features_by_permutation(
        X, y, len(X.columns), task_type, random_state
    )
    perm_importance = perm_importance.set_index('feature')['importance_mean']

    # Combine
    summary = pd.DataFrame({
        'tree_importance': tree_importance,
        'permutation_importance': perm_importance
    })

    # Calculate average rank
    summary['tree_rank'] = summary['tree_importance'].rank(ascending=False)
    summary['perm_rank'] = summary['permutation_importance'].rank(ascending=False)
    summary['avg_rank'] = (summary['tree_rank'] + summary['perm_rank']) / 2

    summary = summary.sort_values('avg_rank')

    return summary
