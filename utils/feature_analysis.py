"""
Feature Importance Analysis and Visualization

Tools for analyzing feature importance, correlations, and distributions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def analyze_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Comprehensive feature importance analysis

    Args:
        X: Feature DataFrame
        y: Target variable
        task_type: 'classification' or 'regression'
        top_n: Number of top features to display

    Returns:
        DataFrame with importance scores
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Train model
    if task_type == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X, y)

    # Get importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Calculate cumulative importance
    importances['cumulative_importance'] = importances['importance'].cumsum()

    # Display top features
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 70)
    for i, row in importances.head(top_n).iterrows():
        bar_length = int(row['importance'] * 100)
        bar = '█' * bar_length
        print(f"{row['feature']:40s} {row['importance']:.4f} {bar}")

    print("-" * 70)

    # Find number of features for 80%, 90%, 95% importance
    for threshold in [0.80, 0.90, 0.95]:
        n_features = (importances['cumulative_importance'] <= threshold).sum() + 1
        print(f"Features needed for {threshold*100:.0f}% cumulative importance: {n_features}")

    print("=" * 70)

    return importances


def analyze_feature_correlations(
    df: pd.DataFrame,
    threshold: float = 0.7,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Analyze feature correlations

    Args:
        df: Feature DataFrame
        threshold: Correlation threshold to report
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        DataFrame with highly correlated feature pairs
    """
    print("=" * 70)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    # Calculate correlation matrix
    corr_matrix = df.corr(method=method).abs()

    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find highly correlated pairs
    high_corr_pairs = []
    for column in upper.columns:
        for index in upper.index:
            corr_value = upper.loc[index, column]
            if pd.notna(corr_value) and corr_value >= threshold:
                high_corr_pairs.append({
                    'feature_1': index,
                    'feature_2': column,
                    'correlation': corr_value
                })

    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            'correlation', ascending=False
        )
        print(f"\nFound {len(high_corr_df)} feature pairs with correlation >= {threshold}")
        print(f"\nTop 10 correlated pairs:")
        print(high_corr_df.head(10).to_string(index=False))
    else:
        print(f"\nNo feature pairs found with correlation >= {threshold}")
        high_corr_df = pd.DataFrame()

    print("=" * 70)

    return high_corr_df


def analyze_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze feature distributions (mean, std, skew, kurtosis)

    Args:
        df: Feature DataFrame
        features: List of features to analyze (default: all)

    Returns:
        DataFrame with distribution statistics
    """
    if features is None:
        features = df.columns.tolist()

    stats = []

    for feature in features:
        if feature in df.columns:
            series = df[feature]

            stat_dict = {
                'feature': feature,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'skew': series.skew(),
                'kurtosis': series.kurtosis(),
                'zeros_pct': (series == 0).sum() / len(series) * 100,
                'missing_pct': series.isna().sum() / len(series) * 100
            }

            stats.append(stat_dict)

    stats_df = pd.DataFrame(stats)

    print("=" * 70)
    print("FEATURE DISTRIBUTION STATISTICS")
    print("=" * 70)
    print(f"\nTotal features analyzed: {len(stats_df)}")

    # Flag potential issues
    high_skew = stats_df[stats_df['skew'].abs() > 2]
    if len(high_skew) > 0:
        print(f"\n⚠ {len(high_skew)} features with high skewness (|skew| > 2)")

    high_missing = stats_df[stats_df['missing_pct'] > 10]
    if len(high_missing) > 0:
        print(f"⚠ {len(high_missing)} features with >10% missing values")

    print("=" * 70)

    return stats_df


def analyze_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Analyze correlation between features and target

    Args:
        X: Feature DataFrame
        y: Target variable
        top_n: Number of top features to display
        method: Correlation method

    Returns:
        DataFrame with feature-target correlations
    """
    print("=" * 70)
    print("FEATURE-TARGET CORRELATION ANALYSIS")
    print("=" * 70)

    correlations = []

    for col in X.columns:
        corr = X[col].corr(y, method=method)
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

    corr_df = pd.DataFrame(correlations).sort_values(
        'abs_correlation', ascending=False
    )

    print(f"\nTop {top_n} Features by Target Correlation:")
    print("-" * 70)
    for i, row in corr_df.head(top_n).iterrows():
        print(f"{row['feature']:40s} {row['correlation']:+.4f}")

    print("=" * 70)

    return corr_df


def detect_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in features

    Args:
        df: Feature DataFrame
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outlier counts per feature
    """
    outlier_stats = []

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                outliers = (z_scores > threshold).sum()

            outlier_pct = outliers / len(df) * 100

            outlier_stats.append({
                'feature': col,
                'outliers': outliers,
                'outlier_pct': outlier_pct
            })

    outlier_df = pd.DataFrame(outlier_stats).sort_values(
        'outlier_pct', ascending=False
    )

    print("=" * 70)
    print(f"OUTLIER DETECTION ({method.upper()} method)")
    print("=" * 70)

    high_outliers = outlier_df[outlier_df['outlier_pct'] > 5]
    if len(high_outliers) > 0:
        print(f"\n⚠ {len(high_outliers)} features with >5% outliers:")
        print(high_outliers.head(10).to_string(index=False))
    else:
        print("\n✓ No features with excessive outliers (>5%)")

    print("=" * 70)

    return outlier_df


def generate_feature_report(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification',
    output_file: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive feature analysis report

    Args:
        X: Feature DataFrame
        y: Target variable
        task_type: 'classification' or 'regression'
        output_file: Optional path to save report

    Returns:
        Dictionary with all analysis results
    """
    print("\n")
    print("=" * 70)
    print("COMPREHENSIVE FEATURE ANALYSIS REPORT")
    print("=" * 70)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Task type: {task_type}")
    print("=" * 70)

    report = {}

    # 1. Feature importance
    report['importance'] = analyze_feature_importance(X, y, task_type, top_n=20)

    # 2. Feature correlations
    report['correlations'] = analyze_feature_correlations(X, threshold=0.7)

    # 3. Feature distributions
    report['distributions'] = analyze_feature_distributions(X)

    # 4. Target correlations
    if task_type == 'regression' or y.nunique() > 10:
        report['target_correlations'] = analyze_target_correlation(X, y, top_n=20)

    # 5. Outlier detection
    report['outliers'] = detect_outliers(X, method='iqr')

    # Save report if requested
    if output_file:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(report, f)
        print(f"\n✓ Report saved to {output_file}")

    return report


def compare_feature_sets(
    original_features: List[str],
    selected_features: List[str]
) -> Dict:
    """
    Compare original and selected feature sets

    Args:
        original_features: List of all features
        selected_features: List of selected features

    Returns:
        Dictionary with comparison statistics
    """
    print("=" * 70)
    print("FEATURE SET COMPARISON")
    print("=" * 70)

    removed_features = set(original_features) - set(selected_features)
    kept_features = set(selected_features)

    print(f"Original features:    {len(original_features)}")
    print(f"Selected features:    {len(selected_features)}")
    print(f"Removed features:     {len(removed_features)}")
    print(f"Reduction:            {len(removed_features)/len(original_features)*100:.1f}%")

    print("\n" + "=" * 70)

    return {
        'original_count': len(original_features),
        'selected_count': len(selected_features),
        'removed_count': len(removed_features),
        'reduction_pct': len(removed_features) / len(original_features) * 100,
        'removed_features': list(removed_features)
    }
