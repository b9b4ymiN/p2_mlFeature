"""
Phase 2: ML Feature Engineering for OI Trading

A comprehensive feature engineering pipeline for cryptocurrency futures trading.
"""

__version__ = '1.0.0'
__author__ = 'AI Trading System'

from features import FeatureEngineer, TargetEngineer, FeatureStore
from utils import (
    time_series_split,
    select_features_combined,
    analyze_feature_importance
)

__all__ = [
    'FeatureEngineer',
    'TargetEngineer',
    'FeatureStore',
    'time_series_split',
    'select_features_combined',
    'analyze_feature_importance'
]
