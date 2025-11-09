"""
Time-series aware data splitting utilities

Critical: NO shuffling for time-series data!
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train/val/test sets

    IMPORTANT: No shuffling! Maintains temporal order.

    Args:
        df: DataFrame (must be sorted by time)
        train_ratio: Fraction of data for training (default 60%)
        val_ratio: Fraction of data for validation (default 20%)
        test_ratio: Fraction of data for testing (default: remainder)

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> train, val, test = time_series_split(df, 0.6, 0.2)
        >>> # train: 0-60%, val: 60-80%, test: 80-100%
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print("=" * 60)
    print("TIME-SERIES SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total samples:        {n:,}")
    print(f"\nTrain samples:        {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    if isinstance(train_df.index, pd.DatetimeIndex):
        print(f"  Date range:         {train_df.index.min()} to {train_df.index.max()}")

    print(f"\nValidation samples:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    if isinstance(val_df.index, pd.DatetimeIndex):
        print(f"  Date range:         {val_df.index.min()} to {val_df.index.max()}")

    print(f"\nTest samples:         {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    if isinstance(test_df.index, pd.DatetimeIndex):
        print(f"  Date range:         {test_df.index.min()} to {test_df.index.max()}")
    print("=" * 60)

    return train_df, val_df, test_df


def time_series_cv_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_ratio: float = 0.2
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Time-series cross-validation with expanding window

    Each fold uses all data up to a point for training,
    and the next chunk for validation.

    Args:
        df: DataFrame (must be sorted by time)
        n_splits: Number of CV splits
        test_ratio: Fraction of data to reserve for final test

    Returns:
        List of (train_df, val_df) tuples

    Example:
        Split 1: [---train---][val]
        Split 2: [------train------][val]
        Split 3: [---------train---------][val]
    """
    # Reserve test set
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    df_cv = df.iloc[:test_start]

    n_cv = len(df_cv)
    val_size = n_cv // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        train_end = val_size * i
        val_end = train_end + val_size

        train_df = df_cv.iloc[:train_end]
        val_df = df_cv.iloc[train_end:val_end]

        splits.append((train_df, val_df))

        print(f"Split {i}: Train={len(train_df):,}, Val={len(val_df):,}")

    return splits


def split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by specific dates

    Args:
        df: DataFrame with DatetimeIndex
        train_end_date: Last date for training (inclusive)
        val_end_date: Last date for validation (inclusive)

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> train, val, test = split_by_date(df, '2023-06-30', '2023-08-31')
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)

    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]

    print("=" * 60)
    print("DATE-BASED SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df):,} samples)")
    print(f"Val:   {val_df.index.min()} to {val_df.index.max()} ({len(val_df):,} samples)")
    print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df):,} samples)")
    print("=" * 60)

    return train_df, val_df, test_df


def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    step_size: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward validation (sliding window)

    Useful for simulating live trading conditions

    Args:
        df: DataFrame (sorted by time)
        train_size: Number of samples for training window
        val_size: Number of samples for validation window
        step_size: Number of samples to move forward each iteration

    Returns:
        List of (train_df, val_df) tuples

    Example:
        Step 1: [---train---][val]
        Step 2:      [---train---][val]
        Step 3:           [---train---][val]
    """
    splits = []
    n = len(df)

    start = 0
    while start + train_size + val_size <= n:
        train_end = start + train_size
        val_end = train_end + val_size

        train_df = df.iloc[start:train_end]
        val_df = df.iloc[train_end:val_end]

        splits.append((train_df, val_df))

        start += step_size

    print(f"Created {len(splits)} walk-forward splits")
    print(f"Train size: {train_size}, Val size: {val_size}, Step: {step_size}")

    return splits


def purge_and_embargo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    purge_periods: int = 0,
    embargo_periods: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove samples to prevent data leakage in time-series

    Purge: Remove last N samples from train set
    Embargo: Remove first N samples from validation set

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        purge_periods: Number of periods to remove from end of train
        embargo_periods: Number of periods to remove from start of val

    Returns:
        Tuple of (purged_train_df, embargoed_val_df)

    Use case:
        If features use future information (e.g., lookahead bias),
        remove boundary samples to prevent leakage
    """
    if purge_periods > 0:
        train_df = train_df.iloc[:-purge_periods]
        print(f"Purged {purge_periods} samples from end of training set")

    if embargo_periods > 0:
        val_df = val_df.iloc[embargo_periods:]
        print(f"Embargoed {embargo_periods} samples from start of validation set")

    return train_df, val_df


def get_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'target'
) -> pd.DataFrame:
    """
    Get statistics comparing train/val/test splits

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_col: Name of target column

    Returns:
        DataFrame with comparison statistics
    """
    stats = []

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if target_col in df.columns:
            stat_dict = {
                'Split': name,
                'Samples': len(df),
                'Target_Mean': df[target_col].mean(),
                'Target_Std': df[target_col].std(),
                'Target_Min': df[target_col].min(),
                'Target_Max': df[target_col].max()
            }

            # If classification, get class distribution
            if df[target_col].dtype in ['int64', 'int32', 'category']:
                for class_val in sorted(df[target_col].unique()):
                    count = (df[target_col] == class_val).sum()
                    pct = count / len(df) * 100
                    stat_dict[f'Class_{class_val}_count'] = count
                    stat_dict[f'Class_{class_val}_pct'] = pct

            stats.append(stat_dict)

    return pd.DataFrame(stats)
