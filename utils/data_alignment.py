"""
Data Alignment & Resampling Utilities

Centralized functions to align timestamps across all data feeds:
- Merge OHLCV, OI, Funding, Liquidations, L/S Ratio on common timeline
- Handle missing intervals with explicit rules (drop/ffill/bfill)
- Generate "missing-by-feature" reports
- Ensure monotonic timestamps and no duplicates

This prevents silent misalignment and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import warnings


class DataAligner:
    """
    Align multiple time-series data feeds to a common timeline
    """

    def __init__(self, base_frequency: str = '5min', timezone: str = 'UTC'):
        """
        Initialize aligner

        Args:
            base_frequency: Base frequency for resampling (default: 5min)
            timezone: Timezone for all timestamps (default: UTC)
        """
        self.base_frequency = base_frequency
        self.timezone = timezone
        self.missing_report = {}

    def align_and_resample(
        self,
        ohlcv: pd.DataFrame,
        oi: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        liquidations: Optional[pd.DataFrame] = None,
        ls_ratio: Optional[pd.DataFrame] = None,
        fill_method: str = 'ffill'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Align all data feeds to common timeline

        Args:
            ohlcv: OHLCV DataFrame (required, serves as base timeline)
            oi: Open Interest DataFrame
            funding: Funding Rate DataFrame
            liquidations: Liquidations DataFrame
            ls_ratio: Long/Short Ratio DataFrame
            fill_method: How to handle missing values ('ffill', 'bfill', 'drop', 'zero')

        Returns:
            Tuple of (aligned_df, missing_report)
        """
        print("=" * 70)
        print("ALIGNING DATA FEEDS")
        print("=" * 70)

        # Step 1: Prepare base timeline from OHLCV
        aligned_df = self._prepare_base_timeline(ohlcv)
        print(f"\n✓ Base timeline prepared: {len(aligned_df)} rows")

        # Step 2: Align each feed
        feeds = {
            'oi': oi,
            'funding': funding,
            'liquidations': liquidations,
            'ls_ratio': ls_ratio
        }

        for feed_name, feed_df in feeds.items():
            if feed_df is not None:
                aligned_df = self._align_feed(
                    aligned_df, feed_df, feed_name, fill_method
                )

        # Step 3: Generate missing data report
        self.missing_report = self._generate_missing_report(aligned_df)

        # Step 4: Print summary
        self._print_alignment_summary(aligned_df)

        print("=" * 70)

        return aligned_df, self.missing_report

    def _prepare_base_timeline(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare base timeline from OHLCV data

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            DataFrame with timestamp index
        """
        df = ohlcv.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Make timezone-aware
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(self.timezone)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)

            # Set as index
            df = df.set_index('timestamp')

        # Check for duplicates
        if df.index.duplicated().any():
            n_duplicates = df.index.duplicated().sum()
            warnings.warn(f"Found {n_duplicates} duplicate timestamps, keeping first")
            df = df[~df.index.duplicated(keep='first')]

        # Sort index
        df = df.sort_index()

        # Check monotonic
        if not df.index.is_monotonic_increasing:
            warnings.warn("Index was not monotonic, sorting applied")

        return df

    def _align_feed(
        self,
        base_df: pd.DataFrame,
        feed_df: pd.DataFrame,
        feed_name: str,
        fill_method: str
    ) -> pd.DataFrame:
        """
        Align a single feed to the base timeline

        Args:
            base_df: Base DataFrame with timeline
            feed_df: Feed DataFrame to align
            feed_name: Name of the feed (for reporting)
            fill_method: How to handle missing values

        Returns:
            Merged DataFrame
        """
        print(f"\n→ Aligning {feed_name}...")

        # Prepare feed
        df = feed_df.copy()

        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Make timezone-aware
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(self.timezone)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)

            df = df.set_index('timestamp')

        # Remove duplicates
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        # Sort
        df = df.sort_index()

        # Merge with base
        before_count = len(base_df)
        base_df = base_df.join(df, how='left', rsuffix=f'_{feed_name}')
        after_count = len(base_df)

        # Count missing values
        feed_cols = [col for col in base_df.columns if col in df.columns or col.endswith(f'_{feed_name}')]
        n_missing = base_df[feed_cols].isna().any(axis=1).sum()
        missing_pct = (n_missing / len(base_df)) * 100

        print(f"   Rows: {before_count} → {after_count}")
        print(f"   Missing: {n_missing} rows ({missing_pct:.2f}%)")

        # Handle missing values
        if fill_method == 'ffill':
            base_df[feed_cols] = base_df[feed_cols].ffill()
            print(f"   Applied: forward fill")
        elif fill_method == 'bfill':
            base_df[feed_cols] = base_df[feed_cols].bfill()
            print(f"   Applied: backward fill")
        elif fill_method == 'zero':
            base_df[feed_cols] = base_df[feed_cols].fillna(0)
            print(f"   Applied: fill with 0")
        elif fill_method == 'drop':
            base_df = base_df.dropna(subset=feed_cols)
            print(f"   Applied: drop missing rows ({len(base_df)} remaining)")
        else:
            print(f"   Applied: no fill (method={fill_method})")

        return base_df

    def _generate_missing_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate detailed missing data report

        Args:
            df: Aligned DataFrame

        Returns:
            Dict with missing data statistics per column
        """
        report = {}

        for col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100

            if n_missing > 0:
                report[col] = {
                    'count': n_missing,
                    'percentage': pct_missing,
                    'first_missing_idx': df[col].isna().idxmax() if n_missing > 0 else None
                }

        return report

    def _print_alignment_summary(self, df: pd.DataFrame):
        """Print alignment summary"""
        print(f"\n{'='*70}")
        print("ALIGNMENT SUMMARY")
        print("=" * 70)
        print(f"Total rows:        {len(df):,}")
        print(f"Total columns:     {len(df.columns)}")
        print(f"Date range:        {df.index.min()} to {df.index.max()}")
        print(f"Frequency:         {self.base_frequency}")
        print(f"Timezone:          {self.timezone}")

        # Missing data summary
        total_missing = sum(df[col].isna().sum() for col in df.columns)
        total_cells = len(df) * len(df.columns)
        pct_missing = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        print(f"\nMissing values:    {total_missing:,} / {total_cells:,} ({pct_missing:.2f}%)")

        if self.missing_report:
            print(f"\nColumns with missing data ({len(self.missing_report)}):")
            for col, stats in sorted(self.missing_report.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
                print(f"   {col:30s}: {stats['count']:6,} ({stats['percentage']:5.2f}%)")

    def get_missing_report(self) -> pd.DataFrame:
        """
        Get missing data report as DataFrame

        Returns:
            DataFrame with missing data statistics
        """
        if not self.missing_report:
            return pd.DataFrame()

        report_df = pd.DataFrame.from_dict(self.missing_report, orient='index')
        report_df = report_df.sort_values('count', ascending=False)
        return report_df


# ========================================
# HELPER FUNCTIONS
# ========================================

def resample_to_frequency(
    df: pd.DataFrame,
    frequency: str = '5min',
    aggregation: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Resample DataFrame to target frequency

    Args:
        df: DataFrame with DatetimeIndex
        frequency: Target frequency (e.g., '5min', '1h')
        aggregation: Dict of {column: method} for custom aggregation
                    Default: OHLCV uses standard OHLC, others use 'last'

    Returns:
        Resampled DataFrame
    """
    if aggregation is None:
        # Default aggregation
        aggregation = {}
        for col in df.columns:
            if col == 'open':
                aggregation[col] = 'first'
            elif col == 'high':
                aggregation[col] = 'max'
            elif col == 'low':
                aggregation[col] = 'min'
            elif col == 'close':
                aggregation[col] = 'last'
            elif col == 'volume':
                aggregation[col] = 'sum'
            else:
                aggregation[col] = 'last'  # Default for other columns

    resampled = df.resample(frequency).agg(aggregation)
    return resampled


def check_time_gaps(
    df: pd.DataFrame,
    expected_frequency: str = '5min',
    tolerance: str = '1min'
) -> pd.DataFrame:
    """
    Check for unexpected time gaps in data

    Args:
        df: DataFrame with DatetimeIndex
        expected_frequency: Expected frequency between rows
        tolerance: Allowed tolerance for gaps

    Returns:
        DataFrame with detected gaps
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Calculate time differences
    time_diff = df.index.to_series().diff()

    # Expected difference
    expected_diff = pd.Timedelta(expected_frequency)
    tolerance_diff = pd.Timedelta(tolerance)

    # Find gaps
    gaps = time_diff[time_diff > (expected_diff + tolerance_diff)]

    if len(gaps) > 0:
        gap_report = pd.DataFrame({
            'timestamp': gaps.index,
            'gap_duration': gaps.values,
            'expected': expected_diff,
            'excess': gaps.values - expected_diff
        })
        return gap_report
    else:
        return pd.DataFrame()


def fill_missing_timestamps(
    df: pd.DataFrame,
    frequency: str = '5min',
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Fill missing timestamps in time-series

    Args:
        df: DataFrame with DatetimeIndex
        frequency: Expected frequency
        fill_method: Method to fill missing values

    Returns:
        DataFrame with complete timestamp sequence
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Create complete date range
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=frequency,
        tz=df.index.tz
    )

    # Reindex to full range
    df_filled = df.reindex(full_range)

    # Fill missing values
    if fill_method == 'ffill':
        df_filled = df_filled.ffill()
    elif fill_method == 'bfill':
        df_filled = df_filled.bfill()
    elif fill_method == 'interpolate':
        df_filled = df_filled.interpolate(method='time')

    return df_filled


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("Testing data alignment...\n")

    # Create mock data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')

    mock_ohlcv = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 41000, 1000),
        'high': np.random.uniform(40000, 41000, 1000),
        'low': np.random.uniform(40000, 41000, 1000),
        'close': np.random.uniform(40000, 41000, 1000),
        'volume': np.random.uniform(100, 1000, 1000)
    })

    mock_oi = pd.DataFrame({
        'timestamp': dates[::2],  # Every other timestamp
        'open_interest': np.random.uniform(1e9, 2e9, 500)
    })

    # Align
    aligner = DataAligner()
    aligned, report = aligner.align_and_resample(
        ohlcv=mock_ohlcv,
        oi=mock_oi,
        fill_method='ffill'
    )

    print(f"\n✓ Aligned data shape: {aligned.shape}")
    print(f"✓ Missing report: {len(report)} columns with missing data")
