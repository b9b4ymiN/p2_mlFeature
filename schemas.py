"""
Data Schema Contracts for ML Pipeline

Defines the single source of truth for all data feeds:
- Required columns, dtypes, units
- Frequency (5-minute bars)
- Timezone (UTC)
- Validation rules

This prevents silent schema drift and ensures data quality.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


class DataSchema:
    """Base class for data schema validation"""

    def __init__(self, name: str, columns: Dict[str, str], required: List[str],
                 frequency: str = '5min', timezone: str = 'UTC'):
        """
        Initialize schema

        Args:
            name: Schema name
            columns: Dict of {column_name: dtype}
            required: List of required column names
            frequency: Expected data frequency
            timezone: Expected timezone
        """
        self.name = name
        self.columns = columns
        self.required = required
        self.frequency = frequency
        self.timezone = timezone

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame against schema

        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        errors = []
        warnings = []

        # Check required columns
        missing_cols = set(self.required) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check data types
        for col, expected_dtype in self.columns.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_matches(actual_dtype, expected_dtype):
                    warnings.append(
                        f"Column '{col}' has dtype '{actual_dtype}', "
                        f"expected '{expected_dtype}'"
                    )

        # Check for duplicates in timestamp
        if 'timestamp' in df.columns:
            if df['timestamp'].duplicated().any():
                errors.append("Duplicate timestamps found")

        # Check monotonic timestamps
        if 'timestamp' in df.columns:
            if isinstance(df['timestamp'], pd.Series):
                timestamps = pd.to_datetime(df['timestamp'])
                if not timestamps.is_monotonic_increasing:
                    errors.append("Timestamps are not monotonic increasing")

        # Check for timezone awareness
        if 'timestamp' in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    warnings.append(f"Timestamps should be timezone-aware ({self.timezone})")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'schema': self.name
        }

    def _dtype_matches(self, actual: str, expected: str) -> bool:
        """Check if dtypes match (with some flexibility)"""
        # Normalize dtype strings
        dtype_mapping = {
            'int64': ['int64', 'int32', 'int'],
            'float64': ['float64', 'float32', 'float'],
            'datetime64[ns]': ['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]'],
            'object': ['object', 'string']
        }

        for expected_group, actual_group in dtype_mapping.items():
            if expected in actual_group and actual in actual_group:
                return True

        return actual == expected


# ========================================
# SCHEMA DEFINITIONS
# ========================================

# OHLCV Schema
OHLCV_SCHEMA = DataSchema(
    name='OHLCV',
    columns={
        'timestamp': 'datetime64[ns]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64'
    },
    required=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
    frequency='5min',
    timezone='UTC'
)

# Open Interest Schema
OI_SCHEMA = DataSchema(
    name='OpenInterest',
    columns={
        'timestamp': 'datetime64[ns]',
        'open_interest': 'float64'
    },
    required=['timestamp', 'open_interest'],
    frequency='5min',
    timezone='UTC'
)

# Funding Rate Schema
FUNDING_SCHEMA = DataSchema(
    name='FundingRate',
    columns={
        'timestamp': 'datetime64[ns]',
        'funding_rate': 'float64'
    },
    required=['timestamp', 'funding_rate'],
    frequency='8h',  # Funding rates are 8-hourly
    timezone='UTC'
)

# Liquidations Schema
LIQUIDATIONS_SCHEMA = DataSchema(
    name='Liquidations',
    columns={
        'timestamp': 'datetime64[ns]',
        'side': 'object',
        'price': 'float64',
        'quantity': 'float64'
    },
    required=['timestamp', 'side', 'price', 'quantity'],
    frequency='irregular',  # Event-based
    timezone='UTC'
)

# Long/Short Ratio Schema
LS_RATIO_SCHEMA = DataSchema(
    name='LongShortRatio',
    columns={
        'timestamp': 'datetime64[ns]',
        'longShortRatio': 'float64'
    },
    required=['timestamp', 'longShortRatio'],
    frequency='5min',
    timezone='UTC'
)

# Features Schema (after engineering)
FEATURES_SCHEMA = DataSchema(
    name='Features',
    columns={
        'timestamp': 'datetime64[ns]',
        # All other columns are float features
    },
    required=['timestamp'],
    frequency='5min',
    timezone='UTC'
)

# Targets Schema
TARGETS_SCHEMA = DataSchema(
    name='Targets',
    columns={
        'timestamp': 'datetime64[ns]',
        'target': 'int64',  # Classification: 0=SHORT, 1=NEUTRAL, 2=LONG
        'future_return': 'float64'  # Regression target
    },
    required=['timestamp', 'target'],
    frequency='5min',
    timezone='UTC'
)


# ========================================
# VALIDATION HELPERS
# ========================================

def validate_all_feeds(
    ohlcv: Optional[pd.DataFrame] = None,
    oi: Optional[pd.DataFrame] = None,
    funding: Optional[pd.DataFrame] = None,
    liquidations: Optional[pd.DataFrame] = None,
    ls_ratio: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Validate all data feeds against their schemas

    Returns:
        Dict with validation results for each feed
    """
    results = {}

    feeds = {
        'ohlcv': (ohlcv, OHLCV_SCHEMA),
        'oi': (oi, OI_SCHEMA),
        'funding': (funding, FUNDING_SCHEMA),
        'liquidations': (liquidations, LIQUIDATIONS_SCHEMA),
        'ls_ratio': (ls_ratio, LS_RATIO_SCHEMA)
    }

    for feed_name, (df, schema) in feeds.items():
        if df is not None:
            results[feed_name] = schema.validate(df)
        else:
            results[feed_name] = {
                'valid': None,
                'errors': [],
                'warnings': ['Feed is None (not provided)'],
                'schema': schema.name
            }

    return results


def print_validation_report(validation_results: Dict[str, Any]):
    """Print a formatted validation report"""
    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)

    total_errors = 0
    total_warnings = 0

    for feed_name, result in validation_results.items():
        print(f"\n📊 {feed_name.upper()}: {result['schema']}")

        if result['valid'] is None:
            print("   ⏭️  Skipped (not provided)")
        elif result['valid']:
            print("   ✅ Valid")
        else:
            print("   ❌ Invalid")

        if result['errors']:
            total_errors += len(result['errors'])
            print(f"\n   🔴 Errors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"      - {error}")

        if result['warnings']:
            total_warnings += len(result['warnings'])
            print(f"\n   ⚠️  Warnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"      - {warning}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_errors} errors, {total_warnings} warnings")
    print("=" * 70)

    return total_errors == 0


def ensure_timezone(df: pd.DataFrame, tz: str = 'UTC') -> pd.DataFrame:
    """
    Ensure DataFrame timestamps are timezone-aware

    Args:
        df: DataFrame with 'timestamp' column
        tz: Target timezone (default: UTC)

    Returns:
        DataFrame with timezone-aware timestamps
    """
    df = df.copy()

    if 'timestamp' in df.columns:
        if not isinstance(df['timestamp'], pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(tz)

    return df


def check_monotonic_time(df: pd.DataFrame, col: str = 'timestamp') -> bool:
    """
    Check if timestamps are monotonically increasing

    Args:
        df: DataFrame
        col: Timestamp column name

    Returns:
        True if monotonic, False otherwise
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    timestamps = pd.to_datetime(df[col])
    return timestamps.is_monotonic_increasing


def find_duplicate_timestamps(df: pd.DataFrame, col: str = 'timestamp') -> pd.DataFrame:
    """
    Find duplicate timestamps in DataFrame

    Args:
        df: DataFrame
        col: Timestamp column name

    Returns:
        DataFrame with duplicate timestamps
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    duplicates = df[df[col].duplicated(keep=False)]
    return duplicates.sort_values(col)


# ========================================
# USAGE EXAMPLES
# ========================================

if __name__ == "__main__":
    # Example: Validate mock data
    print("Testing schema validation...\n")

    # Create mock OHLCV data
    mock_ohlcv = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': np.random.uniform(40000, 41000, 100),
        'high': np.random.uniform(40000, 41000, 100),
        'low': np.random.uniform(40000, 41000, 100),
        'close': np.random.uniform(40000, 41000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })

    # Validate
    result = OHLCV_SCHEMA.validate(mock_ohlcv)

    print(f"Schema: {result['schema']}")
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
