"""
Feature Store for fast feature access in production

Uses Redis for low-latency feature retrieval during live trading
"""

import json
import pickle
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class FeatureStore:
    """
    Fast feature storage and retrieval via Redis

    Features are stored with keys like:
    - features:{symbol}:{timestamp} -> feature dict
    - feature_metadata -> feature names and types
    - latest_features:{symbol} -> most recent features
    """

    def __init__(self, redis_client: Optional[Any] = None, use_mock: bool = False):
        """
        Initialize feature store

        Args:
            redis_client: Redis client instance (redis.Redis)
            use_mock: If True, use in-memory dict instead of Redis (for testing)
        """
        self.redis = redis_client
        self.use_mock = use_mock or (redis_client is None)

        if self.use_mock:
            print("Warning: Using mock in-memory storage (not Redis)")
            self._mock_store = {}

    def save_features(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        features: Dict[str, float]
    ) -> bool:
        """
        Save computed features to store

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timestamp: Feature timestamp
            features: Dictionary of feature_name -> value

        Returns:
            True if successful
        """
        key = f"features:{symbol}:{timestamp.isoformat()}"

        # Convert numpy types to native Python types
        features_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in features.items()
        }

        # Serialize to JSON
        value = json.dumps(features_serializable)

        if self.use_mock:
            self._mock_store[key] = value
            # Also update latest
            self._mock_store[f"latest_features:{symbol}"] = value
        else:
            # Save with TTL (keep for 7 days)
            self.redis.setex(key, 7 * 24 * 3600, value)
            # Also save as latest
            self.redis.set(f"latest_features:{symbol}", value)

        return True

    def get_features(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve features from store

        Args:
            symbol: Trading symbol
            timestamp: Feature timestamp

        Returns:
            Dictionary of features, or None if not found
        """
        key = f"features:{symbol}:{timestamp.isoformat()}"

        if self.use_mock:
            value = self._mock_store.get(key)
        else:
            value = self.redis.get(key)

        if value:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return json.loads(value)

        return None

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get most recent features for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of latest features, or None if not found
        """
        key = f"latest_features:{symbol}"

        if self.use_mock:
            value = self._mock_store.get(key)
        else:
            value = self.redis.get(key)

        if value:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return json.loads(value)

        return None

    def save_feature_metadata(
        self,
        feature_names: List[str],
        feature_types: Optional[Dict[str, str]] = None,
        selected_features: Optional[List[str]] = None
    ) -> bool:
        """
        Save feature metadata (names, types, selection status)

        Args:
            feature_names: List of all feature names
            feature_types: Optional dict mapping feature -> category
            selected_features: Optional list of selected features

        Returns:
            True if successful
        """
        metadata = {
            'feature_names': feature_names,
            'feature_types': feature_types or {},
            'selected_features': selected_features or feature_names,
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_features': len(feature_names),
            'num_selected': len(selected_features) if selected_features else len(feature_names)
        }

        value = json.dumps(metadata)

        if self.use_mock:
            self._mock_store['feature_metadata'] = value
        else:
            self.redis.set('feature_metadata', value)

        return True

    def get_feature_metadata(self) -> Optional[Dict]:
        """
        Retrieve feature metadata

        Returns:
            Dictionary with feature metadata, or None if not found
        """
        if self.use_mock:
            value = self._mock_store.get('feature_metadata')
        else:
            value = self.redis.get('feature_metadata')

        if value:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return json.loads(value)

        return None

    def save_batch_features(
        self,
        symbol: str,
        features_df: pd.DataFrame
    ) -> int:
        """
        Save multiple feature rows in batch

        Args:
            symbol: Trading symbol
            features_df: DataFrame with features (index = timestamp)

        Returns:
            Number of rows saved
        """
        count = 0

        for timestamp, row in features_df.iterrows():
            features = row.to_dict()
            if self.save_features(symbol, timestamp, features):
                count += 1

        print(f"Saved {count} feature rows to store")
        return count

    def get_features_range(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Retrieve features for a time range

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame with features
        """
        if not self.use_mock:
            # For Redis, we'd need to scan keys
            pattern = f"features:{symbol}:*"
            keys = self.redis.keys(pattern)

            # Filter by timestamp range
            features_list = []
            for key in keys:
                # Extract timestamp from key
                timestamp_str = key.decode('utf-8').split(':')[-1]
                timestamp = pd.Timestamp(timestamp_str)

                if start_time <= timestamp <= end_time:
                    features = self.get_features(symbol, timestamp)
                    if features:
                        features['timestamp'] = timestamp
                        features_list.append(features)

            if features_list:
                df = pd.DataFrame(features_list)
                df = df.set_index('timestamp').sort_index()
                return df
        else:
            # For mock store, filter manually
            features_list = []
            for key, value in self._mock_store.items():
                if key.startswith(f"features:{symbol}:"):
                    timestamp_str = key.split(':')[-1]
                    timestamp = pd.Timestamp(timestamp_str)

                    if start_time <= timestamp <= end_time:
                        features = json.loads(value)
                        features['timestamp'] = timestamp
                        features_list.append(features)

            if features_list:
                df = pd.DataFrame(features_list)
                df = df.set_index('timestamp').sort_index()
                return df

        return pd.DataFrame()

    def delete_features(self, symbol: str, timestamp: pd.Timestamp) -> bool:
        """
        Delete features for a specific timestamp

        Args:
            symbol: Trading symbol
            timestamp: Feature timestamp

        Returns:
            True if deleted
        """
        key = f"features:{symbol}:{timestamp.isoformat()}"

        if self.use_mock:
            if key in self._mock_store:
                del self._mock_store[key]
                return True
        else:
            return bool(self.redis.delete(key))

        return False

    def clear_symbol(self, symbol: str) -> int:
        """
        Clear all features for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Number of keys deleted
        """
        if self.use_mock:
            keys_to_delete = [k for k in self._mock_store.keys() if k.startswith(f"features:{symbol}:")]
            for key in keys_to_delete:
                del self._mock_store[key]
            return len(keys_to_delete)
        else:
            pattern = f"features:{symbol}:*"
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0

    def get_storage_info(self) -> Dict:
        """
        Get information about stored features

        Returns:
            Dictionary with storage statistics
        """
        if self.use_mock:
            total_keys = len(self._mock_store)
            feature_keys = [k for k in self._mock_store.keys() if k.startswith("features:")]
            symbols = set(k.split(':')[1] for k in feature_keys if len(k.split(':')) >= 2)

            return {
                'total_keys': total_keys,
                'feature_keys': len(feature_keys),
                'symbols': list(symbols),
                'storage_type': 'mock'
            }
        else:
            # Get all feature keys
            pattern = "features:*"
            keys = self.redis.keys(pattern)
            symbols = set(k.decode('utf-8').split(':')[1] for k in keys if len(k.decode('utf-8').split(':')) >= 2)

            return {
                'total_keys': len(keys),
                'symbols': list(symbols),
                'storage_type': 'redis'
            }


class FileBasedFeatureStore:
    """
    File-based feature store for offline/development use

    Stores features as Parquet files for efficient disk storage
    """

    def __init__(self, base_path: str = "./feature_store"):
        """
        Initialize file-based feature store

        Args:
            base_path: Directory to store feature files
        """
        self.base_path = base_path
        import os
        os.makedirs(base_path, exist_ok=True)

    def save_features(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        partition_by: str = 'date'
    ):
        """
        Save features to parquet file

        Args:
            symbol: Trading symbol
            features_df: DataFrame with features
            partition_by: Partition strategy ('date', 'month', None)
        """
        import os

        if partition_by == 'date':
            # Save one file per date
            for date, group in features_df.groupby(features_df.index.date):
                filepath = os.path.join(
                    self.base_path,
                    symbol,
                    f"{date}.parquet"
                )
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                group.to_parquet(filepath)
                print(f"Saved {len(group)} rows to {filepath}")
        else:
            # Save as single file
            filepath = os.path.join(self.base_path, f"{symbol}_features.parquet")
            features_df.to_parquet(filepath)
            print(f"Saved {len(features_df)} rows to {filepath}")

    def load_features(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load features from parquet file(s)

        Args:
            symbol: Trading symbol
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with features
        """
        import os
        import glob

        symbol_dir = os.path.join(self.base_path, symbol)

        if os.path.isdir(symbol_dir):
            # Load from partitioned files
            parquet_files = glob.glob(os.path.join(symbol_dir, "*.parquet"))

            dfs = []
            for filepath in parquet_files:
                df = pd.read_parquet(filepath)
                dfs.append(df)

            if dfs:
                combined_df = pd.concat(dfs).sort_index()

                # Filter by date range if specified
                if start_date:
                    combined_df = combined_df[combined_df.index >= start_date]
                if end_date:
                    combined_df = combined_df[combined_df.index <= end_date]

                return combined_df
        else:
            # Load from single file
            filepath = os.path.join(self.base_path, f"{symbol}_features.parquet")
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)

                # Filter by date range if specified
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

                return df

        return pd.DataFrame()
