"""
Artifact Manager for Reproducible ML Pipeline

Manages prepared datasets and metadata artifacts:
- Export train/val/test datasets as Parquet files
- Save metadata (feature_set_id, scaler_path, seeds, versions)
- Load prepared datasets with full context
- Ensure reproducibility across experiments

Usage:
    >>> from utils.artifact_manager import export_prepared_datasets, load_prepared_datasets
    >>> export_prepared_datasets(X_train, y_train, X_val, y_val, X_test, y_test, 'abc123')
    >>> data = load_prepared_datasets('abc123')
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime
import sys


class ArtifactManager:
    """
    Manage ML pipeline artifacts
    """

    def __init__(self, artifacts_dir: str = 'artifacts'):
        """
        Initialize artifact manager

        Args:
            artifacts_dir: Base directory for artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def export_datasets(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_set_id: str,
        scaler_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Export prepared datasets with metadata

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            feature_set_id: Feature set ID
            scaler_path: Path to fitted scaler
            metadata: Additional metadata

        Returns:
            Output directory path
        """
        print("=" * 70)
        print("EXPORTING PREPARED DATASETS")
        print("=" * 70)

        # Create output directory
        dataset_dir = self.artifacts_dir / f'datasets_{feature_set_id}'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Export train set
        print("\n→ Exporting train set...")
        X_train.to_parquet(dataset_dir / 'X_train.parquet')
        y_train.to_frame('target').to_parquet(dataset_dir / 'y_train.parquet')
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")

        # Export validation set
        print("\n→ Exporting validation set...")
        X_val.to_parquet(dataset_dir / 'X_val.parquet')
        y_val.to_frame('target').to_parquet(dataset_dir / 'y_val.parquet')
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")

        # Export test set
        print("\n→ Exporting test set...")
        X_test.to_parquet(dataset_dir / 'X_test.parquet')
        y_test.to_frame('target').to_parquet(dataset_dir / 'y_test.parquet')
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")

        # Create metadata
        meta = self._create_metadata(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_set_id=feature_set_id,
            scaler_path=scaler_path,
            custom_metadata=metadata
        )

        # Save metadata
        meta_path = dataset_dir / 'meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n→ Metadata saved: {meta_path}")
        print(f"\n✓ Datasets exported to: {dataset_dir}")
        print("=" * 70)

        return str(dataset_dir)

    def _create_metadata(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_set_id: str,
        scaler_path: Optional[str],
        custom_metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create comprehensive metadata"""

        # Get library versions
        versions = {
            'python': sys.version.split()[0],
            'pandas': pd.__version__,
            'numpy': np.__version__
        }

        # Try to get sklearn version
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except ImportError:
            pass

        # Dataset statistics
        datasets = {
            'train': {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'feature_names': list(X_train.columns),
                'date_range': self._get_date_range(X_train),
                'target_distribution': self._get_target_distribution(y_train)
            },
            'val': {
                'n_samples': len(X_val),
                'n_features': X_val.shape[1],
                'date_range': self._get_date_range(X_val),
                'target_distribution': self._get_target_distribution(y_val)
            },
            'test': {
                'n_samples': len(X_test),
                'n_features': X_test.shape[1],
                'date_range': self._get_date_range(X_test),
                'target_distribution': self._get_target_distribution(y_test)
            }
        }

        # Reproducibility settings
        seeds = {
            'numpy': 42,
            'random': 42
        }

        # Try to get torch seed if available
        try:
            import torch
            seeds['torch'] = 42
        except ImportError:
            pass

        # Combine metadata
        meta = {
            'feature_set_id': feature_set_id,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'scaler_path': scaler_path,
            'versions': versions,
            'seeds': seeds,
            'datasets': datasets
        }

        # Add custom metadata
        if custom_metadata:
            meta['custom'] = custom_metadata

        return meta

    def _get_date_range(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Get date range from DataFrame index"""
        if isinstance(df.index, pd.DatetimeIndex):
            return {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            }
        return None

    def _get_target_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """Get target variable distribution"""
        dist = {
            'dtype': str(y.dtype),
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean()),
            'std': float(y.std())
        }

        # If classification (integer dtype), add class counts
        if y.dtype in ['int64', 'int32', 'int']:
            value_counts = y.value_counts().to_dict()
            dist['class_counts'] = {str(k): int(v) for k, v in value_counts.items()}

        return dist

    def load_datasets(
        self,
        feature_set_id: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
        """
        Load prepared datasets with metadata

        Args:
            feature_set_id: Feature set ID

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        """
        print("=" * 70)
        print("LOADING PREPARED DATASETS")
        print("=" * 70)

        dataset_dir = self.artifacts_dir / f'datasets_{feature_set_id}'

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        # Load datasets
        print(f"\nLoading from: {dataset_dir}")

        X_train = pd.read_parquet(dataset_dir / 'X_train.parquet')
        y_train = pd.read_parquet(dataset_dir / 'y_train.parquet')['target']

        X_val = pd.read_parquet(dataset_dir / 'X_val.parquet')
        y_val = pd.read_parquet(dataset_dir / 'y_val.parquet')['target']

        X_test = pd.read_parquet(dataset_dir / 'X_test.parquet')
        y_test = pd.read_parquet(dataset_dir / 'y_test.parquet')['target']

        print(f"\n✓ Train: X={X_train.shape}, y={y_train.shape}")
        print(f"✓ Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"✓ Test:  X={X_test.shape}, y={y_test.shape}")

        # Load metadata
        meta_path = dataset_dir / 'meta.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        print(f"\n✓ Metadata loaded")
        print(f"  Feature Set ID: {metadata['feature_set_id']}")
        print(f"  Created: {metadata['created_at']}")
        print(f"  Features: {metadata['datasets']['train']['n_features']}")

        print("=" * 70)

        return X_train, y_train, X_val, y_val, X_test, y_test, metadata

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all prepared datasets

        Returns:
            List of dataset metadata dicts
        """
        datasets = []

        for dataset_dir in self.artifacts_dir.glob('datasets_*'):
            if dataset_dir.is_dir():
                meta_path = dataset_dir / 'meta.json'

                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                        datasets.append(metadata)

        # Sort by creation time
        datasets.sort(key=lambda x: x['created_at'], reverse=True)

        return datasets

    def print_datasets(self):
        """Print formatted list of all datasets"""
        datasets = self.list_datasets()

        if not datasets:
            print("No prepared datasets found.")
            return

        print("=" * 100)
        print("AVAILABLE DATASETS")
        print("=" * 100)
        print(f"{'Feature Set ID':<14} {'Created':<20} {'Features':<10} {'Train/Val/Test Samples':<30}")
        print("-" * 100)

        for ds in datasets:
            feature_set_id = ds['feature_set_id']
            created = ds['created_at'][:19].replace('T', ' ')
            n_features = ds['datasets']['train']['n_features']
            n_train = ds['datasets']['train']['n_samples']
            n_val = ds['datasets']['val']['n_samples']
            n_test = ds['datasets']['test']['n_samples']
            samples = f"{n_train:,} / {n_val:,} / {n_test:,}"

            print(f"{feature_set_id:<14} {created:<20} {n_features:<10} {samples:<30}")

        print("=" * 100)


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def export_prepared_datasets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set_id: str,
    scaler_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    artifacts_dir: str = 'artifacts'
) -> str:
    """
    Export prepared datasets

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        feature_set_id: Feature set ID
        scaler_path: Path to fitted scaler
        metadata: Additional metadata
        artifacts_dir: Artifacts directory

    Returns:
        Output directory path
    """
    manager = ArtifactManager(artifacts_dir)
    return manager.export_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_set_id=feature_set_id,
        scaler_path=scaler_path,
        metadata=metadata
    )


def load_prepared_datasets(
    feature_set_id: str,
    artifacts_dir: str = 'artifacts'
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
    """
    Load prepared datasets

    Args:
        feature_set_id: Feature set ID
        artifacts_dir: Artifacts directory

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
    """
    manager = ArtifactManager(artifacts_dir)
    return manager.load_datasets(feature_set_id)


def list_available_datasets(artifacts_dir: str = 'artifacts'):
    """
    List all available prepared datasets

    Args:
        artifacts_dir: Artifacts directory
    """
    manager = ArtifactManager(artifacts_dir)
    manager.print_datasets()


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("Testing artifact manager...\n")

    # Create mock data
    np.random.seed(42)

    X_train = pd.DataFrame({
        'feat1': np.random.randn(1000),
        'feat2': np.random.randn(1000)
    })
    y_train = pd.Series(np.random.randint(0, 3, 1000))

    X_val = pd.DataFrame({
        'feat1': np.random.randn(200),
        'feat2': np.random.randn(200)
    })
    y_val = pd.Series(np.random.randint(0, 3, 200))

    X_test = pd.DataFrame({
        'feat1': np.random.randn(200),
        'feat2': np.random.randn(200)
    })
    y_test = pd.Series(np.random.randint(0, 3, 200))

    # Export
    output_dir = export_prepared_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_set_id='test123',
        metadata={'test': True}
    )

    print(f"\n✓ Exported to: {output_dir}")

    # List datasets
    print("\n")
    list_available_datasets()
