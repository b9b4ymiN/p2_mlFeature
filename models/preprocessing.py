"""
Data Preprocessing & Scaling Pipeline

CRITICAL: Scalers must be fitted on TRAINING data ONLY to prevent data leakage!

Features:
- StandardScaler (zero mean, unit variance)
- MinMaxScaler (scale to range)
- RobustScaler (handles outliers)
- Per-symbol scaling support
- Save/load scaler artifacts

Usage:
    >>> from models.preprocessing import fit_scaler, apply_scaler
    >>> scaler = fit_scaler(X_train, feature_set_id='abc123')
    >>> X_train_scaled = apply_scaler(X_train, scaler)
    >>> X_val_scaled = apply_scaler(X_val, scaler)  # Use same scaler!
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


class FeatureScaler:
    """
    Feature scaling manager with artifact persistence
    """

    def __init__(
        self,
        scaler_type: str = 'standard',
        per_symbol: bool = False
    ):
        """
        Initialize scaler

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            per_symbol: If True, fit separate scalers per symbol
        """
        self.scaler_type = scaler_type
        self.per_symbol = per_symbol
        self.scalers = {}  # {symbol: scaler} if per_symbol else {'global': scaler}
        self.feature_names = None
        self.fitted = False

    def _create_scaler(self):
        """Create new scaler instance"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def fit(
        self,
        X: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> 'FeatureScaler':
        """
        Fit scaler on training data

        IMPORTANT: Only call this on training data!

        Args:
            X: Training features DataFrame
            symbol: Symbol name (if per_symbol=True)

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Store feature names
        self.feature_names = list(X.columns)

        if self.per_symbol:
            if symbol is None:
                raise ValueError("symbol must be provided when per_symbol=True")

            # Fit scaler for this symbol
            scaler = self._create_scaler()
            scaler.fit(X)
            self.scalers[symbol] = scaler

            print(f"✓ Fitted {self.scaler_type} scaler for symbol '{symbol}'")
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Samples: {len(X):,}")

        else:
            # Fit global scaler
            scaler = self._create_scaler()
            scaler.fit(X)
            self.scalers['global'] = scaler

            print(f"✓ Fitted {self.scaler_type} scaler (global)")
            print(f"  Features: {len(self.feature_names)}")
            print(f"  Samples: {len(X):,}")

        self.fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted scaler

        Args:
            X: Features DataFrame
            symbol: Symbol name (if per_symbol=True)

        Returns:
            Scaled DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Get appropriate scaler
        if self.per_symbol:
            if symbol is None:
                raise ValueError("symbol must be provided when per_symbol=True")
            if symbol not in self.scalers:
                warnings.warn(
                    f"Symbol '{symbol}' not in fitted scalers, using first available"
                )
                scaler = list(self.scalers.values())[0]
            else:
                scaler = self.scalers[symbol]
        else:
            scaler = self.scalers['global']

        # Transform
        X_scaled = scaler.transform(X)

        # Return as DataFrame
        return pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step

        IMPORTANT: Only use this on training data!

        Args:
            X: Training features DataFrame
            symbol: Symbol name (if per_symbol=True)

        Returns:
            Scaled DataFrame
        """
        self.fit(X, symbol)
        return self.transform(X, symbol)

    def save(self, filepath: str):
        """
        Save scaler to file

        Args:
            filepath: Path to save scaler
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted scaler")

        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save scaler object
        joblib.dump(self, filepath)

        print(f"✓ Saved scaler: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'FeatureScaler':
        """
        Load scaler from file

        Args:
            filepath: Path to saved scaler

        Returns:
            Loaded FeatureScaler
        """
        scaler = joblib.load(filepath)

        print(f"✓ Loaded scaler: {filepath}")
        print(f"  Type: {scaler.scaler_type}")
        print(f"  Features: {len(scaler.feature_names) if scaler.feature_names else 'N/A'}")
        print(f"  Per-symbol: {scaler.per_symbol}")

        return scaler

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scaling statistics

        Returns:
            Dict with scaling parameters
        """
        if not self.fitted:
            return {}

        stats = {
            'scaler_type': self.scaler_type,
            'per_symbol': self.per_symbol,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'symbols': list(self.scalers.keys()) if self.per_symbol else None
        }

        # Add scaler-specific statistics
        if self.scaler_type == 'standard':
            # Get mean and std from first scaler
            scaler = list(self.scalers.values())[0]
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                stats['mean'] = scaler.mean_.tolist()
                stats['std'] = scaler.scale_.tolist()

        elif self.scaler_type == 'minmax':
            scaler = list(self.scalers.values())[0]
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                stats['min'] = scaler.data_min_.tolist()
                stats['max'] = scaler.data_max_.tolist()

        return stats


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def fit_scaler(
    X_train: pd.DataFrame,
    feature_set_id: str,
    scaler_type: str = 'standard',
    per_symbol: bool = False,
    symbol: Optional[str] = None,
    artifacts_dir: str = 'artifacts'
) -> FeatureScaler:
    """
    Fit scaler on training data and save to artifacts

    Args:
        X_train: Training features DataFrame
        feature_set_id: Feature set ID for filename
        scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        per_symbol: Fit separate scalers per symbol
        symbol: Symbol name (if per_symbol=True)
        artifacts_dir: Directory to save artifacts

    Returns:
        Fitted FeatureScaler
    """
    print("=" * 70)
    print("FITTING SCALER ON TRAINING DATA")
    print("=" * 70)

    # Create scaler
    scaler = FeatureScaler(scaler_type=scaler_type, per_symbol=per_symbol)

    # Fit on training data
    scaler.fit(X_train, symbol=symbol)

    # Save to artifacts
    filename = f'scaler_{feature_set_id}.pkl'
    filepath = Path(artifacts_dir) / filename
    scaler.save(str(filepath))

    print("=" * 70)

    return scaler


def apply_scaler(
    X: pd.DataFrame,
    scaler: FeatureScaler,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply fitted scaler to data

    Args:
        X: Features DataFrame
        scaler: Fitted FeatureScaler
        symbol: Symbol name (if per_symbol scaler)

    Returns:
        Scaled DataFrame
    """
    return scaler.transform(X, symbol=symbol)


def load_scaler(
    feature_set_id: str,
    artifacts_dir: str = 'artifacts'
) -> FeatureScaler:
    """
    Load saved scaler by feature set ID

    Args:
        feature_set_id: Feature set ID
        artifacts_dir: Directory containing artifacts

    Returns:
        Loaded FeatureScaler
    """
    filename = f'scaler_{feature_set_id}.pkl'
    filepath = Path(artifacts_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")

    return FeatureScaler.load(str(filepath))


def scale_train_val_test(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_set_id: str,
    scaler_type: str = 'standard',
    artifacts_dir: str = 'artifacts'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Scale train/val/test sets properly (fit on train only!)

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        feature_set_id: Feature set ID
        scaler_type: Type of scaler
        artifacts_dir: Directory to save artifacts

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    print("=" * 70)
    print("SCALING TRAIN/VAL/TEST SETS")
    print("=" * 70)

    # Fit scaler on TRAIN ONLY
    scaler = fit_scaler(
        X_train,
        feature_set_id=feature_set_id,
        scaler_type=scaler_type,
        artifacts_dir=artifacts_dir
    )

    print("\nApplying scaler to all sets...")

    # Apply to train
    X_train_scaled = apply_scaler(X_train, scaler)
    print(f"✓ Train scaled: {X_train_scaled.shape}")

    # Apply to val
    X_val_scaled = apply_scaler(X_val, scaler)
    print(f"✓ Val scaled: {X_val_scaled.shape}")

    # Apply to test
    X_test_scaled = apply_scaler(X_test, scaler)
    print(f"✓ Test scaled: {X_test_scaled.shape}")

    print("=" * 70)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("Testing preprocessing pipeline...\n")

    # Create mock data
    np.random.seed(42)

    X_train = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.uniform(0, 100, 1000),
        'feature_3': np.random.exponential(2, 1000)
    })

    X_val = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.uniform(0, 100, 200),
        'feature_3': np.random.exponential(2, 200)
    })

    X_test = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.uniform(0, 100, 200),
        'feature_3': np.random.exponential(2, 200)
    })

    # Scale properly
    X_train_s, X_val_s, X_test_s, scaler = scale_train_val_test(
        X_train, X_val, X_test,
        feature_set_id='test123',
        scaler_type='standard'
    )

    print("\n✓ Scaling complete!")
    print(f"\nTrain mean: {X_train_s.mean().mean():.6f} (should be ~0)")
    print(f"Train std:  {X_train_s.std().mean():.6f} (should be ~1)")
