"""
Feature List Versioning System

Manages feature set versions with immutable artifacts:
- Generates unique feature_set_id (hash-based)
- Saves feature lists with metadata
- Tracks git commits, timestamps, configurations
- Ensures reproducibility across experiments

Usage:
    >>> from utils.feature_versioning import save_feature_list, load_feature_list
    >>> feature_set_id = save_feature_list(feature_names, config)
    >>> features, metadata = load_feature_list(feature_set_id)
"""

import json
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import subprocess


def compute_feature_set_hash(
    feature_names: List[str],
    config: Optional[Dict] = None
) -> str:
    """
    Compute unique hash for feature set

    Args:
        feature_names: List of feature names (will be sorted)
        config: Optional configuration dict (window sizes, lookbacks, etc.)

    Returns:
        12-character hash string
    """
    # Sort feature names for consistency
    sorted_features = sorted(feature_names)

    # Create string representation
    feature_str = ','.join(sorted_features)

    # Add config if provided
    if config:
        config_str = json.dumps(config, sort_keys=True)
        combined_str = f"{feature_str}|{config_str}"
    else:
        combined_str = feature_str

    # Compute SHA256 hash
    hash_obj = hashlib.sha256(combined_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()

    # Take first 12 characters
    return hash_hex[:12]


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash

    Returns:
        Git commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_branch() -> Optional[str]:
    """
    Get current git branch name

    Returns:
        Git branch name or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def save_feature_list(
    feature_names: List[str],
    config: Optional[Dict] = None,
    output_dir: str = 'artifacts',
    description: str = '',
    metadata: Optional[Dict] = None
) -> str:
    """
    Save feature list with version metadata

    Args:
        feature_names: List of feature names
        config: Configuration dict (window sizes, parameters, etc.)
        output_dir: Directory to save artifacts
        description: Human-readable description
        metadata: Additional metadata to save

    Returns:
        feature_set_id (hash)
    """
    # Compute hash
    feature_set_id = compute_feature_set_hash(feature_names, config)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    feature_metadata = {
        'feature_set_id': feature_set_id,
        'n_features': len(feature_names),
        'features': sorted(feature_names),
        'config': config or {},
        'description': description,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'git_commit': get_git_commit(),
        'git_branch': get_git_branch(),
    }

    # Add custom metadata
    if metadata:
        feature_metadata['metadata'] = metadata

    # Save to file
    output_path = os.path.join(output_dir, f'feature_list_v{feature_set_id}.json')

    with open(output_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    print(f"✓ Saved feature list: {output_path}")
    print(f"  Feature Set ID: {feature_set_id}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Git commit: {feature_metadata['git_commit']}")

    return feature_set_id


def load_feature_list(
    feature_set_id: str,
    artifacts_dir: str = 'artifacts'
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load feature list by ID

    Args:
        feature_set_id: Feature set hash ID
        artifacts_dir: Directory containing artifacts

    Returns:
        Tuple of (feature_names, metadata)
    """
    file_path = os.path.join(artifacts_dir, f'feature_list_v{feature_set_id}.json')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature list not found: {file_path}")

    with open(file_path, 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['features']

    print(f"✓ Loaded feature list: {file_path}")
    print(f"  Feature Set ID: {metadata['feature_set_id']}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Created: {metadata['created_at']}")

    return feature_names, metadata


def list_feature_sets(artifacts_dir: str = 'artifacts') -> List[Dict[str, Any]]:
    """
    List all saved feature sets

    Args:
        artifacts_dir: Directory containing artifacts

    Returns:
        List of metadata dicts
    """
    if not os.path.exists(artifacts_dir):
        return []

    feature_sets = []

    for filename in os.listdir(artifacts_dir):
        if filename.startswith('feature_list_v') and filename.endswith('.json'):
            file_path = os.path.join(artifacts_dir, filename)

            with open(file_path, 'r') as f:
                metadata = json.load(f)
                feature_sets.append(metadata)

    # Sort by creation time (newest first)
    feature_sets.sort(key=lambda x: x['created_at'], reverse=True)

    return feature_sets


def print_feature_sets(artifacts_dir: str = 'artifacts'):
    """
    Print formatted list of all feature sets

    Args:
        artifacts_dir: Directory containing artifacts
    """
    feature_sets = list_feature_sets(artifacts_dir)

    if not feature_sets:
        print("No feature sets found.")
        return

    print("=" * 90)
    print("AVAILABLE FEATURE SETS")
    print("=" * 90)
    print(f"{'ID':<14} {'Features':<10} {'Created':<20} {'Description':<40}")
    print("-" * 90)

    for fs in feature_sets:
        feature_set_id = fs['feature_set_id']
        n_features = fs['n_features']
        created = fs['created_at'][:19].replace('T', ' ')
        description = fs.get('description', '')[:40]

        print(f"{feature_set_id:<14} {n_features:<10} {created:<20} {description:<40}")

    print("=" * 90)


def compare_feature_sets(
    id1: str,
    id2: str,
    artifacts_dir: str = 'artifacts'
) -> Dict[str, Any]:
    """
    Compare two feature sets

    Args:
        id1: First feature set ID
        id2: Second feature set ID
        artifacts_dir: Directory containing artifacts

    Returns:
        Dict with comparison results
    """
    features1, meta1 = load_feature_list(id1, artifacts_dir)
    features2, meta2 = load_feature_list(id2, artifacts_dir)

    set1 = set(features1)
    set2 = set(features2)

    comparison = {
        'id1': id1,
        'id2': id2,
        'n_features_1': len(features1),
        'n_features_2': len(features2),
        'common_features': list(set1 & set2),
        'only_in_1': list(set1 - set2),
        'only_in_2': list(set2 - set1),
        'n_common': len(set1 & set2),
        'n_only_1': len(set1 - set2),
        'n_only_2': len(set2 - set1)
    }

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """
    Print formatted feature set comparison

    Args:
        comparison: Comparison dict from compare_feature_sets()
    """
    print("=" * 70)
    print("FEATURE SET COMPARISON")
    print("=" * 70)
    print(f"\nSet 1: {comparison['id1']} ({comparison['n_features_1']} features)")
    print(f"Set 2: {comparison['id2']} ({comparison['n_features_2']} features)")

    print(f"\nCommon features:        {comparison['n_common']}")
    print(f"Only in Set 1:          {comparison['n_only_1']}")
    print(f"Only in Set 2:          {comparison['n_only_2']}")

    if comparison['n_only_1'] > 0:
        print(f"\nFeatures only in Set 1 (first 10):")
        for feat in comparison['only_in_1'][:10]:
            print(f"   - {feat}")

    if comparison['n_only_2'] > 0:
        print(f"\nFeatures only in Set 2 (first 10):")
        for feat in comparison['only_in_2'][:10]:
            print(f"   - {feat}")

    print("=" * 70)


def validate_feature_set(
    df_columns: List[str],
    feature_set_id: str,
    artifacts_dir: str = 'artifacts'
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required features

    Args:
        df_columns: List of column names in DataFrame
        feature_set_id: Feature set ID to validate against
        artifacts_dir: Directory containing artifacts

    Returns:
        Tuple of (is_valid, missing_features)
    """
    feature_names, _ = load_feature_list(feature_set_id, artifacts_dir)

    required_features = set(feature_names)
    available_features = set(df_columns)

    missing_features = list(required_features - available_features)

    is_valid = len(missing_features) == 0

    if is_valid:
        print(f"✓ Feature set validation passed")
    else:
        print(f"✗ Feature set validation failed")
        print(f"  Missing {len(missing_features)} features:")
        for feat in missing_features[:10]:
            print(f"    - {feat}")

    return is_valid, missing_features


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == "__main__":
    print("Testing feature versioning...\n")

    # Example feature list
    features = [
        'oi_sma_20',
        'price_vs_vwap',
        'volume_sma_50',
        'rsi_14',
        'macd_line'
    ]

    # Example config
    config = {
        'oi_windows': [20, 50, 100],
        'price_windows': [10, 20, 50],
        'lookback_periods': 48,
        'target_horizon': 48,
        'normalization': 'standard'
    }

    # Save feature list
    feature_set_id = save_feature_list(
        feature_names=features,
        config=config,
        description="Example feature set for testing"
    )

    print(f"\n✓ Feature Set ID: {feature_set_id}")

    # Load it back
    print("\nLoading feature list...")
    loaded_features, metadata = load_feature_list(feature_set_id)

    print(f"\n✓ Loaded {len(loaded_features)} features")

    # List all feature sets
    print("\n" + "="*70)
    print_feature_sets()
