"""
Target Variable Engineering for ML Models

Supports:
- Classification targets (LONG, SHORT, NEUTRAL)
- Regression targets (future returns)
- Multi-horizon targets
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class TargetEngineer:
    """
    Engineer target variables for different ML tasks
    """

    def __init__(self):
        pass

    def create_classification_target(
        self,
        df: pd.DataFrame,
        horizon: int = 48,  # 4 hours in 5-min bars
        threshold: float = 0.005,  # 0.5% move
        n_classes: int = 3
    ) -> pd.DataFrame:
        """
        Create classification target for predicting directional moves

        Args:
            df: DataFrame with 'close' price
            horizon: Number of periods ahead to predict
            threshold: Minimum price move to consider significant
            n_classes: 2 (binary) or 3 (ternary with neutral class)

        Returns:
            DataFrame with added 'target' column

        Target classes:
            3-class: 0=SHORT, 1=NEUTRAL, 2=LONG
            2-class: 0=SHORT/NEUTRAL, 1=LONG
        """
        df = df.copy()

        # Calculate future returns
        future_close = df['close'].shift(-horizon)
        future_return = (future_close - df['close']) / df['close']

        if n_classes == 3:
            # Ternary classification
            df['target'] = np.where(
                future_return > threshold,
                2,  # LONG
                np.where(
                    future_return < -threshold,
                    0,  # SHORT
                    1   # NEUTRAL
                )
            )
        else:
            # Binary classification
            df['target'] = (future_return > threshold).astype(int)

        # Store the actual future return for analysis
        df['future_return'] = future_return

        # Remove last 'horizon' rows (no target available)
        df = df.iloc[:-horizon]

        return df

    def create_regression_target(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [12, 48, 288]  # 1h, 4h, 24h
    ) -> pd.DataFrame:
        """
        Create regression targets for predicting future returns

        Args:
            df: DataFrame with 'close' price
            horizons: List of horizons (in 5-min bars) to predict

        Returns:
            DataFrame with added 'target_return_Xh' columns
        """
        df = df.copy()

        for horizon in horizons:
            future_close = df['close'].shift(-horizon)
            future_return = (future_close - df['close']) / df['close']

            # Convert periods to hours (assuming 5-min bars)
            hours = horizon * 5 / 60
            col_name = f'target_return_{hours:.0f}h'
            df[col_name] = future_return

        # Remove last max(horizons) rows
        max_horizon = max(horizons)
        df = df.iloc[:-max_horizon]

        return df

    def create_oi_target(
        self,
        df: pd.DataFrame,
        horizon: int = 48
    ) -> pd.DataFrame:
        """
        Create target for predicting future OI changes

        Args:
            df: DataFrame with 'open_interest' column
            horizon: Number of periods ahead to predict

        Returns:
            DataFrame with 'target_oi_change' column
        """
        df = df.copy()

        if 'open_interest' not in df.columns:
            raise ValueError("DataFrame must contain 'open_interest' column")

        future_oi = df['open_interest'].shift(-horizon)
        df['target_oi_change'] = (future_oi - df['open_interest']) / df['open_interest']

        # Remove last 'horizon' rows
        df = df.iloc[:-horizon]

        return df

    def create_multi_horizon_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [12, 48, 96, 288],  # 1h, 4h, 8h, 24h
        threshold: float = 0.005
    ) -> pd.DataFrame:
        """
        Create targets for multiple prediction horizons

        Useful for multi-task learning or ensemble models

        Args:
            df: DataFrame with 'close' price
            horizons: List of horizons to create targets for
            threshold: Classification threshold

        Returns:
            DataFrame with targets for each horizon
        """
        df = df.copy()

        for horizon in horizons:
            future_close = df['close'].shift(-horizon)
            future_return = (future_close - df['close']) / df['close']

            hours = horizon * 5 / 60
            suffix = f'{hours:.0f}h'

            # Regression target
            df[f'target_return_{suffix}'] = future_return

            # Classification target
            df[f'target_class_{suffix}'] = np.where(
                future_return > threshold,
                2,  # LONG
                np.where(
                    future_return < -threshold,
                    0,  # SHORT
                    1   # NEUTRAL
                )
            )

        # Remove last max(horizons) rows
        max_horizon = max(horizons)
        df = df.iloc[:-max_horizon]

        return df

    def create_volatility_target(
        self,
        df: pd.DataFrame,
        horizon: int = 48,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Create target for predicting future volatility

        Args:
            df: DataFrame with 'close' price
            horizon: Number of periods ahead to start calculating volatility
            lookback: Number of periods to calculate volatility over

        Returns:
            DataFrame with 'target_volatility' column
        """
        df = df.copy()

        # Calculate log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))

        # Calculate future volatility
        future_vol = log_returns.shift(-horizon).rolling(lookback).std() * np.sqrt(288)
        df['target_volatility'] = future_vol

        # Remove last (horizon + lookback) rows
        df = df.iloc[:-(horizon + lookback)]

        return df

    def create_directional_accuracy_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 48
    ) -> pd.DataFrame:
        """
        Create simple directional labels (up/down/flat)

        Args:
            df: DataFrame with 'close' price
            horizon: Number of periods ahead

        Returns:
            DataFrame with 'direction' column: 1=up, 0=flat, -1=down
        """
        df = df.copy()

        future_close = df['close'].shift(-horizon)
        price_change = future_close - df['close']

        df['direction'] = np.sign(price_change).astype(int)

        # Remove last 'horizon' rows
        df = df.iloc[:-horizon]

        return df

    @staticmethod
    def get_target_distribution(df: pd.DataFrame, target_col: str = 'target') -> pd.Series:
        """
        Get distribution of target classes

        Args:
            df: DataFrame with target column
            target_col: Name of target column

        Returns:
            Series with class counts and percentages
        """
        counts = df[target_col].value_counts().sort_index()
        percentages = df[target_col].value_counts(normalize=True).sort_index() * 100

        distribution = pd.DataFrame({
            'count': counts,
            'percentage': percentages
        })

        return distribution

    @staticmethod
    def balance_classes(
        df: pd.DataFrame,
        target_col: str = 'target',
        method: str = 'undersample'
    ) -> pd.DataFrame:
        """
        Balance class distribution

        Args:
            df: DataFrame with target column
            target_col: Name of target column
            method: 'undersample' or 'oversample'

        Returns:
            Balanced DataFrame
        """
        if method == 'undersample':
            # Undersample to match smallest class
            min_count = df[target_col].value_counts().min()

            balanced_dfs = []
            for class_val in df[target_col].unique():
                class_df = df[df[target_col] == class_val]
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))

            return pd.concat(balanced_dfs).sort_index()

        elif method == 'oversample':
            # Oversample to match largest class
            max_count = df[target_col].value_counts().max()

            balanced_dfs = []
            for class_val in df[target_col].unique():
                class_df = df[df[target_col] == class_val]
                balanced_dfs.append(
                    class_df.sample(n=max_count, replace=True, random_state=42)
                )

            return pd.concat(balanced_dfs).sort_index()

        else:
            raise ValueError(f"Unknown method: {method}. Use 'undersample' or 'oversample'")
