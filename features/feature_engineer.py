"""
Comprehensive feature engineering for OI trading system
Implements 100+ features from OI, Price, Volume, Funding, and other data sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import linregress

# Make pandas_ta optional
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    # Provide basic alternative implementations
    class _BasicTA:
        """Basic technical analysis functions when pandas_ta is not available"""

        @staticmethod
        def rsi(series, length=14):
            """Calculate RSI"""
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        @staticmethod
        def sma(series, length):
            """Simple Moving Average"""
            return series.rolling(window=length).mean()

        @staticmethod
        def ema(series, length):
            """Exponential Moving Average"""
            return series.ewm(span=length, adjust=False).mean()

        @staticmethod
        def macd(series, fast=12, slow=26, signal=9):
            """MACD indicator"""
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            result = pd.DataFrame({
                'MACD_12_26_9': macd_line,
                'MACDs_12_26_9': signal_line,
                'MACDh_12_26_9': histogram
            })
            return result

        @staticmethod
        def stoch(high, low, close, k=14, d=3, smooth_k=3):
            """Stochastic Oscillator"""
            lowest_low = low.rolling(window=k).min()
            highest_high = high.rolling(window=k).max()
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_k = stoch_k.rolling(window=smooth_k).mean()
            stoch_d = stoch_k.rolling(window=d).mean()
            result = pd.DataFrame({
                'STOCHk_14_3_3': stoch_k,
                'STOCHd_14_3_3': stoch_d
            })
            return result

        @staticmethod
        def roc(series, length=10):
            """Rate of Change"""
            return ((series - series.shift(length)) / series.shift(length)) * 100

        @staticmethod
        def willr(high, low, close, length=14):
            """Williams %R"""
            highest_high = high.rolling(window=length).max()
            lowest_low = low.rolling(window=length).min()
            return -100 * (highest_high - close) / (highest_high - lowest_low)

        @staticmethod
        def atr(high, low, close, length=14):
            """Average True Range"""
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=length).mean()

        @staticmethod
        def bbands(series, length=20, std=2):
            """Bollinger Bands"""
            sma = series.rolling(window=length).mean()
            rolling_std = series.rolling(window=length).std()
            upper = sma + (rolling_std * std)
            lower = sma - (rolling_std * std)
            result = pd.DataFrame({
                'BBL_20_2.0': lower,
                'BBM_20_2.0': sma,
                'BBU_20_2.0': upper,
                'BBB_20_2.0': (upper - lower) / sma,
                'BBP_20_2.0': (series - lower) / (upper - lower)
            })
            return result

        @staticmethod
        def adx(high, low, close, length=14):
            """Average Directional Index"""
            # Simplified ADX calculation
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)

            up_move = high - high.shift()
            down_move = low.shift() - low

            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)

            atr_val = tr.rolling(window=length).mean()
            plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr_val)
            minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr_val)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx_val = dx.rolling(window=length).mean()

            result = pd.DataFrame({
                'ADX_14': adx_val,
                'DMP_14': plus_di,
                'DMN_14': minus_di
            })
            return result

        @staticmethod
        def obv(close, volume):
            """On-Balance Volume"""
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            return obv

        @staticmethod
        def cmf(high, low, close, volume, length=20):
            """Chaikin Money Flow"""
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            mfv = mfm * volume
            return mfv.rolling(window=length).sum() / volume.rolling(window=length).sum()

        @staticmethod
        def mfi(high, low, close, volume, length=14):
            """Money Flow Index"""
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume

            positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=high.index)
            negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=high.index)

            positive_mf = positive_flow.rolling(window=length).sum()
            negative_mf = negative_flow.rolling(window=length).sum()

            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi

    ta = _BasicTA()


class FeatureEngineer:
    """
    Comprehensive feature engineering for OI trading

    Features categories:
    - Open Interest Features: 25+
    - Price Action Features: 30+
    - Volume Features: 20+
    - Funding Rate Features: 10+
    - Liquidation Features: 10+
    - Long/Short Ratio Features: 5+
    - Time-Based Features: 10+
    - Interaction Features: 10+

    Total: 100+ features
    """

    def __init__(self):
        self.feature_names = []
        self.feature_categories = {}

    def engineer_all_features(
        self,
        ohlcv: pd.DataFrame,
        oi: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        liquidations: Optional[pd.DataFrame] = None,
        ls_ratio: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Main function to engineer all features

        Args:
            ohlcv: DataFrame with columns [timestamp, open, high, low, close, volume]
            oi: DataFrame with columns [timestamp, open_interest]
            funding: DataFrame with columns [timestamp, funding_rate]
            liquidations: DataFrame with liquidation data
            ls_ratio: DataFrame with columns [timestamp, longShortRatio]

        Returns:
            DataFrame with all engineered features
        """
        df = ohlcv.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Merge OI data
        if oi is not None:
            oi = oi.copy()
            if 'timestamp' in oi.columns:
                oi['timestamp'] = pd.to_datetime(oi['timestamp'])
                oi = oi.set_index('timestamp')
            df = df.join(oi, how='left', rsuffix='_oi')

        # Merge funding rate
        if funding is not None:
            funding = funding.copy()
            if 'timestamp' in funding.columns:
                funding['timestamp'] = pd.to_datetime(funding['timestamp'])
                funding = funding.set_index('timestamp')
            df = df.join(funding, how='left', rsuffix='_fund')

        # Engineer features by category
        print("Engineering OI features...")
        df = self._oi_features(df)

        print("Engineering price features...")
        df = self._price_features(df)

        print("Engineering volume features...")
        df = self._volume_features(df)

        if funding is not None:
            print("Engineering funding features...")
            df = self._funding_features(df)

        if liquidations is not None:
            print("Engineering liquidation features...")
            df = self._liquidation_features(df, liquidations)

        if ls_ratio is not None:
            print("Engineering L/S ratio features...")
            df = self._ls_ratio_features(df, ls_ratio)

        print("Engineering time features...")
        df = self._time_features(df)

        print("Engineering interaction features...")
        df = self._interaction_features(df)

        # Fill NaN values from rolling calculations
        # Use forward fill then backward fill, then 0
        df = df.ffill().bfill().fillna(0)

        print(f"Total features engineered: {len(df.columns)}")

        return df

    def _oi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Open Interest Features (25+)

        Categories:
        - Basic OI metrics (changes, velocity, acceleration)
        - OI momentum & trend (MAs, MACD, trend slope)
        - OI volatility (std, Bollinger Bands, CV)
        - OI divergence (OI-Price divergence)
        - OI extremes (z-score, percentiles, distance from high/low)
        """
        if 'open_interest' not in df.columns:
            print("Warning: open_interest column not found, skipping OI features")
            return df

        oi = df['open_interest']
        price = df['close']

        # ========== Basic OI Metrics ==========
        # Absolute and percentage changes
        df['oi_change_abs'] = oi.diff(1)
        df['oi_pct_change_1'] = oi.pct_change(1)
        df['oi_pct_change_5'] = oi.pct_change(5)
        df['oi_pct_change_20'] = oi.pct_change(20)
        df['oi_log_return'] = np.log(oi / oi.shift(1))

        # Velocity (rate of change)
        df['oi_velocity_1h'] = (oi - oi.shift(12)) / oi.shift(12)  # 1h on 5m data
        df['oi_velocity_4h'] = (oi - oi.shift(48)) / oi.shift(48)  # 4h
        df['oi_velocity_24h'] = (oi - oi.shift(288)) / oi.shift(288)  # 24h

        # Acceleration (change in velocity)
        df['oi_acceleration'] = df['oi_velocity_1h'] - df['oi_velocity_1h'].shift(1)

        # ========== OI Momentum & Trend ==========
        # Moving averages
        df['oi_sma_20'] = oi.rolling(20).mean()
        df['oi_sma_50'] = oi.rolling(50).mean()
        df['oi_sma_200'] = oi.rolling(200).mean()
        df['oi_ema_12'] = oi.ewm(span=12, adjust=False).mean()
        df['oi_ema_26'] = oi.ewm(span=26, adjust=False).mean()

        # OI MACD (like price MACD)
        df['oi_macd'] = df['oi_ema_12'] - df['oi_ema_26']
        df['oi_macd_signal'] = df['oi_macd'].ewm(span=9, adjust=False).mean()
        df['oi_macd_histogram'] = df['oi_macd'] - df['oi_macd_signal']

        # Trend strength (linear regression slope)
        df['oi_trend_slope_20'] = oi.rolling(20).apply(
            lambda x: linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
            raw=True
        )
        df['oi_trend_slope_50'] = oi.rolling(50).apply(
            lambda x: linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
            raw=True
        )

        # R-squared (how strong the trend is)
        df['oi_r_squared_20'] = oi.rolling(20).apply(
            lambda x: linregress(range(len(x)), x)[2]**2 if len(x) > 1 else 0,
            raw=True
        )

        # ========== OI Volatility ==========
        # Standard deviation
        df['oi_std_20'] = oi.rolling(20).std()
        df['oi_std_50'] = oi.rolling(50).std()

        # OI Bollinger Bands
        df['oi_bb_upper'] = df['oi_sma_20'] + 2 * df['oi_std_20']
        df['oi_bb_lower'] = df['oi_sma_20'] - 2 * df['oi_std_20']
        df['oi_bb_position'] = (oi - df['oi_bb_lower']) / (df['oi_bb_upper'] - df['oi_bb_lower'] + 1e-10)
        df['oi_bb_width'] = (df['oi_bb_upper'] - df['oi_bb_lower']) / (df['oi_sma_20'] + 1e-10)

        # Coefficient of Variation
        df['oi_cv'] = df['oi_std_20'] / (df['oi_sma_20'] + 1e-10)

        # ========== OI Divergence (Critical!) ==========
        # OI-Price Divergence
        df['oi_price_divergence_20'] = self._calculate_divergence(oi, price, period=20)
        df['oi_price_divergence_48'] = self._calculate_divergence(oi, price, period=48)  # 4h
        df['oi_price_divergence_288'] = self._calculate_divergence(oi, price, period=288)  # 1d

        # ========== OI Extremes ==========
        # Z-score (how far from mean)
        oi_mean_50 = oi.rolling(50).mean()
        df['oi_zscore'] = (oi - oi_mean_50) / (df['oi_std_50'] + 1e-10)

        # Percentile rank
        df['oi_percentile_100'] = oi.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        # Distance from historical high/low
        oi_high_100 = oi.rolling(100).max()
        oi_low_100 = oi.rolling(100).min()
        df['oi_dist_from_high'] = (oi - oi_high_100) / (oi_high_100 + 1e-10)
        df['oi_dist_from_low'] = (oi - oi_low_100) / (oi_low_100 + 1e-10)

        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price Action Features (30+)

        Categories:
        - Returns (simple, log, volatility)
        - Trend indicators (MAs, crossovers)
        - Momentum (RSI, MACD, Stochastic, ROC)
        - Volatility (ATR, Bollinger Bands, Keltner Channels)
        - Market structure (HH/HL detection, ADX)
        """
        close = df['close']
        high = df['high']
        low = df['low']

        # ========== Returns ==========
        for period in [1, 5, 10, 20, 50, 100]:
            df[f'return_{period}'] = close.pct_change(period)

        # Log returns
        df['log_return_1'] = np.log(close / close.shift(1))
        df['log_return_5'] = np.log(close / close.shift(5))
        df['log_return_20'] = np.log(close / close.shift(20))

        # Realized volatility (annualized)
        df['realized_vol_20'] = df['log_return_1'].rolling(20).std() * np.sqrt(288)  # 288 5-min periods per day

        # ========== Trend Indicators ==========
        # Moving averages
        df['sma_10'] = close.rolling(10).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        df['sma_200'] = close.rolling(200).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

        # Price relative to MAs
        df['price_vs_sma10'] = (close - df['sma_10']) / (df['sma_10'] + 1e-10)
        df['price_vs_sma20'] = (close - df['sma_20']) / (df['sma_20'] + 1e-10)
        df['price_vs_sma50'] = (close - df['sma_50']) / (df['sma_50'] + 1e-10)
        df['price_vs_sma200'] = (close - df['sma_200']) / (df['sma_200'] + 1e-10)

        # MA crossovers
        df['sma_cross_20_50'] = np.sign(df['sma_20'] - df['sma_50'])

        # ========== Momentum ==========
        # RSI
        df['rsi_14'] = ta.rsi(close, length=14)
        df['rsi_50'] = ta.rsi(close, length=50)

        # MACD
        macd_result = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_result is not None:
            df = pd.concat([df, macd_result], axis=1)

        # Stochastic
        stoch_result = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch_result is not None:
            df = pd.concat([df, stoch_result], axis=1)

        # Rate of Change
        df['roc_10'] = ta.roc(close, length=10)
        df['roc_30'] = ta.roc(close, length=30)

        # Williams %R
        df['willr_14'] = ta.willr(high, low, close, length=14)

        # ========== Volatility ==========
        # ATR (Average True Range)
        df['atr_14'] = ta.atr(high, low, close, length=14)
        df['atr_50'] = ta.atr(high, low, close, length=50)

        # Normalized ATR
        df['natr'] = df['atr_14'] / (close + 1e-10)

        # Bollinger Bands
        bbands_result = ta.bbands(close, length=20, std=2)
        if bbands_result is not None:
            df = pd.concat([df, bbands_result], axis=1)
            # Calculate BB width and position
            bb_upper_col = [col for col in bbands_result.columns if 'BBU' in col]
            bb_lower_col = [col for col in bbands_result.columns if 'BBL' in col]
            bb_middle_col = [col for col in bbands_result.columns if 'BBM' in col]

            if bb_upper_col and bb_lower_col and bb_middle_col:
                df['bb_width'] = (df[bb_upper_col[0]] - df[bb_lower_col[0]]) / (df[bb_middle_col[0]] + 1e-10)
                df['bb_position'] = (close - df[bb_lower_col[0]]) / (df[bb_upper_col[0]] - df[bb_lower_col[0]] + 1e-10)

        # Keltner Channels
        ema_20 = close.ewm(span=20, adjust=False).mean()
        df['kc_upper'] = ema_20 + 2 * df['atr_14']
        df['kc_lower'] = ema_20 - 2 * df['atr_14']
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / (ema_20 + 1e-10)

        # ========== Market Structure ==========
        # ADX (Trend Strength)
        adx_result = ta.adx(high, low, close, length=14)
        if adx_result is not None:
            df = pd.concat([df, adx_result], axis=1)

        # Higher High, Higher Low detection
        df['market_structure'] = self._detect_market_structure(high, low, lookback=20)

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume Features (20+)

        Categories:
        - Volume metrics (changes, ratios, momentum)
        - Volume-based indicators (OBV, CMF, MFI, VWAP)
        - OI/Volume interaction
        """
        volume = df['volume']
        close = df['close']
        high = df['high']
        low = df['low']

        # ========== Volume Metrics ==========
        # Volume changes
        df['volume_change'] = volume.pct_change(1)
        df['volume_change_20'] = volume.pct_change(20)

        # Volume moving averages
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_sma_50'] = volume.rolling(50).mean()
        df['volume_ema_20'] = volume.ewm(span=20, adjust=False).mean()

        # Volume ratio
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-10)

        # Volume momentum
        df['volume_roc_10'] = ta.roc(volume, length=10)

        # Volume trend
        df['volume_trend_20'] = volume.rolling(20).apply(
            lambda x: linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
            raw=True
        )

        # Volume volatility
        df['volume_std_20'] = volume.rolling(20).std()
        df['volume_cv'] = df['volume_std_20'] / (df['volume_sma_20'] + 1e-10)

        # ========== Volume-Based Indicators ==========
        # OBV (On-Balance Volume)
        df['obv'] = ta.obv(close, volume)
        df['obv_sma_20'] = df['obv'].rolling(20).mean()
        df['obv_divergence'] = (df['obv'] - df['obv_sma_20']) / (df['obv_sma_20'].abs() + 1e-10)

        # CMF (Chaikin Money Flow)
        df['cmf'] = ta.cmf(high, low, close, volume, length=20)

        # MFI (Money Flow Index)
        df['mfi'] = ta.mfi(high, low, close, volume, length=14)

        # VWAP (Volume Weighted Average Price)
        # Reset VWAP daily or use cumulative
        df['typical_price'] = (high + low + close) / 3
        df['vwap'] = (df['typical_price'] * volume).cumsum() / (volume.cumsum() + 1e-10)
        df['price_vs_vwap'] = (close - df['vwap']) / (df['vwap'] + 1e-10)

        # Price-Volume correlation
        df['price_volume_corr_20'] = close.rolling(20).corr(volume)

        # Volume percentile
        df['volume_percentile_100'] = volume.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        return df

    def _funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Funding Rate Features (10+)
        """
        if 'funding_rate' not in df.columns:
            print("Warning: funding_rate column not found, skipping funding features")
            return df

        funding = df['funding_rate']

        # Current and changes
        df['funding_change'] = funding.diff(1)
        df['funding_change_24h'] = funding.diff(288)  # 24h in 5-min intervals

        # Cumulative funding
        df['cumulative_funding_7d'] = funding.rolling(56*3).sum()  # 56 8-hour periods
        df['cumulative_funding_30d'] = funding.rolling(56*3*4).sum()

        # Funding statistics
        df['funding_mean_100'] = funding.rolling(100).mean()
        df['funding_std_100'] = funding.rolling(100).std()

        # Funding z-score
        df['funding_zscore'] = (funding - df['funding_mean_100']) / (df['funding_std_100'] + 1e-10)

        # Funding percentile
        df['funding_percentile'] = funding.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        # Funding extremes (binary flags)
        df['funding_extreme_positive'] = (funding > 0.05).astype(int)
        df['funding_extreme_negative'] = (funding < -0.02).astype(int)

        # Funding momentum
        df['funding_momentum'] = funding.rolling(10).apply(
            lambda x: linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
            raw=True
        )

        return df

    def _liquidation_features(self, df: pd.DataFrame, liq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Liquidation Features (10+)
        """
        if liq_df is None or len(liq_df) == 0:
            print("Warning: No liquidation data provided, skipping liquidation features")
            return df

        # Prepare liquidation data
        liq_df = liq_df.copy()
        if 'timestamp' in liq_df.columns:
            liq_df['timestamp'] = pd.to_datetime(liq_df['timestamp'])

        # Aggregate liquidations by timestamp
        liq_agg = liq_df.groupby('timestamp').agg({
            'quantity': ['sum', 'mean', 'count']
        })
        liq_agg.columns = ['liq_volume', 'liq_avg_size', 'liq_count']

        # Separate long and short liquidations if side column exists
        if 'side' in liq_df.columns:
            long_liq = liq_df[liq_df['side'] == 'SELL'].groupby('timestamp')['quantity'].sum()
            short_liq = liq_df[liq_df['side'] == 'BUY'].groupby('timestamp')['quantity'].sum()
            liq_agg['long_liq_volume'] = long_liq
            liq_agg['short_liq_volume'] = short_liq
            liq_agg = liq_agg.fillna(0)

        # Merge with main dataframe
        df = df.join(liq_agg, how='left')
        df[['liq_volume', 'liq_avg_size', 'liq_count']] = df[['liq_volume', 'liq_avg_size', 'liq_count']].fillna(0)

        # Rolling aggregations
        df['liq_volume_1h'] = df['liq_volume'].rolling(12).sum()  # 1h = 12 5-min bars
        df['liq_volume_4h'] = df['liq_volume'].rolling(48).sum()
        df['liq_count_1h'] = df['liq_count'].rolling(12).sum()

        # Net liquidation pressure
        if 'long_liq_volume' in df.columns and 'short_liq_volume' in df.columns:
            df['net_liquidation'] = df['short_liq_volume'] - df['long_liq_volume']
            df['net_liquidation_1h'] = df['net_liquidation'].rolling(12).sum()

        # Liquidation spikes
        liq_mean = df['liq_volume'].rolling(24).mean()
        df['liq_spike'] = (df['liq_volume'] > 2 * liq_mean).astype(int)

        # Liquidation momentum
        df['liq_momentum'] = df['liq_volume'].pct_change(12)

        return df

    def _ls_ratio_features(self, df: pd.DataFrame, ls_df: pd.DataFrame) -> pd.DataFrame:
        """
        Long/Short Ratio Features (5+)
        """
        if ls_df is None or len(ls_df) == 0:
            print("Warning: No L/S ratio data provided, skipping L/S ratio features")
            return df

        ls_df = ls_df.copy()
        if 'timestamp' in ls_df.columns:
            ls_df['timestamp'] = pd.to_datetime(ls_df['timestamp'])
            ls_df = ls_df.set_index('timestamp')

        # Merge L/S ratio
        df = df.join(ls_df[['longShortRatio']], how='left')

        if 'longShortRatio' not in df.columns:
            print("Warning: longShortRatio column not found after merge")
            return df

        ls_ratio = df['longShortRatio'].ffill()

        # Changes in ratio
        df['ls_ratio_change_1h'] = ls_ratio.diff(12)
        df['ls_ratio_change_4h'] = ls_ratio.diff(48)
        df['ls_ratio_change_24h'] = ls_ratio.diff(288)

        # Moving averages
        df['ls_ratio_sma_20'] = ls_ratio.rolling(20).mean()
        df['ls_ratio_ema_20'] = ls_ratio.ewm(span=20, adjust=False).mean()

        # Percentile rank
        df['ls_ratio_percentile'] = ls_ratio.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )

        # Extreme levels (binary flags)
        df['ls_ratio_extreme_long'] = (ls_ratio > 1.5).astype(int)  # Too many longs
        df['ls_ratio_extreme_short'] = (ls_ratio < 0.7).astype(int)  # Too many shorts

        # Z-score
        ls_mean = ls_ratio.rolling(100).mean()
        ls_std = ls_ratio.rolling(100).std()
        df['ls_ratio_zscore'] = (ls_ratio - ls_mean) / (ls_std + 1e-10)

        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-Based Features (10+)

        Includes cyclical encoding for time of day, day of week, and market sessions
        """
        # Extract time components
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
        else:
            print("Warning: Index is not DatetimeIndex, using default time features")
            df['hour'] = 0
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1

        # Cyclical encoding for hour (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week (0-6)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Cyclical encoding for month (1-12)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Market sessions (UTC-based, adjust as needed)
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Funding cycle (8-hour cycles for perpetual futures)
        # Assuming funding happens at 00:00, 08:00, 16:00 UTC
        hours_since_midnight = df['hour']
        hours_in_funding_cycle = hours_since_midnight % 8
        df['funding_cycle_sin'] = np.sin(2 * np.pi * hours_in_funding_cycle / 8)
        df['funding_cycle_cos'] = np.cos(2 * np.pi * hours_in_funding_cycle / 8)

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interaction Features (10+)

        Features that combine information from multiple categories
        """
        # OI-Volume ratio (key metric!)
        if 'open_interest' in df.columns and 'volume_sma_20' in df.columns:
            df['oi_volume_ratio'] = df['open_interest'] / (df['volume_sma_20'] + 1e-10)

        # OI-Volume divergence
        if 'oi_pct_change_1' in df.columns and 'volume_change' in df.columns:
            df['oi_volume_divergence'] = np.sign(df['oi_pct_change_1']) - np.sign(df['volume_change'])

        # RSI * Funding interaction
        if 'rsi_14' in df.columns and 'funding_rate' in df.columns:
            df['rsi_funding_interaction'] = df['rsi_14'] * df['funding_rate']

        # OI change * Price momentum
        if 'oi_pct_change_20' in df.columns and 'return_20' in df.columns:
            df['oi_price_momentum'] = df['oi_pct_change_20'] * df['return_20']

        # Volume * Volatility
        if 'volume_ratio' in df.columns and 'natr' in df.columns:
            df['volume_volatility'] = df['volume_ratio'] * df['natr']

        # Funding * Price momentum
        if 'funding_rate' in df.columns and 'return_20' in df.columns:
            df['funding_price_momentum'] = df['funding_rate'] * df['return_20']

        # OI MACD * Price MACD
        if 'oi_macd' in df.columns and 'MACD_12_26_9' in df.columns:
            df['oi_price_macd_interaction'] = df['oi_macd'] * df['MACD_12_26_9']

        # Liquidation * Volatility
        if 'liq_volume_1h' in df.columns and 'realized_vol_20' in df.columns:
            df['liq_volatility'] = df['liq_volume_1h'] * df['realized_vol_20']

        # L/S ratio * RSI
        if 'longShortRatio' in df.columns and 'rsi_14' in df.columns:
            df['ls_rsi_interaction'] = df['longShortRatio'] * (df['rsi_14'] - 50)  # Centered RSI

        # OI extremes * Price extremes
        if 'oi_zscore' in df.columns and 'bb_position' in df.columns:
            df['oi_price_extreme'] = df['oi_zscore'] * (df['bb_position'] - 0.5)

        return df

    @staticmethod
    def _calculate_divergence(series1: pd.Series, series2: pd.Series, period: int) -> pd.Series:
        """
        Calculate divergence between two series

        Returns:
            1: Bullish divergence (series2 down, series1 down)
           -1: Bearish divergence (series2 up, series1 down)
            0: No divergence
        """
        dir1 = np.sign(series1.diff(period))
        dir2 = np.sign(series2.diff(period))

        # Divergence occurs when directions are opposite
        divergence = np.where(
            (dir1 != dir2) & (dir2 > 0),
            -1,  # Bearish: price up but series1 down
            np.where(
                (dir1 != dir2) & (dir2 < 0),
                1,   # Bullish: price down but series1 down
                0    # No divergence
            )
        )

        return pd.Series(divergence, index=series1.index)

    @staticmethod
    def _detect_market_structure(high: pd.Series, low: pd.Series, lookback: int = 20) -> pd.Series:
        """
        Detect market structure (uptrend, downtrend, range)

        Returns:
            1: Uptrend (Higher Highs and Higher Lows)
           -1: Downtrend (Lower Highs and Lower Lows)
            0: Range/Sideways
        """
        structure = []

        for i in range(len(high)):
            if i < lookback:
                structure.append(0)
                continue

            current_high = high.iloc[i]
            current_low = low.iloc[i]
            prev_highs = high.iloc[i-lookback:i]
            prev_lows = low.iloc[i-lookback:i]

            hh = current_high > prev_highs.max()  # Higher high
            hl = current_low > prev_lows.max()     # Higher low
            lh = current_high < prev_highs.min()  # Lower high
            ll = current_low < prev_lows.min()     # Lower low

            if hh and hl:
                structure.append(1)   # Uptrend
            elif lh and ll:
                structure.append(-1)  # Downtrend
            else:
                structure.append(0)   # Range

        return pd.Series(structure, index=high.index)

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature names (excluding OHLCV and timestamp)"""
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                     'open_interest', 'funding_rate', 'longShortRatio']
        feature_cols = [col for col in df.columns if col not in base_cols]
        return feature_cols
