"""
Phase 1 Database Connector

Connects to Phase 1 PostgreSQL/TimescaleDB and fetches market data
"""

import pandas as pd
from typing import Optional, Dict
from datetime import datetime, timedelta


class Phase1DataConnector:
    """
    Connect to Phase 1 PostgreSQL database and fetch market data
    """

    def __init__(self, host='localhost', port=5432, database='trading_db',
                 user='postgres', password='postgres'):
        """
        Initialize database connector

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None

    def connect(self):
        """Establish database connection"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(**self.conn_params)
            print("✓ Connected to Phase 1 database")
        except ImportError:
            print("⚠ psycopg2 not installed. Install with: pip install psycopg2-binary")
            print("  Falling back to mock data mode")
            self.conn = None
        except Exception as e:
            print(f"⚠ Failed to connect to database: {e}")
            print("  Falling back to mock data mode")
            self.conn = None

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Disconnected from database")

    def fetch_ohlcv(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """
        Fetch OHLCV data

        Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        if not self.conn:
            return self._generate_mock_ohlcv(symbol, start_date, end_date)

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} OHLCV records for {symbol}")
        return df

    def fetch_open_interest(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Open Interest data"""
        if not self.conn:
            return self._generate_mock_oi(symbol, start_date, end_date)

        query = """
            SELECT timestamp, open_interest
            FROM open_interest
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} OI records")
        return df

    def fetch_funding_rate(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Funding Rate data"""
        if not self.conn:
            return self._generate_mock_funding(symbol, start_date, end_date)

        query = """
            SELECT timestamp, funding_rate
            FROM funding_rate
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} funding rate records")
        return df

    def fetch_liquidations(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Liquidation data"""
        if not self.conn:
            return self._generate_mock_liquidations(symbol, start_date, end_date)

        query = """
            SELECT timestamp, quantity, side, order_id
            FROM liquidations
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} liquidation records")
        return df

    def fetch_long_short_ratio(self, symbol='BTCUSDT', start_date=None, end_date=None):
        """Fetch Long/Short Ratio data"""
        if not self.conn:
            return self._generate_mock_ls_ratio(symbol, start_date, end_date)

        query = """
            SELECT timestamp, long_short_ratio as "longShortRatio"
            FROM long_short_ratio
            WHERE symbol = %s
        """
        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"✓ Fetched {len(df)} L/S ratio records")
        return df

    def fetch_all_data(self, symbol='BTCUSDT', start_date=None, end_date=None,
                      days_back=None, hours_back=None):
        """
        Fetch all data types at once

        Args:
            symbol: Trading symbol
            start_date: Start date (datetime)
            end_date: End date (datetime)
            days_back: Fetch last N days (alternative to start_date)
            hours_back: Fetch last N hours (alternative to start_date)

        Returns:
            Dictionary with all DataFrames
        """
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
        elif hours_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)

        print(f"\nFetching all data for {symbol}...")
        if start_date and end_date:
            print(f"Date range: {start_date} to {end_date}")

        data = {
            'ohlcv': self.fetch_ohlcv(symbol, start_date, end_date),
            'oi': self.fetch_open_interest(symbol, start_date, end_date),
            'funding': self.fetch_funding_rate(symbol, start_date, end_date),
            'liquidations': self.fetch_liquidations(symbol, start_date, end_date),
            'ls_ratio': self.fetch_long_short_ratio(symbol, start_date, end_date)
        }

        print(f"\n✓ All data fetched successfully!")
        return data

    # Mock data generators for testing without database
    def _generate_mock_ohlcv(self, symbol, start_date, end_date):
        """Generate mock OHLCV data"""
        import numpy as np

        if start_date and end_date:
            n_samples = int((end_date - start_date).total_seconds() / 300)  # 5-min intervals
            start = start_date
        else:
            n_samples = 5000
            start = datetime.now() - timedelta(days=30)

        timestamps = [start + timedelta(minutes=5*i) for i in range(n_samples)]

        np.random.seed(42)
        price_base = 30000
        returns = np.random.normal(0.0001, 0.01, n_samples)
        prices = price_base * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n_samples)),
            'low': prices * (1 + np.random.uniform(-0.01, 0, n_samples)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_samples)
        })

        print(f"✓ Generated {len(df)} mock OHLCV records")
        return df

    def _generate_mock_oi(self, symbol, start_date, end_date):
        """Generate mock OI data"""
        import numpy as np

        ohlcv = self._generate_mock_ohlcv(symbol, start_date, end_date)
        timestamps = ohlcv['timestamp']
        prices = ohlcv['close']

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open_interest': np.random.uniform(10000, 50000, len(timestamps)) * (prices / prices.iloc[0])
        })

        print(f"✓ Generated {len(df)} mock OI records")
        return df

    def _generate_mock_funding(self, symbol, start_date, end_date):
        """Generate mock funding rate data"""
        import numpy as np

        ohlcv = self._generate_mock_ohlcv(symbol, start_date, end_date)
        timestamps = ohlcv['timestamp']

        df = pd.DataFrame({
            'timestamp': timestamps,
            'funding_rate': np.random.normal(0.0001, 0.0005, len(timestamps))
        })

        print(f"✓ Generated {len(df)} mock funding rate records")
        return df

    def _generate_mock_liquidations(self, symbol, start_date, end_date):
        """Generate mock liquidation data"""
        import numpy as np

        ohlcv = self._generate_mock_ohlcv(symbol, start_date, end_date)
        timestamps = ohlcv['timestamp'].values

        n_liquidations = int(len(timestamps) * 0.1)
        liq_timestamps = np.random.choice(timestamps, n_liquidations, replace=False)

        df = pd.DataFrame({
            'timestamp': liq_timestamps,
            'quantity': np.random.uniform(0.1, 10, n_liquidations),
            'side': np.random.choice(['BUY', 'SELL'], n_liquidations),
            'order_id': range(n_liquidations)
        })

        df = df.sort_values('timestamp')

        print(f"✓ Generated {len(df)} mock liquidation records")
        return df

    def _generate_mock_ls_ratio(self, symbol, start_date, end_date):
        """Generate mock L/S ratio data"""
        import numpy as np

        ohlcv = self._generate_mock_ohlcv(symbol, start_date, end_date)
        timestamps = ohlcv['timestamp']

        df = pd.DataFrame({
            'timestamp': timestamps,
            'longShortRatio': np.random.uniform(0.5, 2.0, len(timestamps))
        })

        print(f"✓ Generated {len(df)} mock L/S ratio records")
        return df
