import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class DataFetcher:
    """
    Handles fetching historical and live stock data from various sources.
    """
    
    def __init__(self, ticker: str = None, interval: str = None):
        self.ticker = ticker or Config.DEFAULT_TICKER
        self.interval = interval or Config.DATA_INTERVAL
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fetch_historical_data(self, period: str = None) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Args:
            period: Time period for historical data (e.g., "2y", "1y", "6mo")
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        period = period or Config.HISTORICAL_PERIOD
        
        try:
            self.logger.info(f"Fetching historical data for {self.ticker} with interval {self.interval}")
            
            # Create yfinance ticker object
            ticker_obj = yf.Ticker(self.ticker)
            
            # Fetch historical data
            data = ticker_obj.history(
                period=period,
                interval=self.interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Clean column names
            data.columns = data.columns.str.lower()
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            self.logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def fetch_live_data(self, days: int = 1) -> pd.DataFrame:
        """
        Fetch the most recent stock data.
        
        Args:
            days: Number of days of recent data to fetch
            
        Returns:
            DataFrame with recent OHLCV data
        """
        try:
            self.logger.info(f"Fetching live data for {self.ticker}")
            
            ticker_obj = yf.Ticker(self.ticker)
            
            # Fetch recent data
            data = ticker_obj.history(
                period=f"{days}d",
                interval=self.interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No live data found for ticker {self.ticker}")
            
            # Clean column names
            data.columns = data.columns.str.lower()
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching live data: {str(e)}")
            raise
    
    def get_latest_price(self) -> float:
        """
        Get the latest price for the ticker.
        
        Returns:
            Latest close price
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period="1d", interval="1m")
            
            if data.empty:
                raise ValueError(f"No price data found for {self.ticker}")
            
            return float(data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error fetching latest price: {str(e)}")
            raise


class DataPreprocessor:
    """
    Handles data preprocessing including technical indicator calculation and feature engineering.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.scaler = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            self.logger.info("Calculating technical indicators")
            
            df = data.copy()
            
            # Basic features that model was trained on
            df['returns'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume ratio with handling for zero volume periods
            volume_mean = df['volume'].rolling(window=20).mean()
            # Handle cases where volume mean is 0 or NaN
            volume_mean = volume_mean.fillna(1.0)  # Fill NaN with 1 to avoid division by 0
            volume_mean = volume_mean.replace(0, 1.0)  # Replace 0 with 1 to avoid division by 0
            df['volume_ratio'] = df['volume'] / volume_mean
            
            # Simple moving averages
            df['sma_5'] = ta.sma(df['close'], length=5)
            df['sma_20'] = ta.sma(df['close'], length=20)
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=Config.RSI_PERIOD)
            
            # Volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            self.logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            DataFrame with selected features only
        """
        try:
            # Select only the features we want
            feature_df = data[Config.FEATURES].copy()
            
            # Handle missing values using updated pandas syntax
            feature_df = feature_df.ffill().bfill()
            
            # If any columns still have all NaN values, replace with safe defaults
            for col in feature_df.columns:
                if feature_df[col].isna().all():
                    self.logger.warning(f"Column {col} has all NaN values, replacing with 0")
                    feature_df[col] = 0.0
                elif feature_df[col].isna().any():
                    self.logger.warning(f"Column {col} has some NaN values, forward/backward filling")
                    feature_df[col] = feature_df[col].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def create_sequences(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: DataFrame with features
            target_col: Column name for target variable
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            self.logger.info(f"Creating sequences with length {Config.SEQUENCE_LENGTH}")
            
            # Ensure data is sorted by index
            data = data.sort_index()
            
            # Get feature columns (all except target)
            feature_cols = [col for col in data.columns if col != target_col]
            
            X, y = [], []
            
            for i in range(Config.SEQUENCE_LENGTH, len(data)):
                # Features for sequence
                X.append(data[feature_cols].iloc[i-Config.SEQUENCE_LENGTH:i].values)
                # Target (next period's close price)
                y.append(data[target_col].iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file to save
        """
        try:
            filepath = os.path.join(Config.DATA_DIR, filename)
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            
            data.to_csv(filepath)
            self.logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from CSV file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded DataFrame
        """
        try:
            filepath = os.path.join(Config.DATA_DIR, filename)
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"Data loaded from {filepath}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
