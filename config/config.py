# Configuration file for the Stock Trading System

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Data Configuration
    DEFAULT_TICKER = "AAPL"
    DATA_INTERVAL = "1m"  # 1m, 5m, 15m, 30m, 1h, 1d
    HISTORICAL_PERIOD = "2y"  # 2 years of historical data
    SEQUENCE_LENGTH = 60  # Number of timesteps to look back
    
    # Pattern Recognition Configuration
    PATTERN_SEQUENCE_LENGTH = 60  # Days for pattern recognition
    PATTERN_CLASSES = {
        0: 'Uptrend',
        1: 'Downtrend', 
        2: 'Head-and-Shoulders',
        3: 'Double Bottom'
    }
    
    # Default tickers for pattern recognition dataset
    DEFAULT_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
        'NVDA', 'META', 'JPM', 'XOM', 'NFLX',
        'JNJ', 'V', 'PG', 'UNH', 'HD'
    ]
    
    # Model Configuration
    LSTM_UNITS = [64, 64]  # LSTM layer units for pattern recognition
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.1  # 10% for validation
    TEST_SPLIT = 0.1        # 10% for test
    PATIENCE = 5  # Early stopping patience
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    
    # Trading Signal Configuration
    BUY_THRESHOLD = 0.02  # 2% price increase threshold
    SELL_THRESHOLD = -0.02  # 2% price decrease threshold
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for signals
    
    # Data Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")
    MODELS_DIR = os.path.join(BASE_DIR, "../models")
    LOGS_DIR = os.path.join(BASE_DIR, "../logs")
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    
    # Alert Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    EMAIL_TO = os.getenv("EMAIL_TO")
    
    # Dashboard Configuration
    DASHBOARD_HOST = "0.0.0.0"
    DASHBOARD_PORT = 8501
    AUTO_REFRESH_INTERVAL = 60  # seconds
    
    # Technical Indicators Configuration
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    EMA_SHORT = 12
    EMA_LONG = 26
    SMA_SHORT = 20
    SMA_LONG = 50
    BB_PERIOD = 20
    BB_STD = 2
    STOCH_K = 14
    STOCH_D = 3
    
    # Feature Configuration  
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'high_low_ratio', 'close_open_ratio', 'volume_ratio',
        'sma_5', 'sma_20', 'rsi', 'volatility'
    ]
    
    # Deployment Configuration
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, production
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
