#!/usr/bin/env python3
"""
Real-time Stock Prediction and Signal Generation Script

This script continuously fetches live data, makes predictions, generates trading signals,
and sends alerts. It's designed to run as a background service.

Usage:
    python live_prediction.py --model_name AAPL_1h_regression_model --ticker AAPL

Fixed Version - Includes:
- File locking to prevent race conditions
- Proper error handling and validation
- Graceful shutdown
- Health checks
- Optimized I/O operations
"""

import argparse
import logging
import logging.handlers
import os
import sys
import time
import schedule
import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import deque
import threading
import signal as signal_module
import atexit
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from utils.data_utils import DataFetcher, DataPreprocessor
from utils.model_utils import LSTMModel, TradingSignalGenerator
from utils.alert_utils import AlertManager


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and numpy types."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)

class LivePredictor:
    """
    Handles live prediction and signal generation.
    
    Fixed version with:
    - File locking for thread safety
    - Data validation
    - Graceful shutdown
    - Health monitoring
    """
    
    def __init__(self, model_name: str, ticker: str, interval: str = None):
        self.model_name = model_name
        self.ticker = ticker
        self.interval = interval or Config.DATA_INTERVAL
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher(ticker=self.ticker, interval=self.interval)
        self.data_preprocessor = DataPreprocessor()
        self.model = LSTMModel()
        self.signal_generator = None
        self.alert_manager = AlertManager()
        
        # Shutdown flag
        self.shutdown_flag = threading.Event()
        
        # Status save optimization
        self.status_save_counter = 0
        self.status_save_interval = 5  # Save status every 5 cycles
        
        # Load model and verify initialization
        try:
            self._load_model()
            if self.signal_generator is None:
                raise RuntimeError("Signal generator not initialized after model load")
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise RuntimeError(f"Cannot start without valid model: {e}")
        
        # Initialize data buffer
        self.data_buffer = deque(maxlen=Config.SEQUENCE_LENGTH * 2)
        self.predictions_history = deque(maxlen=100)
        self.signals_history = deque(maxlen=50)
        
        # Performance tracking
        self.last_prediction_time = None
        self.prediction_count = 0
        self.error_count = 0
        self.validation_failures = 0
        
        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal_module.signal(signal_module.SIGINT, self._signal_handler)
        signal_module.signal(signal_module.SIGTERM, self._signal_handler)
        
        self.logger.info(f"LivePredictor initialized for {ticker} using model {model_name}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_flag.set()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with rotation."""
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        logger = logging.getLogger(f"LivePredictor_{self.ticker}")
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            # File handler with rotation (max 5MB, keep 3 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(Config.LOGS_DIR, f'live_prediction_{self.ticker}.log'),
                maxBytes=5*1024*1024,
                backupCount=3
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model.load_model(self.model_name)
            self.signal_generator = TradingSignalGenerator(self.model)
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _fetch_initial_data(self):
        """Fetch initial data to populate the buffer."""
        try:
            self.logger.info("Fetching initial data...")
            
            # Fetch recent data (more than sequence length)
            days_needed = max(7, Config.SEQUENCE_LENGTH // 100)  # Estimate based on interval
            raw_data = self.data_fetcher.fetch_live_data(days=days_needed)
            
            # Process data
            processed_data = self.data_preprocessor.calculate_technical_indicators(raw_data)
            feature_data = self.data_preprocessor.prepare_features(processed_data)
            
            # Populate buffer with recent data
            for idx in range(len(feature_data)):
                self.data_buffer.append({
                    'timestamp': feature_data.index[idx],
                    'data': feature_data.iloc[idx].values,
                    'close_price': processed_data['close'].iloc[idx]
                })
            
            self.logger.info(f"Initialized data buffer with {len(self.data_buffer)} records")
            
        except Exception as e:
            self.logger.error(f"Error fetching initial data: {str(e)}")
            raise
    
    def _fetch_latest_data(self):
        """Fetch and validate the latest data point."""
        try:
            # Fetch recent data
            raw_data = self.data_fetcher.fetch_live_data(days=1)
            
            if raw_data.empty:
                self.logger.warning("No new data available")
                return None
            
            # Get the latest data point
            latest_raw = raw_data.iloc[-1:]
            
            # Process data
            processed_data = self.data_preprocessor.calculate_technical_indicators(raw_data)
            feature_data = self.data_preprocessor.prepare_features(processed_data)
            
            # Get the latest processed data point
            latest_processed = feature_data.iloc[-1:]
            latest_close = processed_data['close'].iloc[-1]
            
            # Validate data quality
            if not self._validate_data_quality(latest_processed.iloc[0].values, latest_close):
                self.logger.warning("Data validation failed")
                self.validation_failures += 1
                return None
            
            # Check if this is truly new data
            if self.data_buffer and self.data_buffer[-1]['timestamp'] >= latest_processed.index[0]:
                self.logger.debug("No new data since last fetch")
                return None
            
            new_data_point = {
                'timestamp': latest_processed.index[0],
                'data': latest_processed.iloc[0].values,
                'close_price': latest_close
            }
            
            return new_data_point
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            return None
    
    def _validate_data_quality(self, feature_array, close_price):
        """Validate data quality before processing."""
        try:
            # Check for NaN/Inf values
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                self.logger.warning("Invalid values (NaN/Inf) detected in features")
                return False
            
            # Check close price is positive
            if close_price <= 0:
                self.logger.warning(f"Invalid close price: {close_price}")
                return False
            
            # Check feature array size matches expected feature count
            expected_features = len(Config.FEATURES)
            if len(feature_array) != expected_features:
                self.logger.warning(f"Feature size mismatch: {len(feature_array)} vs expected {expected_features}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False
    
    def _make_prediction(self):
        """Make prediction and generate trading signal."""
        try:
            if len(self.data_buffer) < Config.SEQUENCE_LENGTH:
                self.logger.warning(f"Insufficient data for prediction. Need {Config.SEQUENCE_LENGTH}, have {len(self.data_buffer)}")
                return None
            
            # Prepare sequence data
            sequence_data = np.array([point['data'] for point in list(self.data_buffer)[-Config.SEQUENCE_LENGTH:]])
            current_price = self.data_buffer[-1]['close_price']
            
            # Null check (defensive programming)
            if self.signal_generator is None:
                self.logger.error("Signal generator is None, cannot make prediction")
                self.error_count += 1
                return None
            
            # Generate signal
            signal_data = self.signal_generator.generate_signal(sequence_data, current_price)
            
            # Validate prediction before using
            if not self._validate_prediction(signal_data):
                self.logger.warning("Prediction validation failed")
                self.validation_failures += 1
                return None
            
            # Add metadata
            signal_data['model_name'] = self.model_name
            signal_data['ticker'] = self.ticker
            signal_data['data_timestamp'] = self.data_buffer[-1]['timestamp']
            
            # Store prediction
            self.predictions_history.append(signal_data)
            
            # Update counters
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            self.logger.info(f"Prediction: {signal_data['signal']} (confidence: {signal_data['confidence']:.2%}) for {self.ticker} at ${current_price:.2f}")
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            self.error_count += 1
            return None
    
    def _validate_prediction(self, signal_data):
        """Validate prediction output for sanity checks."""
        try:
            # Check required fields (common to both regression and classification)
            required_fields = ['signal', 'confidence', 'current_price']
            for field in required_fields:
                if field not in signal_data:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check confidence range [0, 1]
            confidence = signal_data['confidence']
            if not (0 <= confidence <= 1):
                self.logger.warning(f"Confidence out of range: {confidence}")
                return False
            
            # Check signal is valid
            valid_signals = ['BUY', 'SELL', 'HOLD']
            if signal_data['signal'] not in valid_signals:
                self.logger.warning(f"Invalid signal: {signal_data['signal']}")
                return False
            
            # Check current price is positive
            if signal_data['current_price'] <= 0:
                self.logger.warning(f"Invalid current price: {signal_data['current_price']}")
                return False
            
            # For regression models, validate predicted_price
            if 'predicted_price' in signal_data:
                if signal_data['predicted_price'] <= 0:
                    self.logger.warning(f"Invalid predicted price: {signal_data['predicted_price']}")
                    return False
            
            # For classification models, validate pattern probabilities
            if 'pattern_probabilities' in signal_data:
                pattern_probs = list(signal_data['pattern_probabilities'].values())
                prob_sum = sum(pattern_probs)
                if not (0.95 <= prob_sum <= 1.05):  # Allow small numerical error
                    self.logger.warning(f"Pattern probabilities don't sum to 1: {prob_sum}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {e}")
            return False
    
    def _process_signal(self, signal_data):
        """Process trading signal and send alerts if necessary."""
        try:
            # Check if signal is valid (meets confidence threshold)
            if not self.signal_generator.is_signal_valid(signal_data):
                self.logger.debug(f"Signal confidence too low: {signal_data['confidence']:.2%}")
                return
            
            # Store signal
            self.signals_history.append(signal_data)
            
            # Send alert
            alert_sent = self.alert_manager.send_signal_alert(signal_data, self.ticker)
            
            if alert_sent:
                self.logger.info(f"Alert sent for {signal_data['signal']} signal")
            
            # Save signal to file
            self._save_signal_to_file(signal_data)
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
    
    def _save_signal_to_file(self, signal_data):
        """Save signal data to file with simple file operations."""
        try:
            signals_file = os.path.join(Config.DATA_DIR, f"live_signals_{self.ticker}.json")
            
            # Simple approach: read, modify, write
            signals = []
            if os.path.exists(signals_file):
                try:
                    with open(signals_file, 'r') as f:
                        signals = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.logger.warning("Could not read existing signals file, starting fresh")
                    signals = []
            
            # Add new signal
            signals.append(signal_data)
            
            # Keep only recent signals (last 100)
            signals = signals[-100:]
            
            # Write updated content
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2, cls=DateTimeEncoder)
            
            # Check file size for rotation (if > 1MB, archive old data)
            if os.path.exists(signals_file) and os.path.getsize(signals_file) > 1024 * 1024:  # 1MB
                self._rotate_signal_file(signals_file)
            
        except Exception as e:
            self.logger.error(f"Error saving signal to file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _rotate_signal_file(self, signals_file):
        """Rotate signal file when it gets too large."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = signals_file.replace(".json", f"_archive_{timestamp}.json")
            
            # Move current file to archive
            os.rename(signals_file, archive_file)
            self.logger.info(f"Rotated signal file to {archive_file}")
            
            # Keep only last 5 archives
            archive_pattern = signals_file.replace(".json", "_archive_*.json")
            archives = sorted(glob.glob(archive_pattern))
            for old_archive in archives[:-5]:
                os.remove(old_archive)
                self.logger.info(f"Removed old archive: {old_archive}")
                
        except Exception as e:
            self.logger.error(f"Error rotating signal file: {e}")
    
    def _save_status(self):
        """Save current status for monitoring (optimized to reduce I/O)."""
        try:
            # Only save every N cycles to reduce disk I/O
            self.status_save_counter += 1
            if self.status_save_counter < self.status_save_interval:
                return
            
            self.status_save_counter = 0  # Reset counter
            
            status = {
                'ticker': self.ticker,
                'model_name': self.model_name,
                'last_update': datetime.now(),
                'last_prediction_time': self.last_prediction_time,
                'prediction_count': self.prediction_count,
                'error_count': self.error_count,
                'validation_failures': self.validation_failures,
                'buffer_size': len(self.data_buffer),
                'recent_signals': list(self.signals_history)[-5:],  # Last 5 signals
                'current_price': self.data_buffer[-1]['close_price'] if self.data_buffer else None,
                'status': 'running'
            }
            
            status_file = os.path.join(Config.DATA_DIR, f"live_status_{self.ticker}.json")
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, cls=DateTimeEncoder)
            
        except Exception as e:
            self.logger.error(f"Error saving status: {str(e)}")
    
    def run_prediction_cycle(self):
        """Run one prediction cycle."""
        try:
            self.logger.debug("Starting prediction cycle")
            
            # Fetch latest data
            new_data = self._fetch_latest_data()
            
            if new_data is not None:
                # Add to buffer
                self.data_buffer.append(new_data)
                self.logger.debug(f"Added new data point: {new_data['timestamp']}")
                
                # Make prediction
                signal_data = self._make_prediction()
                
                if signal_data is not None:
                    # Process signal
                    self._process_signal(signal_data)
            
            # Save status
            self._save_status()
            
        except Exception as e:
            self.logger.error(f"Error in prediction cycle: {str(e)}")
            self.error_count += 1
    
    def start_live_prediction(self, update_interval_minutes: int = 1):
        """Start the live prediction loop with graceful shutdown."""
        try:
            self.logger.info(f"Starting live prediction for {self.ticker}")
            self.logger.info(f"Update interval: {update_interval_minutes} minutes")
            
            # Fetch initial data
            self._fetch_initial_data()
            
            # Schedule prediction cycles
            schedule.every(update_interval_minutes).minutes.do(self.run_prediction_cycle)
            
            # Send startup notification
            startup_message = f"""
Live prediction started for {self.ticker}

Model: {self.model_name}
Update Interval: {update_interval_minutes} minutes
Prediction Type: {self.model.prediction_type}
"""
            self.alert_manager.alert_system.send_system_alert(startup_message, "INFO")
            
            # Run initial prediction
            self.run_prediction_cycle()
            
            # Main loop with shutdown flag checking
            while not self.shutdown_flag.is_set():
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            
            self.logger.info("Shutdown flag detected, cleaning up...")
            self.shutdown()
                
        except KeyboardInterrupt:
            self.logger.info("Stopping live prediction (KeyboardInterrupt)")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Error in live prediction loop: {str(e)}")
            
            # Send error notification
            error_message = f"Live prediction stopped due to error: {str(e)}"
            self.alert_manager.alert_system.send_system_alert(error_message, "ERROR")
            
            self.shutdown()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the predictor."""
        try:
            self.logger.info("Initiating graceful shutdown...")
            
            # Save final status
            status = {
                'ticker': self.ticker,
                'model_name': self.model_name,
                'last_update': datetime.now(),
                'last_prediction_time': self.last_prediction_time,
                'prediction_count': self.prediction_count,
                'error_count': self.error_count,
                'validation_failures': self.validation_failures,
                'buffer_size': len(self.data_buffer),
                'recent_signals': list(self.signals_history)[-5:],
                'current_price': self.data_buffer[-1]['close_price'] if self.data_buffer else None,
                'status': 'stopped'
            }
            
            status_file = os.path.join(Config.DATA_DIR, f"live_status_{self.ticker}.json")
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, cls=DateTimeEncoder)
            
            # Send shutdown notification
            shutdown_message = f"""
Live prediction stopped for {self.ticker}

Total Predictions: {self.prediction_count}
Total Errors: {self.error_count}
Validation Failures: {self.validation_failures}
"""
            self.alert_manager.alert_system.send_system_alert(shutdown_message, "INFO")
            
            self.logger.info("Graceful shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return {
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'validation_failures': self.validation_failures,
            'error_rate': self.error_count / max(self.prediction_count, 1),
            'validation_failure_rate': self.validation_failures / max(self.prediction_count, 1),
            'last_prediction_time': self.last_prediction_time,
            'uptime': datetime.now() - (self.last_prediction_time or datetime.now()),
            'buffer_size': len(self.data_buffer),
            'signals_generated': len(self.signals_history)
        }


def main():
    parser = argparse.ArgumentParser(description='Live stock prediction and signal generation')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the trained model to use')
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--interval', type=str, default=Config.DATA_INTERVAL,
                       help='Data interval (1m, 5m, 1h, 1d)')
    parser.add_argument('--update_interval', type=int, default=1,
                       help='Update interval in minutes')
    
    args = parser.parse_args()
    
    try:
        # Create live predictor
        predictor = LivePredictor(
            model_name=args.model_name,
            ticker=args.ticker,
            interval=args.interval
        )
        
        # Start live prediction
        predictor.start_live_prediction(update_interval_minutes=args.update_interval)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
