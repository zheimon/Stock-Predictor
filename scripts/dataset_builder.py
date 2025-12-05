#!/usr/bin/env python3
"""
Dataset Builder for Chart Pattern Recognition

This script downloads stock data, labels patterns, and preprocesses sequences
for LSTM training according to the specified requirements.

Features:
- Downloads data for 10-15 well-known stocks from different sectors
- Implements rule-based pattern labeling (Uptrend, Downtrend, Head-and-Shoulders, Double Bottom)
- Creates sliding window sequences for LSTM training
- Handles missing data and normalization
- Balanced dataset splitting

Usage:
    python dataset_builder.py --sequence_length 60 --save_data
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config

class PatternLabeler:
    """Rule-based pattern labeling for chart patterns."""
    
    def __init__(self):
        self.pattern_classes = {
            0: 'Uptrend',
            1: 'Downtrend', 
            2: 'Head-and-Shoulders',
            3: 'Double Bottom'
        }
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def identify_uptrend(self, data):
        """Identify uptrend pattern: series of higher highs and higher lows."""
        highs = data['high'].values
        lows = data['low'].values
        
        # Check for at least 3 higher highs and higher lows
        higher_highs = 0
        higher_lows = 0
        
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                higher_highs += 1
            if lows[i] > lows[i-1]:
                higher_lows += 1
        
        # Uptrend if more than 60% of the sequence shows higher highs/lows
        uptrend_score = (higher_highs + higher_lows) / (2 * (len(highs) - 1))
        return uptrend_score > 0.6
    
    def identify_downtrend(self, data):
        """Identify downtrend pattern: series of lower highs and lower lows."""
        highs = data['high'].values
        lows = data['low'].values
        
        # Check for at least 3 lower highs and lower lows
        lower_highs = 0
        lower_lows = 0
        
        for i in range(1, len(highs)):
            if highs[i] < highs[i-1]:
                lower_highs += 1
            if lows[i] < lows[i-1]:
                lower_lows += 1
        
        # Downtrend if more than 60% of the sequence shows lower highs/lows
        downtrend_score = (lower_highs + lower_lows) / (2 * (len(highs) - 1))
        return downtrend_score > 0.6
    
    def identify_head_and_shoulders(self, data):
        """Identify head-and-shoulders pattern: left shoulder, peak, right shoulder."""
        highs = data['high'].values
        closes = data['close'].values
        
        if len(highs) < 20:  # Need minimum length for pattern
            return False
        
        # Find the highest point (head)
        head_idx = np.argmax(highs)
        
        # Head should be in the middle third of the sequence
        seq_len = len(highs)
        if head_idx < seq_len * 0.3 or head_idx > seq_len * 0.7:
            return False
        
        # Find left shoulder (highest point before head)
        left_shoulder_idx = np.argmax(highs[:head_idx]) if head_idx > 5 else 0
        
        # Find right shoulder (highest point after head)
        right_shoulder_idx = head_idx + np.argmax(highs[head_idx+1:]) + 1 if head_idx < seq_len - 5 else seq_len - 1
        
        # Check if shoulders are roughly equal height and lower than head
        left_shoulder_height = highs[left_shoulder_idx]
        head_height = highs[head_idx]
        right_shoulder_height = highs[right_shoulder_idx]
        
        # Shoulders should be within 10% of each other and at least 5% lower than head
        shoulder_diff = abs(left_shoulder_height - right_shoulder_height) / max(left_shoulder_height, right_shoulder_height)
        head_dominance = (head_height - max(left_shoulder_height, right_shoulder_height)) / head_height
        
        # Final price should be lower than the neckline
        neckline = (closes[left_shoulder_idx] + closes[right_shoulder_idx]) / 2
        final_break = closes[-1] < neckline * 0.98
        
        return shoulder_diff < 0.1 and head_dominance > 0.05 and final_break
    
    def identify_double_bottom(self, data):
        """Identify double bottom pattern: two distinct lows separated by a peak."""
        lows = data['low'].values
        highs = data['high'].values
        closes = data['close'].values
        
        if len(lows) < 20:  # Need minimum length for pattern
            return False
        
        # Find two lowest points
        low_indices = np.argsort(lows)[:3]  # Get 3 lowest points
        low_indices = np.sort(low_indices)
        
        if len(low_indices) < 2:
            return False
        
        first_low_idx = low_indices[0]
        second_low_idx = low_indices[1]
        
        # Ensure lows are separated by at least 25% of sequence length
        separation = abs(second_low_idx - first_low_idx)
        if separation < len(lows) * 0.25:
            return False
        
        # Find peak between the two lows
        start_idx = min(first_low_idx, second_low_idx)
        end_idx = max(first_low_idx, second_low_idx)
        peak_idx = start_idx + np.argmax(highs[start_idx:end_idx])
        
        # Check if lows are roughly equal (within 5%)
        first_low = lows[first_low_idx]
        second_low = lows[second_low_idx]
        low_diff = abs(first_low - second_low) / min(first_low, second_low)
        
        # Check if peak is significantly higher than lows (at least 10%)
        peak_height = highs[peak_idx]
        peak_dominance = (peak_height - max(first_low, second_low)) / peak_height
        
        # Final price should break above the peak
        breakout = closes[-1] > peak_height * 1.02
        
        return low_diff < 0.05 and peak_dominance > 0.1 and breakout
    
    def label_sequence(self, data):
        """Label a sequence with the most prominent pattern."""
        # Check patterns in order of priority
        if self.identify_head_and_shoulders(data):
            return 2  # Head-and-Shoulders
        elif self.identify_double_bottom(data):
            return 3  # Double Bottom
        elif self.identify_uptrend(data):
            return 0  # Uptrend
        elif self.identify_downtrend(data):
            return 1  # Downtrend
        else:
            # Default to trend based on overall price movement
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            if end_price > start_price * 1.05:
                return 0  # Uptrend
            else:
                return 1  # Downtrend

class DatasetBuilder:
    """Main class for building the pattern recognition dataset."""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'NVDA', 'META', 'JPM', 'XOM', 'NFLX',
            'JNJ', 'V', 'PG', 'UNH', 'HD'  # 15 diverse stocks
        ]
        self.pattern_labeler = PatternLabeler()
        self.scaler = MinMaxScaler()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def download_stock_data(self, start_date='2010-01-01'):
        """Download stock data for all tickers."""
        all_data = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"📥 Downloading data for {len(self.tickers)} stocks from {start_date} to {end_date}")
        
        for ticker in self.tickers:
            try:
                self.logger.info(f"Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date, interval='1d')
                
                if data.empty:
                    self.logger.warning(f"No data found for {ticker}")
                    continue
                
                # Clean column names
                data.columns = [col.lower() for col in data.columns]
                
                # Handle missing data
                data = self._handle_missing_data(data)
                
                if len(data) < self.sequence_length * 2:
                    self.logger.warning(f"Insufficient data for {ticker}: {len(data)} days")
                    continue
                
                all_data[ticker] = data
                self.logger.info(f"✓ {ticker}: {len(data)} days of data")
                
            except Exception as e:
                self.logger.error(f"Error fetching {ticker}: {str(e)}")
                continue
        
        return all_data
    
    def _handle_missing_data(self, data):
        """Handle missing data in stock data."""
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Drop rows with excessive gaps (more than 5 consecutive NaN values)
        data = data.dropna()
        
        return data
    
    def create_features(self, data):
        """Create features from OHLCV data."""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Technical indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Drop NaN values created by rolling operations
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_sequences(self, all_data):
        """Create sliding window sequences with pattern labels."""
        sequences = []
        labels = []
        tickers_list = []
        
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                          'high_low_ratio', 'close_open_ratio', 'volume_ratio',
                          'sma_5', 'sma_20', 'rsi', 'volatility']
        
        self.logger.info(f"🔄 Creating sequences with length {self.sequence_length}")
        
        for ticker, data in all_data.items():
            # Create features
            data_with_features = self.create_features(data)
            
            # Ensure we have all required features
            missing_features = [col for col in feature_columns if col not in data_with_features.columns]
            if missing_features:
                self.logger.warning(f"Missing features for {ticker}: {missing_features}")
                continue
            
            # Select feature columns
            feature_data = data_with_features[feature_columns]
            
            # Create sliding windows
            for i in range(len(feature_data) - self.sequence_length + 1):
                sequence_data = data_with_features.iloc[i:i + self.sequence_length]
                
                # Check for any remaining NaN values
                if sequence_data.isnull().any().any():
                    continue
                
                # Get features for this sequence
                sequence_features = feature_data.iloc[i:i + self.sequence_length].values
                
                # Label the sequence
                pattern_label = self.pattern_labeler.label_sequence(sequence_data)
                
                sequences.append(sequence_features)
                labels.append(pattern_label)
                tickers_list.append(ticker)
        
        self.logger.info(f"✓ Created {len(sequences)} sequences")
        
        return np.array(sequences), np.array(labels), tickers_list, feature_columns
    
    def normalize_sequences(self, sequences):
        """Normalize sequences using percentage change."""
        normalized_sequences = []
        
        for seq in sequences:
            # Convert to percentage change per day for price columns (0-4: OHLCV)
            normalized_seq = seq.copy()
            
            # Normalize price columns (0-4) using percentage change
            for col in range(4):  # OHLC
                if seq[0, col] != 0:
                    normalized_seq[:, col] = (seq[:, col] / seq[0, col] - 1) * 100
            
            # Volume: use log transformation and normalize
            if seq[:, 4].max() > 0:
                normalized_seq[:, 4] = np.log1p(seq[:, 4])
                normalized_seq[:, 4] = (normalized_seq[:, 4] - normalized_seq[:, 4].min()) / (normalized_seq[:, 4].max() - normalized_seq[:, 4].min())
            
            # Other features: already normalized or ratios
            
            normalized_sequences.append(normalized_seq)
        
        return np.array(normalized_sequences)
    
    def create_synthetic_sample(self, X, pattern_indices, augmentation_type='smote'):
        """
        Create a synthetic sample using SMOTE-like technique.
        
        Args:
            X (np.ndarray): All feature sequences
            pattern_indices (np.ndarray): Indices of samples from same class
            augmentation_type (str): Type of augmentation ('smote', 'noise', 'scale', 'interpolate')
        
        Returns:
            np.ndarray: Synthetic sample
        """
        if augmentation_type == 'smote':
            # SMOTE: Interpolate between two samples
            if len(pattern_indices) < 2:
                # If only one sample, add noise
                idx = pattern_indices[0]
                synthetic = X[idx].copy()
                noise = np.random.normal(0, 0.02, synthetic.shape)
                return synthetic + noise
            
            # Pick two random samples from same class
            idx1, idx2 = np.random.choice(pattern_indices, size=2, replace=False)
            
            # Interpolate with random weight
            alpha = np.random.uniform(0.3, 0.7)
            synthetic = alpha * X[idx1] + (1 - alpha) * X[idx2]
            
            # Add small noise for extra variation
            noise = np.random.normal(0, 0.01, synthetic.shape)
            synthetic = synthetic + noise
            
        elif augmentation_type == 'noise':
            # Add Gaussian noise to existing sample
            idx = np.random.choice(pattern_indices)
            synthetic = X[idx].copy()
            noise = np.random.normal(0, 0.02, synthetic.shape)
            synthetic = synthetic + noise
            
        elif augmentation_type == 'scale':
            # Scale features slightly
            idx = np.random.choice(pattern_indices)
            synthetic = X[idx].copy()
            
            # Scale prices (columns 0-3: open, high, low, close)
            price_scale = np.random.uniform(0.98, 1.02)
            synthetic[:, :4] *= price_scale
            
            # Scale volume (column 4) more aggressively
            volume_scale = np.random.uniform(0.7, 1.3)
            if synthetic.shape[1] > 4:
                synthetic[:, 4] *= volume_scale
            
        elif augmentation_type == 'interpolate':
            # Time-aware interpolation
            idx = np.random.choice(pattern_indices)
            synthetic = X[idx].copy()
            
            # Add slight trend drift
            drift = np.random.uniform(-0.01, 0.01)
            trend = np.linspace(0, drift, len(synthetic))
            synthetic[:, :4] *= (1 + trend[:, np.newaxis])  # Apply to OHLC
            
        else:
            # Default: simple copy with noise
            idx = np.random.choice(pattern_indices)
            synthetic = X[idx].copy()
            noise = np.random.normal(0, 0.01, synthetic.shape)
            synthetic = synthetic + noise
        
        return synthetic
    
    def balance_dataset(self, X, y, tickers, target_count=None, use_smote=True):
        """
        Balance dataset using SMOTE-like augmentation instead of naive duplication.
        
        Args:
            X (np.ndarray): Feature sequences
            y (np.ndarray): Pattern labels
            tickers (list): Stock tickers corresponding to each sequence
            target_count (int): Target count per class (default: median * 1.2)
            use_smote (bool): Use SMOTE augmentation vs naive duplication
        
        Returns:
            np.ndarray, np.ndarray, list
        """
        from collections import Counter
        
        label_counts = Counter(y)
        self.logger.info(f"📊 Pattern distribution before balancing: {label_counts}")
        
        # Calculate imbalance ratio
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        self.logger.info(f"⚖️  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Use median-based target to avoid over-representing majority class
        if target_count is None:
            counts = list(label_counts.values())
            if use_smote:
                # Use median * 1.2 for SMOTE to avoid too much synthetic data
                target_count = int(np.median(counts) * 1.2)
            else:
                # Use max for naive duplication (old behavior)
                target_count = max(counts)
        
        self.logger.info(f"🎯 Balancing to {target_count} samples per class")
        if use_smote:
            self.logger.info(f"🔬 Using SMOTE-like augmentation (creates synthetic variations)")
        else:
            self.logger.info(f"📋 Using naive duplication (creates exact copies)")
        
        balanced_X, balanced_y, balanced_tickers = [], [], []
        
        for pattern_id in range(len(self.pattern_labeler.pattern_classes)):
            pattern_indices = np.where(y == pattern_id)[0]
            pattern_name = self.pattern_labeler.pattern_classes[pattern_id]
            current_count = len(pattern_indices)
            
            if current_count == 0:
                self.logger.warning(f"⚠️ No samples found for pattern {pattern_id} ({pattern_name})")
                continue
            
            # Add all original samples
            balanced_X.extend(X[pattern_indices])
            balanced_y.extend(y[pattern_indices])
            balanced_tickers.extend([tickers[i] for i in pattern_indices])
            
            # Augment if needed
            if current_count < target_count:
                samples_needed = target_count - current_count
                self.logger.info(f"   {pattern_name}: {current_count} → {target_count} (adding {samples_needed} synthetic samples)")
                
                if use_smote:
                    # Create synthetic samples using different augmentation techniques
                    augmentation_types = ['smote', 'noise', 'scale', 'interpolate']
                    
                    for i in range(samples_needed):
                        # Rotate through augmentation types for diversity
                        aug_type = augmentation_types[i % len(augmentation_types)]
                        synthetic = self.create_synthetic_sample(X, pattern_indices, aug_type)
                        
                        balanced_X.append(synthetic)
                        balanced_y.append(pattern_id)
                        # Use original ticker + synthetic marker
                        original_ticker = tickers[pattern_indices[i % len(pattern_indices)]]
                        balanced_tickers.append(f"{original_ticker}_syn{i}")
                else:
                    # Old behavior: naive duplication
                    for i in range(samples_needed):
                        idx = pattern_indices[i % len(pattern_indices)]
                        balanced_X.append(X[idx])
                        balanced_y.append(pattern_id)
                        balanced_tickers.append(tickers[idx])
            
            elif current_count > target_count:
                # Downsample if too many
                samples_to_remove = current_count - target_count
                self.logger.info(f"   {pattern_name}: {current_count} → {target_count} (removing {samples_to_remove} samples)")
                # Already added all samples, need to remove some
                # Remove the last samples_to_remove that were just added
                balanced_X = balanced_X[:-samples_to_remove]
                balanced_y = balanced_y[:-samples_to_remove]
                balanced_tickers = balanced_tickers[:-samples_to_remove]
        
        balanced_X = np.array(balanced_X)
        balanced_y = np.array(balanced_y)
        
        # Report final distribution
        final_counts = Counter(balanced_y)
        self.logger.info(f"✅ Final distribution: {final_counts}")
        
        # Calculate duplicate statistics
        unique_samples = len(np.unique(balanced_X.reshape(len(balanced_X), -1), axis=0))
        duplicate_ratio = (len(balanced_X) - unique_samples) / len(balanced_X) * 100
        self.logger.info(f"📊 Unique samples: {unique_samples}/{len(balanced_X)} (duplicate ratio: {duplicate_ratio:.1f}%)")
        
        return balanced_X, balanced_y, balanced_tickers

    
    def split_dataset(self, X, y, tickers):
        """Split dataset into train/validation/test sets."""
        # First split: 80% train, 20% temp
        X_train, X_temp, y_train, y_temp, tickers_train, tickers_temp = train_test_split(
            X, y, tickers, test_size=0.2, random_state=42, stratify=y
        )
        
        # Second split: 10% validation, 10% test
        X_val, X_test, y_val, y_test, tickers_val, tickers_test = train_test_split(
            X_temp, y_temp, tickers_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), (tickers_train, tickers_val, tickers_test)
    
    def save_dataset(self, X_splits, y_splits, tickers_splits, feature_columns, save_dir='data'):
        """Save the processed dataset."""
        os.makedirs(save_dir, exist_ok=True)
        
        X_train, X_val, X_test = X_splits
        y_train, y_val, y_test = y_splits
        tickers_train, tickers_val, tickers_test = tickers_splits
        
        # Save arrays
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'feature_columns': feature_columns,
            'pattern_classes': self.pattern_labeler.pattern_classes,
            'tickers': self.tickers,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'n_features': X_train.shape[2]
        }
        
        import json
        with open(os.path.join(save_dir, 'dataset_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save tickers for each split
        with open(os.path.join(save_dir, 'tickers_train.txt'), 'w') as f:
            f.write('\n'.join(tickers_train))
        with open(os.path.join(save_dir, 'tickers_val.txt'), 'w') as f:
            f.write('\n'.join(tickers_val))
        with open(os.path.join(save_dir, 'tickers_test.txt'), 'w') as f:
            f.write('\n'.join(tickers_test))
        
        self.logger.info(f"💾 Dataset saved to {save_dir}/")
        self.logger.info(f"   Train: {len(X_train)} samples")
        self.logger.info(f"   Val:   {len(X_val)} samples") 
        self.logger.info(f"   Test:  {len(X_test)} samples")
    
    def plot_pattern_distribution(self, y, save_path=None):
        """Plot distribution of patterns in the dataset."""
        from collections import Counter
        
        pattern_counts = Counter(y)
        pattern_names = [self.pattern_labeler.pattern_classes[i] for i in sorted(pattern_counts.keys())]
        counts = [pattern_counts[i] for i in sorted(pattern_counts.keys())]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(pattern_names, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        plt.title('Distribution of Chart Patterns in Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Pattern Type', fontsize=12)
        plt.ylabel('Number of Sequences', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Pattern distribution plot saved to {save_path}")
        
        plt.show()
        return pattern_counts

def main():
    parser = argparse.ArgumentParser(description='Build dataset for chart pattern recognition')
    parser.add_argument('--sequence_length', type=int, default=60,
                       help='Length of sequences for LSTM training')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                       help='Start date for data download (YYYY-MM-DD)')
    parser.add_argument('--save_data', action='store_true',
                       help='Save the processed dataset')
    parser.add_argument('--balance_dataset', action='store_true', default=True,
                       help='Balance the dataset across pattern classes')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for saved data')
    
    args = parser.parse_args()
    
    # Create dataset builder
    builder = DatasetBuilder(sequence_length=args.sequence_length)
    
    try:
        print("🚀 Starting dataset building process...")
        
        # 1. Download stock data
        print("\n📥 Step 1: Downloading stock data...")
        all_data = builder.download_stock_data(start_date=args.start_date)
        
        if not all_data:
            print("❌ No data downloaded. Exiting.")
            return
        
        # 2. Create sequences and labels
        print("\n🔄 Step 2: Creating sequences and pattern labels...")
        X, y, tickers, feature_columns = builder.create_sequences(all_data)
        
        if len(X) == 0:
            print("❌ No sequences created. Exiting.")
            return
        
        # 3. Normalize sequences
        print("\n📊 Step 3: Normalizing sequences...")
        X_normalized = builder.normalize_sequences(X)
        
        # 4. Balance dataset if requested
        if args.balance_dataset:
            print("\n⚖️ Step 4: Balancing dataset...")
            X_normalized, y, tickers = builder.balance_dataset(X_normalized, y, tickers)
        
        # 5. Plot pattern distribution
        print("\n📈 Step 5: Analyzing pattern distribution...")
        pattern_counts = builder.plot_pattern_distribution(y, 
                                                         save_path=os.path.join(args.output_dir, 'pattern_distribution.png'))
        
        # 6. Split dataset
        print("\n✂️ Step 6: Splitting dataset...")
        X_splits, y_splits, tickers_splits = builder.split_dataset(X_normalized, y, tickers)
        
        # 7. Save dataset if requested
        if args.save_data:
            print("\n💾 Step 7: Saving dataset...")
            builder.save_dataset(X_splits, y_splits, tickers_splits, feature_columns, args.output_dir)
        
        # Summary
        print("\n🎉 DATASET BUILDING COMPLETED SUCCESSFULLY!")
        print(f"📋 Dataset Summary:")
        print(f"   Total sequences: {len(X_normalized)}")
        print(f"   Sequence length: {args.sequence_length}")
        print(f"   Features per timestep: {X_normalized.shape[2]}")
        print(f"   Pattern classes: {len(builder.pattern_labeler.pattern_classes)}")
        print(f"   Train/Val/Test split: {len(X_splits[0])}/{len(X_splits[1])}/{len(X_splits[2])}")
        
        print(f"\n📊 Pattern Distribution:")
        for pattern_id, count in pattern_counts.items():
            pattern_name = builder.pattern_labeler.pattern_classes[pattern_id]
            print(f"   {pattern_name}: {count} sequences")
        
        if args.save_data:
            print(f"\n📁 Data saved to: {args.output_dir}/")
            print("\nNext steps:")
            print("1. Run train_model.py to train the LSTM model")
            print("2. Use predict.py for pattern prediction on new data")
        
    except Exception as e:
        print(f"\n❌ Error during dataset building: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
