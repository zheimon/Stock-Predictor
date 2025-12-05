#!/bin/bash
"""
Setup script for the Stock Trading System

This script helps set up the environment and train initial models.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def setup_environment():
    """Set up Python environment and install dependencies."""
    print("Setting up Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✓ Python {python_version.major}.{python_version.minor} detected")
    
    # Install requirements
    if run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return True
    return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = ['data', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment_file():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\nSetting up environment file...")
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✓ Created .env file from .env.example")
        print("⚠️  Please edit .env file with your actual API keys and configuration")
        return True
    elif env_file.exists():
        print("✓ .env file already exists")
        return True
    else:
        print("❌ No .env.example file found")
        return False

def train_sample_model(ticker="AAPL", interval="1h", prediction_type="regression"):
    """Train a sample model."""
    print(f"\nTraining sample model for {ticker}...")
    
    command = f"python scripts/train_model.py --ticker {ticker} --interval {interval} --prediction_type {prediction_type}"
    
    if run_command(command, f"Training {prediction_type} model for {ticker}"):
        print(f"✓ Model trained successfully: {ticker}_{interval}_{prediction_type}_model")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Setup Stock Trading System")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--ticker", default="AAPL", help="Ticker for sample model")
    parser.add_argument("--interval", default="1h", help="Interval for sample model")
    parser.add_argument("--prediction-type", default="regression", choices=["regression", "classification"], 
                       help="Prediction type for sample model")
    
    args = parser.parse_args()
    
    print("🚀 Stock Trading System Setup")
    print("=" * 50)
    
    # Setup steps
    success = True
    
    # 1. Create directories
    create_directories()
    
    # 2. Setup environment
    if not setup_environment():
        success = False
    
    # 3. Setup environment file
    if not setup_environment_file():
        success = False
    
    # 4. Train sample model (optional)
    if not args.skip_training and success:
        if not train_sample_model(args.ticker, args.interval, args.prediction_type):
            print("⚠️  Model training failed, but setup can continue")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Train models: python scripts/train_model.py --ticker AAPL")
        print("3. Start live prediction: python scripts/live_prediction.py --model_name AAPL_1h_regression_model --ticker AAPL")
        print("4. Run dashboard: streamlit run dashboard/dashboard_app.py")
        print("\nOr use Docker:")
        print("docker-compose up -d")
    else:
        print("❌ Setup completed with errors")
        print("Please check the error messages above and try again")

if __name__ == "__main__":
    main()
