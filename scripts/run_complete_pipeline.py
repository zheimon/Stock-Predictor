#!/usr/bin/env python3
"""
Complete LSTM Pattern Recognition Pipeline Example

This script demonstrates the complete end-to-end pipeline for chart pattern recognition.
It builds a dataset, trains the model, and makes predictions.

Usage:
    python run_complete_pipeline.py --quick_demo
    python run_complete_pipeline.py --full_pipeline
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Failed after {elapsed_time:.1f} seconds")
        print(f"Error: {e}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'tensorflow', 'pandas', 'numpy', 'sklearn', 
        'yfinance', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def quick_demo():
    """Run a quick demo with minimal data."""
    print("🎯 Running Quick Demo (Fast demonstration)")
    print("This demo uses minimal data for faster execution.")
    
    steps = [
        {
            "command": "python dataset_builder.py --sequence_length 30 --start_date 2020-01-01 --save_data --balance_dataset",
            "description": "Building small dataset (2020-present, 30-day sequences)"
        },
        {
            "command": "python train_model.py --epochs 10 --batch_size 16 --save_model --model_name demo_model",
            "description": "Training model (10 epochs for demo)"
        },
        {
            "command": "python predict.py --model models/demo_model.h5 --ticker AAPL --save_plot --save_result",
            "description": "Making prediction for AAPL"
        }
    ]
    
    return run_pipeline_steps(steps)

def full_pipeline():
    """Run the complete pipeline with full dataset."""
    print("🎯 Running Full Pipeline (Complete training)")
    print("This will download full historical data and train the complete model.")
    
    steps = [
        {
            "command": "python dataset_builder.py --sequence_length 60 --start_date 2010-01-01 --save_data --balance_dataset",
            "description": "Building complete dataset (2010-present, 60-day sequences)"
        },
        {
            "command": "python train_model.py --epochs 50 --batch_size 32 --save_model --model_name lstm_pattern_classifier",
            "description": "Training model (50 epochs)"
        },
        {
            "command": "python predict.py --model models/lstm_pattern_classifier.h5 --ticker AAPL --save_plot --save_result",
            "description": "Making prediction for AAPL"
        },
        {
            "command": "python predict.py --model models/lstm_pattern_classifier.h5 --ticker GOOGL --save_plot --save_result", 
            "description": "Making prediction for GOOGL"
        },
        {
            "command": "python predict.py --model models/lstm_pattern_classifier.h5 --ticker TSLA --save_plot --save_result",
            "description": "Making prediction for TSLA"
        }
    ]
    
    return run_pipeline_steps(steps)

def run_pipeline_steps(steps):
    """Execute pipeline steps."""
    total_start_time = time.time()
    completed_steps = 0
    
    for i, step in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{len(steps)}: {step['description']}")
        
        success = run_command(step['command'], step['description'])
        
        if success:
            completed_steps += 1
        else:
            print(f"\n❌ Pipeline failed at step {i}")
            print("You can continue manually or check the error above.")
            
            user_input = input("\nContinue with next step? (y/n): ").lower()
            if user_input != 'y':
                break
    
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print(f"📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {completed_steps}/{len(steps)}")
    print(f"⏱️ Total time: {total_elapsed/60:.1f} minutes")
    
    if completed_steps == len(steps):
        print("🎉 Pipeline completed successfully!")
        
        print(f"\n📁 Generated files:")
        print(f"   📊 data/ - Training dataset")
        print(f"   🧠 models/ - Trained LSTM model")
        print(f"   🔮 predictions/ - Pattern predictions")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Examine the training results in models/")
        print(f"   2. Check prediction visualizations in predictions/")
        print(f"   3. Try predictions on other stocks")
        print(f"   4. Experiment with different parameters")
        
    else:
        print("⚠️ Pipeline completed with some failures.")
        print("Check the error messages above and try running individual steps.")
    
    return completed_steps == len(steps)

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models', 'predictions', 'logs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}/")

def main():
    parser = argparse.ArgumentParser(description='Complete LSTM Pattern Recognition Pipeline')
    parser.add_argument('--quick_demo', action='store_true',
                       help='Run quick demo with minimal data (faster)')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Run complete pipeline with full dataset')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check dependencies without running pipeline')
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 LSTM Chart Pattern Recognition Pipeline")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install requirements first.")
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ Dependency check passed. System ready!")
        return
    
    # Determine which pipeline to run
    if args.quick_demo:
        success = quick_demo()
    elif args.full_pipeline:
        success = full_pipeline()
    else:
        print("\n❓ Which pipeline would you like to run?")
        print("1. Quick Demo (minimal data, faster)")
        print("2. Full Pipeline (complete data, slower)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = quick_demo()
        elif choice == "2":
            success = full_pipeline()
        else:
            print("❌ Invalid choice. Use --quick_demo or --full_pipeline")
            sys.exit(1)
    
    # Final message
    if success:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n⚠️ Pipeline completed with issues.")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
