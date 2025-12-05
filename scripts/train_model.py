#!/usr/bin/env python3
"""
Test script to rebuild dataset with SMOTE augmentation and train with improved methods.

This script:
1. Rebuilds the dataset using SMOTE-like augmentation (no naive duplication)
2. Trains the model with class weights enabled
3. Optionally uses focal loss for better handling of hard examples
4. Monitors per-class metrics during training
5. Evaluates final performance across all pattern classes

Usage:
    # Basic: SMOTE + class weights
    python test_improved_training.py
    
    # Advanced: SMOTE + class weights + focal loss
    python test_improved_training.py --use_focal_loss
    
    # Quick test with fewer epochs
    python test_improved_training.py --epochs 20 --quick_test
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from utils.model_utils import LSTMModel

def plot_comparison(history, save_dir='models'):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Train Accuracy')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_improved.png'), dpi=150)
    print(f"📊 Training curves saved to {os.path.join(save_dir, 'training_curves_improved.png')}")
    plt.close()

def plot_confusion_matrix_improved(y_true, y_pred, class_names, save_dir='models'):
    """Plot confusion matrix with percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Confusion Matrix (% of True Class)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_improved.png'), dpi=150)
    print(f"📊 Confusion matrix saved to {os.path.join(save_dir, 'confusion_matrix_improved.png')}")
    plt.close()

def evaluate_model(model, X_test, y_test, class_names):
    """Comprehensive model evaluation."""
    print("\n" + "="*80)
    print("🎯 FINAL MODEL EVALUATION")
    print("="*80)
    
    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📊 Detailed Classification Report:")
    print("-"*80)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Per-class accuracy
    print("\n📈 Per-Class Accuracy:")
    print("-"*80)
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
            print(f"  {class_name:<25}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n🔢 Confusion Matrix:")
    print("-"*80)
    print(f"{'':>25}", end='')
    for name in class_names:
        print(f"{name[:10]:>12}", end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:<25}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12d}", end='')
        print()
    
    # Check balance
    print("\n⚖️  Class Balance in Predictions:")
    print("-"*80)
    from collections import Counter
    pred_counts = Counter(y_pred)
    total_preds = len(y_pred)
    
    for i, class_name in enumerate(class_names):
        count = pred_counts.get(i, 0)
        percentage = (count / total_preds) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {class_name:<25}: {count:>6d} ({percentage:>5.1f}%) {bar}")
    
    # Check if still biased
    max_pred = max(pred_counts.values())
    min_pred = min(pred_counts.values())
    pred_ratio = max_pred / min_pred if min_pred > 0 else float('inf')
    
    print(f"\n📊 Prediction Balance Ratio: {pred_ratio:.2f}:1")
    
    if pred_ratio > 3.0:
        print("⚠️  WARNING: Model still shows significant bias!")
        print("   Consider:")
        print("   - Using focal loss (--use_focal_loss flag)")
        print("   - Increasing class weights for underrepresented classes")
        print("   - Collecting more diverse training data")
    elif pred_ratio > 1.5:
        print("⚡ NOTICE: Minor prediction imbalance detected")
        print("   This is acceptable but could be improved")
    else:
        print("✅ EXCELLENT: Predictions are well-balanced across all classes!")
    
    print("="*80 + "\n")
    
    return y_pred, cm

def main():
    parser = argparse.ArgumentParser(description='Test improved training with SMOTE and class weights')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of categorical crossentropy')
    parser.add_argument('--no_class_weights', action='store_true',
                       help='Disable automatic class weighting')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode: 10 epochs only')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing dataset files')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 10
        print("⚡ Quick test mode: 10 epochs only\n")
    
    print("="*80)
    print("🚀 IMPROVED MODEL TRAINING TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Focal Loss: {'Yes' if args.use_focal_loss else 'No'}")
    print(f"  Class Weights: {'No' if args.no_class_weights else 'Yes (Auto)'}")
    print(f"  Data Directory: {args.data_dir}")
    print("="*80 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    
    print(f"✅ Dataset loaded:")
    print(f"   Training: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Check for duplicates
    unique_train = len(np.unique(X_train.reshape(len(X_train), -1), axis=0))
    duplicate_ratio = (len(X_train) - unique_train) / len(X_train) * 100
    print(f"\n📊 Training set duplicate ratio: {duplicate_ratio:.1f}%")
    
    if duplicate_ratio > 30:
        print("⚠️  WARNING: High duplicate ratio detected!")
        print("   Consider rebuilding dataset with:")
        print("   python dataset_builder.py --balance_dataset")
        print("   (New version uses SMOTE-like augmentation)\n")
    
    # Class names
    class_names = ['Uptrend', 'Downtrend', 'Head-Shoulders', 'Double Bottom']
    
    # Create model
    print(f"\n🏗️  Building LSTM model...")
    model = LSTMModel(
        prediction_type="classification",
        use_focal_loss=args.use_focal_loss
    )
    
    # Train model
    print(f"\n🚀 Starting training...")
    start_time = datetime.now()
    
    history = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights
    )
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Training completed in {training_time}")
    
    # Evaluate
    y_pred, cm = evaluate_model(model, X_test, y_test, class_names)
    
    # Save visualizations
    print("\n📊 Generating visualizations...")
    plot_comparison(history, save_dir='models')
    plot_confusion_matrix_improved(y_test, y_pred, class_names, save_dir='models')
    
    # Save model
    model_name = f"lstm_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.use_focal_loss:
        model_name += "_focal"
    
    print(f"\n💾 Saving model as '{model_name}'...")
    model.save_model(model_name)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review training curves: models/training_curves_improved.png")
    print(f"2. Review confusion matrix: models/confusion_matrix_improved.png")
    print(f"3. If results are good, use this model for live prediction:")
    print(f"   python scripts/live_prediction.py --model_name {model_name} --ticker AAPL")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
