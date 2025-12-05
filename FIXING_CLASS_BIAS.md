# 🔧 Fixing Class Bias in Pattern Recognition Model

## Problem Identified
Model is biased towards predicting Uptrend (Class 0) despite having a perfectly balanced dataset (22,941 samples per class).

## **ACTUAL ROOT CAUSE** ⚠️
The original dataset was **heavily skewed** (probably 80%+ Uptrend) and was balanced using **naive upsampling** (duplicating minority samples). This causes:

1. **Overfitting on Duplicates**: Model memorizes the few unique Downtrend/H&S/Double Bottom samples
2. **No Real Diversity**: 22,941 "balanced" samples but maybe only 2,000 unique Downtrend patterns
3. **Default to Majority**: When uncertain, model predicts Uptrend (the naturally abundant class)
4. **Data Leakage**: Same exact sequences appear in train/val/test (just duplicated)

## Secondary Causes
1. **Natural Market Bias**: Stock markets trend upward ~70% of the time historically
2. **Pattern Detection Difficulty**: Uptrends are simpler (linear), H&S/Double Bottom are complex (multi-peak)
3. **Feature Dominance**: Some features may be more predictive of uptrends
4. **No Class Weighting**: Training treats all classes equally even if learning difficulty varies

