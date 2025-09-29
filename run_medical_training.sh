#!/bin/bash
# AiMedRes Medical AI Training Script
# Runs training for ALS, Alzheimer's, and Parkinson's disease prediction models

set -e  # Exit on any error

echo "🏥 AiMedRes Medical AI Training Pipeline"
echo "========================================"
echo

# Check Python and dependencies
echo "🔍 Checking dependencies..."
python -c "import torch, sklearn, xgboost, pandas, numpy; print('✅ All dependencies available')"
echo

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "📅 Training session: $TIMESTAMP"
echo

# Training 1: ALS
echo "🧠 Training ALS (Amyotrophic Lateral Sclerosis) Model..."
echo "--------------------------------------------------------"
python training/train_als.py \
    --dataset-choice als-progression \
    --output-dir "als_results_${TIMESTAMP}" \
    --epochs 50 \
    --folds 5
echo "✅ ALS training completed"
echo

# Training 2: Alzheimer's  
echo "🧠 Training Alzheimer's Disease Model..."
echo "----------------------------------------"
python files/training/train_alzheimers.py \
    --output-dir "alzheimer_results_${TIMESTAMP}" \
    --epochs 50 \
    --folds 5
echo "✅ Alzheimer's training completed"
echo

# Training 3: Parkinson's
echo "🧠 Training Parkinson's Disease Model..."
echo "----------------------------------------"
python training/train_parkinsons.py \
    --dataset-choice vikasukani \
    --output-dir "parkinsons_results_${TIMESTAMP}" \
    --epochs 50 \
    --folds 5
echo "✅ Parkinson's training completed"
echo

# Summary
echo "🎉 All Medical AI Training Completed Successfully!"
echo "================================================="
echo "📁 Results saved in directories with timestamp: $TIMESTAMP"
echo "🔬 Models are ready for medical applications!"
echo
echo "📊 To view detailed results, run:"
echo "   python training_results_summary.py"