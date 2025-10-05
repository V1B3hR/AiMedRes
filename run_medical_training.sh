#!/bin/bash
# AiMedRes Medical AI Training Script
# Runs training for ALL medical AI models using the orchestrator

set -e  # Exit on any error

echo "🏥 AiMedRes Medical AI Training Pipeline"
echo "========================================"
echo

# Check Python and dependencies
echo "🔍 Checking dependencies..."
if ! python -c "import numpy, pandas, sklearn" 2>/dev/null; then
    echo "❌ Missing dependencies. Please install with:"
    echo "   pip install -r requirements-ml.txt"
    exit 1
fi
echo "✅ Core dependencies available"
echo

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "📅 Training session: $TIMESTAMP"
echo

# Run all training using the orchestrator
echo "🚀 Running ALL medical AI training models..."
echo "-------------------------------------------"
echo "This will train:"
echo "  • ALS (Amyotrophic Lateral Sclerosis)"
echo "  • Alzheimer's Disease"
echo "  • Parkinson's Disease"
echo "  • Brain MRI Classification"
echo "  • Cardiovascular Disease"
echo "  • Diabetes Prediction"
echo

python run_all_training.py \
    --epochs 50 \
    --folds 5 \
    --verbose

# Summary
echo
echo "🎉 All Medical AI Training Completed Successfully!"
echo "================================================="
echo "📁 Results saved in: results/"
echo "📝 Logs saved in: logs/"
echo "📊 Summary saved in: summaries/"
echo "🔬 Models are ready for medical applications!"
echo
echo "To run specific models only:"
echo "   python run_all_training.py --only als alzheimers"
echo
echo "To run in parallel (faster):"
echo "   python run_all_training.py --parallel --max-workers 4"
echo
echo "To see all options:"
echo "   python run_all_training.py --help"