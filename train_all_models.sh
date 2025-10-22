#!/bin/bash
# Simple wrapper to train ALL medical AI models using the orchestrator
# Usage: ./train_all_models.sh [OPTIONS]
#
# Examples:
#   ./train_all_models.sh                           # Train all models sequentially
#   ./train_all_models.sh --parallel                # Train all models in parallel
#   ./train_all_models.sh --parallel --epochs 50    # Train with custom epochs

set -e  # Exit on error

echo "========================================"
echo "Training ALL Medical AI Models"
echo "========================================"
echo ""
echo "This will train all 7 medical AI models:"
echo "  1. ALS (Amyotrophic Lateral Sclerosis)"
echo "  2. Alzheimer's Disease"
echo "  3. Parkinson's Disease"
echo "  4. Brain MRI Classification"
echo "  5. Cardiovascular Disease"
echo "  6. Diabetes Prediction"
echo "  7. Specialized Medical Agents"
echo ""

# Default options
PARALLEL=""
MAX_WORKERS="4"
EPOCHS=""
FOLDS=""
BATCH=""
DRY_RUN=""
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --parallel)
      PARALLEL="--parallel"
      shift
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="--epochs $2"
      shift 2
      ;;
    --folds)
      FOLDS="--folds $2"
      shift 2
      ;;
    --batch)
      BATCH="--batch $2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --parallel           Run models in parallel"
      echo "  --max-workers N      Number of parallel workers (default: 4)"
      echo "  --epochs N           Number of training epochs"
      echo "  --folds N            Number of cross-validation folds"
      echo "  --batch N            Batch size for training"
      echo "  --dry-run            Show commands without executing"
      echo "  --verbose            Verbose logging"
      echo "  --help, -h           Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Train all models sequentially"
      echo "  $0 --parallel                         # Train in parallel"
      echo "  $0 --parallel --max-workers 6         # Train with 6 workers"
      echo "  $0 --epochs 50 --folds 5              # Custom training parameters"
      echo "  $0 --dry-run                          # Preview without execution"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build command
CMD="./aimedres train --no-auto-discover"

if [ -n "$PARALLEL" ]; then
  CMD="$CMD $PARALLEL --max-workers $MAX_WORKERS"
fi

if [ -n "$EPOCHS" ]; then
  CMD="$CMD $EPOCHS"
fi

if [ -n "$FOLDS" ]; then
  CMD="$CMD $FOLDS"
fi

if [ -n "$BATCH" ]; then
  CMD="$CMD $BATCH"
fi

if [ -n "$DRY_RUN" ]; then
  CMD="$CMD $DRY_RUN"
fi

if [ -n "$VERBOSE" ]; then
  CMD="$CMD $VERBOSE"
fi

# Show command
echo "Executing: $CMD"
echo ""

# Execute
eval $CMD

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Results can be found in:"
echo "  - results/           (training outputs)"
echo "  - logs/              (detailed logs)"
echo "  - summaries/         (training summaries)"
