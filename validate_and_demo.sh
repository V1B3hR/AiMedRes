#!/bin/bash
# Quick validation script for training command
# This runs verification and demo to show everything works

set -e

echo "======================================================================"
echo "Training Command Validation & Demo"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Verify command structure (dry-run tests)"
echo "  2. Run a quick training demo (1 epoch, 2 folds)"
echo "  3. Show that the system is ready for full training"
echo ""
echo "----------------------------------------------------------------------"
echo ""

# Step 1: Verification
echo "Step 1: Running verification tests..."
echo "----------------------------------------------------------------------"
python verify_training_command.py
echo ""

# Step 2: Demo
echo "Step 2: Running training demonstration..."
echo "----------------------------------------------------------------------"
python demo_training_command.py
echo ""

# Step 3: Summary
echo "======================================================================"
echo "VALIDATION COMPLETE"
echo "======================================================================"
echo ""
echo "✅ All tests passed!"
echo "✅ Training execution validated!"
echo "✅ Command is ready for production use!"
echo ""
echo "To run full training:"
echo "  python run_all_training.py --epochs 50 --folds 5"
echo ""
echo "Or use the convenience script:"
echo "  ./run_medical_training.sh"
echo ""
echo "For more information:"
echo "  - TRAINING_EXECUTION_SUMMARY.md"
echo "  - TRAINING_COMMAND_VERIFIED.md"
echo "  - RUN_ALL_GUIDE.md"
echo ""
