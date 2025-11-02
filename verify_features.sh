#!/bin/bash
# Feature Verification Script
# Runs all tests for verified features

echo "=========================================="
echo "AiMedRes Feature Verification Test Suite"
echo "=========================================="
echo ""

# Set Python path for PHI scrubber tests
export PYTHONPATH=/home/runner/work/AiMedRes/AiMedRes/src:$PYTHONPATH

echo "1. Testing Immutable Audit Trail (Blockchain)..."
echo "   File: security/blockchain_records.py"
python -m pytest tests/test_phase2b_security.py::TestBlockchainMedicalRecords -v --tb=short
BLOCKCHAIN_RESULT=$?

echo ""
echo "2. Testing PHI Scrubber..."
echo "   File: src/aimedres/security/phi_scrubber.py"
python -m pytest tests/test_phi_detection.py::TestPHIScrubber -v --tb=short
PHI_RESULT=$?

echo ""
echo "3. Testing Bias Detector..."
echo "   File: files/safety/decision_validation/bias_detector.py"
python -m pytest tests/test_bias_detector.py -v --tb=short
BIAS_RESULT=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo ""

if [ $BLOCKCHAIN_RESULT -eq 0 ]; then
    echo "✅ Blockchain Audit Trail: PASS (15 tests)"
else
    echo "❌ Blockchain Audit Trail: FAIL"
fi

if [ $PHI_RESULT -eq 0 ]; then
    echo "✅ PHI Scrubber: PASS (14 tests)"
else
    echo "❌ PHI Scrubber: FAIL"
fi

if [ $BIAS_RESULT -eq 0 ]; then
    echo "✅ Bias Detector: PASS (21 tests)"
else
    echo "❌ Bias Detector: FAIL"
fi

echo ""

# Overall result
if [ $BLOCKCHAIN_RESULT -eq 0 ] && [ $PHI_RESULT -eq 0 ] && [ $BIAS_RESULT -eq 0 ]; then
    echo "=========================================="
    echo "✅ ALL FEATURES VERIFIED AND TESTED"
    echo "=========================================="
    echo ""
    echo "Total: 50 tests passing"
    echo ""
    echo "See FEATURE_STATUS.md for detailed documentation."
    exit 0
else
    echo "=========================================="
    echo "❌ SOME TESTS FAILED"
    echo "=========================================="
    exit 1
fi
