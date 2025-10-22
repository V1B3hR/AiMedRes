#!/bin/bash
# Demonstration script for running ALL medical AI models using the orchestrator
# This shows the capability to train all 7 medical AI models

set -e  # Exit on error

echo "=========================================="
echo "AiMedRes - Train All Medical AI Models"
echo "=========================================="
echo ""

echo "This script demonstrates running ALL medical AI models using the orchestrator."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: List all available models${NC}"
echo "Command: ./aimedres train --list --no-auto-discover"
echo ""
./aimedres train --list --no-auto-discover
echo ""

echo -e "${BLUE}Step 2: Dry-run - Preview training commands for all models${NC}"
echo "Command: ./aimedres train --dry-run --no-auto-discover --epochs 5 --folds 2"
echo ""
./aimedres train --dry-run --no-auto-discover --epochs 5 --folds 2
echo ""

echo -e "${BLUE}Step 3: Dry-run - Parallel execution of all models${NC}"
echo "Command: ./aimedres train --dry-run --parallel --max-workers 4 --epochs 10 --folds 3 --no-auto-discover"
echo ""
./aimedres train --dry-run --parallel --max-workers 4 --epochs 10 --folds 3 --no-auto-discover
echo ""

echo -e "${BLUE}Step 4: Dry-run - Train specific models (subset)${NC}"
echo "Command: ./aimedres train --dry-run --only als alzheimers parkinsons --epochs 20 --folds 5"
echo ""
./aimedres train --dry-run --only als alzheimers parkinsons --epochs 20 --folds 5
echo ""

echo -e "${GREEN}=========================================="
echo "Demo Complete!"
echo "==========================================${NC}"
echo ""
echo "The orchestrator can successfully:"
echo "  ✓ Discover and list all 7 medical AI models"
echo "  ✓ Generate training commands for all models"
echo "  ✓ Support parallel execution"
echo "  ✓ Apply custom parameters (epochs, folds, batch size)"
echo "  ✓ Filter and select specific models"
echo ""
echo -e "${YELLOW}To actually train all models (not dry-run):${NC}"
echo "  Sequential: ./aimedres train --no-auto-discover"
echo "  Parallel:   ./aimedres train --parallel --max-workers 4 --no-auto-discover"
echo ""
echo -e "${YELLOW}With custom parameters:${NC}"
echo "  ./aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128"
echo ""
