#!/usr/bin/env python3
"""
Diabetes Classification Training Demonstration

This script demonstrates both diabetes datasets specified in the problem statement:
1. Early diabetes classification dataset
2. Early-stage diabetes risk prediction dataset

Both systems train with 20 epochs matching the existing training configuration.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_diabetes_training(dataset_choice: str, output_dir: str) -> bool:
    """Run diabetes training with specified dataset"""
    logger.info("=" * 60)
    logger.info(f"DEMONSTRATING DIABETES TRAINING WITH {dataset_choice.upper()} DATASET")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable, "train_diabetes.py", 
        "--epochs", "20",
        "--folds", "3",
        "--dataset-choice", dataset_choice,
        "--output-dir", output_dir
    ]
    
    logger.info(f"Running diabetes training with {dataset_choice} dataset...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"✅ Diabetes training with {dataset_choice} completed successfully!")
        # Show key metrics
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Neural Network:' in line or 'Accuracy=' in line:
                print(f"  {line.strip()}")
    else:
        logger.error(f"❌ Diabetes training with {dataset_choice} failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_alzheimers_comparison() -> bool:
    """Run alzheimers training for comparison"""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING ALZHEIMER'S TRAINING WITH 20 EPOCHS (FOR COMPARISON)")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable, "train_alzheimers.py", 
        "--epochs", "20",
        "--folds", "3",
        "--output-dir", "/tmp/demo_alzheimers_comparison"
    ]
    
    logger.info("Running Alzheimer's training for comparison...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Alzheimer's training completed successfully!")
        # Show key metrics
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Neural Network:' in line or 'Accuracy=' in line:
                print(f"  {line.strip()}")
    else:
        logger.error("❌ Alzheimer's training failed")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Main demonstration function"""
    print("🩺 Diabetes Classification Training with 20 Epochs Demonstration")
    print("📋 Problem Statement Implementation:")
    print("   - learning, training and tests same amount of epochs as last time")
    print("   - datasets: https://www.kaggle.com/datasets/andrewmvd/early-diabetes-classification")
    print("   - datasets: https://www.kaggle.com/datasets/tanshihjen/early-stage-diabetes-risk-prediction")
    print()
    
    # Run first diabetes dataset
    early_diabetes_success = run_diabetes_training("early-diabetes", "/tmp/demo_early_diabetes")
    
    print()
    
    # Run second diabetes dataset
    early_stage_success = run_diabetes_training("early-stage", "/tmp/demo_early_stage")
    
    print()
    
    # Run Alzheimer's for comparison
    alzheimers_success = run_alzheimers_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Early diabetes classification (20 epochs): {'SUCCESS' if early_diabetes_success else 'FAILED'}")
    print(f"✅ Early-stage diabetes risk prediction (20 epochs): {'SUCCESS' if early_stage_success else 'FAILED'}")
    print(f"✅ Alzheimer's classification (20 epochs): {'SUCCESS' if alzheimers_success else 'FAILED'}")
    print()
    print("All training systems now use 20 epochs consistently.")
    print("The diabetes classification systems support both specified datasets.")
    print("This matches the 'same amount of epochs as last time' requirement.")
    
    return early_diabetes_success and early_stage_success and alzheimers_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)