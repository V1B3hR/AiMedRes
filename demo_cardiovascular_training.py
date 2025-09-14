#!/usr/bin/env python3
"""
Cardiovascular Disease Classification Training Demonstration

This script demonstrates both cardiovascular datasets specified in the problem statement:
1. Cardiovascular diseases risk prediction dataset
2. Cardiovascular disease dataset

Both systems train with 20 epochs matching the existing training configuration.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cardiovascular_training(dataset_choice: str, output_dir: str) -> bool:
    """Run cardiovascular training with specified dataset"""
    logger.info("=" * 60)
    logger.info(f"DEMONSTRATING CARDIOVASCULAR TRAINING WITH {dataset_choice.upper()} DATASET")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable, "train_cardiovascular.py", 
        "--epochs", "20",
        "--folds", "3",
        "--dataset-choice", dataset_choice,
        "--output-dir", output_dir
    ]
    
    logger.info(f"Running cardiovascular training with {dataset_choice} dataset...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info(f"✅ Cardiovascular training with {dataset_choice} completed successfully!")
        # Show key metrics
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Neural Network:' in line or 'Accuracy=' in line:
                print(f"  {line.strip()}")
    else:
        logger.error(f"❌ Cardiovascular training with {dataset_choice} failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_diabetes_comparison() -> bool:
    """Run diabetes training for comparison"""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING DIABETES TRAINING WITH 20 EPOCHS (FOR COMPARISON)")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable, "train_diabetes.py", 
        "--epochs", "20",
        "--folds", "3",
        "--output-dir", "/tmp/demo_diabetes_comparison"
    ]
    
    logger.info("Running diabetes training for comparison...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Diabetes training completed successfully!")
        # Show key metrics
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Neural Network:' in line or 'Accuracy=' in line:
                print(f"  {line.strip()}")
    else:
        logger.error("❌ Diabetes training failed")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Main demonstration function"""
    print("💓 Cardiovascular Disease Classification Training with 20 Epochs Demonstration")
    print("📋 Problem Statement Implementation:")
    print("   - learning, training and tests with 20 epochs")
    print("   - datasets: https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset")
    print("   - datasets: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
    print()
    
    # Run first cardiovascular dataset
    prediction_success = run_cardiovascular_training("cardiovascular-prediction", "/tmp/demo_cardiovascular_prediction")
    
    print()
    
    # Run second cardiovascular dataset
    disease_success = run_cardiovascular_training("cardiovascular-disease", "/tmp/demo_cardiovascular_disease")
    
    print()
    
    # Run diabetes for comparison
    diabetes_success = run_diabetes_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Cardiovascular diseases risk prediction (20 epochs): {'SUCCESS' if prediction_success else 'FAILED'}")
    print(f"✅ Cardiovascular disease dataset (20 epochs): {'SUCCESS' if disease_success else 'FAILED'}")
    print(f"✅ Diabetes classification (20 epochs): {'SUCCESS' if diabetes_success else 'FAILED'}")
    print()
    print("All training systems now use 20 epochs consistently.")
    print("The cardiovascular classification systems support both specified datasets.")
    print("This matches the 'same amount of epochs as last time' requirement.")
    
    return prediction_success and disease_success and diabetes_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)