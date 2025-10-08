#!/usr/bin/env python3
"""
Verification test for specialized medical agents training integration
Tests the command: python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5
"""

import subprocess
import sys
from pathlib import Path


def test_training_script_help():
    """Test that the training script has correct CLI flags"""
    print("Test 1: Verifying training script CLI...")
    result = subprocess.run(
        [sys.executable, "src/aimedres/training/train_specialized_agents.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert "--epochs" in result.stdout, "Missing --epochs flag"
    assert "--folds" in result.stdout, "Missing --folds flag"
    assert "--output-dir" in result.stdout, "Missing --output-dir flag"
    print("✓ Training script has all required flags")


def test_discovery():
    """Test that run_all_training.py discovers the script"""
    print("\nTest 2: Verifying auto-discovery...")
    result = subprocess.run(
        [sys.executable, "run_all_training.py", "--list"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    output = result.stdout + result.stderr
    assert "train_specialized_agents" in output, "Script not discovered"
    assert "epochs=True" in output, "epochs support not detected"
    assert "folds=True" in output, "folds support not detected"
    assert "outdir=True" in output, "output-dir support not detected"
    print("✓ Training script discovered with correct capabilities")


def test_dry_run():
    """Test dry-run execution with specified parameters"""
    print("\nTest 3: Verifying command construction...")
    result = subprocess.run(
        [
            sys.executable, "run_all_training.py",
            "--parallel", "--max-workers", "6",
            "--epochs", "50", "--folds", "5",
            "--only", "train_specialized_agents",
            "--dry-run"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
    assert "--epochs 50" in output, "epochs parameter not passed"
    assert "--folds 5" in output, "folds parameter not passed"
    assert "train_specialized_agents" in output, "Job not selected"
    print("✓ Command constructed correctly with --epochs 50 and --folds 5")


def test_actual_training():
    """Test actual training execution with minimal parameters"""
    print("\nTest 4: Testing actual training execution...")
    
    # Create synthetic test data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 100
    data = {
        'PatientID': range(1, n + 1),
        'Age': np.random.randint(50, 90, n),
        'M/F': np.random.choice([0, 1], n),
        'EDUC': np.random.randint(8, 20, n),
        'SES': np.random.randint(1, 5, n),
        'MMSE': np.random.randint(10, 30, n),
        'CDR': np.random.choice([0, 0.5, 1, 2], n),
        'eTIV': np.random.randint(1200, 1800, n),
        'nWBV': np.random.uniform(0.65, 0.85, n),
        'ASF': np.random.uniform(1.0, 1.4, n),
        'Diagnosis': np.random.choice(['Nondemented', 'Demented'], n, p=[0.6, 0.4])
    }
    df = pd.DataFrame(data)
    test_data_path = Path("/tmp/test_specialized_agents_data.csv")
    df.to_csv(test_data_path, index=False)
    
    # Run training with minimal parameters for speed
    result = subprocess.run(
        [
            sys.executable, "src/aimedres/training/train_specialized_agents.py",
            "--data-path", str(test_data_path),
            "--output-dir", "/tmp/test_specialized_agents_output",
            "--folds", "2",
            "--epochs", "3"
        ],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    assert result.returncode == 0, f"Training failed: {result.stderr}\nStdout: {result.stdout}"
    output = result.stdout + result.stderr
    assert "TRAINING COMPLETED SUCCESSFULLY" in output, "Training did not complete"
    
    # Verify outputs
    output_dir = Path("/tmp/test_specialized_agents_output")
    assert (output_dir / "agent_models").exists(), "agent_models directory not created"
    assert (output_dir / "metrics").exists(), "metrics directory not created"
    
    # Check for model files
    models_dir = output_dir / "agent_models"
    model_files = list(models_dir.glob("*.pkl"))
    assert len(model_files) >= 3, f"Expected at least 3 models, found {len(model_files)}"
    
    # Check for summary file
    assert (output_dir / "agent_training_summary.txt").exists(), "Summary file not created"
    
    print("✓ Training execution successful with all outputs created")


def main():
    """Run all tests"""
    print("="*80)
    print("SPECIALIZED MEDICAL AGENTS TRAINING VERIFICATION")
    print("="*80)
    
    try:
        test_training_script_help()
        test_discovery()
        test_dry_run()
        test_actual_training()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nThe specialized medical agents training is fully integrated and ready to use.")
        print("\nTo run the full training, execute:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5")
        print("\nTo run only the specialized agents training:")
        print("  python run_all_training.py --parallel --max-workers 6 --epochs 50 --folds 5 --only train_specialized_agents")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
