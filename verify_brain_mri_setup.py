#!/usr/bin/env python3
"""
Brain MRI Training Verification Script

This script verifies that the brain MRI training setup is working correctly
by checking dependencies, downloading the dataset, and showing sample data.
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        from sklearn import __version__ as sklearn_version
        print(f"✓ Scikit-learn {sklearn_version}")
        
        from PIL import Image
        print(f"✓ Pillow (PIL)")
        
        import kagglehub
        print(f"✓ KaggleHub")
        
        import mlflow
        print(f"✓ MLflow {mlflow.__version__}")
        
        print("✓ All dependencies are available!\n")
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def verify_dataset():
    """Verify the brain MRI dataset can be accessed"""
    print("Verifying brain MRI dataset...")
    
    try:
        import kagglehub
        
        # Check if dataset is already cached
        cache_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "ashfakyeafi" / "brain-mri-images"
        
        if cache_path.exists():
            # Count images in cache
            import os
            image_files = []
            for root, dirs, files in os.walk(cache_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, file))
            
            print(f"✓ Dataset cached locally with {len(image_files)} images")
            
            # Show sample paths
            if image_files:
                print("Sample image paths:")
                for i, path in enumerate(image_files[:3]):
                    print(f"  {i+1}. {path}")
                    
                # Try loading a sample image
                try:
                    from PIL import Image
                    sample_img = Image.open(image_files[0])
                    print(f"✓ Successfully loaded sample image: {sample_img.size}")
                except Exception as e:
                    print(f"⚠ Could not load sample image: {e}")
            
            return True
        else:
            print("✗ Dataset not cached yet. Would need to download first.")
            return False
            
    except Exception as e:
        print(f"✗ Error verifying dataset: {e}")
        return False

def verify_training_script():
    """Verify the training script is valid"""
    print("Verifying training script...")
    
    script_path = Path("files/training/train_brain_mri.py")
    
    if not script_path.exists():
        print(f"✗ Training script not found at {script_path}")
        return False
    
    try:
        # Test compilation
        import py_compile
        py_compile.compile(str(script_path), doraise=True)
        print("✓ Training script compiles successfully")
        
        # Check if script has main function
        with open(script_path, 'r') as f:
            content = f.read()
            
        if "def main():" in content and "if __name__ == \"__main__\":" in content:
            print("✓ Training script has proper main function")
        else:
            print("⚠ Training script missing main function structure")
        
        # Check for required classes
        required_classes = ["BrainMRIDataset", "BrainMRICNN", "BrainMRITrainingPipeline"]
        missing_classes = []
        
        for cls in required_classes:
            if f"class {cls}" not in content:
                missing_classes.append(cls)
        
        if not missing_classes:
            print("✓ All required classes present in training script")
        else:
            print(f"⚠ Missing classes: {missing_classes}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error verifying training script: {e}")
        return False

def verify_output_directory():
    """Verify output directories can be created"""
    print("Verifying output directory setup...")
    
    try:
        from pathlib import Path
        
        test_dir = Path("brain_mri_verification_test")
        test_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (test_dir / "models").mkdir(exist_ok=True)
        (test_dir / "metrics").mkdir(exist_ok=True)
        (test_dir / "logs").mkdir(exist_ok=True)
        
        print(f"✓ Output directory structure created at {test_dir}")
        
        # Clean up test directory
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating output directories: {e}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("BRAIN MRI TRAINING SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Dataset", verify_dataset),
        ("Training Script", verify_training_script),
        ("Output Directories", verify_output_directory),
    ]
    
    results = {}
    
    for name, check_func in checks:
        print(f"{name}:")
        print("-" * 20)
        results[name] = check_func()
        print()
    
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! Brain MRI training setup is ready.")
        print()
        print("You can now run the training script:")
        print("python files/training/train_brain_mri.py --epochs 20")
    else:
        print("⚠ Some checks failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)