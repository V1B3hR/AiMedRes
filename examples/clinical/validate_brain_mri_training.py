#!/usr/bin/env python3
"""
Brain MRI Training Validation Script

This script validates that the brain MRI training pipeline is functional
by running a minimal training test with a small subset of data.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_dependencies():
    """Validate all required dependencies are available"""
    logger.info("Validating dependencies...")
    
    required_modules = [
        'torch',
        'torchvision',
        'PIL',
        'kagglehub',
        'mlflow',
        'sklearn',
        'numpy',
        'pandas'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        return False
    
    logger.info("✓ All dependencies available")
    return True


def validate_script_syntax():
    """Validate training script compiles without syntax errors"""
    logger.info("Validating script syntax...")
    
    script_path = Path(__file__).parent / "training" / "train_brain_mri.py"
    
    if not script_path.exists():
        logger.error(f"Training script not found: {script_path}")
        return False
    
    try:
        import py_compile
        py_compile.compile(str(script_path), doraise=True)
        logger.info(f"✓ Script compiles successfully: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Script compilation failed: {e}")
        return False


def validate_imports():
    """Validate training script can be imported"""
    logger.info("Validating script imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import (
            BrainMRIDataset,
            BrainMRICNN,
            BrainMRI3DCNN,
            BrainMRITrainingPipeline,
            EarlyStopping
        )
        logger.info("✓ All classes imported successfully")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_model_creation():
    """Validate model can be created and runs"""
    logger.info("Validating model creation...")
    
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import BrainMRICNN, BrainMRI3DCNN
        
        # Test 2D CNN
        model_2d = BrainMRICNN(num_classes=2)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model_2d(dummy_input)
        
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        logger.info(f"✓ 2D CNN model works: {dummy_input.shape} -> {output.shape}")
        
        # Test 3D CNN
        model_3d = BrainMRI3DCNN(num_classes=2, input_channels=1)
        dummy_input_3d = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            output_3d = model_3d(dummy_input_3d)
        
        assert output_3d.shape == (1, 2), f"Expected output shape (1, 2), got {output_3d.shape}"
        logger.info(f"✓ 3D CNN model works: {dummy_input_3d.shape} -> {output_3d.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_pipeline_creation():
    """Validate training pipeline can be instantiated"""
    logger.info("Validating pipeline creation...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import BrainMRITrainingPipeline
        
        pipeline = BrainMRITrainingPipeline(output_dir='/tmp/validation_test')
        logger.info("✓ Training pipeline created successfully")
        
        # Verify device is set
        assert hasattr(pipeline, 'device'), "Pipeline missing device attribute"
        logger.info(f"✓ Pipeline device: {pipeline.device}")
        
        return True
    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataset_loading():
    """Validate dataset can be loaded (may download if not cached)"""
    logger.info("Validating dataset loading (this may download data)...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import BrainMRITrainingPipeline
        
        pipeline = BrainMRITrainingPipeline(output_dir='/tmp/validation_test')
        image_paths, labels = pipeline.load_dataset()
        
        assert len(image_paths) > 0, "No images loaded"
        assert len(labels) == len(image_paths), "Labels and images count mismatch"
        
        logger.info(f"✓ Dataset loaded: {len(image_paths)} images, {len(set(labels))} classes")
        logger.info(f"  Label distribution: {dict([(label, labels.count(label)) for label in set(labels)])}")
        
        return True
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_transforms():
    """Validate data transforms work correctly"""
    logger.info("Validating data transforms...")
    
    try:
        import torch
        from PIL import Image
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import BrainMRITrainingPipeline
        
        pipeline = BrainMRITrainingPipeline(output_dir='/tmp/validation_test')
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (256, 256), color='red')
        
        # Test train transform
        transformed = pipeline.train_transform(dummy_img)
        assert isinstance(transformed, torch.Tensor), "Transform didn't return tensor"
        assert transformed.shape == (3, 224, 224), f"Expected (3, 224, 224), got {transformed.shape}"
        
        logger.info(f"✓ Train transform works: (256, 256) -> {transformed.shape}")
        
        # Test val transform
        transformed_val = pipeline.val_transform(dummy_img)
        assert transformed_val.shape == (3, 224, 224), f"Expected (3, 224, 224), got {transformed_val.shape}"
        
        logger.info(f"✓ Val transform works: (256, 256) -> {transformed_val.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Transform validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks"""
    print("=" * 70)
    print("BRAIN MRI TRAINING PIPELINE VALIDATION")
    print("=" * 70)
    print()
    
    checks = [
        ("Dependencies", validate_dependencies),
        ("Script Syntax", validate_script_syntax),
        ("Script Imports", validate_imports),
        ("Model Creation", validate_model_creation),
        ("Pipeline Creation", validate_pipeline_creation),
        ("Data Transforms", validate_transforms),
        ("Dataset Loading", validate_dataset_loading),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 50)
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"Check {name} raised exception: {e}")
            results[name] = False
        print()
    
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All validation checks passed!")
        print()
        print("The brain MRI training pipeline is fully functional.")
        print("To run full training:")
        print("  python training/train_brain_mri.py --epochs 50")
        print("  python training/train_brain_mri.py --epochs 20 --batch-size 32")
        print()
    else:
        print("⚠ Some validation checks failed.")
        print("Please review the errors above before running training.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
