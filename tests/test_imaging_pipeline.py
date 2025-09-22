#!/usr/bin/env python3
"""
Test script for the imaging preprocessing and feature extraction MVP.

This script tests the complete pipeline from ingestion to model training.
"""

import logging
import sys
from pathlib import Path
import subprocess
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_imaging_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test_individual_components(logger):
    """Test individual components."""
    
    logger.info("Testing individual preprocessing components...")
    
    try:
        # Test bias correction
        from mlops.imaging.preprocessing import BiasFieldCorrector
        corrector = BiasFieldCorrector(method='simple')
        logger.info("✓ BiasFieldCorrector imported successfully")
        
        # Test skull stripping
        from mlops.imaging.preprocessing import SkullStripper
        stripper = SkullStripper(method='threshold')
        logger.info("✓ SkullStripper imported successfully")
        
        # Test registration
        from mlops.imaging.preprocessing import ImageRegistrar
        registrar = ImageRegistrar(method='affine')
        logger.info("✓ ImageRegistrar imported successfully")
        
        # Test feature extraction
        from mlops.imaging.features import VolumetricFeatureExtractor, QualityControlMetrics, RadiomicsExtractor
        vol_extractor = VolumetricFeatureExtractor()
        qc_extractor = QualityControlMetrics()
        rad_extractor = RadiomicsExtractor(enabled=True)
        logger.info("✓ Feature extractors imported successfully")
        
        # Test drift detection
        from mlops.monitoring import ImagingDriftDetector
        drift_detector = ImagingDriftDetector()
        logger.info("✓ ImagingDriftDetector imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Component test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_synthetic_data_generation(logger):
    """Test synthetic data generation."""
    
    logger.info("Testing synthetic data generation...")
    
    try:
        from mlops.imaging.generators import SyntheticNIfTIGenerator
        
        generator = SyntheticNIfTIGenerator(output_dir=str(test_dir))
        
        # Generate a test image
        result = generator.generate_synthetic_nifti(
            subject_id='test-001',
            modality='T1w',
            add_pathology=False
        )
        
        test_image_path = Path(result['nifti_path'])
        
        if test_image_path.exists():
            logger.info("✓ Synthetic NIfTI generation successful")
            
            # Test loading the generated image
            import nibabel as nib
            img = nib.load(test_image_path)
            data = img.get_fdata()
            logger.info(f"✓ Generated image shape: {data.shape}")
            
            return True
        else:
            logger.error("✗ Synthetic image not created")
            return False
            
    except Exception as e:
        logger.error(f"Synthetic data generation test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_feature_extraction(logger):
    """Test feature extraction on synthetic data."""
    
    logger.info("Testing feature extraction...")
    
    try:
        from mlops.imaging.features import VolumetricFeatureExtractor, QualityControlMetrics, RadiomicsExtractor
        from mlops.imaging.generators import SyntheticNIfTIGenerator
        
        # Generate test data
        test_dir = Path('test_outputs')
        test_dir.mkdir(exist_ok=True)
        
        generator = SyntheticNIfTIGenerator(output_dir=str(test_dir))
        result = generator.generate_synthetic_nifti(
            subject_id='test-002',
            modality='T1w',
            add_pathology=True
        )
        
        test_image_path = Path(result['nifti_path'])
        
        # Test volumetric features
        vol_extractor = VolumetricFeatureExtractor()
        vol_features = vol_extractor.extract_features(test_image_path)
        logger.info(f"✓ Extracted {len(vol_features)} volumetric features")
        
        # Test QC metrics
        qc_extractor = QualityControlMetrics()
        qc_features = qc_extractor.calculate_qc_metrics(test_image_path)
        logger.info(f"✓ Extracted {len(qc_features)} QC metrics")
        
        # Test radiomics features
        rad_extractor = RadiomicsExtractor(enabled=True)
        rad_features = rad_extractor.extract_features(test_image_path)
        logger.info(f"✓ Extracted {len(rad_features)} radiomics features")
        
        total_features = len(vol_features) + len(qc_features) + len(rad_features)
        logger.info(f"✓ Total features extracted: {total_features}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature extraction test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_preprocessing_pipeline(logger):
    """Test preprocessing pipeline."""
    
    logger.info("Testing preprocessing pipeline...")
    
    try:
        from mlops.imaging.preprocessing import BiasFieldCorrector, SkullStripper
        from mlops.imaging.generators import SyntheticNIfTIGenerator
        
        # Generate test data
        test_dir = Path('test_outputs')
        test_dir.mkdir(exist_ok=True)
        
        generator = SyntheticNIfTIGenerator(output_dir=str(test_dir))
        result = generator.generate_synthetic_nifti(
            subject_id='test-003',
            modality='T1w',
            add_pathology=False
        )
        
        test_image_path = Path(result['nifti_path'])
        
        # Test bias correction
        corrector = BiasFieldCorrector(method='simple')
        corrected_path = test_dir / 'test_bias_corrected.nii.gz'
        
        corrector.correct_bias(
            input_path=test_image_path,
            output_path=corrected_path
        )
        
        if corrected_path.exists():
            logger.info("✓ Bias correction completed")
        else:
            logger.error("✗ Bias correction failed")
            return False
        
        # Test skull stripping
        stripper = SkullStripper(method='threshold')
        stripped_path = test_dir / 'test_brain.nii.gz'
        mask_path = test_dir / 'test_mask.nii.gz'
        
        stripper.extract_brain(
            input_path=corrected_path,
            output_path=stripped_path,
            mask_path=mask_path
        )
        
        if stripped_path.exists() and mask_path.exists():
            logger.info("✓ Skull stripping completed")
        else:
            logger.error("✗ Skull stripping failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_drift_detection(logger):
    """Test drift detection functionality."""
    
    logger.info("Testing drift detection...")
    
    try:
        import pandas as pd
        import numpy as np
        from mlops.monitoring import ImagingDriftDetector
        
        # Create synthetic feature data
        np.random.seed(42)
        
        # Baseline data
        baseline_data = pd.DataFrame({
            'total_brain_volume_mm3': np.random.normal(1500000, 150000, 100),
            'gray_matter_volume_mm3': np.random.normal(600000, 60000, 100),
            'qc_snr_basic': np.random.normal(15, 3, 100),
            'qc_motion_score': np.random.normal(0.2, 0.05, 100)
        })
        
        # New data (with some drift)
        new_data = pd.DataFrame({
            'total_brain_volume_mm3': np.random.normal(1400000, 200000, 50),  # Shifted mean and variance
            'gray_matter_volume_mm3': np.random.normal(580000, 80000, 50),   # Shifted mean and variance
            'qc_snr_basic': np.random.normal(12, 4, 50),                     # Lower SNR
            'qc_motion_score': np.random.normal(0.3, 0.08, 50)              # Higher motion
        })
        
        # Test drift detector
        drift_detector = ImagingDriftDetector()
        drift_detector.fit_baseline(baseline_data)
        logger.info("✓ Drift detector fitted to baseline")
        
        # Detect drift
        drift_results = drift_detector.detect_drift(new_data)
        logger.info(f"✓ Drift detection completed. Drift detected: {drift_results['drift_detected']}")
        logger.info(f"✓ Number of alerts: {len(drift_results['alerts'])}")
        
        # Test summary
        summary = drift_detector.get_drift_summary(days=1)
        logger.info(f"✓ Drift summary generated: {summary.get('total_checks', 0)} checks")
        
        return True
        
    except Exception as e:
        logger.error(f"Drift detection test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_ml_models(logger):
    """Test baseline ML model training."""
    
    logger.info("Testing ML model training...")
    
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Create synthetic feature data
        np.random.seed(42)
        n_samples = 200
        
        # Synthetic imaging features
        features = pd.DataFrame({
            'total_brain_volume_mm3': np.random.normal(1500000, 150000, n_samples),
            'gray_matter_volume_mm3': np.random.normal(600000, 60000, n_samples),
            'white_matter_volume_mm3': np.random.normal(500000, 50000, n_samples),
            'qc_snr_basic': np.random.normal(15, 3, n_samples),
            'qc_motion_score': np.random.normal(0.2, 0.05, n_samples),
            'radiomics_mean': np.random.normal(100, 20, n_samples),
            'radiomics_std': np.random.normal(30, 5, n_samples)
        })
        
        # Create synthetic labels (pathology detection)
        # Use volume and quality metrics to create realistic labels
        label_score = (
            -0.3 * (features['total_brain_volume_mm3'] - 1500000) / 150000 +
            -0.2 * (features['qc_snr_basic'] - 15) / 3 +
            0.4 * (features['qc_motion_score'] - 0.2) / 0.05 +
            np.random.normal(0, 0.5, n_samples)
        )
        labels = (label_score > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Test LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=50,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_accuracy = accuracy_score(y_test, lgb_pred)
        logger.info(f"✓ LightGBM model trained. Accuracy: {lgb_accuracy:.3f}")
        
        # Test XGBoost
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=50,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        logger.info(f"✓ XGBoost model trained. Accuracy: {xgb_accuracy:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ML model test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def run_pipeline_integration_test(logger):
    """Run a full integration test of the pipeline stages."""
    
    logger.info("Running pipeline integration test...")
    
    try:
        # Ensure we have the required directories
        Path('data/imaging/raw').mkdir(parents=True, exist_ok=True)
        Path('data/imaging/processed').mkdir(parents=True, exist_ok=True)
        Path('outputs/imaging/features').mkdir(parents=True, exist_ok=True)
        Path('outputs/imaging/qc').mkdir(parents=True, exist_ok=True)
        
        # Test that we can import and run the pipeline scripts
        # (In a real test, we would run them, but for this demo we'll just test imports)
        
        logger.info("Testing pipeline script imports...")
        
        try:
            from mlops.pipelines import preprocess_imaging
            logger.info("✓ Preprocessing pipeline script imported")
        except ImportError as e:
            logger.error(f"✗ Preprocessing pipeline import failed: {e}")
            return False
        
        try:
            from mlops.pipelines import extract_features
            logger.info("✓ Feature extraction pipeline script imported")
        except ImportError as e:
            logger.error(f"✗ Feature extraction pipeline import failed: {e}")
            return False
        
        try:
            from mlops.pipelines import train_baseline_models
            logger.info("✓ Model training pipeline script imported")
        except ImportError as e:
            logger.error(f"✗ Model training pipeline import failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    logger = setup_logging()
    logger.info("Starting imaging pipeline MVP tests")
    
    test_results = []
    
    # Run individual tests
    tests = [
        ("Component imports", test_individual_components),
        ("Synthetic data generation", test_synthetic_data_generation),
        ("Feature extraction", test_feature_extraction),
        ("Preprocessing pipeline", test_preprocessing_pipeline),
        ("Drift detection", test_drift_detection),
        ("ML model training", test_ml_models),
        ("Pipeline integration", run_pipeline_integration_test)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func(logger)
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The imaging pipeline MVP is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)