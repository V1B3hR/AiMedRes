#!/usr/bin/env python3
"""
Demo script showcasing the Imaging Preprocessing & Feature Extraction MVP

This script demonstrates the key capabilities implemented:
1. Synthetic data generation
2. Preprocessing (bias correction, skull stripping)
3. Feature extraction (volumetric, QC, basic radiomics)
4. Drift detection
5. Baseline ML model training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🧠 DuetMind Adaptive - Imaging Preprocessing & Feature Extraction MVP Demo")
    print("=" * 80)
    
    # 1. Synthetic Data Generation
    print("\n1️⃣  SYNTHETIC DATA GENERATION")
    print("-" * 40)
    
    from mlops.imaging.generators import SyntheticNIfTIGenerator
    
    generator = SyntheticNIfTIGenerator(output_dir="demo_outputs")
    result = generator.generate_synthetic_nifti(
        subject_id="demo-001",
        modality="T1w", 
        add_pathology=True
    )
    
    print(f"✅ Generated synthetic brain image: {result['nifti_path']}")
    print(f"   📊 QC Metrics: SNR={result['qc_metrics']['snr_estimate']:.2f}")
    
    # 2. Preprocessing Pipeline
    print("\n2️⃣  PREPROCESSING PIPELINE")
    print("-" * 40)
    
    from mlops.imaging.preprocessing import BiasFieldCorrector, SkullStripper
    
    # Bias correction
    corrector = BiasFieldCorrector(method='simple')
    corrected_path = Path("demo_outputs") / "bias_corrected.nii.gz"
    corrector.correct_bias(
        input_path=result['nifti_path'],
        output_path=corrected_path
    )
    print(f"✅ Bias correction completed: {corrected_path}")
    
    # Skull stripping
    stripper = SkullStripper(method='threshold')
    brain_path = Path("demo_outputs") / "brain_extracted.nii.gz" 
    mask_path = Path("demo_outputs") / "brain_mask.nii.gz"
    
    stripper.extract_brain(
        input_path=corrected_path,
        output_path=brain_path,
        mask_path=mask_path
    )
    print(f"✅ Skull stripping completed: {brain_path}")
    
    # 3. Feature Extraction
    print("\n3️⃣  FEATURE EXTRACTION")
    print("-" * 40)
    
    from mlops.imaging.features import VolumetricFeatureExtractor, QualityControlMetrics, RadiomicsExtractor
    
    # Volumetric features
    vol_extractor = VolumetricFeatureExtractor()
    vol_features = vol_extractor.extract_features(brain_path, mask_path)
    print(f"✅ Extracted {len(vol_features)} volumetric features")
    print(f"   🧠 Total brain volume: {vol_features.get('total_brain_volume_mm3', 0):.0f} mm³")
    
    # QC metrics
    qc_extractor = QualityControlMetrics()
    qc_features = qc_extractor.calculate_qc_metrics(brain_path, mask_path)
    print(f"✅ Extracted {len(qc_features)} QC metrics")
    print(f"   📊 SNR: {qc_features.get('snr_basic', 0):.2f}")
    
    # Basic radiomics
    rad_extractor = RadiomicsExtractor(enabled=True)
    rad_features = rad_extractor.extract_features(brain_path, mask_path)
    print(f"✅ Extracted {len(rad_features)} radiomics features")
    
    # 4. Drift Detection Demo
    print("\n4️⃣  DRIFT DETECTION")
    print("-" * 40)
    
    from mlops.monitoring import ImagingDriftDetector
    
    # Create synthetic baseline and new data
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'total_brain_volume_mm3': np.random.normal(1500000, 150000, 100),
        'gray_matter_volume_mm3': np.random.normal(600000, 60000, 100),
        'qc_snr_basic': np.random.normal(15, 3, 100),
        'qc_motion_score': np.random.normal(0.2, 0.05, 100)
    })
    
    new_data = pd.DataFrame({
        'total_brain_volume_mm3': np.random.normal(1400000, 200000, 20),  # Drifted
        'gray_matter_volume_mm3': np.random.normal(580000, 80000, 20),
        'qc_snr_basic': np.random.normal(12, 4, 20),  # Lower SNR
        'qc_motion_score': np.random.normal(0.3, 0.08, 20)  # More motion
    })
    
    # Test drift detection
    drift_detector = ImagingDriftDetector()
    drift_detector.fit_baseline(baseline_data)
    drift_results = drift_detector.detect_drift(new_data)
    
    print(f"✅ Drift detection completed")
    print(f"   🚨 Drift detected: {drift_results['drift_detected']}")
    print(f"   ⚠️  Number of alerts: {len(drift_results['alerts'])}")
    if drift_results['alerts']:
        print(f"   📋 Sample alert: {drift_results['alerts'][0]}")
    
    # 5. Baseline ML Models Demo
    print("\n5️⃣  BASELINE ML MODELS")
    print("-" * 40)
    
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Create synthetic feature data for demo
    np.random.seed(42)
    n_samples = 200
    
    # Combine all our extracted features into a demo dataset
    demo_features = pd.DataFrame({
        'total_brain_volume_mm3': np.random.normal(1500000, 150000, n_samples),
        'gray_matter_volume_mm3': np.random.normal(600000, 60000, n_samples),
        'white_matter_volume_mm3': np.random.normal(500000, 50000, n_samples),
        'qc_snr_basic': np.random.normal(15, 3, n_samples),
        'qc_motion_score': np.random.normal(0.2, 0.05, n_samples),
        'radiomics_mean': np.random.normal(100, 20, n_samples),
        'sphericity': np.random.uniform(0.5, 0.9, n_samples)
    })
    
    # Create realistic synthetic labels (pathology detection)
    label_score = (
        -0.3 * (demo_features['total_brain_volume_mm3'] - 1500000) / 150000 +
        -0.2 * (demo_features['qc_snr_basic'] - 15) / 3 +
        0.4 * (demo_features['qc_motion_score'] - 0.2) / 0.05 +
        np.random.normal(0, 0.5, n_samples)
    )
    labels = (label_score > 0).astype(int)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        demo_features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_accuracy = accuracy_score(y_test, lgb_model.predict(X_test))
    
    # XGBoost  
    xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
    
    print(f"✅ LightGBM baseline model trained - Accuracy: {lgb_accuracy:.3f}")
    print(f"✅ XGBoost baseline model trained - Accuracy: {xgb_accuracy:.3f}")
    
    # Summary
    print("\n🎉 MVP DEMO COMPLETE!")
    print("=" * 80)
    print("✅ Synthetic data generation")
    print("✅ Bias correction & skull stripping")
    print("✅ Volumetric, QC & radiomics feature extraction")
    print("✅ Statistical drift detection")  
    print("✅ LightGBM & XGBoost baseline models")
    print("\n🚀 All major MVP requirements successfully implemented!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ Demo completed successfully!")
        else:
            print("\n❌ Demo failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)