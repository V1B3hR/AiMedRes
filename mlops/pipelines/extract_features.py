#!/usr/bin/env python3
"""
Feature Extraction Pipeline

Extracts volumetric and radiomics features from preprocessed medical images.
Includes quality control metrics and feature validation.
"""

import logging
import yaml
from pathlib import Path
import json
import pandas as pd
import traceback
from typing import Dict, Any, List
import sys
import os
import mlflow
import mlflow.sklearn

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops.imaging.features import VolumetricFeatureExtractor, QualityControlMetrics, RadiomicsExtractor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_extraction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from params_imaging.yaml."""
    config_path = Path('params_imaging.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_mlflow(config: Dict[str, Any]) -> None:
    """Setup MLflow for experiment tracking."""
    if config.get('mlops', {}).get('track_experiments', False):
        experiment_name = config['mlops'].get('experiment_name', 'imaging_feature_extraction')
        
        try:
            # Set or create experiment
            mlflow.set_experiment(experiment_name)
            logging.info(f"MLflow experiment set: {experiment_name}")
        except Exception as e:
            logging.warning(f"MLflow setup failed: {e}")


def extract_features_from_image(
    image_path: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
    mask_path: Path = None
) -> Dict[str, Any]:
    """Extract all features from a single image."""
    
    features = {}
    metadata = {
        'image_path': str(image_path),
        'mask_path': str(mask_path) if mask_path else None,
        'extraction_success': {},
        'errors': []
    }
    
    feature_config = config.get('feature_extraction', {})
    
    # Extract volumetric features
    if 'volumetric' in feature_config.get('pipelines', []):
        logger.info(f"Extracting volumetric features from {image_path}")
        try:
            volumetric_extractor = VolumetricFeatureExtractor(
                atlas=feature_config.get('volumetric', {}).get('atlas', 'AAL3')
            )
            
            volumetric_features = volumetric_extractor.extract_features(
                image_path=image_path,
                mask_path=mask_path
            )
            
            features.update(volumetric_features)
            metadata['extraction_success']['volumetric'] = True
            logger.info(f"Extracted {len(volumetric_features)} volumetric features")
            
        except Exception as e:
            error_msg = f"Volumetric feature extraction failed: {e}"
            logger.error(error_msg)
            metadata['errors'].append(error_msg)
            metadata['extraction_success']['volumetric'] = False
    
    # Extract radiomics features (if enabled)
    radiomics_enabled = feature_config.get('radiomics', {}).get('enabled', True)
    if 'radiomics' in feature_config.get('pipelines', []) and radiomics_enabled:
        logger.info(f"Extracting radiomics features from {image_path}")
        try:
            radiomics_extractor = RadiomicsExtractor(enabled=True)
            
            radiomics_features = radiomics_extractor.extract_features(
                image_path=image_path,
                mask_path=mask_path
            )
            
            features.update(radiomics_features)
            metadata['extraction_success']['radiomics'] = True
            logger.info(f"Extracted {len(radiomics_features)} radiomics features")
            
        except Exception as e:
            error_msg = f"Radiomics feature extraction failed: {e}"
            logger.error(error_msg)
            metadata['errors'].append(error_msg)
            metadata['extraction_success']['radiomics'] = False
    
    # Extract quality control metrics (always include)
    logger.info(f"Extracting QC metrics from {image_path}")
    try:
        qc_extractor = QualityControlMetrics()
        
        qc_features = qc_extractor.calculate_qc_metrics(
            image_path=image_path,
            mask_path=mask_path
        )
        
        # Add QC classification
        qc_classification = qc_extractor.classify_quality(qc_features)
        qc_features.update(qc_classification)
        
        # Prefix QC features
        qc_features_prefixed = {f"qc_{k}": v for k, v in qc_features.items()}
        features.update(qc_features_prefixed)
        
        metadata['extraction_success']['quality_control'] = True
        logger.info(f"Extracted {len(qc_features)} QC metrics")
        
    except Exception as e:
        error_msg = f"QC feature extraction failed: {e}"
        logger.error(error_msg)
        metadata['errors'].append(error_msg)
        metadata['extraction_success']['quality_control'] = False
    
    return features, metadata


def validate_features(features: Dict[str, float], logger: logging.Logger) -> Dict[str, Any]:
    """Validate extracted features for quality and completeness."""
    validation_results = {
        'total_features': len(features),
        'valid_features': 0,
        'invalid_features': 0,
        'missing_features': 0,
        'infinite_features': 0,
        'zero_features': 0,
        'issues': []
    }
    
    for feature_name, feature_value in features.items():
        if feature_value is None:
            validation_results['missing_features'] += 1
            validation_results['issues'].append(f"Missing value: {feature_name}")
        elif not isinstance(feature_value, (int, float)):
            validation_results['invalid_features'] += 1
            validation_results['issues'].append(f"Non-numeric value: {feature_name}")
        elif not isinstance(feature_value, str) and (
            not isinstance(feature_value, (int, float)) or 
            (isinstance(feature_value, float) and (
                pd.isna(feature_value) or 
                pd.isinf(feature_value)
            ))
        ):
            validation_results['infinite_features'] += 1
            validation_results['issues'].append(f"Invalid numeric value: {feature_name}")
        elif feature_value == 0:
            validation_results['zero_features'] += 1
        else:
            validation_results['valid_features'] += 1
    
    validation_results['success_rate'] = (
        validation_results['valid_features'] / validation_results['total_features']
        if validation_results['total_features'] > 0 else 0
    )
    
    if validation_results['issues']:
        logger.warning(f"Feature validation found {len(validation_results['issues'])} issues")
    
    return validation_results


def create_feature_summary(all_features: List[Dict[str, float]], logger: logging.Logger) -> Dict[str, Any]:
    """Create summary statistics of all extracted features."""
    if not all_features:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Calculate summary statistics
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'feature_types': {},
        'feature_stats': {}
    }
    
    # Categorize features
    feature_categories = {
        'volume': len([col for col in df.columns if 'volume' in col.lower()]),
        'intensity': len([col for col in df.columns if 'intensity' in col.lower() or 'mean' in col.lower()]),
        'shape': len([col for col in df.columns if any(x in col.lower() for x in ['sphericity', 'elongation', 'surface'])]),
        'texture': len([col for col in df.columns if any(x in col.lower() for x in ['glcm', 'radiomics', 'gradient'])]),
        'qc': len([col for col in df.columns if col.startswith('qc_')])
    }
    
    summary['feature_types'] = feature_categories
    
    # Calculate basic statistics for numeric features
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    
    for col in numeric_columns[:20]:  # Limit to first 20 for brevity
        if len(df[col].dropna()) > 0:
            summary['feature_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'non_null_count': int(df[col].count())
            }
    
    logger.info(f"Feature summary created for {summary['n_samples']} samples with {summary['n_features']} features")
    
    return summary


def main():
    """Main feature extraction pipeline."""
    logger = setup_logging()
    logger.info("Starting feature extraction pipeline")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Setup MLflow
        setup_mlflow(config)
        
        # Find processed images
        processed_dir = Path(config['data']['processed_dir'])
        features_dir = Path(config['data']['features_dir'])
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all processed images
        processed_images = []
        
        # Look for processed images in subdirectories
        for subject_dir in processed_dir.iterdir():
            if subject_dir.is_dir():
                # Look for the final processed image
                final_images = list(subject_dir.glob('*registered.nii.gz'))
                if not final_images:
                    final_images = list(subject_dir.glob('*brain.nii.gz'))
                if not final_images:
                    final_images = list(subject_dir.glob('*bias_corrected.nii.gz'))
                if not final_images:
                    final_images = list(subject_dir.glob('*.nii.gz'))
                
                if final_images:
                    image_path = final_images[0]  # Take the first one
                    
                    # Look for corresponding brain mask
                    mask_path = None
                    mask_files = list(subject_dir.glob('*mask.nii.gz'))
                    if mask_files:
                        mask_path = mask_files[0]
                    
                    processed_images.append((image_path, mask_path))
        
        if not processed_images:
            logger.warning(f"No processed images found in {processed_dir}")
            # Try to find any NIfTI files in processed dir
            nifti_files = list(processed_dir.glob('**/*.nii.gz'))
            if nifti_files:
                processed_images = [(f, None) for f in nifti_files]
            else:
                logger.error("No images found for feature extraction")
                return
        
        logger.info(f"Found {len(processed_images)} images for feature extraction")
        
        # Start MLflow run
        with mlflow.start_run(run_name="feature_extraction_pipeline"):
            
            # Extract features from all images
            all_features = []
            all_metadata = []
            successful_extractions = 0
            
            for i, (image_path, mask_path) in enumerate(processed_images):
                logger.info(f"Processing image {i+1}/{len(processed_images)}: {image_path}")
                
                try:
                    features, metadata = extract_features_from_image(
                        image_path, config, logger, mask_path
                    )
                    
                    # Add image identifier
                    features['image_id'] = image_path.parent.name
                    features['image_path'] = str(image_path)
                    
                    # Validate features
                    validation_results = validate_features(features, logger)
                    metadata['validation'] = validation_results
                    
                    all_features.append(features)
                    all_metadata.append(metadata)
                    
                    if validation_results['success_rate'] > 0.8:  # 80% success rate threshold
                        successful_extractions += 1
                        logger.info(f"Successfully extracted features from {image_path}")
                    else:
                        logger.warning(f"Feature extraction had issues for {image_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to extract features from {image_path}: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Add empty entry to maintain alignment
                    all_metadata.append({
                        'image_path': str(image_path),
                        'extraction_success': {},
                        'errors': [str(e)]
                    })
            
            # Create feature summary
            feature_summary = create_feature_summary(all_features, logger)
            
            # Save features as parquet (efficient for ML)
            if all_features:
                features_df = pd.DataFrame(all_features)
                features_parquet_path = features_dir / 'features.parquet'
                features_df.to_parquet(features_parquet_path, index=False)
                logger.info(f"Features saved to {features_parquet_path}")
                
                # Also save as CSV for inspection
                features_csv_path = features_dir / 'features.csv'
                features_df.to_csv(features_csv_path, index=False)
                logger.info(f"Features also saved as CSV: {features_csv_path}")
            
            # Save metadata
            metadata_path = features_dir / 'extraction_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            
            # Save feature summary
            summary_path = features_dir / 'feature_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(feature_summary, f, indent=2)
            
            # Log metrics to MLflow
            if config.get('mlops', {}).get('track_experiments', False):
                mlflow.log_metrics({
                    'total_images': len(processed_images),
                    'successful_extractions': successful_extractions,
                    'extraction_success_rate': successful_extractions / len(processed_images),
                    'total_features': feature_summary.get('n_features', 0),
                    'volume_features': feature_summary.get('feature_types', {}).get('volume', 0),
                    'texture_features': feature_summary.get('feature_types', {}).get('texture', 0),
                    'qc_features': feature_summary.get('feature_types', {}).get('qc', 0)
                })
                
                # Log artifacts
                if config.get('mlops', {}).get('log_artifacts', False):
                    mlflow.log_artifacts(str(features_dir), artifact_path="features")
            
            logger.info(f"Feature extraction pipeline completed")
            logger.info(f"Extracted features from {successful_extractions}/{len(processed_images)} images")
            logger.info(f"Total features extracted: {feature_summary.get('n_features', 0)}")
            
    except Exception as e:
        logger.error(f"Feature extraction pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()