#!/usr/bin/env python3
"""
Imaging Data Ingestion Pipeline

Ingests and validates raw imaging data, converts DICOM to NIfTI if needed,
and prepares data for preprocessing pipeline.
"""

import logging
import yaml
from pathlib import Path
import json
import traceback
from typing import Dict, Any, List
import sys
import os
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mlops.imaging.converters import DICOMToNIfTIConverter
from mlops.imaging.validators import BIDSComplianceValidator
from mlops.imaging.generators import SyntheticNIfTIGenerator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ingestion.log'),
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


def ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure all required directories exist."""
    directories = [
        config['data']['raw_imaging_dir'],
        config['data']['nifti_dir'],
        config['data']['bids_dir'],
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def find_input_data(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, List[Path]]:
    """Find input data files (DICOM or NIfTI)."""
    input_files = {
        'dicom': [],
        'nifti': [],
        'other': []
    }
    
    # Check if raw imaging directory exists and has data
    raw_dir = Path(config['data']['raw_imaging_dir'])
    
    if raw_dir.exists():
        # Look for DICOM files
        dicom_patterns = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
        for pattern in dicom_patterns:
            input_files['dicom'].extend(raw_dir.rglob(pattern))
        
        # Look for NIfTI files
        nifti_patterns = ['*.nii', '*.nii.gz']
        for pattern in nifti_patterns:
            input_files['nifti'].extend(raw_dir.rglob(pattern))
        
        # Look for other medical imaging formats
        other_patterns = ['*.img', '*.hdr', '*.mgh', '*.mgz']
        for pattern in other_patterns:
            input_files['other'].extend(raw_dir.rglob(pattern))
    
    logger.info(f"Found {len(input_files['dicom'])} DICOM files")
    logger.info(f"Found {len(input_files['nifti'])} NIfTI files")
    logger.info(f"Found {len(input_files['other'])} other format files")
    
    return input_files


def generate_synthetic_data(config: Dict[str, Any], logger: logging.Logger) -> List[Path]:
    """Generate synthetic NIfTI data for demonstration."""
    logger.info("Generating synthetic imaging data...")
    
    synthetic_config = config.get('synthetic', {})
    output_dir = Path(config['data']['raw_imaging_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticNIfTIGenerator()
    generated_files = []
    
    num_subjects = synthetic_config.get('num_subjects', 5)
    modalities = synthetic_config.get('modalities', ['T1w'])
    pathology_rate = synthetic_config.get('pathology_rate', 0.3)
    
    for subject_id in range(1, num_subjects + 1):
        for modality in modalities:
            output_path = output_dir / f"sub-{subject_id:03d}_{modality}.nii.gz"
            
            try:
                generator.generate_synthetic_nifti(
                    output_path=str(output_path),
                    modality=modality,
                    add_pathology=(subject_id / num_subjects) < pathology_rate,
                    noise_level=0.1
                )
                
                generated_files.append(output_path)
                logger.info(f"Generated: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate {output_path}: {e}")
    
    logger.info(f"Generated {len(generated_files)} synthetic files")
    return generated_files


def convert_dicom_to_nifti(
    dicom_files: List[Path], 
    config: Dict[str, Any], 
    logger: logging.Logger
) -> List[Path]:
    """Convert DICOM files to NIfTI format."""
    if not dicom_files:
        return []
    
    logger.info(f"Converting {len(dicom_files)} DICOM files to NIfTI")
    
    output_dir = Path(config['data']['nifti_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converter = DICOMToNIfTIConverter()
    converted_files = []
    
    # Group DICOM files by directory (assuming each directory is a series)
    dicom_dirs = {}
    for dicom_file in dicom_files:
        parent_dir = dicom_file.parent
        if parent_dir not in dicom_dirs:
            dicom_dirs[parent_dir] = []
        dicom_dirs[parent_dir].append(dicom_file)
    
    for dicom_dir, files in dicom_dirs.items():
        try:
            # Create output filename based on directory name
            output_name = f"{dicom_dir.name}.nii.gz"
            output_path = output_dir / output_name
            
            # Convert the first file (converter will handle the series)
            result = converter.convert_dicom_series(
                input_path=str(files[0]),
                output_path=str(output_path)
            )
            
            if result and Path(output_path).exists():
                converted_files.append(Path(output_path))
                logger.info(f"Converted DICOM series to: {output_path}")
            else:
                logger.warning(f"DICOM conversion failed for: {dicom_dir}")
                
        except Exception as e:
            logger.error(f"Failed to convert DICOM series {dicom_dir}: {e}")
    
    logger.info(f"Successfully converted {len(converted_files)} DICOM series")
    return converted_files


def validate_nifti_files(
    nifti_files: List[Path], 
    config: Dict[str, Any], 
    logger: logging.Logger
) -> Dict[str, Any]:
    """Validate NIfTI files for quality and compliance."""
    logger.info(f"Validating {len(nifti_files)} NIfTI files")
    
    validation_results = {
        'total_files': len(nifti_files),
        'valid_files': 0,
        'invalid_files': 0,
        'warnings': 0,
        'file_results': []
    }
    
    for nifti_file in nifti_files:
        file_result = {
            'file_path': str(nifti_file),
            'valid': False,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Try to load and validate the NIfTI file
            import nibabel as nib
            
            img = nib.load(nifti_file)
            
            # Basic validation
            data = img.get_fdata()
            
            # Check for valid data
            if data.size == 0:
                file_result['errors'].append("Empty image data")
            elif not data.any():
                file_result['warnings'].append("Image contains only zeros")
            
            # Check dimensions
            if len(data.shape) < 3:
                file_result['errors'].append(f"Invalid dimensions: {data.shape}")
            elif len(data.shape) > 4:
                file_result['warnings'].append(f"Unusual dimensions: {data.shape}")
            
            # Check voxel sizes
            voxel_sizes = img.header.get_zooms()
            if any(size <= 0 for size in voxel_sizes[:3]):
                file_result['errors'].append(f"Invalid voxel sizes: {voxel_sizes}")
            
            # Store metadata
            file_result['metadata'] = {
                'shape': data.shape,
                'voxel_sizes': voxel_sizes,
                'data_type': str(data.dtype),
                'file_size_mb': nifti_file.stat().st_size / (1024 * 1024)
            }
            
            # Mark as valid if no errors
            if not file_result['errors']:
                file_result['valid'] = True
                validation_results['valid_files'] += 1
            else:
                validation_results['invalid_files'] += 1
            
            if file_result['warnings']:
                validation_results['warnings'] += 1
                
        except Exception as e:
            file_result['errors'].append(f"Failed to load file: {e}")
            validation_results['invalid_files'] += 1
        
        validation_results['file_results'].append(file_result)
    
    logger.info(f"Validation complete: {validation_results['valid_files']} valid, "
                f"{validation_results['invalid_files']} invalid, "
                f"{validation_results['warnings']} with warnings")
    
    return validation_results


def organize_bids_structure(
    nifti_files: List[Path], 
    config: Dict[str, Any], 
    logger: logging.Logger
) -> Dict[str, Any]:
    """Organize files into BIDS-like structure."""
    logger.info(f"Organizing {len(nifti_files)} files into BIDS structure")
    
    bids_dir = Path(config['data']['bids_dir'])
    bids_dir.mkdir(parents=True, exist_ok=True)
    
    organization_results = {
        'organized_files': 0,
        'failed_files': 0,
        'subjects': [],
        'file_mappings': []
    }
    
    for i, nifti_file in enumerate(nifti_files):
        try:
            # Extract subject information from filename
            filename = nifti_file.stem.replace('.nii', '')
            
            # Try to parse subject ID and modality
            if 'sub-' in filename:
                parts = filename.split('_')
                subject_id = next((part for part in parts if part.startswith('sub-')), f'sub-{i+1:03d}')
                modality = next((part for part in parts if part in ['T1w', 'T2w', 'FLAIR']), 'T1w')
            else:
                subject_id = f'sub-{i+1:03d}'
                modality = 'T1w'
            
            # Create BIDS directory structure
            subject_dir = bids_dir / subject_id / 'anat'
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Create BIDS-compliant filename
            bids_filename = f"{subject_id}_{modality}.nii.gz"
            bids_path = subject_dir / bids_filename
            
            # Copy file to BIDS location
            shutil.copy2(nifti_file, bids_path)
            
            organization_results['organized_files'] += 1
            organization_results['file_mappings'].append({
                'original': str(nifti_file),
                'bids': str(bids_path),
                'subject_id': subject_id,
                'modality': modality
            })
            
            if subject_id not in organization_results['subjects']:
                organization_results['subjects'].append(subject_id)
            
            logger.info(f"Organized: {nifti_file} -> {bids_path}")
            
        except Exception as e:
            logger.error(f"Failed to organize {nifti_file}: {e}")
            organization_results['failed_files'] += 1
    
    # Create dataset_description.json
    dataset_description = {
        "Name": "DuetMind Adaptive Imaging Dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "Authors": ["DuetMind Adaptive System"],
        "GeneratedBy": [
            {
                "Name": "DuetMind Adaptive Ingestion Pipeline",
                "Version": "1.0.0"
            }
        ]
    }
    
    with open(bids_dir / 'dataset_description.json', 'w') as f:
        json.dump(dataset_description, f, indent=2)
    
    logger.info(f"BIDS organization complete: {organization_results['organized_files']} files, "
                f"{len(organization_results['subjects'])} subjects")
    
    return organization_results


def main():
    """Main ingestion pipeline."""
    logger = setup_logging()
    logger.info("Starting imaging data ingestion pipeline")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Ensure directories exist
        ensure_directories(config)
        
        # Find existing input data
        input_files = find_input_data(config, logger)
        
        # Generate synthetic data if no real data exists
        all_nifti_files = input_files['nifti'].copy()
        
        if not any(input_files.values()):
            logger.info("No input data found, generating synthetic data")
            synthetic_files = generate_synthetic_data(config, logger)
            all_nifti_files.extend(synthetic_files)
        else:
            # Convert DICOM files if any exist
            if input_files['dicom']:
                converted_files = convert_dicom_to_nifti(input_files['dicom'], config, logger)
                all_nifti_files.extend(converted_files)
        
        if not all_nifti_files:
            logger.error("No NIfTI files available for processing")
            return
        
        logger.info(f"Processing {len(all_nifti_files)} NIfTI files")
        
        # Validate NIfTI files
        validation_results = validate_nifti_files(all_nifti_files, config, logger)
        
        # Filter valid files
        valid_files = [
            Path(result['file_path']) 
            for result in validation_results['file_results'] 
            if result['valid']
        ]
        
        if not valid_files:
            logger.error("No valid NIfTI files found")
            return
        
        # Organize into BIDS structure
        organization_results = organize_bids_structure(valid_files, config, logger)
        
        # Save ingestion summary
        summary = {
            'ingestion_timestamp': str(pd.Timestamp.now()),
            'input_summary': {
                'dicom_files': len(input_files['dicom']),
                'nifti_files': len(input_files['nifti']),
                'other_files': len(input_files['other'])
            },
            'validation_summary': validation_results,
            'organization_summary': organization_results,
            'final_dataset': {
                'total_subjects': len(organization_results['subjects']),
                'total_files': organization_results['organized_files'],
                'bids_directory': str(config['data']['bids_dir'])
            }
        }
        
        # Save summary
        output_dir = Path(config['data']['raw_imaging_dir']).parent / 'ingestion_outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / 'ingestion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Ingestion pipeline completed successfully")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Final dataset: {summary['final_dataset']['total_subjects']} subjects, "
                   f"{summary['final_dataset']['total_files']} files")
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Add pandas import that was missing
    import pandas as pd
    main()