"""
DuetMind Adaptive - Medical Imaging Module

This module provides comprehensive medical imaging capabilities including:
- DICOM processing and conversion
- NIfTI generation and manipulation
- BIDS compliance validation
- Medical image preprocessing (bias correction, skull stripping, registration)
- Feature extraction (volumetric, radiomics, quality control)
- De-identification workflows
"""

from .generators.synthetic_nifti import SyntheticNIfTIGenerator
from .converters.dicom_to_nifti import DICOMToNIfTIConverter
from .validators.bids_validator import BIDSComplianceValidator
from .utils.deidentify import MedicalImageDeidentifier

# Preprocessing modules
from .preprocessing.bias_correction import BiasFieldCorrector
from .preprocessing.skull_stripping import SkullStripper
from .preprocessing.registration import ImageRegistrar

# Feature extraction modules
from .features.volumetric import VolumetricFeatureExtractor
from .features.quality_control import QualityControlMetrics
from .features.radiomics import RadiomicsExtractor

__version__ = "1.1.0"
__all__ = [
    # Original modules
    "SyntheticNIfTIGenerator",
    "DICOMToNIfTIConverter", 
    "BIDSComplianceValidator",
    "MedicalImageDeidentifier",
    # Preprocessing modules
    "BiasFieldCorrector",
    "SkullStripper",
    "ImageRegistrar",
    # Feature extraction modules
    "VolumetricFeatureExtractor",
    "QualityControlMetrics", 
    "RadiomicsExtractor"
]