"""
DuetMind Adaptive - Medical Imaging Module

This module provides comprehensive medical imaging capabilities including:
- DICOM processing and conversion
- NIfTI generation and manipulation
- BIDS compliance validation
- Medical image feature extraction
- Quality control assessment
- De-identification workflows
"""

from .generators.synthetic_nifti import SyntheticNIfTIGenerator
from .converters.dicom_to_nifti import DICOMToNIfTIConverter
from .validators.bids_validator import BIDSComplianceValidator
from .utils.deidentify import MedicalImageDeidentifier

__version__ = "1.0.0"
__all__ = [
    "SyntheticNIfTIGenerator",
    "DICOMToNIfTIConverter", 
    "BIDSComplianceValidator",
    "MedicalImageDeidentifier"
]