"""
External system integration modules.

This package contains integrations with EHR systems, DICOM imaging workflows,
and multimodal data sources.
"""

from .dicom_handler import DICOMAnonymizer, DICOMMetadata, DICOMWorkflowManager, CTScanProcessor

__all__ = [
    "ehr",
    "multimodal",
    "DICOMWorkflowManager",
    "CTScanProcessor",
    "DICOMAnonymizer",
    "DICOMMetadata",
]
