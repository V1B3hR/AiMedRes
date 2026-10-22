"""
Open-Access Biomedical Dataset Connectors for Scientific Research.
Conforms to Phase 0.1 & 3.1 of the Consolidated Roadmap.

Provides zero-wait, leakage-free data loaders for:
- OASIS-2 (Longitudinal Brain MRI and Clinical Demographics)
- PRO-ACT (ALS Clinical Trials Consortium Database)
- PTB-XL (12-Lead Diagnostic Electrocardiography Database)
- Parkinson's Telemonitoring (Oxford/UCI Longitudinal Vocal Telemetry)
"""

from ..data.open_datasets import (
    OASISDataset,
    ParkinsonsTelemonitoringDataset,
    PROACTALSDataset,
    PTBXLECGDataset,
    get_leakage_safe_cv,
    get_open_dataset,
)

__all__ = [
    "OASISDataset",
    "ParkinsonsTelemonitoringDataset",
    "PROACTALSDataset",
    "PTBXLECGDataset",
    "get_open_dataset",
    "get_leakage_safe_cv",
]
