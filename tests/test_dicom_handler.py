"""Unit tests for DICOM handler workflow utilities."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydicom.dataset import Dataset

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "aimedres" / "integration" / "dicom_handler.py"
)
spec = importlib.util.spec_from_file_location("dicom_handler", MODULE_PATH)
dicom_handler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dicom_handler)

CTScanProcessor = dicom_handler.CTScanProcessor
DICOMAnonymizer = dicom_handler.DICOMAnonymizer
DICOMWorkflowManager = dicom_handler.DICOMWorkflowManager


def test_dicom_anonymizer_clears_phi_fields():
    anonymizer = DICOMAnonymizer()
    ds = Dataset()
    ds.PatientName = "John Doe"
    ds.PatientBirthDate = "19600101"
    ds.PatientAddress = "123 Main St"
    ds.PatientPhone = "555-1212"
    ds.InstitutionName = "General Hospital"
    ds.ReferringPhysicianName = "Dr Smith"
    ds.OperatorsName = "Tech A"
    ds.PatientID = "PAT-001"

    result = anonymizer.anonymize_dataset(ds)

    assert result.PatientName == ""
    assert result.PatientBirthDate == ""
    assert result.PatientAddress == ""
    assert result.PatientPhone == ""
    assert result.InstitutionName == ""
    assert result.ReferringPhysicianName == ""
    assert result.OperatorsName == ""
    assert result.PatientID
    assert result.PatientID != "PAT-001"


def test_ct_processor_apply_windowing_in_unit_interval():
    processor = CTScanProcessor(config={})
    array = np.array([[-1000.0, 0.0, 400.0, 1000.0]], dtype=np.float32)

    windowed = processor.apply_windowing(array, window_center=50.0, window_width=400.0)

    assert np.min(windowed) >= 0.0
    assert np.max(windowed) <= 1.0


def test_ct_processor_window_presets():
    processor = CTScanProcessor(config={})

    assert processor.get_window_preset("lung") == (-600.0, 1500.0)
    assert processor.get_window_preset("bone") == (400.0, 1800.0)
    assert processor.get_window_preset("brain") == (40.0, 80.0)
    assert processor.get_window_preset("soft_tissue") == (50.0, 400.0)
    assert processor.get_window_preset("default") == (40.0, 400.0)


def test_ct_processor_stack_series_shape():
    processor = CTScanProcessor(config={})
    slices = [
        (2, np.zeros((3, 4), dtype=np.float32)),
        (1, np.ones((3, 4), dtype=np.float32)),
    ]

    volume = processor.stack_series(slices)

    assert volume.shape == (2, 3, 4)
    assert np.allclose(volume[0], 1.0)


@patch.object(dicom_handler, "dcmread", side_effect=Exception("bad bytes"))
def test_dicom_workflow_validate_invalid_file_returns_false(_mock_dcmread):
    manager = DICOMWorkflowManager(config={})

    valid, reason = manager.validate_dicom_file(b"not-a-dicom")

    assert valid is False
    assert reason == "Unable to parse DICOM file"
