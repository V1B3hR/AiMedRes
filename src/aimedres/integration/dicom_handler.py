"""
DICOM CT Scan Integration Module for AiMedRes

Provides full DICOM imaging workflow:
- Multi-file DICOM series upload and parsing
- CT scan pixel data extraction and volumetric stacking
- Hounsfield Unit (HU) conversion and windowing presets (lung, bone, brain, soft tissue)
- PHI/PII anonymization of DICOM metadata
- FHIR ImagingStudy resource generation
- Preprocessing pipeline ready for AI model inference
"""

import base64
import io
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pydicom import Dataset, dcmread
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


@dataclass
class DICOMMetadata:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    study_date: str
    study_description: str
    series_description: str
    rows: int
    columns: int
    slice_thickness: float
    pixel_spacing: List[float]
    instance_number: int
    window_center: float
    window_width: float
    manufacturer: str
    body_part_examined: str


class DICOMAnonymizer:
    """Anonymizes DICOM datasets and extracts safe metadata."""

    PHI_TAGS = [
        "PatientName",
        "PatientBirthDate",
        "PatientAddress",
        "PatientPhone",
        "InstitutionName",
        "ReferringPhysicianName",
        "OperatorsName",
    ]

    def anonymize_dataset(self, ds: Dataset) -> Dataset:
        """Remove or replace PHI tags while preserving imaging utility."""
        for tag in self.PHI_TAGS:
            if hasattr(ds, tag):
                setattr(ds, tag, "")
                logger.info("DICOM anonymization: cleared %s", tag)

        anonymized_patient_id = str(uuid.uuid4())
        ds.PatientID = anonymized_patient_id
        logger.info("DICOM anonymization: replaced PatientID with generated UUID")

        return ds

    def extract_safe_metadata(self, ds: Dataset) -> DICOMMetadata:
        """Extract safe metadata fields from a (preferably anonymized) dataset."""

        def _as_float(value: Any, default: float = 0.0) -> float:
            try:
                if isinstance(value, (list, tuple)):
                    value = value[0]
                return float(value)
            except (TypeError, ValueError, IndexError):
                return default

        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        if not isinstance(pixel_spacing, (list, tuple)):
            pixel_spacing = [1.0, 1.0]

        return DICOMMetadata(
            patient_id=str(getattr(ds, "PatientID", "")),
            study_instance_uid=str(getattr(ds, "StudyInstanceUID", "")),
            series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "")),
            sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
            modality=str(getattr(ds, "Modality", "")),
            study_date=str(getattr(ds, "StudyDate", "")),
            study_description=str(getattr(ds, "StudyDescription", "")),
            series_description=str(getattr(ds, "SeriesDescription", "")),
            rows=int(getattr(ds, "Rows", 0) or 0),
            columns=int(getattr(ds, "Columns", 0) or 0),
            slice_thickness=_as_float(getattr(ds, "SliceThickness", 0.0)),
            pixel_spacing=[_as_float(v, 1.0) for v in list(pixel_spacing)[:2]],
            instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
            window_center=_as_float(getattr(ds, "WindowCenter", 40.0), 40.0),
            window_width=_as_float(getattr(ds, "WindowWidth", 400.0), 400.0),
            manufacturer=str(getattr(ds, "Manufacturer", "")),
            body_part_examined=str(getattr(ds, "BodyPartExamined", "")),
        )


class CTScanProcessor:
    """Processing utilities for CT DICOM series."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_dicom_file(self, file_bytes: bytes) -> Dataset:
        """Load a DICOM file from bytes."""
        return dcmread(io.BytesIO(file_bytes))

    def extract_pixel_array(self, ds: Dataset) -> np.ndarray:
        """Extract HU-scaled pixel array as float32."""
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        return (pixel_array * slope + intercept).astype(np.float32)

    def apply_windowing(
        self, pixel_array: np.ndarray, window_center: float, window_width: float
    ) -> np.ndarray:
        """Apply CT windowing and normalize to [0, 1]."""
        width = max(float(window_width), 1.0)
        center = float(window_center)
        lower = center - width / 2.0
        upper = center + width / 2.0
        windowed = np.clip(pixel_array, lower, upper)
        normalized = (windowed - lower) / max((upper - lower), 1e-6)
        return normalized.astype(np.float32)

    def get_window_preset(self, preset: str) -> Tuple[float, float]:
        """Get standard CT window presets as (center, width)."""
        presets = {
            "lung": (-600.0, 1500.0),
            "bone": (400.0, 1800.0),
            "brain": (40.0, 80.0),
            "soft_tissue": (50.0, 400.0),
            "default": (40.0, 400.0),
        }
        return presets.get(preset, presets["default"])

    def stack_series(self, slices: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Sort slices by instance number and stack into a 3D volume."""
        if not slices:
            return np.empty((0, 0, 0), dtype=np.float32)
        sorted_slices = sorted(slices, key=lambda item: item[0])
        arrays = [arr.astype(np.float32) for _, arr in sorted_slices]
        return np.stack(arrays, axis=0).astype(np.float32)

    def preprocess_for_inference(
        self, volume: np.ndarray, target_shape: Tuple[int, int, int] = (64, 224, 224)
    ) -> np.ndarray:
        """Resize depth/spatial dimensions and normalize to zero mean, unit std."""
        if volume.size == 0:
            return np.empty(target_shape, dtype=np.float32)

        target_depth, target_h, target_w = target_shape
        depth_indices = np.linspace(0, max(volume.shape[0] - 1, 0), target_depth).astype(int)

        resized_slices = []
        for idx in depth_indices:
            slice_arr = volume[idx].astype(np.float32)
            min_val = float(np.min(slice_arr))
            max_val = float(np.max(slice_arr))
            if max_val > min_val:
                slice_norm = (slice_arr - min_val) / (max_val - min_val)
            else:
                slice_norm = np.zeros_like(slice_arr, dtype=np.float32)

            image = Image.fromarray((slice_norm * 255).astype(np.uint8), mode="L")
            resized = image.resize((target_w, target_h), Image.BILINEAR)
            resized_slices.append(np.array(resized, dtype=np.float32) / 255.0)

        processed = np.stack(resized_slices, axis=0).astype(np.float32)
        mean = float(np.mean(processed))
        std = float(np.std(processed))
        processed = (processed - mean) / max(std, 1e-6)
        return processed.astype(np.float32)

    def volume_to_base64_slices(self, volume: np.ndarray, max_slices: int = 10) -> List[str]:
        """Sample slices from volume and return base64-encoded PNG previews."""
        if volume.size == 0:
            return []

        count = min(max_slices, volume.shape[0])
        slice_indices = np.linspace(0, volume.shape[0] - 1, count).astype(int)
        previews: List[str] = []

        for idx in slice_indices:
            slice_arr = volume[idx].astype(np.float32)
            min_val = float(np.min(slice_arr))
            max_val = float(np.max(slice_arr))
            if max_val > min_val:
                slice_norm = (slice_arr - min_val) / (max_val - min_val)
            else:
                slice_norm = np.zeros_like(slice_arr, dtype=np.float32)

            image = Image.fromarray((slice_norm * 255).astype(np.uint8), mode="L")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            previews.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return previews


class DICOMWorkflowManager:
    """End-to-end DICOM CT workflow manager."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anonymizer = DICOMAnonymizer()
        self.processor = CTScanProcessor(config)

    def process_dicom_upload(self, files: List[bytes]) -> Dict[str, Any]:
        """Process uploaded DICOM series into model-ready representation."""
        metadata_list: List[DICOMMetadata] = []
        slices: List[Tuple[int, np.ndarray]] = []

        study_uid = ""
        series_uid = ""
        modality = ""

        for file_bytes in files:
            ds = self.processor.load_dicom_file(file_bytes)
            anonymized_ds = self.anonymizer.anonymize_dataset(ds)
            metadata = self.anonymizer.extract_safe_metadata(anonymized_ds)
            pixel_array = self.processor.extract_pixel_array(anonymized_ds)

            metadata_list.append(metadata)
            slices.append((metadata.instance_number, pixel_array))

            if not study_uid:
                study_uid = metadata.study_instance_uid
            if not series_uid:
                series_uid = metadata.series_instance_uid
            if not modality:
                modality = metadata.modality

        volume = self.processor.stack_series(slices)
        preprocessed_volume = self.processor.preprocess_for_inference(volume)
        preview_slices = self.processor.volume_to_base64_slices(volume)

        return {
            "study_uid": study_uid or str(uuid.uuid4()),
            "series_uid": series_uid or str(uuid.uuid4()),
            "modality": modality,
            "metadata": [asdict(meta) for meta in metadata_list],
            "slice_count": len(metadata_list),
            "volume_shape": list(volume.shape),
            "preprocessed_volume": preprocessed_volume,
            "preview_slices": preview_slices,
            "processing_status": "success",
        }

    def generate_fhir_imaging_study(self, metadata: DICOMMetadata, patient_id: str) -> Dict[str, Any]:
        """Generate FHIR R4 ImagingStudy resource for CT series."""
        started: Optional[str] = None
        if metadata.study_date:
            try:
                started = datetime.strptime(metadata.study_date, "%Y%m%d").date().isoformat()
            except ValueError:
                started = metadata.study_date

        return {
            "resourceType": "ImagingStudy",
            "id": str(uuid.uuid4()),
            "status": "available",
            "subject": {"reference": f"Patient/{patient_id}"},
            "started": started,
            "numberOfSeries": 1,
            "numberOfInstances": 1,
            "series": [
                {
                    "uid": metadata.series_instance_uid,
                    "number": 1,
                    "modality": {
                        "system": "http://dicom.nema.org/resources/ontology/DCM",
                        "code": metadata.modality or "CT",
                        "display": "Computed Tomography",
                    },
                    "description": metadata.series_description,
                    "instance": [
                        {
                            "uid": metadata.sop_instance_uid,
                            "sopClass": {
                                "system": "http://dicom.nema.org/resources/ontology/DCM",
                                "code": "CT",
                                "display": "CT Image Storage",
                            },
                            "number": metadata.instance_number,
                        }
                    ],
                }
            ],
        }

    def validate_dicom_file(self, file_bytes: bytes) -> Tuple[bool, str]:
        """Validate uploaded bytes as CT DICOM file."""
        try:
            ds = self.processor.load_dicom_file(file_bytes)
            modality = str(getattr(ds, "Modality", "")).upper()
            if modality != "CT":
                return False, f"Unsupported modality: {modality or 'unknown'}"
            return True, "valid"
        except InvalidDicomError:
            return False, "Invalid DICOM file"
        except Exception as exc:
            logger.error("DICOM validation error: %s", exc)
            return False, "Unable to parse DICOM file"
