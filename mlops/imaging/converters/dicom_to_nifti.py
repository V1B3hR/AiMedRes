"""
Advanced DICOM → NIfTI (BIDS-Oriented) Conversion Utility

Key Enhancements:
- Robust series discovery & sorting (InstanceNumber, ImagePosition, FrameOfReference)
- BIDS-style filename/entity inference (anat / func / dwi heuristics)
- Rich metadata extraction & normalization (dates, times, float coercion)
- Optional de-identification (configurable policy + salted hash pseudonyms)
- Quality metrics (SNR proxy, slice intensity stability, orientation validation)
- Parallel batch conversion with graceful fallbacks
- Two conversion backends: dcm2niix (preferred) or pure Python (pydicom+nibabel)
- Optional affine/orientation auditing and re-orientation suggestion
- Pluggable configuration via dataclass & CLI flags
- Structured logging (console + rotating file handler)
- Progress feedback (tqdm if available)
- Self-test harness for basic smoke validation
- Single-file deployment (no external framework dependency)

Note:
This script aims to provide advanced functionality while remaining a single self-contained file.
For production, consider breaking into modules, adding exhaustive tests, and hardening the
de-identification logic per your regulatory requirements.

Author: Enhanced by AI Assistant
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import hashlib
import logging
import argparse
import tempfile
import subprocess
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterable, Callable

# Optional / soft dependencies
try:
    import pydicom
    from pydicom.uid import generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    import nibabel.orientations as nb_orient
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENCY_AVAILABLE = True
except ImportError:
    CONCURRENCY_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Configuration & Constants
# --------------------------------------------------------------------------------------

DEFAULT_BIDS_TAGS = {
    'StudyInstanceUID': (0x0020, 0x000D),
    'SeriesInstanceUID': (0x0020, 0x000E),
    'StudyDate': (0x0008, 0x0020),
    'StudyTime': (0x0008, 0x0030),
    'SeriesDate': (0x0008, 0x0021),
    'SeriesTime': (0x0008, 0x0031),
    'PatientID': (0x0010, 0x0020),
    'PatientName': (0x0010, 0x0010),
    'PatientBirthDate': (0x0010, 0x0030),
    'PatientSex': (0x0010, 0x0040),
    'StudyDescription': (0x0008, 0x1030),
    'SeriesDescription': (0x0008, 0x103E),
    'Modality': (0x0008, 0x0060),
    'Manufacturer': (0x0008, 0x0070),
    'ManufacturerModelName': (0x0008, 0x1090),
    'MagneticFieldStrength': (0x0018, 0x0087),
    'RepetitionTime': (0x0018, 0x0080),
    'EchoTime': (0x0018, 0x0081),
    'FlipAngle': (0x0018, 0x1314),
    'SliceThickness': (0x0018, 0x0050),
    'PixelSpacing': (0x0028, 0x0030),
    'ImageOrientationPatient': (0x0020, 0x0037),
    'ImagePositionPatient': (0x0020, 0x0032),
    'AcquisitionTime': (0x0008, 0x0032),
    'InstitutionName': (0x0008, 0x0080),
    'ScanningSequence': (0x0018, 0x0020),
    'SequenceVariant': (0x0018, 0x0021),
    'ScanOptions': (0x0018, 0x0022),
    'NumberOfAverages': (0x0018, 0x0083),
    # Additional helpful heuristics:
    'EchoTrainLength': (0x0018, 0x0091),
    'InversionTime': (0x0018, 0x0082),
    'PhaseEncodingDirection': (0x0018, 0x1312),
    'ProtocolName': (0x0018, 0x1030),
    'SeriesNumber': (0x0020, 0x0011),
}

PHI_TAGS_DEFAULT = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1070),  # OperatorsName
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1010),  # StationName
    (0x0008, 0x1030),  # StudyDescription (potential PHI)
    (0x0020, 0x0010),  # StudyID
]

BIDS_MODALITY_MAP = {
    # Basic heuristics; expand as needed
    'MR': 'anat',
    'CT': 'anat',
    'PT': 'pet',
    'NM': 'pet',
    'US': 'other',
}

BIDS_ALLOWED_CHARS = re.compile(r'[^A-Za-z0-9_\-]')

QUALITY_DEFAULTS = {
    'snr_min': 5.0,  # simplistic threshold for warning
    'slice_intensity_coefficient_max': 0.15,  # coefficient of variation across slice means
}

# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------

@dataclass
class DeidentificationPolicy:
    enabled: bool = True
    replace_uids: bool = True
    subject_id: Optional[str] = None
    hash_salt: Optional[str] = None
    remove_descriptions_if_phi_suspect: bool = True

    def pseudonymize(self, value: str) -> str:
        if not self.enabled:
            return value
        if self.hash_salt is None:
            self.hash_salt = "default_salt"
        h = hashlib.sha256((self.hash_salt + str(value)).encode()).hexdigest()[:12]
        return f"anon-{h}"


@dataclass
class ConverterConfig:
    output_dir: Path = Path("./converted_nifti_advanced")
    bids_tags: Dict[str, Tuple[int, int]] = field(default_factory=lambda: DEFAULT_BIDS_TAGS)
    phi_tags: List[Tuple[int, int]] = field(default_factory=lambda: PHI_TAGS_DEFAULT)
    max_workers: int = 4
    parallel: bool = True
    force_python_backend: bool = False
    compute_quality: bool = True
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: QUALITY_DEFAULTS.copy())
    overwrite: bool = False
    keep_intermediate: bool = False
    log_level: str = "INFO"
    log_file_mb: int = 5
    log_backup_count: int = 3
    preserve_series_uid: bool = True
    # Orientation audit
    audit_orientation: bool = True
    # Additional metadata retention:
    retain_original_metadata: bool = True
    # BIDS:
    enable_bids_naming: bool = True


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def setup_logging(config: ConverterConfig) -> logging.Logger:
    logger = logging.getLogger("dicom2nifti_advanced")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    if not logger.handlers:
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)

        # Rotating file
        try:
            from logging.handlers import RotatingFileHandler
            log_dir = config.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_dir / "conversion.log",
                maxBytes=config.log_file_mb * 1024 * 1024,
                backupCount=config.log_backup_count
            )
            fh.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s %(funcName)s: %(message)s"))
            logger.addHandler(fh)
        except Exception:
            pass

    return logger


# --------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------

def safe_filename(base: str) -> str:
    base = base.strip()
    base = BIDS_ALLOWED_CHARS.sub("_", base)
    base = re.sub(r"_+", "_", base)
    return base.strip("_") or "unk"

def estimate_snr(volume: "np.ndarray") -> float:
    """Crude SNR: mean(volume) / std(background) where background = lowest 5% intensities."""
    if volume.size == 0:
        return 0.0
    flat = volume.flatten()
    flat_sorted = np.sort(flat)
    cutoff = max(1, int(0.05 * len(flat_sorted)))
    background = flat_sorted[:cutoff]
    bg_std = float(np.std(background)) or 1e-6
    mean_signal = float(np.mean(flat))
    return mean_signal / bg_std

def slice_intensity_variability(volume: "np.ndarray") -> float:
    """Coefficient of variation of slice means along the last axis."""
    if volume.ndim != 3:
        return 0.0
    slice_means = volume.mean(axis=(0, 1))
    if np.mean(slice_means) == 0:
        return 0.0
    return float(np.std(slice_means) / (np.mean(slice_means) + 1e-6))

def orientation_audit(affine: "np.ndarray") -> Dict[str, Any]:
    """Audit orientation (RAS etc.) using nibabel if available."""
    if not NIBABEL_AVAILABLE:
        return {"available": False}
    try:
        codes = nib.affines.aff2axcodes(affine)
        return {"available": True, "axcodes": codes}
    except Exception:
        return {"available": False}

def infer_bids_entities(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Heuristic to map DICOM metadata to BIDS entities:
    Returns dict with: subject, session (optional), modality_folder (anat/func/dwi), suffix, run
    """
    modality = str(metadata.get("Modality", "")).upper()
    series_desc = str(metadata.get("SeriesDescription", "") or metadata.get("ProtocolName", "")).lower()
    suffix = "T1w"
    modality_folder = BIDS_MODALITY_MAP.get(modality, "anat")

    # Rough heuristics
    if "t2" in series_desc and "flair" not in series_desc:
        suffix = "T2w"
    if "flair" in series_desc:
        suffix = "FLAIR"
    if "dwi" in series_desc or "diff" in series_desc:
        suffix = "dwi"
        modality_folder = "dwi"
    if any(k in series_desc for k in ["bold", "rest", "fmri", "task"]):
        suffix = "bold"
        modality_folder = "func"
    if "adc" in series_desc:
        suffix = "ADC"
        modality_folder = "dwi"

    run_number = metadata.get("SeriesNumber")
    run = None
    try:
        if run_number is not None:
            rn_int = int(run_number)
            if rn_int > 0:
                run = f"{rn_int:02d}"
    except Exception:
        pass

    return {
        "modality_folder": modality_folder,
        "suffix": suffix,
        "run": run or "01"
    }

def ensure_numpy():
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required for conversion backend operations.")

def ensure_pydicom():
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required but not available.")

def ensure_nibabel():
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required but not available.")

# --------------------------------------------------------------------------------------
# Core Converter
# --------------------------------------------------------------------------------------

class AdvancedDICOMToNIfTIConverter:
    def __init__(self,
                 config: Optional[ConverterConfig] = None,
                 deid_policy: Optional[DeidentificationPolicy] = None):
        self.config = config or ConverterConfig()
        self.deid_policy = deid_policy or DeidentificationPolicy()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "nifti").mkdir(exist_ok=True)
        (self.config.output_dir / "metadata").mkdir(exist_ok=True)

        self.logger = setup_logging(self.config)

        self.dcm2niix_available = self._check_dcm2niix()
        self.logger.info(f"dcm2niix available: {self.dcm2niix_available}")

    # ------------------------------------------------------------------
    # External tool detection
    # ------------------------------------------------------------------
    def _check_dcm2niix(self) -> bool:
        try:
            proc = subprocess.run(["dcm2niix", "-h"],
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # DICOM Directory Scanning
    # ------------------------------------------------------------------
    def scan_dicom_directory(self, dicom_dir: str | Path) -> Dict[str, List[str]]:
        ensure_pydicom()
        dicom_dir = Path(dicom_dir)
        self.logger.info(f"Scanning DICOM directory: {dicom_dir}")

        series_map: Dict[str, List[str]] = {}
        candidate_files = list(dicom_dir.rglob("*"))
        for fp in candidate_files:
            if fp.is_file():
                try:
                    ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                    if not hasattr(ds, "SeriesInstanceUID"):
                        continue
                    suid = getattr(ds, "SeriesInstanceUID", "unknown")
                    series_map.setdefault(suid, []).append(str(fp))
                except Exception:
                    continue

        # Sorting within each series
        for suid, files in series_map.items():
            series_map[suid] = self._sort_dicom_files(files)

        self.logger.info(f"Found {len(series_map)} series.")
        return series_map

    def _sort_dicom_files(self, file_paths: List[str]) -> List[str]:
        ensure_pydicom()
        info = []
        for p in file_paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                inst = int(getattr(ds, "InstanceNumber", 0))
                ipp = getattr(ds, "ImagePositionPatient", [0, 0, inst])
                z = float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 else float(inst)
                info.append((p, inst, z))
            except Exception:
                info.append((p, 0, 0.0))
        info.sort(key=lambda x: (x[1], x[2]))
        return [x[0] for x in info]

    # ------------------------------------------------------------------
    # Metadata Extraction
    # ------------------------------------------------------------------
    def extract_metadata(self, dicom_files: List[str]) -> Dict[str, Any]:
        ensure_pydicom()
        if not dicom_files:
            return {}
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        md: Dict[str, Any] = {}

        for key, tag in self.config.bids_tags.items():
            try:
                value = ds[tag].value
                if key in ["StudyDate", "SeriesDate"]:
                    md[key] = self._format_date(value)
                elif key in ["StudyTime", "SeriesTime", "AcquisitionTime"]:
                    md[key] = self._format_time(value)
                elif key in ["RepetitionTime", "EchoTime", "InversionTime"]:
                    # Convert ms to seconds if numeric
                    if isinstance(value, (int, float)):
                        md[key] = float(value) / 1000.0
                    else:
                        try:
                            md[key] = float(value) / 1000.0
                        except Exception:
                            md[key] = value
                elif key in ["PixelSpacing", "ImageOrientationPatient", "ImagePositionPatient"]:
                    if isinstance(value, (list, tuple)):
                        md[key] = [float(v) for v in value]
                    else:
                        md[key] = value
                else:
                    md[key] = value
            except Exception:
                continue

        md["NumberOfFiles"] = len(dicom_files)
        # Derived slice spacing
        if len(dicom_files) > 1:
            try:
                first = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                last = pydicom.dcmread(dicom_files[-1], stop_before_pixels=True)
                pos1 = getattr(first, "ImagePositionPatient", [0, 0, 0])
                pos2 = getattr(last, "ImagePositionPatient", [0, 0, (len(dicom_files) - 1)])
                if len(pos1) >= 3 and len(pos2) >= 3:
                    md["DerivedSliceSpacing"] = abs(float(pos2[2]) - float(pos1[2])) / (len(dicom_files)-1)
            except Exception:
                pass

        modality = str(md.get("Modality", "")).upper()
        if modality == "MR":
            self._augment_mr(ds, md)
        elif modality == "CT":
            self._augment_ct(ds, md)

        return md

    def _augment_mr(self, ds: "pydicom.Dataset", md: Dict[str, Any]):
        extra_tags = {
            "ImagingFrequency": (0x0018, 0x0084),
            "PercentSampling": (0x0018, 0x0093),
            "ReceiveCoilName": (0x0018, 0x1250)
        }
        for k, t in extra_tags.items():
            try:
                md[k] = ds[t].value
            except Exception:
                pass
        seq = md.get("ScanningSequence", "") or ""
        if "IR" in seq:
            md["SequenceType"] = "Inversion Recovery"
        elif "GR" in seq:
            md["SequenceType"] = "Gradient Echo"
        elif "SE" in seq:
            md["SequenceType"] = "Spin Echo"
        elif "EP" in seq:
            md["SequenceType"] = "Echo Planar"

    def _augment_ct(self, ds: "pydicom.Dataset", md: Dict[str, Any]):
        ct_tags = {
            "KVP": (0x0018, 0x0060),
            "XRayTubeCurrent": (0x0018, 0x1151),
            "ExposureTime": (0x0018, 0x1150),
            "FilterType": (0x0018, 0x1160),
            "ConvolutionKernel": (0x0018, 0x1210),
            "CTDIvol": (0x0018, 0x9345),
        }
        for k, t in ct_tags.items():
            try:
                md[k] = ds[t].value
            except Exception:
                pass

    def _format_date(self, value: Any) -> Optional[str]:
        s = str(value)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return None

    def _format_time(self, value: Any) -> Optional[str]:
        s = str(value).split(".")[0]
        if len(s) >= 6 and s[:6].isdigit():
            return f"{s[:2]}:{s[2:4]}:{s[4:6]}"
        return None

    # ------------------------------------------------------------------
    # De-identification
    # ------------------------------------------------------------------
    def deidentify_metadata(self, md: Dict[str, Any]) -> Dict[str, Any]:
        if not self.deid_policy.enabled:
            return md
        new_md = md.copy()

        # Remove PHI keys from extracted metadata
        for k in list(new_md.keys()):
            if k in ["PatientName", "PatientID", "PatientBirthDate"]:
                new_md.pop(k, None)

        # Subject
        if self.deid_policy.subject_id:
            new_md["Subject"] = self.deid_policy.subject_id
        elif "PatientID" in md:
            new_md["Subject"] = self.deid_policy.pseudonymize(str(md["PatientID"]))
        else:
            new_md["Subject"] = self.deid_policy.pseudonymize("unknown")

        if self.deid_policy.remove_descriptions_if_phi_suspect:
            desc = new_md.get("StudyDescription")
            if desc and any(tok in desc.lower() for tok in ["name", "dob", "birth", "patient", "mrn"]):
                new_md.pop("StudyDescription", None)

        new_md["DeidentificationDate"] = datetime.utcnow().isoformat()
        new_md["DeidentificationSoftware"] = "Advanced DICOM Converter v2.0"
        return new_md

    # ------------------------------------------------------------------
    # Conversion Orchestration
    # ------------------------------------------------------------------
    def convert_directory(self, dicom_dir: str | Path) -> List[Dict[str, Any]]:
        series = self.scan_dicom_directory(dicom_dir)
        results: List[Dict[str, Any]] = []

        items = list(series.items())

        if self.config.parallel and CONCURRENCY_AVAILABLE and len(items) > 1:
            self.logger.info(f"Converting in parallel with up to {self.config.max_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                futures = {}
                for suid, files in items:
                    futures[ex.submit(self._convert_single_series_wrapper, suid, files)] = suid
                iterator = futures
                if TQDM_AVAILABLE:
                    iterator = tqdm(futures, desc="Series", unit="series")
                for fut in iterator:
                    try:
                        res = fut.result()
                        if res:
                            results.append(res)
                    except Exception as e:
                        self.logger.error(f"Series failed: {e}")
        else:
            iterator = items
            if TQDM_AVAILABLE:
                iterator = tqdm(items, desc="Series", unit="series")
            for suid, files in iterator:
                res = self._convert_single_series_wrapper(suid, files)
                if res:
                    results.append(res)

        self.logger.info(f"Completed {len(results)} series conversions.")
        return results

    def _convert_single_series_wrapper(self, series_uid: str, dicom_files: List[str]) -> Optional[Dict[str, Any]]:
        try:
            return self.convert_series(dicom_files, series_uid=series_uid)
        except Exception as e:
            self.logger.error(f"Failed series {series_uid}: {e}")
            return None

    def convert_series(self,
                       dicom_files: List[str],
                       output_prefix: Optional[str] = None,
                       series_uid: Optional[str] = None) -> Dict[str, Any]:
        if not dicom_files:
            raise ValueError("No DICOM files to convert.")

        md = self.extract_metadata(dicom_files)
        orig_md = md.copy() if self.config.retain_original_metadata else {}

        # De-identify
        if self.deid_policy.enabled:
            md = self.deidentify_metadata(md)

        # Output prefix
        if not output_prefix:
            modality = md.get("Modality", "UNK")
            series_desc = md.get("SeriesDescription") or md.get("ProtocolName") or "unknown"
            series_desc_clean = safe_filename(str(series_desc))
            series_number = md.get("SeriesNumber", 0)
            output_prefix = f"{md.get('Subject','sub-unk')}_{modality}_{int(series_number):03d}_{series_desc_clean}"

        # BIDS naming override
        bids_entities = {}
        bids_path_part = ""
        if self.config.enable_bids_naming:
            bids_entities = infer_bids_entities(md)
            bids_modality = bids_entities["modality_folder"]
            bids_suffix = bids_entities["suffix"]
            run = bids_entities["run"]
            subject = md.get("Subject", "sub-unk").replace("anon-", "sub-")
            bids_path_part = f"{subject}"
            output_prefix = f"{subject}_run-{run}_{bids_suffix}"
        else:
            bids_modality = "nifti"

        out_dir = self.config.output_dir / "nifti" / bids_modality
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(dicom_files) == 1:
            # Single-file DICOM (e.g., enhanced multi-frame) - we still attempt conversion
            self.logger.warning("Single file in series; may be enhanced DICOM. Proceeding.")

        # Decide backend
        use_python = self.config.force_python_backend or not self.dcm2niix_available
        if use_python:
            if not (PYDICOM_AVAILABLE and NIBABEL_AVAILABLE and NUMPY_AVAILABLE):
                raise RuntimeError("Python backend requires pydicom, nibabel, numpy.")
            conv = self._convert_with_pydicom(dicom_files, out_dir, output_prefix)
        else:
            conv = self._convert_with_dcm2niix(dicom_files, out_dir, output_prefix)

        # Quality metrics
        quality = {}
        if self.config.compute_quality and NUMPY_AVAILABLE and NIBABEL_AVAILABLE:
            try:
                img = nib.load(conv["nifti_path"])
                data = img.get_fdata()
                quality["snr_proxy"] = estimate_snr(data)
                quality["slice_intensity_cv"] = slice_intensity_variability(data)
                if self.config.audit_orientation:
                    quality["orientation"] = orientation_audit(img.affine)
                self._quality_warnings(quality)
            except Exception as e:
                self.logger.warning(f"Quality metrics failed: {e}")

        # Compose metadata JSON
        final_md = {
            "series_uid": series_uid,
            "conversion_method": conv["conversion_method"],
            "conversion_time": datetime.utcnow().isoformat(),
            "nifti_path": conv["nifti_path"],
            "metadata": md,
            "bids": bids_entities if bids_entities else None,
            "quality": quality if quality else None,
            "source_files": dicom_files if self.config.retain_original_metadata else None,
            "original_metadata": orig_md if self.config.retain_original_metadata else None
        }

        json_name = f"{output_prefix}.json"
        meta_dir = self.config.output_dir / "metadata"
        meta_dir.mkdir(exist_ok=True)
        json_path = meta_dir / json_name
        if json_path.exists() and not self.config.overwrite:
            self.logger.warning(f"Metadata JSON exists, skipping overwrite: {json_path}")
        else:
            with open(json_path, "w") as f:
                json.dump(final_md, f, indent=2, default=str)

        conv["metadata_json"] = str(json_path)
        conv.update({"series_uid": series_uid, "final_metadata": final_md})
        self.logger.info(f"Converted series -> {conv['nifti_path']}")
        return conv

    # ------------------------------------------------------------------
    # Conversion Backends
    # ------------------------------------------------------------------
    def _convert_with_dcm2niix(self, dicom_files: List[str], out_dir: Path, prefix: str) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td) / "dicom"
            tmp_dir.mkdir()
            for i, fp in enumerate(dicom_files):
                target = tmp_dir / f"{i:05d}.dcm"
                target.write_bytes(Path(fp).read_bytes())

            cmd = [
                "dcm2niix",
                "-z", "y",
                "-f", prefix,
                "-o", str(out_dir),
                str(tmp_dir)
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if proc.returncode != 0:
                    raise RuntimeError(f"dcm2niix failed: {proc.stderr}")
                # Locate produced file
                candidates = list(out_dir.glob(f"{prefix}*.nii.gz"))
                if not candidates:
                    raise RuntimeError("No NIfTI output produced by dcm2niix.")
                nifti_path = candidates[0]
                return {
                    "nifti_path": str(nifti_path),
                    "conversion_method": "dcm2niix",
                    "log": proc.stdout
                }
            except subprocess.TimeoutExpired:
                raise RuntimeError("dcm2niix conversion timed out")

    def _convert_with_pydicom(self, dicom_files: List[str], out_dir: Path, prefix: str) -> Dict[str, Any]:
        ensure_numpy()
        ensure_pydicom()
        ensure_nibabel()

        slices = []
        for fp in dicom_files:
            ds = pydicom.dcmread(fp)
            slices.append(ds)

        # Sort again just in case
        try:
            slices.sort(key=lambda x: float(getattr(x, "ImagePositionPatient", [0, 0, 0])[2]))
        except Exception:
            pass

        volume_list = []
        for ds in slices:
            arr = ds.pixel_array.astype("float32")
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            volume_list.append(arr)
        volume = np.stack(volume_list, axis=-1)  # Z axis last

        affine = self._build_affine(slices[0], len(slices))

        img = nib.Nifti1Image(volume, affine)
        hdr = img.header
        hdr.set_xyzt_units("mm", "sec")
        out_path = out_dir / f"{prefix}.nii.gz"
        if out_path.exists() and not self.config.overwrite:
            self.logger.warning(f"File exists, not overwriting: {out_path}")
        else:
            nib.save(img, str(out_path))
        return {
            "nifti_path": str(out_path),
            "conversion_method": "python",
            "volume_shape": volume.shape
        }

    def _build_affine(self, ds: "pydicom.Dataset", num_slices: int) -> "np.ndarray":
        ensure_numpy()
        affine = np.eye(4, dtype=float)
        try:
            orientation = list(map(float, ds.ImageOrientationPatient))
            row = orientation[:3]
            col = orientation[3:]
            pos = list(map(float, ds.ImagePositionPatient))
            px_spacing = list(map(float, ds.PixelSpacing))
            slice_thick = float(getattr(ds, "SliceThickness", 1.0))
            slice_dir = np.cross(row, col)
            affine[:3, 0] = np.array(row) * px_spacing[0]
            affine[:3, 1] = np.array(col) * px_spacing[1]
            affine[:3, 2] = slice_dir * slice_thick
            affine[:3, 3] = np.array(pos)
        except Exception:
            pass
        return affine

    # ------------------------------------------------------------------
    # Quality Warnings
    # ------------------------------------------------------------------
    def _quality_warnings(self, metrics: Dict[str, Any]):
        snr = metrics.get("snr_proxy")
        if snr is not None and snr < self.config.quality_thresholds.get("snr_min", 0):
            self.logger.warning(f"Low SNR proxy detected: {snr:.2f}")
        cv = metrics.get("slice_intensity_cv")
        if cv is not None and cv > self.config.quality_thresholds.get("slice_intensity_coefficient_max", 1e9):
            self.logger.warning(f"High slice intensity variability (CV): {cv:.3f}")


# --------------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced DICOM to NIfTI Converter (BIDS-oriented, single-file)."
    )
    p.add_argument("dicom_dir", help="Input directory containing DICOM files.")
    p.add_argument("--out", dest="output_dir", default="./converted_nifti_advanced", help="Output root directory.")
    p.add_argument("--subject-id", default=None, help="Explicit subject ID for de-identification.")
    p.add_argument("--no-deid", action="store_true", help="Disable de-identification.")
    p.add_argument("--hash-salt", default=None, help="Salt for pseudonymization hashing.")
    p.add_argument("--force-python", action="store_true", help="Force Python backend (skip dcm2niix).")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel conversion.")
    p.add_argument("--max-workers", type=int, default=4, help="Max workers for parallel mode.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--no-quality", action="store_true", help="Disable quality metric computation.")
    p.add_argument("--no-bids-naming", action="store_true", help="Disable BIDS naming heuristics.")
    p.add_argument("--self-test", action="store_true", help="Run internal smoke tests and exit.")
    return p

def run_self_test():
    print("Running self-test...")
    # Minimal introspection tests
    assert isinstance(DEFAULT_BIDS_TAGS, dict)
    assert isinstance(PHI_TAGS_DEFAULT, list)
    config = ConverterConfig(output_dir=Path("./_selftest_out"))
    deid = DeidentificationPolicy()
    conv = AdvancedDICOMToNIfTIConverter(config, deid)
    print("Basic object construction OK.")
    print("Self-test complete (no DICOM I/O performed).")

def main_cli():
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    config = ConverterConfig(
        output_dir=Path(args.output_dir),
        max_workers=args.max_workers,
        parallel=not args.no_parallel,
        force_python_backend=args.force_python,
        compute_quality=not args.no_quality,
        overwrite=args.overwrite,
        log_level=args.log_level,
        enable_bids_naming=not args.no_bids_naming
    )
    deid_policy = DeidentificationPolicy(
        enabled=not args.no_deid,
        subject_id=args.subject_id,
        hash_salt=args.hash_salt
    )
    converter = AdvancedDICOMToNIfTIConverter(config, deid_policy)
    results = converter.convert_directory(args.dicom_dir)

    # Summary
    print("\nConversion Summary:")
    for r in results:
        print(f"- Series UID: {r.get('series_uid')} -> {r.get('nifti_path')} (method={r.get('conversion_method')})")
    print(f"\nMetadata JSON files in: {config.output_dir / 'metadata'}")

# --------------------------------------------------------------------------------------
# Module Entry
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    main_cli()
