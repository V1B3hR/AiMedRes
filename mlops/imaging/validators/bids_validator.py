"""
Advanced BIDS Compliance Validator

This module provides an extensible, performant, and developer-friendly validator
for neuroimaging datasets following the BIDS (Brain Imaging Data Structure) specification.

Key Enhancements Over Previous Version:
- Structured results using dataclasses
- Distinct severity levels (ERROR, WARNING, INFO)
- Concurrency for file validation (configurable)
- Robust filename entity parsing & pattern validation
- Plugin architecture for extensible custom validations
- Duplicate file & orphan JSON detection
- Improved JSON sidecar pairing logic
- Cross-validation of participants.tsv vs filesystem subjects
- Rich reporting formats: text (default), JSON, Markdown, and (optionally) HTML
- Optional strict mode (treat warnings as errors)
- Configurable ignore directories (e.g., derivatives/, code/, sourcedata/)
- Caching & compiled regex patterns for speed
- Graceful optional dependency integration (nibabel, pydantic if available)
- CLI entry point for direct command-line usage
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Iterable, Set, Callable

# Optional imports (safe fallback if not installed)
try:
    import nibabel as nib  # type: ignore
except ImportError:  # pragma: no cover
    nib = None

try:
    from pydantic import BaseModel  # type: ignore
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore


# ---------------------------------------------------------------------------
# Enumerations & Dataclasses
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    path: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class FileValidationResult:
    file_path: str
    subject: Optional[str] = None
    session: Optional[str] = None
    datatype: Optional[str] = None
    suffix: Optional[str] = None
    modality: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    def is_valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "subject": self.subject,
            "session": self.session,
            "datatype": self.datatype,
            "suffix": self.suffix,
            "modality": self.modality,
            "valid": self.is_valid(),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class DatasetValidationSummary:
    total_files: int
    subjects: List[str]
    sessions: List[str]
    modalities: List[str]
    issues_count: Dict[str, int]
    valid: bool
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetValidationResult:
    dataset_path: str
    issues: List[ValidationIssue]
    files: List[FileValidationResult]
    summary: DatasetValidationSummary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "issues": [i.to_dict() for i in self.issues],
            "files": [f.to_dict() for f in self.files],
            "summary": self.summary.to_dict(),
        }

    def is_valid(self) -> bool:
        return self.summary.valid


# ---------------------------------------------------------------------------
# Plugin Architecture
# ---------------------------------------------------------------------------

class ValidationPlugin:
    """Base class for dataset-level validation plugins."""
    name: str = "base"

    def apply(self, dataset_path: Path, result: DatasetValidationResult) -> List[ValidationIssue]:
        return []


class DuplicateFilePlugin(ValidationPlugin):
    """Detects duplicate logical acquisitions (same entities)."""

    name = "duplicate-file"

    def apply(self, dataset_path: Path, result: DatasetValidationResult) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        seen: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)

        for f in result.files:
            key = (
                f.subject or "",
                f.session or "",
                f.datatype or "",
                f.suffix or "",
            )
            seen[key].append(f.file_path)

        for key, paths in seen.items():
            if len(paths) > 1:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="duplicate_files",
                        message=f"Multiple files share the same entity combination {key}: {paths}",
                        path=None,
                        context={"paths": paths},
                    )
                )
        return issues


class OrphanJSONPlugin(ValidationPlugin):
    """JSON sidecars without corresponding NIfTI."""

    name = "orphan-json"

    def apply(self, dataset_path: Path, result: DatasetValidationResult) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        nifti_stems = {Path(f.file_path).with_suffix("").with_suffix("").name for f in result.files}

        for json_file in dataset_path.rglob("*.json"):
            # Skip dataset description, participants, etc.
            if json_file.name in ("dataset_description.json",) or json_file.parent == dataset_path:
                continue
            stem = json_file.with_suffix("").with_suffix("").name
            if stem not in nifti_stems:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="orphan_json",
                        message="JSON sidecar has no matching NIfTI file",
                        path=str(json_file.relative_to(dataset_path)),
                        context={"stem": stem},
                    )
                )
        return issues


# ---------------------------------------------------------------------------
# Core Validator
# ---------------------------------------------------------------------------

class BIDSComplianceValidator:
    """
    Advanced BIDS compliance validator with plugin and concurrency support.
    """

    DEFAULT_IGNORE_DIRS = {"derivatives", "code", "sourcedata", ".git", ".datalad"}

    def __init__(
        self,
        strict: bool = False,
        max_workers: int = 0,
        ignore_dirs: Optional[Iterable[str]] = None,
        enable_plugins: Optional[Iterable[ValidationPlugin]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            strict: If True, treat warnings as errors for overall validity.
            max_workers: Parallel workers for file validation (0 => automatic).
            ignore_dirs: Directory names to ignore recursively.
            enable_plugins: List of plugin instances to use.
            logger: Optional external logger.
        """
        self.strict = strict
        self.max_workers = max_workers if max_workers >= 0 else 0
        self.ignore_dirs = set(ignore_dirs or self.DEFAULT_IGNORE_DIRS)
        self.logger = logger or logging.getLogger(__name__)

        # Compile regex patterns once
        self._compile_patterns()

        # File naming patterns per datatype/suffix
        self.filename_patterns: Dict[str, Dict[str, re.Pattern]] = {
            "anat": {
                "T1w": re.compile(r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_ce-[A-Za-z0-9]+)?(_rec-[A-Za-z0-9]+)?(_run-[0-9]+)?_T1w\.nii(\.gz)?$"),
                "T2w": re.compile(r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_ce-[A-Za-z0-9]+)?(_rec-[A-Za-z0-9]+)?(_run-[0-9]+)?_T2w\.nii(\.gz)?$"),
                "FLAIR": re.compile(r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_ce-[A-Za-z0-9]+)?(_rec-[A-Za-z0-9]+)?(_run-[0-9]+)?_FLAIR\.nii(\.gz)?$"),
                "PD": re.compile(r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_ce-[A-Za-z0-9]+)?(_rec-[A-Za-z0-9]+)?(_run-[0-9]+)?_PD\.nii(\.gz)?$"),
            },
            "func": {
                "bold": re.compile(
                    r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?_task-[A-Za-z0-9]+"
                    r"(_acq-[A-Za-z0-9]+)?(_ce-[A-Za-z0-9]+)?(_dir-[A-Za-z0-9]+)?(_rec-[A-Za-z0-9]+)?"
                    r"(_run-[0-9]+)?(_echo-[0-9]+)?(_part-[A-Za-z0-9]+)?_bold\.nii(\.gz)?$"
                )
            },
            "dwi": {
                "dwi": re.compile(
                    r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_dir-[A-Za-z0-9]+)?(_run-[0-9]+)?_dwi\.nii(\.gz)?$"
                )
            },
            "fmap": {
                "phasediff": re.compile(
                    r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_run-[0-9]+)?_phasediff\.nii(\.gz)?$"
                ),
                "magnitude": re.compile(
                    r"^sub-[A-Za-z0-9]+(_ses-[A-Za-z0-9]+)?(_acq-[A-Za-z0-9]+)?(_run-[0-9]+)?_magnitude[0-9]+\.nii(\.gz)?$"
                ),
            },
        }

        # Required metadata fields
        self.required_metadata: Dict[str, Dict[str, List[str]]] = {
            "anat": {
                "T1w": ["RepetitionTime", "EchoTime", "FlipAngle"],
                "T2w": ["RepetitionTime", "EchoTime", "FlipAngle"],
                "FLAIR": ["RepetitionTime", "EchoTime", "InversionTime"],
            },
            "func": {
                "bold": ["RepetitionTime", "EchoTime", "TaskName", "SliceTiming"],
            },
            "dwi": {
                "dwi": ["RepetitionTime", "EchoTime", "BValue", "BVector"],
            },
            "fmap": {
                "phasediff": ["EchoTime1", "EchoTime2"],
                "magnitude": ["EchoTime"],
            },
        }

        self.dataset_descriptor_required = ["Name", "BIDSVersion", "DatasetType"]

        # Entity extraction patterns
        self.entity_patterns: Dict[str, re.Pattern] = {
            "subject": re.compile(r"sub-([A-Za-z0-9]+)"),
            "session": re.compile(r"ses-([A-Za-z0-9]+)"),
            "task": re.compile(r"task-([A-Za-z0-9]+)"),
            "acquisition": re.compile(r"acq-([A-Za-z0-9]+)"),
            "contrast": re.compile(r"ce-([A-Za-z0-9]+)"),
            "reconstruction": re.compile(r"rec-([A-Za-z0-9]+)"),
            "direction": re.compile(r"dir-([A-Za-z0-9]+)"),
            "run": re.compile(r"run-([0-9]+)"),
            "echo": re.compile(r"echo-([0-9]+)"),
            "part": re.compile(r"part-([A-Za-z0-9]+)"),
        }

        # Plugins
        self.plugins: List[ValidationPlugin] = list(enable_plugins or [])
        if not any(isinstance(p, DuplicateFilePlugin) for p in self.plugins):
            self.plugins.append(DuplicateFilePlugin())
        if not any(isinstance(p, OrphanJSONPlugin) for p in self.plugins):
            self.plugins.append(OrphanJSONPlugin())

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def validate_dataset(self, dataset_path: str) -> DatasetValidationResult:
        dataset_root = Path(dataset_path).resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            summary = DatasetValidationSummary(
                total_files=0,
                subjects=[],
                sessions=[],
                modalities=[],
                issues_count={"ERROR": 1, "WARNING": 0, "INFO": 0},
                valid=False,
                generated_at=datetime.utcnow().isoformat(),
            )
            return DatasetValidationResult(
                dataset_path=str(dataset_root),
                issues=[
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="path_missing",
                        message=f"Dataset path does not exist or is not a directory: {dataset_root}",
                    )
                ],
                files=[],
                summary=summary,
            )

        self.logger.info(f"Starting BIDS validation for dataset: {dataset_root}")

        issues: List[ValidationIssue] = []
        file_results: List[FileValidationResult] = []

        # Structure & descriptor checks
        issues.extend(self._validate_dataset_structure(dataset_root))
        issues.extend(self._validate_dataset_description(dataset_root))
        issues.extend(self._validate_participants_file(dataset_root))

        # File discovery & validation
        nifti_files = list(self._discover_nifti_files(dataset_root))
        self.logger.debug(f"Discovered {len(nifti_files)} NIfTI files")

        file_results.extend(self._validate_files_concurrently(nifti_files, dataset_root))

        # Cross-file consistency & participants cross-check
        issues.extend(self._cross_validate(dataset_root, file_results))

        # Plugin execution
        dataset_result_stub = DatasetValidationResult(
            dataset_path=str(dataset_root),
            issues=[],
            files=file_results,
            summary=DatasetValidationSummary(
                total_files=len(file_results),
                subjects=[],
                sessions=[],
                modalities=[],
                issues_count={},
                valid=False,
                generated_at=datetime.utcnow().isoformat(),
            ),
        )

        for plugin in self.plugins:
            self.logger.debug(f"Running plugin: {plugin.name}")
            try:
                plugin_issues = plugin.apply(dataset_root, dataset_result_stub)
                issues.extend(plugin_issues)
            except Exception as e:  # pragma: no cover
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="plugin_failure",
                        message=f"Plugin {plugin.name} failed: {e}",
                        context={"plugin": plugin.name},
                    )
                )

        # Aggregate stats
        subjects = sorted({f.subject for f in file_results if f.subject})
        sessions = sorted({f.session for f in file_results if f.session})
        modalities = sorted({f.suffix for f in file_results if f.suffix})

        all_issues = issues + [iss for f in file_results for iss in f.issues]

        counter = Counter(i.severity.value for i in all_issues)
        valid = (counter.get("ERROR", 0) == 0) and (not self.strict or counter.get("WARNING", 0) == 0)

        summary = DatasetValidationSummary(
            total_files=len(file_results),
            subjects=subjects,
            sessions=sessions,
            modalities=modalities,
            issues_count=dict(counter),
            valid=valid,
            generated_at=datetime.utcnow().isoformat(),
        )

        result = DatasetValidationResult(
            dataset_path=str(dataset_root),
            issues=issues,
            files=file_results,
            summary=summary,
        )

        self.logger.info(
            f"Validation complete. Valid={result.summary.valid} "
            f"Errors={counter.get('ERROR',0)} Warnings={counter.get('WARNING',0)}"
        )
        return result

    def validate_file(self, file_path: str) -> FileValidationResult:
        file_p = Path(file_path).resolve()
        if not file_p.exists():
            return FileValidationResult(
                file_path=str(file_p),
                issues=[
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="file_missing",
                        message=f"File does not exist: {file_p}",
                    )
                ],
            )

        dataset_root = self._find_dataset_root(file_p)
        return self._validate_single_file(file_p, dataset_root)

    # ---------------------------------------------------------------------
    # Discovery & Concurrency
    # ---------------------------------------------------------------------

    def _discover_nifti_files(self, dataset_root: Path) -> Iterable[Path]:
        for path in dataset_root.rglob("*.nii"):
            if self._should_include(path, dataset_root):
                yield path
        for path in dataset_root.rglob("*.nii.gz"):
            if self._should_include(path, dataset_root):
                yield path

    def _should_include(self, path: Path, dataset_root: Path) -> bool:
        try:
            rel = path.relative_to(dataset_root)
        except ValueError:
            return False
        for part in rel.parts:
            if part in self.ignore_dirs:
                return False
        return True

    def _validate_files_concurrently(self, files: List[Path], dataset_root: Path) -> List[FileValidationResult]:
        if not files:
            return []

        workers = self.max_workers or min(32, (os.cpu_count() or 2) + 2)
        self.logger.debug(f"Validating {len(files)} files with {workers} workers")

        results: List[FileValidationResult] = []
        if workers == 1:
            for f in files:
                results.append(self._validate_single_file(f, dataset_root))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(self._validate_single_file, f, dataset_root): f for f in files}
            for future in concurrent.futures.as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as e:  # pragma: no cover
                    file_path = str(future_map[future])
                    results.append(
                        FileValidationResult(
                            file_path=file_path,
                            issues=[
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    code="unexpected_error",
                                    message=f"Unhandled error validating file: {e}",
                                )
                            ],
                        )
                    )
        return results

    # ---------------------------------------------------------------------
    # Single File Validation
    # ---------------------------------------------------------------------

    def _validate_single_file(self, file_path: Path, dataset_root: Path) -> FileValidationResult:
        rel_path = file_path.relative_to(dataset_root)
        filename = file_path.name

        entities = self._extract_entities(filename)
        suffix = self._extract_suffix(filename)
        datatype = self._infer_datatype(file_path)
        result = FileValidationResult(
            file_path=str(rel_path),
            subject=entities.get("subject"),
            session=entities.get("session"),
            datatype=datatype,
            suffix=suffix,
            modality=suffix,
        )

        if not result.subject:
            result.issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="missing_subject",
                    message="No subject entity (sub-*) found in filename",
                    path=str(rel_path),
                )
            )
            return result

        # Filename convention
        if datatype and suffix:
            pattern = self.filename_patterns.get(datatype, {}).get(suffix)
            if pattern and not pattern.match(filename):
                result.issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="invalid_filename",
                        message="Filename does not follow BIDS naming convention",
                        path=str(rel_path),
                    )
                )

        # JSON sidecar
        json_sidecar = self._match_json_sidecar(file_path)
        if not json_sidecar.exists():
            result.issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="missing_json",
                    message="Missing JSON sidecar",
                    path=str(rel_path),
                )
            )
        else:
            result.issues.extend(
                self._validate_json(json_sidecar, datatype, suffix, rel_path)
            )

        # Optional deeper validation (nibabel)
        if nib:
            try:
                hdr = nib.load(str(file_path)).header
                _ = hdr.get_data_shape()
            except Exception as e:  # pragma: no cover
                result.issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="nifti_load_failed",
                        message=f"Could not read NIfTI header: {e}",
                        path=str(rel_path),
                    )
                )

        return result

    # ---------------------------------------------------------------------
    # Dataset-Level Validation Helpers
    # ---------------------------------------------------------------------

    def _validate_dataset_structure(self, dataset_root: Path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # Required file: dataset_description.json
        if not (dataset_root / "dataset_description.json").exists():
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="missing_dataset_description",
                    message="dataset_description.json is missing",
                )
            )

        # Subject directories
        sub_dirs = [
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-") and d.name not in self.ignore_dirs
        ]
        if not sub_dirs:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="no_subjects",
                    message="No subject directories (sub-*) found",
                )
            )
        else:
            for sd in sub_dirs:
                if not re.match(r"^sub-[A-Za-z0-9]+$", sd.name):
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            code="invalid_subject_dir",
                            message=f"Invalid subject directory name: {sd.name}",
                            path=str(sd.relative_to(dataset_root)),
                        )
                    )

        # Recommended files
        recommended = ["README", "participants.tsv", "CHANGES"]
        for rec in recommended:
            if not any((dataset_root / f"{rec}{ext}").exists() for ext in ("", ".txt", ".md")):
                issues.append(
                    ValidationIssue(
                        severity=Severity.INFO,
                        code="missing_recommended_file",
                        message=f"Recommended file missing: {rec}",
                    )
                )

        return issues

    def _validate_dataset_description(self, dataset_root: Path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        desc = dataset_root / "dataset_description.json"
        if not desc.exists():
            return issues  # Already recorded as error in structure

        try:
            data = json.loads(desc.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="invalid_json",
                    message=f"Invalid JSON in dataset_description.json: {e}",
                    path=str(desc.relative_to(dataset_root)),
                )
            )
            return issues

        for field in self.dataset_descriptor_required:
            if field not in data:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="missing_descriptor_field",
                        message=f"Missing required field: {field}",
                        path=str(desc.relative_to(dataset_root)),
                        context={"field": field},
                    )
                )

        bv = data.get("BIDSVersion")
        if bv and not re.match(r"^\d+\.\d+(\.\d+)?$", bv):
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="bids_version_format",
                    message=f"BIDSVersion format may be invalid: {bv}",
                    path=str(desc.relative_to(dataset_root)),
                )
            )

        recommended = [
            "Authors",
            "Acknowledgements",
            "Funding",
            "ReferencesAndLinks",
            "DatasetDOI",
        ]
        for rec in recommended:
            if rec not in data:
                issues.append(
                    ValidationIssue(
                        severity=Severity.INFO,
                        code="missing_descriptor_recommended",
                        message=f"Recommended field missing: {rec}",
                        path=str(desc.relative_to(dataset_root)),
                    )
                )

        return issues

    def _validate_participants_file(self, dataset_root: Path) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        participants = dataset_root / "participants.tsv"
        if not participants.exists():
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="participants_missing",
                    message="participants.tsv missing (recommended)",
                )
            )
            return issues

        try:
            lines = participants.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="participants_read_error",
                    message=f"Could not read participants.tsv: {e}",
                    path=str(participants.relative_to(dataset_root)),
                )
            )
            return issues

        if not lines:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="participants_empty",
                    message="participants.tsv is empty",
                    path=str(participants.relative_to(dataset_root)),
                )
            )
            return issues

        header = lines[0].strip().split("\t")
        if "participant_id" not in header:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="participants_missing_column",
                    message="participants.tsv missing 'participant_id' column",
                    path=str(participants.relative_to(dataset_root)),
                )
            )
            return issues

        pid_index = header.index("participant_id")
        for line_no, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) != len(header):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="participants_column_mismatch",
                        message=f"Line {line_no}: column count mismatch",
                        path=str(participants.relative_to(dataset_root)),
                    )
                )
                continue
            pid = cols[pid_index]
            if not re.match(r"^sub-[A-Za-z0-9]+$", pid):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="participants_invalid_id",
                        message=f"Line {line_no}: invalid participant_id '{pid}'",
                        path=str(participants.relative_to(dataset_root)),
                        context={"line": line_no, "participant_id": pid},
                    )
                )
        return issues

    def _cross_validate(self, dataset_root: Path, file_results: List[FileValidationResult]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        participants = dataset_root / "participants.tsv"
        if not participants.exists():
            return issues

        try:
            lines = participants.read_text(encoding="utf-8").splitlines()
            if not lines:
                return issues
            header = lines[0].split("\t")
            if "participant_id" not in header:
                return issues
            pid_index = header.index("participant_id")
            listed = {
                line.split("\t")[pid_index]
                for line in lines[1:]
                if line.strip() and len(line.split("\t")) > pid_index
            }
        except Exception:
            return issues

        found = {f"sub-{f.subject}" for f in file_results if f.subject}

        missing_data = listed - found
        for md in sorted(missing_data):
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="participant_no_data",
                    message=f"{md} listed in participants.tsv but no imaging data found",
                )
            )

        missing_listed = found - listed
        for ml in sorted(missing_listed):
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    code="unlisted_participant",
                    message=f"{ml} has data but not listed in participants.tsv",
                )
            )
        return issues

    # ---------------------------------------------------------------------
    # JSON Sidecar Validation
    # ---------------------------------------------------------------------

    def _validate_json(
        self, json_path: Path, datatype: Optional[str], suffix: Optional[str], rel_path: Path
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="invalid_json_sidecar",
                    message=f"Invalid JSON format: {e}",
                    path=str(json_path),
                )
            )
            return issues
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="json_read_error",
                    message=f"Error reading JSON: {e}",
                    path=str(json_path),
                )
            )
            return issues

        # Required metadata (warn if missing)
        if datatype and suffix:
            for field in self.required_metadata.get(datatype, {}).get(suffix, []):
                if field not in metadata:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            code="missing_metadata",
                            message=f"Missing recommended metadata field: {field}",
                            path=str(json_path),
                        )
                    )

        # Field validations
        def numeric_check(key: str, positive: bool = False, non_negative: bool = False):
            if key in metadata:
                try:
                    val = float(metadata[key])
                    if positive and val <= 0:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                code="invalid_metadata_value",
                                message=f"{key} should be > 0",
                                path=str(json_path),
                            )
                        )
                    if non_negative and val < 0:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                code="invalid_metadata_value",
                                message=f"{key} should be >= 0",
                                path=str(json_path),
                            )
                        )
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            code="invalid_metadata_type",
                            message=f"{key} should be numeric",
                            path=str(json_path),
                        )
                    )

        numeric_check("RepetitionTime", positive=True)
        numeric_check("EchoTime", non_negative=True)

        if "SliceTiming" in metadata:
            st = metadata["SliceTiming"]
            if not isinstance(st, list):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="invalid_slice_timing_type",
                        message="SliceTiming should be an array",
                        path=str(json_path),
                    )
                )
            elif not all(isinstance(x, (int, float)) for x in st):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="invalid_slice_timing_value",
                        message="SliceTiming should contain only numbers",
                        path=str(json_path),
                    )
                )
        return issues

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def _compile_patterns(self) -> None:
        # Placeholder if future compilation is needed centrally
        pass

    def _extract_entities(self, filename: str) -> Dict[str, str]:
        base = filename.replace(".nii.gz", "").replace(".nii", "")
        parts = base.split("_")
        entities: Dict[str, str] = {}
        for token in parts:
            for name, pattern in self.entity_patterns.items():
                m = pattern.fullmatch(token)
                if m:
                    entities[name] = m.group(1)
        return entities

    def _extract_suffix(self, filename: str) -> Optional[str]:
        stem = filename.replace(".nii.gz", "").replace(".nii", "")
        if "_" not in stem:
            return None
        return stem.split("_")[-1]

    def _infer_datatype(self, file_path: Path) -> Optional[str]:
        for part in file_path.parts:
            if part in ("anat", "func", "dwi", "fmap", "perf", "pet"):
                return part
        return None

    def _match_json_sidecar(self, file_path: Path) -> Path:
        # Handles both .nii and .nii.gz
        if file_path.name.endswith(".nii.gz"):
            stem = file_path.name[:-7]
        else:
            stem = file_path.stem
        return file_path.with_name(f"{stem}.json")

    def _find_dataset_root(self, file_path: Path) -> Path:
        current = file_path.parent
        while current != current.parent:
            if (current / "dataset_description.json").exists():
                return current
            current = current.parent
        return file_path.parent

    # ---------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------

    def generate_report(
        self,
        result: DatasetValidationResult,
        format: str = "text",
        include_files: bool = True,
    ) -> str:
        format = format.lower()
        if format == "json":
            return json.dumps(result.to_dict(), indent=2)
        elif format == "markdown":
            return self._report_markdown(result, include_files)
        elif format == "html":
            return self._report_html(result, include_files)
        else:
            return self._report_text(result, include_files)

    def _report_text(self, result: DatasetValidationResult, include_files: bool) -> str:
        lines: List[str] = []
        lines.append("BIDS Dataset Validation Report")
        lines.append("=" * 60)
        lines.append(f"Dataset: {result.dataset_path}")
        lines.append(f"Generated: {result.summary.generated_at}")
        lines.append(f"Overall Status: {'VALID' if result.summary.valid else 'INVALID'}")
        lines.append("")
        lines.append("Summary:")
        for k, v in result.summary.issues_count.items():
            lines.append(f"  {k.capitalize()}: {v}")
        lines.append(f"  Files: {result.summary.total_files}")
        lines.append(f"  Subjects: {len(result.summary.subjects)} ({', '.join(result.summary.subjects)})")
        lines.append(f"  Sessions: {len(result.summary.sessions)} ({', '.join(result.summary.sessions)})")
        lines.append(f"  Modalities: {len(result.summary.modalities)} ({', '.join(result.summary.modalities)})")
        lines.append("")

        if result.issues:
            lines.append("Dataset-Level Issues:")
            for i, iss in enumerate(result.issues, 1):
                lines.append(f"  {i}. [{iss.severity}] {iss.code}: {iss.message}")
            lines.append("")

        if include_files:
            lines.append("File Validation Results:")
            for f in result.files:
                status = "OK" if f.is_valid() else "HAS ISSUES"
                lines.append(f"  - {f.file_path} [{status}]")
                for iss in f.issues:
                    lines.append(f"      * [{iss.severity}] {iss.code}: {iss.message}")
            lines.append("")

        lines.append("Recommendations:")
        if result.summary.valid:
            lines.append("  - Dataset appears valid under current settings.")
            if result.summary.issues_count.get("WARNING", 0):
                lines.append("  - Address warnings to improve robustness.")
        else:
            lines.append("  - Resolve all errors before downstream processing.")
        lines.append("  - Refer to BIDS spec: https://bids-specification.readthedocs.io/")
        return "\n".join(lines)

    def _report_markdown(self, result: DatasetValidationResult, include_files: bool) -> str:
        lines: List[str] = []
        lines.append(f"# BIDS Dataset Validation Report\n")
        lines.append(f"**Dataset:** `{result.dataset_path}`  ")
        lines.append(f"**Generated:** `{result.summary.generated_at}`  ")
        lines.append(f"**Overall Status:** {'✅ VALID' if result.summary.valid else '❌ INVALID'}\n")
        lines.append("## Summary")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Errors | {result.summary.issues_count.get('ERROR',0)} |")
        lines.append(f"| Warnings | {result.summary.issues_count.get('WARNING',0)} |")
        lines.append(f"| Info | {result.summary.issues_count.get('INFO',0)} |")
        lines.append(f"| Files | {result.summary.total_files} |")
        lines.append(f"| Subjects | {len(result.summary.subjects)} |")
        lines.append(f"| Sessions | {len(result.summary.sessions)} |")
        lines.append(f"| Modalities | {len(result.summary.modalities)} |")
        if result.summary.subjects:
            lines.append(f"\n**Subjects:** `{', '.join(result.summary.subjects)}`")
        if result.summary.sessions:
            lines.append(f"\n**Sessions:** `{', '.join(result.summary.sessions)}`")
        if result.summary.modalities:
            lines.append(f"\n**Modalities:** `{', '.join(result.summary.modalities)}`")

        if result.issues:
            lines.append("\n## Dataset-Level Issues")
            for iss in result.issues:
                lines.append(f"- **[{iss.severity}] {iss.code}**: {iss.message}")

        if include_files:
            lines.append("\n## Files")
            for f in result.files:
                icon = "✅" if f.is_valid() else "⚠️"
                lines.append(f"- `{f.file_path}` {icon}")
                for iss in f.issues:
                    lines.append(f"  - **[{iss.severity}] {iss.code}**: {iss.message}")

        lines.append("\n## Recommendations")
        if result.summary.valid:
            lines.append("- Dataset passes validation. Address warnings if present.")
        else:
            lines.append("- Fix all errors. Re-run validation after corrections.")
        lines.append("- Refer to [BIDS Specification](https://bids-specification.readthedocs.io/)")
        return "\n".join(lines)

    def _report_html(self, result: DatasetValidationResult, include_files: bool) -> str:
        # Lightweight HTML; for more complex needs integrate a template engine
        html = []
        status_color = "#2e7d32" if result.summary.valid else "#c62828"
        html.append("<html><head><meta charset='utf-8'><title>BIDS Validation Report</title>")
        html.append(
            "<style>body{font-family:Arial, sans-serif;margin:1.5rem;} "
            "table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 8px;} "
            ".err{color:#c62828;} .warn{color:#e65100;} .info{color:#1565c0;} .ok{color:#2e7d32;}"
            "</style></head><body>"
        )
        html.append(f"<h1>BIDS Dataset Validation Report</h1>")
        html.append(f"<p><strong>Dataset:</strong> {result.dataset_path}<br>")
        html.append(f"<strong>Generated:</strong> {result.summary.generated_at}<br>")
        html.append(f"<strong>Status:</strong> <span style='color:{status_color}'>"
                    f"{'VALID' if result.summary.valid else 'INVALID'}</span></p>")
        html.append("<h2>Summary</h2><table>")
        for k, v in [
            ("Errors", result.summary.issues_count.get("ERROR", 0)),
            ("Warnings", result.summary.issues_count.get("WARNING", 0)),
            ("Info", result.summary.issues_count.get("INFO", 0)),
            ("Files", result.summary.total_files),
            ("Subjects", len(result.summary.subjects)),
            ("Sessions", len(result.summary.sessions)),
            ("Modalities", len(result.summary.modalities)),
        ]:
            html.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html.append("</table>")

        if result.issues:
            html.append("<h2>Dataset-Level Issues</h2><ul>")
            for iss in result.issues:
                cls = "err" if iss.severity == Severity.ERROR else "warn" if iss.severity == Severity.WARNING else "info"
                html.append(f"<li class='{cls}'><strong>[{iss.severity}] {iss.code}</strong>: {iss.message}</li>")
            html.append("</ul>")

        if include_files:
            html.append("<h2>Files</h2><ul>")
            for f in result.files:
                cls = "ok" if f.is_valid() else "warn"
                html.append(f"<li class='{cls}'>{f.file_path}")
                if f.issues:
                    html.append("<ul>")
                    for iss in f.issues:
                        cls_i = "err" if iss.severity == Severity.ERROR else "warn" if iss.severity == Severity.WARNING else "info"
                        html.append(f"<li class='{cls_i}'><strong>[{iss.severity}] {iss.code}</strong>: {iss.message}</li>")
                    html.append("</ul>")
                html.append("</li>")
            html.append("</ul>")

        html.append("<h2>Recommendations</h2><ul>")
        if result.summary.valid:
            html.append("<li>Dataset passes validation. Consider resolving warnings.</li>")
        else:
            html.append("<li>Resolve all errors before proceeding.</li>")
        html.append("<li>Refer to <a href='https://bids-specification.readthedocs.io/'>BIDS Specification</a></li>")
        html.append("</ul></body></html>")
        return "\n".join(html)

    # ---------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Advanced BIDS Dataset Validator")
        p.add_argument("dataset", help="Path to BIDS dataset root")
        p.add_argument(
            "--format",
            default="text",
            choices=["text", "json", "markdown", "html"],
            help="Report output format",
        )
        p.add_argument(
            "--strict",
            action="store_true",
            help="Treat warnings as errors for overall validity",
        )
        p.add_argument(
            "--max-workers",
            type=int,
            default=0,
            help="Max parallel workers (0 = auto)",
        )
        p.add_argument(
            "--no-file-details",
            action="store_true",
            help="Exclude per-file details from report",
        )
        p.add_argument(
            "--output",
            "-o",
            help="Write report to file path (based on format)",
        )
        p.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging verbosity",
        )
        return p

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> int:
        parser = cls.build_arg_parser()
        args = parser.parse_args(argv)

        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        validator = cls(strict=args.strict, max_workers=args.max_workers)
        result = validator.validate_dataset(args.dataset)
        report = validator.generate_report(
            result,
            format=args.format,
            include_files=not args.no_file_details,
        )

        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(report)

        return 0 if result.summary.valid else 1


# ---------------------------------------------------------------------------
# Module Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(BIDSComplianceValidator.main())
