"""
Cryptographic Provenance Manifest Generator for Parkinson's Disease Pipeline.
Guarantees reproducible lineage, software environment hashes, and data integrity.
Conforms to Phase 2.1 & 2.3 of the Consolidated Roadmap.
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_file_sha256(filepath: Union[str, Path]) -> Optional[str]:
    """Compute cryptographic SHA-256 hash for a given file."""
    path = Path(filepath)
    if not path.is_file():
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_experiment_manifest(
    data_path: Optional[Union[str, Path]],
    output_dir: Union[str, Path],
    config_dict: Dict[str, Any],
    metrics_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate complete cryptographic provenance manifest capturing:
    - Data artifact hash
    - Saved model checkpoint hashes
    - Software dependencies and platform runtime
    - Pipeline hyperparameters and deterministic seeds
    """
    out_path = Path(output_dir)
    models_dir = out_path / "models"
    preprocessors_dir = out_path / "preprocessors"

    model_hashes: Dict[str, str] = {}
    if models_dir.exists():
        for mf in sorted(models_dir.glob("*")):
            if mf.is_file():
                h = compute_file_sha256(mf)
                if h:
                    model_hashes[mf.name] = h

    preprocessor_hashes: Dict[str, str] = {}
    if preprocessors_dir.exists():
        for pf in sorted(preprocessors_dir.glob("*")):
            if pf.is_file():
                h = compute_file_sha256(pf)
                if h:
                    preprocessor_hashes[pf.name] = h

    # Runtime environment dependencies
    dependencies: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }

    for pkg_name in ["numpy", "pandas", "scikit-learn", "torch", "pydantic"]:
        try:
            mod = __import__(pkg_name.replace("-", "_"))
            dependencies[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            dependencies[pkg_name] = "not_installed"

    manifest = {
        "manifest_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": "ppmi_parkinsons_progression_research",
        "data_input": {
            "path": str(data_path) if data_path else "synthetic_internal_fallback",
            "sha256": compute_file_sha256(data_path) if data_path else None,
        },
        "configuration": config_dict,
        "runtime_environment": dependencies,
        "model_artifacts": model_hashes,
        "preprocessor_artifacts": preprocessor_hashes,
        "metrics_summary": metrics_summary or {},
        "governance": {
            "license": "MIT",
            "intended_use": "Retrospective research validation only",
            "clinical_clearance": "None (Not approved by FDA for diagnosis)",
            "leakage_controls": [
                "StratifiedGroupKFold grouped on PATNO",
                "Fold-encapsulated ColumnTransformer fit strictly on train split",
                "Target defined from independent clinical progression outcome",
            ],
        },
    }

    manifest_path = out_path / "provenance_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
