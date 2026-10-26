#!/usr/bin/env python3
"""
PPMI Parkinson's Reproducibility Benchmark Runner
AiMedRes Phase 1 Master Verification Script

Executes the leakage-safe Parkinson's motor-progression research pipeline on
the synthetic PPMI-schema fixture (or a user-provided, authorized dataset),
generating bootstrap confidence intervals, calibration scores, and
reproducibility metrics. It does not establish clinical performance.

Usage:
    python scripts/reproduce_ppmi_benchmark.py
    python scripts/reproduce_ppmi_benchmark.py --data-path data/ppmi/ppmi_sbr_longitudinal.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import yaml

# Add src to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SYNTHETIC_FIXTURE = _REPO_ROOT / "ParkinsonDatasets" / "synthetic_pet_sbr_fixture.csv"
_EVIDENCE_SOURCE_FILES = (
    "configs/parkinsons_ppmi_study.yml",
    "scripts/reproduce_ppmi_benchmark.py",
    "src/aimedres/research/parkinsons/config.py",
    "src/aimedres/research/parkinsons/manifest.py",
    "src/aimedres/research/parkinsons/pipeline.py",
    "src/aimedres/training/train_parkinsons.py",
)
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aimedres.research.parkinsons import (
    ParkinsonsPipelineConfig,
    ParkinsonsResearchPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PPMI_REPRODUCIBILITY")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated, runnable settings for the research-only PPMI benchmark."""

    data_path: Path
    output_dir: Path
    target_column: str
    n_folds: int
    epochs: int
    random_state: int
    n_bootstraps: int
    alpha: float
    batch_size: int
    learning_rate: float
    source_config_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.target_column.strip():
            raise ValueError("pipeline.target_column must be a non-empty string")
        if self.n_folds < 2:
            raise ValueError("pipeline.n_folds must be at least 2")
        if self.epochs < 1:
            raise ValueError("pipeline.epochs must be at least 1")
        if self.random_state < 0:
            raise ValueError("pipeline.random_state must be at least 0")
        if self.n_bootstraps < 100:
            raise ValueError("pipeline.n_bootstraps must be at least 100")
        if not 0.0 < self.alpha <= 0.5:
            raise ValueError("pipeline.alpha must be greater than 0 and at most 0.5")
        if self.batch_size < 1:
            raise ValueError("pipeline.batch_size must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("pipeline.learning_rate must be greater than 0")


def _default_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        data_path=_SYNTHETIC_FIXTURE,
        output_dir=_REPO_ROOT / "results" / "ppmi_reproducibility_benchmark",
        target_column="motor_progression_label",
        n_folds=5,
        epochs=25,
        random_state=42,
        n_bootstraps=1000,
        alpha=0.05,
        batch_size=16,
        learning_rate=0.001,
        source_config_path=_REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml",
    )


def _require_mapping(raw: object, name: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return raw


def _require_text(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(section: dict, key: str, minimum: int) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer greater than or equal to {minimum}")
    return value


def _require_float(section: dict, key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    """Load the versioned synthetic-study configuration without executing training."""
    config_path = config_path.expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.is_file():
        raise ValueError(f"Benchmark configuration does not exist: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid benchmark YAML in {config_path}: {exc}") from exc

    raw = _require_mapping(raw, "benchmark configuration")
    required_sections = {"version", "study", "data", "pipeline", "output"}
    missing_sections = required_sections.difference(raw)
    if missing_sections:
        raise ValueError(
            "Benchmark configuration is missing required sections: "
            + ", ".join(sorted(missing_sections))
        )
    unknown_sections = set(raw).difference(required_sections)
    if unknown_sections:
        raise ValueError(
            "Benchmark configuration contains unsupported sections: "
            + ", ".join(sorted(unknown_sections))
        )
    if raw["version"] != 1:
        raise ValueError("Benchmark configuration version must be 1")

    study = _require_mapping(raw["study"], "study")
    _require_text(study, "name")
    intended_use = _require_text(study, "intended_use")
    if "research" not in intended_use.lower():
        raise ValueError("study.intended_use must explicitly state the research-only purpose")

    data = _require_mapping(raw["data"], "data")
    pipeline = _require_mapping(raw["pipeline"], "pipeline")
    output = _require_mapping(raw["output"], "output")

    return BenchmarkConfig(
        data_path=_resolve_repo_path(_require_text(data, "path")),
        output_dir=_resolve_repo_path(_require_text(output, "directory")),
        target_column=_require_text(pipeline, "target_column"),
        n_folds=_require_positive_int(pipeline, "n_folds", minimum=2),
        epochs=_require_positive_int(pipeline, "epochs", minimum=1),
        random_state=_require_positive_int(pipeline, "random_state", minimum=0),
        n_bootstraps=_require_positive_int(pipeline, "n_bootstraps", minimum=100),
        alpha=_require_float(pipeline, "alpha"),
        batch_size=_require_positive_int(pipeline, "batch_size", minimum=1),
        learning_rate=_require_float(pipeline, "learning_rate"),
        source_config_path=config_path.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research-only PPMI Parkinson's reproducibility benchmark"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Versioned YAML study configuration. Paths inside it are relative to the repository root.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Dataset CSV override. Use controlled data only when authorized by the applicable DUA.",
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory override.")
    parser.add_argument("--target", type=str, help="Target endpoint override.")
    parser.add_argument("--folds", type=int, help="Cross-validation fold-count override.")
    parser.add_argument("--epochs", type=int, help="Neural-network epoch-count override.")
    parser.add_argument("--random-state", type=int, help="Random seed override.")
    parser.add_argument("--batch-size", type=int, help="Neural-network batch-size override.")
    parser.add_argument("--learning-rate", type=float, help="Neural-network learning-rate override.")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help=(
            "Empty directory for a safe-to-commit evidence pack. This is permitted only when "
            "running the committed synthetic fixture."
        ),
    )
    return parser


def resolve_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Merge explicit CLI overrides over a validated file or the legacy defaults."""
    base = load_benchmark_config(args.config) if args.config else _default_benchmark_config()
    return BenchmarkConfig(
        data_path=args.data_path or base.data_path,
        output_dir=args.output_dir or base.output_dir,
        target_column=args.target or base.target_column,
        n_folds=args.folds if args.folds is not None else base.n_folds,
        epochs=args.epochs if args.epochs is not None else base.epochs,
        random_state=args.random_state if args.random_state is not None else base.random_state,
        n_bootstraps=base.n_bootstraps,
        alpha=base.alpha,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        learning_rate=args.learning_rate if args.learning_rate is not None else base.learning_rate,
        source_config_path=base.source_config_path,
    )


def print_banner():
    banner = """
================================================================================
          AiMedRes: PPMI Parkinson's Research Reproducibility Benchmark
          Research-Only Reproducibility Runner (Synthetic Fixture)
================================================================================
    """
    print(banner)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_context() -> dict:
    """Return a bounded Git snapshot without executing a shell command."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "head": head,
            "worktree_clean": not bool(status.strip()),
            "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "worktree_clean": None, "worktree_status_sha256": None}


def _evidence_source_hashes(config: BenchmarkConfig) -> dict:
    source_paths = [_REPO_ROOT / relative for relative in _EVIDENCE_SOURCE_FILES]
    if config.source_config_path:
        source_paths.append(config.source_config_path)

    hashes = {}
    for source_path in source_paths:
        if source_path.is_file():
            hashes[_repo_relative(source_path)] = _sha256_file(source_path)
    return hashes


def _require_canonical_evidence_configuration(config: BenchmarkConfig) -> None:
    """Keep public evidence packs comparable by rejecting parameter overrides."""
    canonical_path = _REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"
    canonical = load_benchmark_config(canonical_path)
    if not config.source_config_path or config.source_config_path.resolve() != canonical_path.resolve():
        raise ValueError(
            "Evidence-pack output requires the committed canonical study configuration; "
            "do not use an external or copied configuration file."
        )

    comparable_fields = (
        "data_path",
        "target_column",
        "n_folds",
        "epochs",
        "random_state",
        "n_bootstraps",
        "alpha",
        "batch_size",
        "learning_rate",
    )
    if any(getattr(config, field) != getattr(canonical, field) for field in comparable_fields):
        raise ValueError(
            "Evidence-pack output requires the committed canonical synthetic-study parameters; "
            "run parameter overrides without --evidence-dir instead."
        )


def write_evidence_pack(
    evidence_dir: Path, config: BenchmarkConfig, summary_payload: dict
) -> Path:
    """Write a minimal, non-clinical evidence pack for the committed synthetic fixture."""
    if config.data_path.resolve() != _SYNTHETIC_FIXTURE.resolve():
        raise ValueError(
            "Evidence-pack output is restricted to the committed synthetic fixture; "
            "do not create a public pack from controlled or patient-level data."
        )
    _require_canonical_evidence_configuration(config)

    evidence_dir = evidence_dir.expanduser()
    if not evidence_dir.is_absolute():
        evidence_dir = Path.cwd() / evidence_dir
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise ValueError(f"Evidence-pack directory must be empty: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    config_path = config.source_config_path or _REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"
    manifest = {
        "evidence_pack_version": "1.0.0",
        "scope": "Synthetic-fixture reproducibility evidence only; not clinical validation.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --evidence-dir <directory>",
        "study_configuration": {
            "path": _repo_relative(config_path),
            "sha256": _sha256_file(config_path) if config_path.is_file() else None,
            "values": {
                "target_column": config.target_column,
                "n_folds": config.n_folds,
                "epochs": config.epochs,
                "random_state": config.random_state,
                "n_bootstraps": config.n_bootstraps,
                "alpha": config.alpha,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
        },
        "synthetic_input": {
            "path": _repo_relative(config.data_path),
            "sha256": _sha256_file(config.data_path),
        },
        "source_snapshot_sha256": _evidence_source_hashes(config),
        "git": _git_context(),
        "benchmark_summary": summary_payload,
        "limitations": [
            "Synthetic data only; no PPMI, patient-level, external-cohort, or clinical-performance claim.",
            "Local repeatability does not establish independent, cross-platform, or institutional reproducibility.",
            "This pack contains no model artifact, participant-level data, or controlled dataset metadata.",
        ],
    }
    manifest_path = evidence_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = summary_payload["summary"]
    result_rows = "\n".join(
        f"| {row['model']} | {row['roc_auc']:.3f} | "
        f"{row['roc_auc_ci_95'][0]:.3f}–{row['roc_auc_ci_95'][1]:.3f} | "
        f"{row['brier_score']:.3f} |"
        for row in rows
    )
    evidence_readme = f"""# AiMedRes Evidence Pack — synthetic PPMI-schema fixture

> **Research-only.** This pack demonstrates one local, reproducible execution on a non-identifiable synthetic fixture. It is not PPMI evidence, patient-level performance, clinical validation, regulatory evidence, or a deployable model.

## Frozen inputs

- Study configuration: `{manifest['study_configuration']['path']}` (`SHA-256`: `{manifest['study_configuration']['sha256']}`)
- Synthetic fixture: `{manifest['synthetic_input']['path']}` (`SHA-256`: `{manifest['synthetic_input']['sha256']}`)
- Git HEAD: `{manifest['git']['head'] or 'unavailable'}`
- Worktree clean: `{manifest['git']['worktree_clean']}`. The source-file hashes in `evidence_manifest.json` are the authoritative snapshot for this run.

## Control-run result

Execution time: {summary_payload['execution_seconds']:.2f} seconds.  Five-fold `StratifiedGroupKFold` was used with `PATNO` as group identifier.

| Model | ROC-AUC | 95% CI | Brier score |
|---|---:|---:|---:|
{result_rows}

## Reproduction

```powershell
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --evidence-dir evidence/ppmi_synthetic_fixture/<new-empty-directory>
```

The directory must be new or empty; the runner refuses to overwrite an existing evidence pack. Full parameters, runtime context, Git state and source hashes are in `evidence_manifest.json`.

## Explicit limits

- The source fixture is synthetic; the values above must never be reported as PPMI or clinical-model metrics.
- A dirty worktree cannot substitute for a release commit or signed tag.
- An independent rerun, methodological review, authorized controlled-data work and external validation are still required.
"""
    (evidence_dir / "EVIDENCE_PACK.md").write_text(evidence_readme, encoding="utf-8")
    return manifest_path


def run_benchmark(config: BenchmarkConfig, evidence_dir: Optional[Path] = None) -> int:
    """Run one fully versioned benchmark configuration."""
    data_path = config.data_path
    output_dir = config.output_dir
    target_column = config.target_column
    n_folds = config.n_folds
    epochs = config.epochs
    print_banner()
    logger.info(f"Using Dataset Path: {data_path}")
    logger.info(f"Target Column: {target_column}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Cross-Validation: {n_folds}-Fold StratifiedGroupKFold on PATNO")

    if not data_path.exists():
        logger.error(f"Dataset path does not exist: {data_path}")
        return 1

    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = ParkinsonsPipelineConfig(
        data_path=str(data_path),
        output_dir=str(output_dir),
        target_column=target_column,
        n_folds=n_folds,
        epochs=epochs,
        dataset_choice="local",
        random_state=config.random_state,
        n_bootstraps=config.n_bootstraps,
        alpha=config.alpha,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    # Use the maintained research wrapper so the benchmark records its
    # experiment lineage and cryptographic provenance alongside its metrics.
    report = ParkinsonsResearchPipeline(pipeline_config).run()

    elapsed = time.time() - start_time
    classical = report.get("classical_results", {})

    print("\n" + "=" * 80)
    print("                     BENCHMARK EVALUATION SUMMARY                     ")
    print("=" * 80)
    print(f"{'Model':<20} | {'ROC-AUC (95% CI)':<25} | {'PR-AUC (95% CI)':<25} | {'Brier':<8}")
    print("-" * 80)

    summary_rows = []
    for model_name, metrics in classical.items():
        auc_mean = metrics.get("roc_auc_mean", 0.0)
        auc_ci = metrics.get("roc_auc_ci_95", [0.0, 0.0])
        pr_mean = metrics.get("pr_auc_mean", 0.0)
        pr_ci = metrics.get("pr_auc_ci_95", [0.0, 0.0])
        brier = metrics.get("brier_score", 0.0)

        auc_str = f"{auc_mean:.3f} [{auc_ci[0]:.3f} - {auc_ci[1]:.3f}]"
        pr_str = f"{pr_mean:.3f} [{pr_ci[0]:.3f} - {pr_ci[1]:.3f}]"
        brier_str = f"{brier:.3f}"

        print(f"{model_name:<20} | {auc_str:<25} | {pr_str:<25} | {brier_str:<8}")
        summary_rows.append({
            "model": model_name,
            "roc_auc": auc_mean,
            "roc_auc_ci_95": auc_ci,
            "pr_auc": pr_mean,
            "pr_auc_ci_95": pr_ci,
            "brier_score": brier,
            "sensitivity": metrics.get("sensitivity", 0.0),
            "specificity": metrics.get("specificity", 0.0),
        })

    print("-" * 80)
    print(f"Total Benchmark Execution Time: {elapsed:.2f} seconds")
    print(f"Artifacts and Models Saved To:  {output_dir}")
    print("=" * 80 + "\n")

    # Persist summary JSON in the local, potentially artifact-bearing run directory.
    summary_path = output_dir / "benchmark_summary.json"
    summary_payload = {
        "timestamp": report.get("timestamp"),
        "dataset": _repo_relative(data_path),
        "target": target_column,
        "n_folds": n_folds,
        "execution_seconds": elapsed,
        "summary": summary_rows,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info(f"Saved benchmark summary to: {summary_path}")
    if evidence_dir:
        manifest_path = write_evidence_pack(evidence_dir, config, summary_payload)
        logger.info(f"Saved safe-to-commit evidence manifest to: {manifest_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_benchmark_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    return run_benchmark(config, evidence_dir=args.evidence_dir)


if __name__ == "__main__":
    sys.exit(main())
