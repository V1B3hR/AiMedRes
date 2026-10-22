#!/usr/bin/env python3
"""
PPMI Parkinson's Reproducibility Benchmark Runner
AiMedRes Phase 1 Master Verification Script

Executes the leakage-safe Parkinson's motor progression pipeline on the
calibrated PPMI benchmark fixture (or user-provided real PPMI dataset),
generating 95% bootstrap confidence intervals, calibration scores, and
reproducibility metrics.

Usage:
    python scripts/reproduce_ppmi_benchmark.py
    python scripts/reproduce_ppmi_benchmark.py --data-path data/ppmi/ppmi_sbr_longitudinal.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from aimedres.training.train_parkinsons import ParkinsonTrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PPMI_REPRODUCIBILITY")


def print_banner():
    banner = """
================================================================================
          AiMedRes: PPMI Parkinson's Research Reproducibility Benchmark
          Phase 1 Flagship Scientific Validation Runner
================================================================================
    """
    print(banner)


def run_benchmark(
    data_path: Path,
    output_dir: Path,
    target_column: str = "motor_progression_label",
    n_folds: int = 5,
    epochs: int = 30,
) -> int:
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

    pipeline = ParkinsonTrainingPipeline(output_dir=str(output_dir))

    # Run full leakage-safe pipeline
    report = pipeline.run_full_pipeline(
        data_path=str(data_path),
        target_column=target_column,
        dataset_choice="local",
        epochs=epochs,
        n_folds=n_folds,
    )

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

    # Persist summary JSON
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": report.get("timestamp"),
                "dataset": str(data_path),
                "target": target_column,
                "n_folds": n_folds,
                "execution_seconds": elapsed,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved benchmark summary to: {summary_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run PPMI Parkinson's Reproducibility Benchmark"
    )
    default_data = _REPO_ROOT / "ParkinsonDatasets" / "synthetic_pet_sbr_fixture.csv"
    default_out = _REPO_ROOT / "results" / "ppmi_reproducibility_benchmark"

    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data,
        help=f"Path to dataset CSV (default: {default_data})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Output directory for results (default: {default_out})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="motor_progression_label",
        help="Target endpoint (default: motor_progression_label)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Neural network epochs (default: 25)",
    )

    args = parser.parse_args()
    ret = run_benchmark(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_column=args.target,
        n_folds=args.folds,
        epochs=args.epochs,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
