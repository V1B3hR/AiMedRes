"""
Modularized, typed Parkinson's Disease research pipeline.
Conforms to Phase 2.1 of the Consolidated Roadmap.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ...training.train_parkinsons import ParkinsonsTrainingPipeline
from ..tracking import LocalExperimentTracker
from .config import ParkinsonsPipelineConfig
from .manifest import compute_file_sha256, generate_experiment_manifest

logger = logging.getLogger(__name__)


class ParkinsonsResearchPipeline:
    """
    Modular, reproducible research pipeline for Parkinson's disease progression modeling.
    Enforces Pydantic configurations, deterministic random states,
    live experiment tracking (SQLite/JSON), and automatic cryptographic manifest generation.
    """

    def __init__(self, config: Optional[Union[ParkinsonsPipelineConfig, Dict[str, Any]]] = None):
        if config is None:
            self.config = ParkinsonsPipelineConfig()
        elif isinstance(config, dict):
            self.config = ParkinsonsPipelineConfig(**config)
        elif isinstance(config, ParkinsonsPipelineConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        self.internal_pipeline = ParkinsonsTrainingPipeline(
            output_dir=self.config.output_dir
        )
        self.tracker: Optional[LocalExperimentTracker] = None
        if self.config.track_experiments:
            db_path = Path(self.config.output_dir) / "experiments.db"
            self.tracker = LocalExperimentTracker(
                db_path=db_path, experiment_name=self.config.experiment_name
            )

    def load_data(self) -> pd.DataFrame:
        """Load dataset using configuration parameters."""
        return self.internal_pipeline.load_data(
            data_path=self.config.data_path,
            dataset_choice=self.config.dataset_choice,
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute full leakage-proof pipeline and record provenance manifest and experiment run.

        Returns:
            Dictionary containing cross-validation metrics, bootstrap CIs,
            calibration Brier scores, experiment tracking record, and manifest metadata.
        """
        logger.info(
            f"Executing ParkinsonsResearchPipeline with output_dir: {self.config.output_dir}"
        )

        run_id = None
        if self.tracker:
            data_hash = (
                compute_file_sha256(self.config.data_path)
                if self.config.data_path
                else None
            )
            run_id = self.tracker.start_run(
                run_name=f"ppmi_{self.config.dataset_choice}_k{self.config.n_folds}",
                params=self.config.model_dump(),
                tags={"framework": "aimedres.research.parkinsons", "target": self.config.target_column},
                dataset_sha256=data_hash,
                output_dir=self.config.output_dir,
            )

        try:
            results = self.internal_pipeline.run_full_pipeline(
                data_path=self.config.data_path,
                target_column=self.config.target_column,
                epochs=self.config.epochs,
                n_folds=self.config.n_folds,
                dataset_choice=self.config.dataset_choice,
            )

            # Log metrics to tracker
            if self.tracker and run_id:
                classical = results.get("classical_results", {})
                for model_name, mdict in classical.items():
                    if isinstance(mdict, dict):
                        for k, v in mdict.items():
                            if isinstance(v, (int, float)):
                                self.tracker.log_metric(f"{model_name}_{k}", float(v), run_id=run_id)

                # Log saved model artifacts
                models_dir = Path(self.config.output_dir) / "models"
                if models_dir.exists():
                    for mf in models_dir.glob("*.pkl"):
                        self.tracker.log_artifact(mf.stem, mf, run_id=run_id)

                run_summary = self.tracker.end_run(status="COMPLETED", run_id=run_id)
                results["experiment_run"] = run_summary

        except Exception as e:
            if self.tracker and run_id:
                self.tracker.end_run(status="FAILED", run_id=run_id)
            raise e

        # Generate cryptographic provenance manifest
        manifest = generate_experiment_manifest(
            data_path=self.config.data_path,
            output_dir=self.config.output_dir,
            config_dict=self.config.model_dump(),
            metrics_summary=results.get("classical_results"),
        )
        if run_id:
            manifest["experiment_run_id"] = run_id

        results["manifest"] = manifest
        return results
