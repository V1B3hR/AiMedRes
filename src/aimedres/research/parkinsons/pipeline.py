"""
Modularized, typed Parkinson's Disease research pipeline.
Conforms to Phase 2.1 of the Consolidated Roadmap.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from ...training.train_parkinsons import ParkinsonsTrainingPipeline
from .config import ParkinsonsPipelineConfig
from .manifest import generate_experiment_manifest

logger = logging.getLogger(__name__)


class ParkinsonsResearchPipeline:
    """
    Modular, reproducible research pipeline for Parkinson's disease progression modeling.
    Enforces Pydantic configurations, deterministic random states,
    and automatic cryptographic manifest generation.
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

    def load_data(self) -> pd.DataFrame:
        """Load dataset using configuration parameters."""
        return self.internal_pipeline.load_data(
            data_path=self.config.data_path,
            dataset_choice=self.config.dataset_choice,
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute full leakage-proof pipeline and record provenance manifest.

        Returns:
            Dictionary containing cross-validation metrics, bootstrap CIs,
            calibration Brier scores, and manifest metadata.
        """
        logger.info(
            f"Executing ParkinsonsResearchPipeline with output_dir: {self.config.output_dir}"
        )

        results = self.internal_pipeline.run_full_pipeline(
            data_path=self.config.data_path,
            target_column=self.config.target_column,
            epochs=self.config.epochs,
            n_folds=self.config.n_folds,
            dataset_choice=self.config.dataset_choice,
        )

        # Generate cryptographic provenance manifest
        manifest = generate_experiment_manifest(
            data_path=self.config.data_path,
            output_dir=self.config.output_dir,
            config_dict=self.config.model_dump(),
            metrics_summary=results.get("classical_results"),
        )

        results["manifest"] = manifest
        return results
