"""
Unit tests for decoupled Parkinson's Research Package (Phase 2.1).
Validates Pydantic configuration, cryptographic provenance manifest, and pipeline decoupling.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aimedres.research.parkinsons.config import ParkinsonsPipelineConfig
from aimedres.research.parkinsons.manifest import (
    compute_file_sha256,
    generate_experiment_manifest,
)
from aimedres.research.parkinsons.pipeline import ParkinsonsResearchPipeline


class TestParkinsonsPipelineConfig:
    """Validate Pydantic configuration schema constraints."""

    def test_default_configuration_validity(self):
        config = ParkinsonsPipelineConfig()
        assert config.n_folds == 5
        assert config.random_state == 42
        assert config.target_column == "motor_progression_label"
        assert config.group_column == "PATNO"
        assert config.alpha == 0.05
        assert config.epochs == 25

    def test_custom_parameters_validation(self):
        config = ParkinsonsPipelineConfig(
            n_folds=10,
            random_state=123,
            output_dir="tmp/custom_output",
            epochs=5,
        )
        assert config.n_folds == 10
        assert config.random_state == 123
        assert config.output_dir == "tmp/custom_output"
        assert config.epochs == 5

    def test_invalid_parameters_raise_validation_error(self):
        with pytest.raises(ValidationError):
            # n_folds must be >= 2
            ParkinsonsPipelineConfig(n_folds=1)

        with pytest.raises(ValidationError):
            # alpha must be between 0.001 and 0.5
            ParkinsonsPipelineConfig(alpha=0.9)


class TestProvenanceManifest:
    """Validate cryptographic manifest creation and hashing."""

    def test_compute_file_sha256(self, tmp_path):
        test_file = tmp_path / "sample_data.csv"
        test_file.write_text("colA,colB\n1.0,2.0\n")

        sha = compute_file_sha256(test_file)
        assert sha is not None
        assert len(sha) == 64
        assert int(sha, 16) > 0

    def test_generate_experiment_manifest(self, tmp_path):
        dummy_data = tmp_path / "dummy.csv"
        dummy_data.write_text("header\nvalue\n")

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "dummy_model.pkl").write_bytes(b"dummy_model_bytes")

        config = ParkinsonsPipelineConfig(output_dir=str(tmp_path))
        manifest = generate_experiment_manifest(
            data_path=dummy_data,
            output_dir=tmp_path,
            config_dict=config.model_dump(),
            metrics_summary={"logreg": {"auc": 0.90}},
        )

        assert manifest["manifest_version"] == "1.0.0"
        assert "timestamp_utc" in manifest
        assert manifest["governance"]["license"] == "MIT"
        assert "dummy_model.pkl" in manifest["model_artifacts"]
        assert manifest["runtime_environment"]["python"] is not None

        # Verify saved manifest file
        manifest_file = tmp_path / "provenance_manifest.json"
        assert manifest_file.exists()
        loaded = json.loads(manifest_file.read_text())
        assert loaded["manifest_version"] == "1.0.0"


class TestParkinsonsResearchPipeline:
    """Validate research pipeline initialization and delegation."""

    def test_pipeline_initialization(self, tmp_path):
        config = ParkinsonsPipelineConfig(output_dir=str(tmp_path))
        pipeline = ParkinsonsResearchPipeline(config)

        assert pipeline.config.output_dir == str(tmp_path)
        assert pipeline.internal_pipeline is not None
        assert (tmp_path / "models").exists()
        assert (tmp_path / "metrics").exists()

    def test_pipeline_initialization_from_dict(self, tmp_path):
        config_dict = {"output_dir": str(tmp_path), "n_folds": 3}
        pipeline = ParkinsonsResearchPipeline(config_dict)

        assert pipeline.config.n_folds == 3
        assert pipeline.config.output_dir == str(tmp_path)
