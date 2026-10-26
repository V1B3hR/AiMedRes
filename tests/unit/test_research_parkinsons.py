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
        assert manifest["runtime_environment"]["scikit-learn"] != "not_installed"
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
        assert pipeline.tracker is not None

    def test_pipeline_forwards_reproducibility_parameters(self, tmp_path, monkeypatch):
        data_path = tmp_path / "fixture.csv"
        data_path.write_text("PATNO,motor_progression_label\n1,0\n2,1\n", encoding="utf-8")
        config = ParkinsonsPipelineConfig(
            data_path=str(data_path),
            output_dir=str(tmp_path / "outputs"),
            n_folds=3,
            random_state=17,
            n_bootstraps=200,
            alpha=0.1,
            epochs=2,
            batch_size=4,
            learning_rate=0.002,
            track_experiments=False,
        )
        pipeline = ParkinsonsResearchPipeline(config)
        received = {}

        def fake_run_full_pipeline(**kwargs):
            received.update(kwargs)
            return {"classical_results": {}}

        monkeypatch.setattr(pipeline.internal_pipeline, "run_full_pipeline", fake_run_full_pipeline)
        monkeypatch.setattr(
            "aimedres.research.parkinsons.pipeline.generate_experiment_manifest",
            lambda **_kwargs: {},
        )

        pipeline.run()

        assert received["random_state"] == 17
        assert received["n_bootstraps"] == 200
        assert received["alpha"] == 0.1
        assert received["batch_size"] == 4
        assert received["learning_rate"] == 0.002


class TestLocalExperimentTracker:
    """Validate SQLite and JSON experiment tracking and model lineage (Phase 2.3)."""

    def test_tracker_lifecycle_and_provenance(self, tmp_path):
        from aimedres.research.tracking import LocalExperimentTracker

        db_file = tmp_path / "test_experiments.db"
        tracker = LocalExperimentTracker(db_path=db_file, experiment_name="test_parkinsons")

        # Start run
        run_id = tracker.start_run(
            run_name="benchmark_k5",
            params={"n_folds": 5, "learning_rate": 0.001, "model": "logistic_regression"},
            tags={"env": "pytest", "author": "researcher"},
            dataset_sha256="abcdef1234567890",
            output_dir=tmp_path,
        )
        assert run_id.startswith("run_")
        assert tracker.active_run_id == run_id

        # Log metrics across steps
        tracker.log_metric("roc_auc", 0.88, step=1)
        tracker.log_metric("roc_auc", 0.92, step=2)
        tracker.log_metrics({"brier_score": 0.11, "accuracy": 0.86})

        # Log artifact
        artifact_file = tmp_path / "model_weights.pkl"
        artifact_file.write_bytes(b"sample_serialized_model")
        sha = tracker.log_artifact("best_model", artifact_file)
        assert sha is not None
        assert len(sha) == 64

        # End run
        summary = tracker.end_run(status="COMPLETED")
        assert summary["status"] == "COMPLETED"
        assert summary["git_commit"] is not None
        assert summary["metrics"]["roc_auc"] == 0.92
        assert summary["metrics"]["brier_score"] == 0.11
        assert summary["params"]["n_folds"] == "5"
        assert summary["tags"]["env"] == "pytest"

        # Verify JSON card exported
        card_file = tmp_path / "runs" / f"{run_id}.json"
        assert card_file.exists()
        loaded_card = json.loads(card_file.read_text())
        assert loaded_card["run_id"] == run_id

    def test_tracker_list_and_best_run(self, tmp_path):
        from aimedres.research.tracking import LocalExperimentTracker

        db_file = tmp_path / "test_query.db"
        tracker = LocalExperimentTracker(db_path=db_file, experiment_name="sweep_exp")

        # Run 1
        r1 = tracker.start_run(run_name="run_1", params={"lr": 0.01})
        tracker.log_metric("roc_auc_mean", 0.78)
        tracker.end_run()

        # Run 2
        r2 = tracker.start_run(run_name="run_2", params={"lr": 0.001})
        tracker.log_metric("roc_auc_mean", 0.91)
        tracker.end_run()

        runs = tracker.list_runs("sweep_exp")
        assert len(runs) == 2

        best = tracker.get_best_run("roc_auc_mean", higher_is_better=True)
        assert best is not None
        assert best["run_id"] == r2
        assert best["metrics"]["roc_auc_mean"] == 0.91
