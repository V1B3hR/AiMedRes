"""Tests for the committed, research-only PPMI benchmark configuration."""

from pathlib import Path

import pytest

from scripts.reproduce_ppmi_benchmark import (
    BenchmarkConfig,
    _REPO_ROOT,
    build_parser,
    load_benchmark_config,
    resolve_benchmark_config,
    write_evidence_pack,
)


def test_committed_study_configuration_is_runnable():
    config = load_benchmark_config(_REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml")

    assert config.data_path == _REPO_ROOT / "ParkinsonDatasets" / "synthetic_pet_sbr_fixture.csv"
    assert config.data_path.is_file()
    assert config.output_dir == _REPO_ROOT / "results" / "ppmi_reproducibility_benchmark"
    assert config.target_column == "motor_progression_label"
    assert config.n_folds == 5
    assert config.epochs == 25
    assert config.random_state == 42
    assert config.n_bootstraps == 1000
    assert config.alpha == 0.05
    assert config.batch_size == 16
    assert config.learning_rate == 0.001
    assert config.source_config_path == _REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"


def test_cli_overrides_do_not_mutate_the_committed_study_configuration(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(_REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"),
            "--output-dir",
            str(tmp_path / "benchmark_output"),
            "--folds",
            "3",
            "--epochs",
            "2",
        ]
    )

    config = resolve_benchmark_config(args)

    assert config.output_dir == tmp_path / "benchmark_output"
    assert config.n_folds == 3
    assert config.epochs == 2
    assert config.random_state == 42
    assert config.batch_size == 16
    assert config.learning_rate == 0.001
    assert config.data_path == _REPO_ROOT / "ParkinsonDatasets" / "synthetic_pet_sbr_fixture.csv"
    assert config.source_config_path == _REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"


def test_study_configuration_requires_an_explicit_research_only_intended_use(tmp_path):
    config_path = tmp_path / "unsafe_study.yml"
    config_path.write_text(
        """
version: 1
study:
  name: unsafe-study
  intended_use: diagnosis
data:
  path: ParkinsonDatasets/synthetic_pet_sbr_fixture.csv
pipeline:
  target_column: motor_progression_label
  n_folds: 5
  epochs: 25
output:
  directory: results/test
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="research-only purpose"):
        load_benchmark_config(config_path)


def test_evidence_pack_is_limited_to_the_committed_synthetic_fixture(tmp_path):
    config = load_benchmark_config(_REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml")
    summary = {
        "timestamp": "2026-10-25T00:00:00Z",
        "dataset": "ParkinsonDatasets/synthetic_pet_sbr_fixture.csv",
        "target": "motor_progression_label",
        "n_folds": 5,
        "execution_seconds": 1.0,
        "summary": [
            {
                "model": "Logistic Regression",
                "roc_auc": 0.9,
                "roc_auc_ci_95": [0.8, 0.95],
                "brier_score": 0.1,
            }
        ],
    }

    manifest_path = write_evidence_pack(tmp_path / "evidence", config, summary)

    assert manifest_path.is_file()
    assert (manifest_path.parent / "EVIDENCE_PACK.md").is_file()
    assert "Synthetic-fixture reproducibility evidence only" in manifest_path.read_text(encoding="utf-8")

    non_synthetic = BenchmarkConfig(
        **{**config.__dict__, "data_path": tmp_path / "controlled.csv"}
    )
    non_synthetic.data_path.write_text("PATNO,motor_progression_label\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="restricted to the committed synthetic fixture"):
        write_evidence_pack(tmp_path / "controlled_evidence", non_synthetic, summary)


def test_evidence_pack_rejects_parameter_overrides(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(_REPO_ROOT / "configs" / "parkinsons_ppmi_study.yml"),
            "--folds",
            "3",
        ]
    )
    config = resolve_benchmark_config(args)
    summary = {
        "timestamp": "2026-10-25T00:00:00Z",
        "dataset": "ParkinsonDatasets/synthetic_pet_sbr_fixture.csv",
        "target": "motor_progression_label",
        "n_folds": 3,
        "execution_seconds": 1.0,
        "summary": [],
    }

    with pytest.raises(ValueError, match="canonical synthetic-study parameters"):
        write_evidence_pack(tmp_path / "evidence", config, summary)
