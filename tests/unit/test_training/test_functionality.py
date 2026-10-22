"""Executable checks for the supported Alzheimer training entry points."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from files.training.train_alzheimers import AlzheimerTrainingPipeline


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def write_fixture_dataset(path: Path, n_samples: int = 60) -> Path:
    rng = np.random.default_rng(42)
    dataset = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(n_samples)],
            "age": rng.integers(50, 90, n_samples),
            "gender": rng.choice(["M", "F"], n_samples),
            "education": rng.integers(8, 20, n_samples),
            "mmse_score": rng.integers(10, 30, n_samples),
            "cdr_score": rng.choice([0.0, 0.5, 1.0, 2.0], n_samples),
            "diagnosis": np.resize(["Normal", "MCI", "Dementia"], n_samples),
        }
    )
    dataset.to_csv(path, index=False)
    return path


def test_basic_training(tmp_path):
    pipeline = AlzheimerTrainingPipeline(output_dir=tmp_path)
    dataset_path = write_fixture_dataset(tmp_path / "alzheimer.csv")

    report = pipeline.run_full_pipeline(
        data_path=dataset_path,
        target_column="diagnosis",
        epochs=1,
        n_folds=2,
    )

    assert report["classical_models"]
    assert "neural_network" in report


def test_custom_dataset_compatibility(tmp_path):
    dataset_path = tmp_path / "alzheimer.csv"
    write_fixture_dataset(dataset_path, n_samples=100)
    pipeline = AlzheimerTrainingPipeline(output_dir=tmp_path / "output")

    pipeline.load_data(dataset_path)
    features, target = pipeline.preprocess_data(target_column="diagnosis")

    assert features.shape[0] == 100
    assert len(target) == 100
    assert len(np.unique(target)) == 3


def test_command_line_interface():
    script = REPOSITORY_ROOT / "files" / "training" / "train_alzheimers.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Alzheimer's Disease Classification Training Pipeline" in result.stdout
