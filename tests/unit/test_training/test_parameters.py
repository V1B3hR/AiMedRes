"""Regression tests for training-command parameters."""

import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def run_training_command(*arguments: str) -> str:
    result = subprocess.run(
        [sys.executable, "run_all_training.py", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_sample_parameter_in_help():
    output = run_training_command("--help")

    assert "--sample SAMPLE" in output
    assert "Global default sample size (if supported)" in output


def test_sample_parameter_is_accepted():
    run_training_command("--dry-run", "--sample", "3000", "--only", "als")


def test_parallel_dry_run_propagates_requested_parameters():
    output = run_training_command(
        "--parallel",
        "--max-workers",
        "6",
        "--epochs",
        "70",
        "--folds",
        "5",
        "--sample",
        "3000",
        "--dry-run",
        "--only",
        "als",
        "alzheimers",
    )

    assert "Parallel mode enabled" in output
    assert "--epochs 70" in output
    assert "--folds 5" in output
    assert "[als] (dry-run) Command:" in output
    assert "[alzheimers] (dry-run) Command:" in output


def test_list_displays_sample_support():
    output = run_training_command("--list", "--only", "als")

    assert "sample=" in output
