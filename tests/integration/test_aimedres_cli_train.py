#!/usr/bin/env python3
"""Integration test for the aimedres CLI train command.

This test verifies that the command:
  aimedres train --parallel --max-workers 6 --epochs 50 --folds 5 --batch 128
works correctly across platforms.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_CMD = [sys.executable, str(_REPO_ROOT / "aimedres")]


def test_aimedres_train_basic():
    """Test basic aimedres train command with dry-run."""
    result = subprocess.run(
        _CLI_CMD
        + [
            "train",
            "--parallel",
            "--max-workers",
            "6",
            "--epochs",
            "50",
            "--folds",
            "5",
            "--batch",
            "128",
            "--dry-run",
            "--only",
            "als",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"FAILED with exit code {result.returncode}: {result.stderr}"
    output = result.stdout + result.stderr

    assert "--epochs 50" in output, "--epochs 50 not in command"
    assert "--folds 5" in output, "--folds 5 not in command"
    assert "--batch-size 128" in output, "--batch 128 parameter applied (as --batch-size)"


def test_aimedres_train_help():
    """Test that aimedres train --help works."""
    result = subprocess.run(
        _CLI_CMD + ["train", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"FAILED with exit code {result.returncode}: {result.stderr}"
    output = result.stdout + result.stderr

    required_params = ["--parallel", "--max-workers", "--epochs", "--folds", "--batch"]
    for param in required_params:
        assert param in output, f"{param} not in help text"


def test_aimedres_train_list():
    """Test that aimedres train --list works with parameters."""
    result = subprocess.run(
        _CLI_CMD
        + [
            "train",
            "--parallel",
            "--max-workers",
            "6",
            "--epochs",
            "50",
            "--folds",
            "5",
            "--batch",
            "128",
            "--list",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"FAILED with exit code {result.returncode}: {result.stderr}"
    output = result.stdout + result.stderr

    key_jobs = ["als", "alzheimers", "parkinsons"]
    for job in key_jobs:
        assert f"- {job}:" in output.lower(), f"{job.upper()} job not found in list output"


def test_aimedres_train_multiple_jobs():
    """Test aimedres train with multiple jobs in parallel."""
    result = subprocess.run(
        _CLI_CMD
        + [
            "train",
            "--parallel",
            "--max-workers",
            "6",
            "--epochs",
            "50",
            "--folds",
            "5",
            "--batch",
            "128",
            "--dry-run",
            "--only",
            "als",
            "alzheimers",
            "parkinsons",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
    )

    assert result.returncode == 0, f"FAILED with exit code {result.returncode}: {result.stderr}"
    output = result.stdout + result.stderr

    job_commands = 0
    for job in ["als", "alzheimers", "parkinsons"]:
        if f"[{job}] (dry-run) Command:" in output:
            job_commands += 1

    assert job_commands == 3, f"Expected 3 job commands, got {job_commands}"

    epochs_count = output.count("--epochs 50")
    assert epochs_count >= 3, f"--epochs 50 should appear at least 3 times, found {epochs_count}"


if __name__ == "__main__":
    test_aimedres_train_help()
    test_aimedres_train_basic()
    test_aimedres_train_list()
    test_aimedres_train_multiple_jobs()
    print("ALL TESTS PASSED")
