"""
Structured Local Experiment Tracker and Model Lineage Store.
Conforms to Phase 2.3 of the Consolidated Roadmap.

Provides verifiable cryptographic lineage, parameter logging, metric tracking,
artifact checksums, and SQLite/JSON persistence without heavy cloud dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .parkinsons.manifest import compute_file_sha256

logger = logging.getLogger(__name__)


def _get_git_commit_sha() -> str:
    """Safely obtain the current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "untracked_workspace"


class LocalExperimentTracker:
    """
    Lightweight, verifiable experiment tracker persisting to SQLite and JSON run cards.
    Tracks hyperparameters, evaluation metrics (fold sweeps, bootstrap CIs, calibration),
    and artifact provenance with SHA-256 hashes.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "results/experiments.db",
        experiment_name: str = "parkinsons_progression",
    ):
        self.db_path = Path(db_path).resolve()
        self.experiment_name = experiment_name
        self.active_run_id: Optional[str] = None
        self._ensure_database_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Create database connection with foreign key support."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_database_schema(self) -> None:
        """Initialize SQLite relational tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    run_name TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    git_commit TEXT,
                    dataset_sha256 TEXT,
                    output_dir TEXT,
                    tags_json TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS params (
                    run_id TEXT NOT NULL,
                    param_key TEXT NOT NULL,
                    param_value TEXT,
                    PRIMARY KEY (run_id, param_key),
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    sha256 TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def start_run(
        self,
        run_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        dataset_sha256: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Begin a new tracking run.

        Args:
            run_name: Optional descriptive human-readable label.
            params: Dictionary of initial hyperparameters to log.
            tags: Arbitrary metadata tags.
            dataset_sha256: Cryptographic hash of input training data.
            output_dir: Destination directory for artifacts.

        Returns:
            Unique run ID (UUID4).
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        git_sha = _get_git_commit_sha()
        tags_json = json.dumps(tags or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_name, run_name, start_time,
                    status, git_commit, dataset_sha256, output_dir, tags_json
                ) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self.experiment_name,
                    run_name or run_id,
                    now,
                    git_sha,
                    dataset_sha256,
                    str(output_dir) if output_dir else None,
                    tags_json,
                ),
            )
            conn.commit()

        self.active_run_id = run_id
        logger.info(f"Started experiment run {run_id} under experiment '{self.experiment_name}'")

        if params:
            self.log_params(params, run_id=run_id)

        return run_id

    def log_param(self, key: str, value: Any, run_id: Optional[str] = None) -> None:
        """Log a single parameter string value."""
        target_run = run_id or self.active_run_id
        if not target_run:
            raise ValueError("No active run found. Call start_run() first or specify run_id.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO params (run_id, param_key, param_value)
                VALUES (?, ?, ?)
                """,
                (target_run, str(key), str(value)),
            )
            conn.commit()

    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """Log multiple parameters simultaneously."""
        for k, v in params.items():
            self.log_param(k, v, run_id=run_id)

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Log a numeric evaluation metric with optional step/epoch index."""
        target_run = run_id or self.active_run_id
        if not target_run:
            raise ValueError("No active run found. Call start_run() first or specify run_id.")

        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (run_id, metric_name, metric_value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (target_run, str(name), float(value), step, now),
            )
            conn.commit()

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Log a dictionary of numeric metrics, flattening nested dicts."""
        target_run = run_id or self.active_run_id

        def _flatten_and_log(d: Dict[str, Any], prefix: str = ""):
            for k, v in d.items():
                m_name = f"{prefix}{k}" if prefix else str(k)
                if isinstance(v, (int, float)):
                    self.log_metric(m_name, float(v), step=step, run_id=target_run)
                elif isinstance(v, dict):
                    _flatten_and_log(v, prefix=f"{m_name}_")

        _flatten_and_log(metrics)

    def log_artifact(
        self,
        name: str,
        path: Union[str, Path],
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record a model checkpoint or data artifact with cryptographic SHA-256.

        Returns:
            The computed SHA-256 hash or None if file does not exist.
        """
        target_run = run_id or self.active_run_id
        if not target_run:
            raise ValueError("No active run found. Call start_run() first or specify run_id.")

        file_path = Path(path).resolve()
        sha256 = compute_file_sha256(file_path) if file_path.is_file() else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO artifacts (run_id, artifact_name, artifact_path, sha256)
                VALUES (?, ?, ?, ?)
                """,
                (target_run, str(name), str(file_path), sha256),
            )
            conn.commit()

        return sha256

    def end_run(
        self,
        status: str = "COMPLETED",
        run_id: Optional[str] = None,
        export_card: bool = True,
    ) -> Dict[str, Any]:
        """
        Conclude an experiment run and optionally export its JSON run card.

        Returns:
            Dictionary containing the run summary.
        """
        target_run = run_id or self.active_run_id
        if not target_run:
            raise ValueError("No active run found to end.")

        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs
                SET end_time = ?, status = ?
                WHERE run_id = ?
                """,
                (now, status, target_run),
            )
            conn.commit()

        run_summary = self.get_run(target_run) or {}

        if export_card and run_summary.get("output_dir"):
            try:
                card_dir = Path(run_summary["output_dir"]) / "runs"
                card_dir.mkdir(parents=True, exist_ok=True)
                card_file = card_dir / f"{target_run}.json"
                with open(card_file, "w", encoding="utf-8") as f:
                    json.dump(run_summary, f, indent=2)
                logger.info(f"Exported run card to {card_file}")
            except Exception as e:
                logger.warning(f"Could not export JSON run card: {e}")

        if self.active_run_id == target_run:
            self.active_run_id = None

        return run_summary

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full structured run details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            run_row = cursor.fetchone()
            if not run_row:
                return None

            run_dict = dict(run_row)
            try:
                run_dict["tags"] = json.loads(run_dict.pop("tags_json") or "{}")
            except Exception:
                run_dict["tags"] = {}

            # Parameters
            cursor.execute("SELECT param_key, param_value FROM params WHERE run_id = ?", (run_id,))
            run_dict["params"] = {row["param_key"]: row["param_value"] for row in cursor.fetchall()}

            # Metrics
            cursor.execute(
                "SELECT metric_name, metric_value, step, timestamp FROM metrics WHERE run_id = ?",
                (run_id,),
            )
            metrics_list = [dict(row) for row in cursor.fetchall()]
            run_dict["metrics_history"] = metrics_list
            # Latest value per metric
            run_dict["metrics"] = {m["metric_name"]: m["metric_value"] for m in metrics_list}

            # Artifacts
            cursor.execute(
                "SELECT artifact_name, artifact_path, sha256 FROM artifacts WHERE run_id = ?",
                (run_id,),
            )
            run_dict["artifacts"] = [dict(row) for row in cursor.fetchall()]

            return run_dict

    def list_runs(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List summary of all recorded runs."""
        exp = experiment_name or self.experiment_name
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT run_id, experiment_name, run_name, start_time, end_time, status, git_commit
                FROM runs
                WHERE experiment_name = ?
                ORDER BY start_time DESC
                """,
                (exp,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_best_run(
        self,
        metric_name: str = "roc_auc_mean",
        higher_is_better: bool = True,
        experiment_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query the best run by a specific evaluation metric."""
        runs = self.list_runs(experiment_name=experiment_name)
        best_run_id = None
        best_value: Optional[float] = None

        for r in runs:
            details = self.get_run(r["run_id"])
            if not details:
                continue
            val = details["metrics"].get(metric_name)
            if val is None:
                continue

            if best_value is None:
                best_value = val
                best_run_id = r["run_id"]
            elif higher_is_better and val > best_value:
                best_value = val
                best_run_id = r["run_id"]
            elif not higher_is_better and val < best_value:
                best_value = val
                best_run_id = r["run_id"]

        return self.get_run(best_run_id) if best_run_id else None
