#!/usr/bin/env python3
"""
Unified Medical AI Training Orchestrator (Enhanced)

Features:
- Configurable job list (inline or YAML)
- Robust logging (file + console), per-job logs
- Structured JSON summary + manifest
- Optional parallel execution
- Retry logic
- Environment + GPU detection
- Command-line flexibility (filters, dry-run, list)
- Git commit + environment metadata capture
- Graceful interrupt handling
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# --------------- Data Structures -----------------

@dataclass
class TrainingJob:
    name: str
    script: str
    output: str
    id: str
    args: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False  # If True, failure won't trigger non-zero exit
    status: str = "PENDING"  # Updated dynamically
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None
    attempts: int = 0
    error: Optional[str] = None
    command: List[str] = field(default_factory=list)

    def build_command(
        self,
        python_exec: str,
        global_epochs: Optional[int],
        global_folds: Optional[int],
        extra_args: List[str],
        base_output_dir: Path,
    ) -> List[str]:
        cmd = [python_exec, self.script, "--output-dir", str(base_output_dir / self.output)]
        # Merge args: explicit job-level overrides
        epochs = self.args.get("epochs", global_epochs)
        folds = self.args.get("folds", global_folds)
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]
        if folds is not None:
            cmd += ["--folds", str(folds)]
        # Convert remaining key-value args
        for k, v in self.args.items():
            if k in ("epochs", "folds"):
                continue
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]
        # Append global extra args at end
        cmd.extend(extra_args)
        self.command = cmd
        return cmd


# --------------- Global State -----------------

INTERRUPTED = False
INTERRUPT_LOCK = threading.Lock()

def handle_interrupt(signum, frame):
    global INTERRUPTED
    with INTERRUPT_LOCK:
        if not INTERRUPTED:
            print("\n⚠️  Interrupt signal received. Finishing current tasks gracefully...")
            INTERRUPTED = True
        else:
            print("Second interrupt received. Exiting immediately.")
            sys.exit(130)

signal.signal(signal.SIGINT, handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

# --------------- Logging Setup -----------------

def setup_logging(log_root: Path, verbose: bool = False) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if re-run in same interpreter
    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console_fmt = logging.Formatter("%(message)s")
    console.setFormatter(console_fmt)

    file_handler = logging.FileHandler(log_root / "orchestrator.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

def get_job_logger(job: TrainingJob, log_root: Path) -> logging.Logger:
    logger_name = f"trainer.job.{job.id}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    job_dir = log_root / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(job_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# --------------- Utility Functions -----------------

def detect_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None

def detect_gpu() -> Dict[str, Any]:
    info = {"framework": None, "cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch  # noqa: F401
        import torch.cuda as cuda
        info["framework"] = "torch"
        info["cuda_available"] = cuda.is_available()
        if info["cuda_available"]:
            count = cuda.device_count()
            info["device_count"] = count
            for i in range(count):
                info["devices"].append(cuda.get_device_name(i))
    except Exception:
        # Try nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
            names = [l.strip() for l in smi.stdout.splitlines() if l.strip()]
            if names:
                info["framework"] = "system"
                info["cuda_available"] = True
                info["device_count"] = len(names)
                info["devices"] = names
        except Exception:
            pass
    return info

def load_config_yaml(path: Path) -> List[TrainingJob]:
    import yaml  # Lazy import
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    jobs = []
    for item in raw.get("jobs", []):
        jobs.append(
            TrainingJob(
                name=item["name"],
                script=item["script"],
                output=item["output"],
                id=item.get("id") or derive_id(item["name"]),
                args=item.get("args", {}) or {},
                optional=item.get("optional", False),
            )
        )
    return jobs

def derive_id(name: str) -> str:
    return (
        name.lower()
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("&", "and")
        .replace(" ", "_")
    )

def default_jobs() -> List[TrainingJob]:
    return [
        TrainingJob(
            name="ALS (Amyotrophic Lateral Sclerosis)",
            script="training/train_als.py",
            output="als_comprehensive_results",
            id="als",
            args={"dataset-choice": "als-progression"},
        ),
        TrainingJob(
            name="Alzheimer's Disease",
            script="files/training/train_alzheimers.py",
            output="alzheimer_comprehensive_results",
            id="alzheimers",
            args={},
        ),
        TrainingJob(
            name="Parkinson's Disease",
            script="training/train_parkinsons.py",
            output="parkinsons_comprehensive_results",
            id="parkinsons",
            args={"dataset-choice": "vikasukani"},
        ),
    ]

def filter_jobs(
    jobs: List[TrainingJob],
    only: List[str],
    exclude: List[str],
) -> List[TrainingJob]:
    result = []
    only_set = {o.lower() for o in only} if only else None
    exclude_set = {e.lower() for e in exclude} if exclude else set()
    for job in jobs:
        jid = job.id.lower()
        if only_set and jid not in only_set:
            continue
        if jid in exclude_set:
            continue
        result.append(job)
    return result

# --------------- Core Execution -----------------

def run_job(
    job: TrainingJob,
    repo_root: Path,
    base_output_dir: Path,
    python_exec: str,
    global_epochs: Optional[int],
    global_folds: Optional[int],
    extra_args: List[str],
    retries: int,
    dry_run: bool,
    orchestrator_logger: logging.Logger,
    log_root: Path,
) -> TrainingJob:
    job_logger = get_job_logger(job, log_root)
    script_path = repo_root / job.script
    if not script_path.exists():
        job.status = "FAILED"
        job.error = f"Script not found: {script_path}"
        orchestrator_logger.error(f"[{job.id}] ❌ Script missing: {script_path}")
        return job

    job.build_command(python_exec, global_epochs, global_folds, extra_args, base_output_dir)

    if dry_run:
        orchestrator_logger.info(f"[{job.id}] (dry-run) Command: {' '.join(job.command)}")
        job.status = "SKIPPED"
        return job

    attempts_allowed = retries + 1
    for attempt in range(1, attempts_allowed + 1):
        if INTERRUPTED:
            job.status = "INTERRUPTED"
            orchestrator_logger.warning(f"[{job.id}] ⏹ Interrupted before start (attempt {attempt}).")
            return job

        job.attempts = attempt
        job.start_time = datetime.utcnow().isoformat()
        start_t = time.time()
        orchestrator_logger.info(f"[{job.id}] 🚀 Starting attempt {attempt}/{attempts_allowed}")
        job_logger.info(f"Command: {' '.join(job.command)}")

        try:
            proc = subprocess.run(
                job.command,
                cwd=str(repo_root),
                stdout=job_logger.handlers[0].stream,  # direct to file
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
            code = proc.returncode
            job.end_time = datetime.utcnow().isoformat()
            job.duration_sec = round(time.time() - start_t, 2)
            if code == 0:
                job.status = "SUCCESS"
                orchestrator_logger.info(
                    f"[{job.id}] ✅ Success in {job.duration_sec:.2f}s (attempt {attempt})"
                )
                break
            else:
                job.status = "FAILED"
                job.error = f"Non-zero exit code: {code}"
                orchestrator_logger.error(
                    f"[{job.id}] ❌ Failure (code={code}) attempt {attempt}/{attempts_allowed}"
                )
        except Exception as e:
            job.end_time = datetime.utcnow().isoformat()
            job.duration_sec = round(time.time() - start_t, 2)
            job.status = "FAILED"
            job.error = repr(e)
            orchestrator_logger.exception(f"[{job.id}] ❌ Exception attempt {attempt}: {e}")

        if job.status != "SUCCESS" and attempt < attempts_allowed:
            orchestrator_logger.info(f"[{job.id}] 🔁 Retrying in 2s...")
            time.sleep(2)

    return job

# --------------- Summary & Reporting -----------------

def summarize(
    jobs: List[TrainingJob],
    start_time: str,
    repo_root: Path,
    summary_dir: Path,
    gpu_info: Dict[str, Any],
    git_commit: Optional[str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    summary_dir.mkdir(parents=True, exist_ok=True)
    end_time = datetime.utcnow().isoformat()
    summary = {
        "pipeline": "AiMedRes Medical AI Training",
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "duration_sec": (datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds(),
        "git_commit": git_commit,
        "python": sys.version.replace("\n", " "),
        "working_directory": str(repo_root),
        "gpu_info": gpu_info,
        "arguments": vars(args),
        "jobs": [
            {
                "id": j.id,
                "name": j.name,
                "status": j.status,
                "attempts": j.attempts,
                "start_time": j.start_time,
                "end_time": j.end_time,
                "duration_sec": j.duration_sec,
                "optional": j.optional,
                "command": j.command,
                "error": j.error,
            }
            for j in jobs
        ],
    }
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = summary_dir / f"training_summary_{timestamp}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"📄 Summary written: {out_file}")
    return out_file

def print_console_summary(jobs: List[TrainingJob], logger: logging.Logger):
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 Training Pipeline Summary")
    logger.info("=" * 80)
    success = sum(1 for j in jobs if j.status == "SUCCESS")
    failed = [j for j in jobs if j.status == "FAILED"]
    skipped = [j for j in jobs if j.status == "SKIPPED"]
    interrupted = [j for j in jobs if j.status == "INTERRUPTED"]

    logger.info(f"✅ Successful: {success}")
    if failed:
        logger.info(f"❌ Failed: {len(failed)} -> {', '.join(j.id for j in failed)}")
    if skipped:
        logger.info(f"⏭ Skipped: {len(skipped)} -> {', '.join(j.id for j in skipped)}")
    if interrupted:
        logger.info(f"⏹ Interrupted: {len(interrupted)} -> {', '.join(j.id for j in interrupted)}")

    if failed:
        logger.info("⚠️  Some training pipelines failed. Inspect logs for details.")
    elif interrupted:
        logger.info("⚠️  Pipeline interrupted before completing all jobs.")
    else:
        logger.info("🎉 All selected training pipelines completed successfully!")

# --------------- Argument Parsing -----------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified orchestrator for medical AI training jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config with job definitions.")
    parser.add_argument("--epochs", type=int, help="Global default epochs (overridden per-job if specified).")
    parser.add_argument("--folds", type=int, help="Global default folds (overridden per-job if specified).")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these job IDs (space separated).")
    parser.add_argument("--exclude", nargs="*", default=[], help="Exclude these job IDs.")
    parser.add_argument("--list", action="store_true", help="List available jobs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution.")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum workers when --parallel is set.")
    parser.add_argument("--retries", type=int, default=0, help="Retry attempts per job on failure.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw args appended to all commands (use multiple times).")
    parser.add_argument("--base-output-dir", type=str, default="results", help="Base directory for training outputs.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for logs.")
    parser.add_argument("--summary-dir", type=str, default="summaries", help="Directory for JSON summary files.")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    parser.add_argument("--allow-partial-success", action="store_true",
                        help="Exit with code 0 even if some non-optional jobs fail.")
    return parser.parse_args()

# --------------- Main Orchestration -----------------

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    start_time = datetime.utcnow().isoformat()

    # Logging
    logs_dir = repo_root / args.logs_dir
    logger = setup_logging(logs_dir, verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("AiMedRes Comprehensive Medical AI Training Pipeline (Enhanced)")
    logger.info("=" * 80)
    logger.info(f"⏰ Started at (UTC): {start_time}")

    # Environment / metadata
    git_commit = detect_git_commit(repo_root)
    if git_commit:
        logger.info(f"🔐 Git commit: {git_commit}")
    gpu_info = detect_gpu()
    if gpu_info["cuda_available"]:
        logger.info(f"🧮 GPU(s): {gpu_info['devices']}")
    else:
        logger.info("🧮 GPU: Not detected (running on CPU)")

    # Load jobs
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logger.error(f"Config file not found: {cfg_path}")
            return 2
        try:
            jobs = load_config_yaml(cfg_path)
            logger.info(f"🗂 Loaded {len(jobs)} jobs from config.")
        except Exception as e:
            logger.exception(f"Failed to parse config: {e}")
            return 2
    else:
        jobs = default_jobs()
        logger.info(f"🗂 Using built-in job definitions: {len(jobs)}")

    # Filter
    original_count = len(jobs)
    jobs = filter_jobs(jobs, args.only, args.exclude)
    logger.info(f"🎯 Selected jobs: {len(jobs)} (filtered from {original_count})")
    if args.list:
        logger.info("")
        logger.info("Available Jobs:")
        for j in jobs:
            logger.info(f"- {j.id}: {j.name}  (script={j.script})")
        return 0

    if not jobs:
        logger.warning("No jobs selected. Exiting.")
        return 0

    base_output_dir = repo_root / args.base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Extra args sanitized: allow raw pass-through (already tokenized)
    extra_args = args.extra_arg

    # Run
    executed_jobs: List[TrainingJob] = []
    if args.parallel and len(jobs) > 1:
        logger.info("⚠️  Parallel mode enabled. Ensure GPU memory can handle multiple runs.")
        max_workers = min(args.max_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job") as pool:
            future_map = {
                pool.submit(
                    run_job,
                    job,
                    repo_root,
                    base_output_dir,
                    sys.executable,
                    args.epochs,
                    args.folds,
                    extra_args,
                    args.retries,
                    args.dry_run,
                    logger,
                    logs_dir,
                ): job
                for job in jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    result_job = future.result()
                    executed_jobs.append(result_job)
                except Exception as e:
                    logger.exception(f"[{job.id}] Unhandled exception: {e}")
    else:
        for job in jobs:
            if INTERRUPTED:
                job.status = "INTERRUPTED"
                executed_jobs.append(job)
                continue
            result_job = run_job(
                job,
                repo_root,
                base_output_dir,
                sys.executable,
                args.epochs,
                args.folds,
                extra_args,
                args.retries,
                args.dry_run,
                logger,
                logs_dir,
            )
            executed_jobs.append(result_job)

    # Summary
    summarize(
        executed_jobs,
        start_time,
        repo_root,
        repo_root / args.summary_dir,
        gpu_info,
        git_commit,
        args,
        logger,
    )
    print_console_summary(executed_jobs, logger)

    # Exit code logic
    non_optional_failures = [
        j for j in executed_jobs if j.status not in ("SUCCESS", "SKIPPED") and not j.optional
    ]
    if INTERRUPTED and non_optional_failures:
        logger.info("Exit due to interrupt with failures.")
        return 130
    if non_optional_failures and not args.allow_partial_success:
        logger.info("Exiting with non-zero due to failed mandatory jobs.")
        return 1
    logger.info("Exiting successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
