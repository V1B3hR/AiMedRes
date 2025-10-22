#!/usr/bin/env python3
"""
Unified Medical AI Training Orchestrator (Enhanced + Auto-Discovery)

New in this version:
- Automatic discovery of training scripts (train_*.py) across the repository.
- Heuristic inference of:
    * Job name
    * Output directory
    * Whether script supports --epochs / --folds / --output-dir
- Merges discovered jobs with (optional) built-in defaults or YAML config.
- CLI controls for discovery behavior & filtering patterns.

Discovery Heuristics:
1. A "training script" is any Python file whose basename matches:
      train_*.py   (starts with 'train_')
   You may broaden via --include-pattern.
2. Skipped directories (default): .git, __pycache__, venv, env, .venv, build, dist, .mypy_cache, .pytest_cache, .idea, .vscode, node_modules
   Also skips legacy/duplicate paths: files/training, training (use src/aimedres/training instead)
3. For each candidate file, we read the first ~40000 characters to look for argparse definitions or literal flag strings:
      '--epochs'   -> supports_epochs
      '--folds'    -> supports_folds
      '--output-dir' or '--output_dir' -> use_output_dir
4. Output directory is derived from base name minus 'train_' prefix + '_results'
5. Job id is the sanitized base filename (without .py)

Priority / Merge Order:
- If --config provided: jobs from config are loaded first.
- If auto-discovery (enabled by default) runs, newly discovered job IDs
  that clash with previously defined jobs are skipped unless
  --allow-discovery-overrides is set (then discovery overwrites earlier one).
- If neither config nor discovery yields anything, built-in default_jobs() is used.

Examples:
  List discovered jobs only:
    python run_all_training.py --list

  Disable auto-discovery (use only defaults or config):
    python run_all_training.py --no-auto-discover

  Restrict discovery to 'training/' and 'files/training':
    python run_all_training.py --discover-root training files/training

  Include additional pattern (e.g., any file containing 'model_train'):
    python run_all_training.py --include-pattern model_train

  Exclude certain scripts by ID:
    python run_all_training.py --exclude alzheimers_enhanced

  Parallel run of discovered + defaults:
    python run_all_training.py --parallel --max-workers 4

  Append raw args to every job:
    python run_all_training.py --extra-arg --batch-size=32 --extra-arg --learning-rate=3e-4

Caveats:
- Some scripts (e.g., MLOps pipelines) may not accept --output-dir; heuristic attempts to detect this.
- If detection is wrong, adjust with a YAML config or extend heuristics.
- Discovery does not currently inspect subcommands or dynamic argparse setups.

"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable


# --------------- Data Structures -----------------

@dataclass
class TrainingJob:
    name: str
    script: str
    output: str
    id: str
    args: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False

    # Extended compatibility flags
    use_output_dir: bool = True
    supports_epochs: bool = True
    supports_folds: bool = True
    supports_sample: bool = True
    supports_batch: bool = True

    # Runtime metadata
    status: str = "PENDING"
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
        global_sample: Optional[int],
        global_batch: Optional[int],
        extra_args: List[str],
        base_output_dir: Path,
    ) -> List[str]:
        cmd = [python_exec, self.script]
        if self.use_output_dir:
            cmd += ["--output-dir", str(base_output_dir / self.output)]

        # Respect per-job override vs global
        if self.supports_epochs:
            epochs = self.args.get("epochs", global_epochs)
            if epochs is not None:
                cmd += ["--epochs", str(epochs)]
        if self.supports_folds:
            folds = self.args.get("folds", global_folds)
            if folds is not None:
                cmd += ["--folds", str(folds)]
        if self.supports_sample:
            sample = self.args.get("sample", global_sample)
            if sample is not None:
                cmd += ["--sample", str(sample)]
        if self.supports_batch:
            batch = self.args.get("batch", global_batch)
            if batch is not None:
                cmd += ["--batch-size", str(batch)]

        # Other args
        for k, v in self.args.items():
            if k in ("epochs", "folds", "sample", "batch"):
                continue
            flag = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd += [flag, str(v)]

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
    fh = logging.FileHandler(
        job_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        mode="w",
        encoding="utf-8"
    )
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
                use_output_dir=item.get("use_output_dir", True),
                supports_epochs=item.get("supports_epochs", True),
                supports_folds=item.get("supports_folds", True),
                supports_sample=item.get("supports_sample", True),
                supports_batch=item.get("supports_batch", True),
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


def humanize_script_name(path: Path) -> str:
    base = path.stem  # e.g., train_alzheimers
    if base.startswith("train_"):
        base = base[6:]
    base = base.replace("_", " ")
    return base.title().strip()


def default_jobs() -> List[TrainingJob]:
    # Core disease prediction models - all 7 main models
    # Using canonical location: src/aimedres/training/
    return [
        TrainingJob(
            name="ALS (Amyotrophic Lateral Sclerosis)",
            script="src/aimedres/training/train_als.py",
            output="als_comprehensive_results",
            id="als",
            args={"dataset-choice": "als-progression"},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=True,  # Batch size parameter supported as --batch-size
        ),
        TrainingJob(
            name="Alzheimer's Disease",
            script="src/aimedres/training/train_alzheimers.py",
            output="alzheimer_comprehensive_results",
            id="alzheimers",
            args={},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
        TrainingJob(
            name="Parkinson's Disease",
            script="src/aimedres/training/train_parkinsons.py",
            output="parkinsons_comprehensive_results",
            id="parkinsons",
            args={"data-path": "ParkinsonDatasets"},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
        TrainingJob(
            name="Brain MRI Classification",
            script="src/aimedres/training/train_brain_mri.py",
            output="brain_mri_comprehensive_results",
            id="brain_mri",
            args={},
            supports_folds=False,  # Brain MRI doesn't support --folds
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
        TrainingJob(
            name="Cardiovascular Disease Prediction",
            script="src/aimedres/training/train_cardiovascular.py",
            output="cardiovascular_comprehensive_results",
            id="cardiovascular",
            args={},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
        TrainingJob(
            name="Diabetes Prediction",
            script="src/aimedres/training/train_diabetes.py",
            output="diabetes_comprehensive_results",
            id="diabetes",
            args={},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
        TrainingJob(
            name="Specialized Medical Agents",
            script="src/aimedres/training/train_specialized_agents.py",
            output="specialized_agents_comprehensive_results",
            id="specialized_agents",
            args={},
            supports_sample=False,  # Sample parameter not supported yet
            supports_batch=False,  # Batch parameter not supported yet
        ),
    ]


# --------------- Auto-Discovery Logic -----------------

SKIP_DIR_NAMES = {
    ".git", "__pycache__", "venv", "env", ".venv", "build", "dist",
    ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "node_modules"
}

# Directories to skip from the repository root to avoid discovering duplicate training scripts
# These contain legacy or duplicate versions of training scripts
SKIP_PATHS_FROM_ROOT = {
    "files/training",  # Duplicate of src/aimedres/training
    "training",        # Legacy location, use src/aimedres/training instead
}

DEFAULT_INCLUDE_PATTERNS = ["train_*.py"]


def read_head(path: Path, max_chars: int = 40000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def infer_support_flags(file_text: str) -> Dict[str, bool]:
    """
    Heuristic: if the script mentions the flag tokens, assume support.
    This avoids running the script with --help (which could have side-effects).
    """
    lower = file_text.lower()
    return {
        "supports_epochs": "--epochs" in lower,
        "supports_folds": "--folds" in lower,
        "supports_sample": "--sample" in lower,
        "supports_batch": ("--batch" in lower) or ("--batch-size" in lower),
        "use_output_dir": ("--output-dir" in lower) or ("--output_dir" in lower),
    }


def discover_training_scripts(
    roots: List[Path],
    include_patterns: List[str],
    exclude_regex: Optional[re.Pattern],
    logger: logging.Logger,
    limit: Optional[int] = None,
) -> List[Path]:
    discovered: List[Path] = []
    for root in roots:
        if not root.exists():
            logger.debug(f"[DISCOVERY] Root not found: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip directories in-place
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            
            # Also skip specific paths from root (to avoid duplicates)
            current_path = Path(dirpath)
            try:
                rel_to_root = current_path.relative_to(root)
                if str(rel_to_root) in SKIP_PATHS_FROM_ROOT or any(
                    str(rel_to_root).startswith(skip_path) for skip_path in SKIP_PATHS_FROM_ROOT
                ):
                    dirnames[:] = []  # Don't descend into this directory
                    continue
            except ValueError:
                pass  # Not relative to root, continue normally
            
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                full_path = Path(dirpath) / filename
                rel_path = full_path.relative_to(root.parent if root.parent else root)
                # Pattern match
                if not any(fnmatch.fnmatch(filename, pat) for pat in include_patterns):
                    continue
                if exclude_regex and exclude_regex.search(str(rel_path)):
                    logger.debug(f"[DISCOVERY] Excluded by regex: {rel_path}")
                    continue
                discovered.append(full_path)
                if limit and len(discovered) >= limit:
                    return discovered
    return discovered


def build_jobs_from_discovery(
    script_paths: List[Path],
    repo_root: Path,
    logger: logging.Logger,
    mark_optional_unknown: bool = True,
    base_output_suffix: str = "_results",
) -> List[TrainingJob]:
    jobs: List[TrainingJob] = []
    for sp in script_paths:
        rel_script = sp.relative_to(repo_root).as_posix()
        text = read_head(sp)
        flags = infer_support_flags(text)
        name = humanize_script_name(sp)
        job_id = derive_id(sp.stem)
        if sp.stem.startswith("train_"):
            output_base = sp.stem[6:]  # remove 'train_'
        else:
            output_base = sp.stem
        output_dir = f"{output_base}{base_output_suffix}"
        # Heuristic: mark optional for scripts outside 'training/' or that lack output-dir support
        optional = mark_optional_unknown and (not flags["use_output_dir"] or "mlops" in rel_script or "scripts/" in rel_script)
        job = TrainingJob(
            name=name,
            script=rel_script,
            output=output_dir,
            id=job_id,
            args={},
            optional=optional,
            use_output_dir=flags["use_output_dir"],
            supports_epochs=flags["supports_epochs"],
            supports_folds=flags["supports_folds"],
            supports_sample=flags["supports_sample"],
            supports_batch=flags["supports_batch"],
        )
        logger.debug(
            f"[DISCOVERY] Job: id={job.id} script={job.script} "
            f"epochs={job.supports_epochs} folds={job.supports_folds} sample={job.supports_sample} batch={job.supports_batch} out_dir={job.use_output_dir} optional={job.optional}"
        )
        jobs.append(job)
    return jobs


def merge_jobs(
    base: List[TrainingJob],
    discovered: List[TrainingJob],
    allow_overrides: bool,
    logger: logging.Logger,
) -> List[TrainingJob]:
    by_id: Dict[str, TrainingJob] = {j.id: j for j in base}
    for dj in discovered:
        if dj.id in by_id and not allow_overrides:
            logger.debug(f"[DISCOVERY] Skipping duplicate id (override disabled): {dj.id}")
            continue
        if dj.id in by_id and allow_overrides:
            logger.info(f"[DISCOVERY] Overriding existing job id={dj.id} with discovered version.")
        by_id[dj.id] = dj
    return list(by_id.values())


# --------------- Filtering & Execution -----------------

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


def run_job(
    job: TrainingJob,
    repo_root: Path,
    base_output_dir: Path,
    python_exec: str,
    global_epochs: Optional[int],
    global_folds: Optional[int],
    global_sample: Optional[int],
    global_batch: Optional[int],
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

    job.build_command(python_exec, global_epochs, global_folds, global_sample, global_batch, extra_args, base_output_dir)

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
        job.start_time = datetime.now(timezone.utc).isoformat()
        start_t = time.time()
        orchestrator_logger.info(f"[{job.id}] 🚀 Starting attempt {attempt}/{attempts_allowed}")
        job_logger.info(f"Command: {' '.join(job.command)}")

        try:
            proc = subprocess.run(
                job.command,
                cwd=str(repo_root),
                stdout=job_logger.handlers[0].stream,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
            code = proc.returncode
            job.end_time = datetime.now(timezone.utc).isoformat()
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
            job.end_time = datetime.now(timezone.utc).isoformat()
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
    end_time = datetime.now(timezone.utc).isoformat()
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
                "use_output_dir": j.use_output_dir,
                "supports_epochs": j.supports_epochs,
                "supports_folds": j.supports_folds,
                "supports_sample": j.supports_sample,
                "supports_batch": j.supports_batch,
                "command": j.command,
                "error": j.error,
            }
            for j in jobs
        ],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified orchestrator for medical AI training jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core config
    parser.add_argument("--config", type=str, help="Path to YAML config with job definitions.")
    parser.add_argument("--epochs", type=int, help="Global default epochs (if supported).")
    parser.add_argument("--folds", type=int, help="Global default folds (if supported).")
    parser.add_argument("--sample", type=int, help="Global default sample size (if supported).")
    parser.add_argument("--batch", type=int, help="Global default batch size (if supported).")
    parser.add_argument("--only", nargs="*", default=[], help="Run only these job IDs.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Exclude these job IDs.")
    parser.add_argument("--list", action="store_true", help="List selected jobs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution.")
    parser.add_argument("--max-workers", type=int, default=4, help="Workers for parallel mode.")
    parser.add_argument("--retries", type=int, default=0, help="Retry attempts per job.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw args appended to all job commands (repeatable).")
    parser.add_argument("--base-output-dir", type=str, default="results", help="Base directory for training outputs.")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for logs.")
    parser.add_argument("--summary-dir", type=str, default="summaries", help="Directory for summaries.")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    parser.add_argument("--allow-partial-success", action="store_true",
                        help="Exit 0 even if some non-optional jobs fail.")

    # Auto-discovery controls
    parser.add_argument("--no-auto-discover", action="store_true", help="Disable auto-discovery entirely.")
    parser.add_argument("--discover-root", nargs="*", default=[],
                        help="Root directories to search (default: repo root). Can pass multiple.")
    parser.add_argument("--include-pattern", action="append", default=[],
                        help="Additional filename patterns to include (e.g., *trainer.py).")
    parser.add_argument("--exclude-regex", type=str,
                        help="Regex to exclude discovered script paths.")
    parser.add_argument("--discovery-limit", type=int, help="Max number of scripts to discover (debug/testing).")
    parser.add_argument("--allow-discovery-overrides", action="store_true",
                        help="Allow auto-discovered jobs to override existing job IDs from config/default.")
    parser.add_argument("--no-default-jobs", action="store_true",
                        help="Do not load built-in default jobs (use only config + discovery).")
    parser.add_argument("--mark-discovered-optional", action="store_true",
                        help="Force all discovered jobs to be optional.")
    parser.add_argument("--strict-discovery", action="store_true",
                        help="If set, and no jobs discovered (and no config), exit with non-zero instead of using defaults.")

    return parser.parse_args(argv)


# --------------- Main Orchestration -----------------

def main(argv=None):
    args = parse_args(argv)
    # Find repository root (go up from src/aimedres/cli to the repo root)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    start_time = datetime.now(timezone.utc).isoformat()

    logs_dir = repo_root / args.logs_dir
    logger = setup_logging(logs_dir, verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("AiMedRes Comprehensive Medical AI Training Pipeline (Auto-Discovery Enabled)")
    logger.info("=" * 80)
    logger.info(f"⏰ Started at (UTC): {start_time}")

    git_commit = detect_git_commit(repo_root)
    if git_commit:
        logger.info(f"🔐 Git commit: {git_commit}")
    gpu_info = detect_gpu()
    if gpu_info["cuda_available"]:
        logger.info(f"🧮 GPU(s): {gpu_info['devices']}")
    else:
        logger.info("🧮 GPU: Not detected (running on CPU)")

    jobs: List[TrainingJob] = []

    # 1. Load from config if provided
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            logger.error(f"Config file not found: {cfg_path}")
            return 2
        try:
            cfg_jobs = load_config_yaml(cfg_path)
            logger.info(f"🗂 Loaded {len(cfg_jobs)} jobs from config.")
            jobs.extend(cfg_jobs)
        except Exception as e:
            logger.exception(f"Failed to parse config: {e}")
            return 2

    # 2. Load defaults unless suppressed
    if not args.no_default_jobs:
        base_defaults = default_jobs()
        logger.info(f"🧩 Added {len(base_defaults)} built-in default jobs.")
        jobs.extend(base_defaults)
    else:
        logger.info("ℹ️  Skipping built-in default jobs (--no-default-jobs specified).")

    # Build a map for existing IDs before discovery (for conflict resolution)
    existing_ids = {j.id for j in jobs}

    # 3. Auto-discovery
    if not args.no_auto_discover:
        include_patterns = DEFAULT_INCLUDE_PATTERNS[:]
        if args.include_pattern:
            include_patterns.extend(args.include_pattern)

        discovery_roots: List[Path] = [repo_root]
        if args.discover_root:
            discovery_roots = [Path(r).resolve() if not Path(r).is_absolute() else Path(r) for r in args.discover_root]

        exclude_regex = re.compile(args.exclude_regex) if args.exclude_regex else None

        logger.info(f"🔍 Auto-discovery scanning roots: {[str(r) for r in discovery_roots]}")
        logger.info(f"🔍 Include patterns: {include_patterns}")
        if exclude_regex:
            logger.info(f"🔍 Exclude regex: {exclude_regex.pattern}")

        script_paths = discover_training_scripts(
            roots=discovery_roots,
            include_patterns=include_patterns,
            exclude_regex=exclude_regex,
            logger=logger,
            limit=args.discovery_limit,
        )
        logger.info(f"🔍 Discovered {len(script_paths)} candidate training scripts.")

        discovered_jobs = build_jobs_from_discovery(
            script_paths,
            repo_root=repo_root,
            logger=logger,
            mark_optional_unknown=args.mark_discovered_optional
        )

        jobs = merge_jobs(
            base=jobs,
            discovered=discovered_jobs,
            allow_overrides=args.allow_discovery_overrides,
            logger=logger,
        )
        logger.info(f"🧪 Total jobs after merge: {len(jobs)}")
    else:
        logger.info("ℹ️  Auto-discovery disabled (--no-auto-discover).")

    # 4. If no jobs found, fallback or fail
    if not jobs:
        if args.strict_discovery:
            logger.error("No jobs discovered/loaded and --strict-discovery set.")
            return 3
        logger.warning("No jobs discovered/loaded. Falling back to built-in defaults.")
        jobs = default_jobs()

    original_count = len(jobs)

    # 5. Apply filtering
    jobs = filter_jobs(jobs, args.only, args.exclude)
    logger.info(f"🎯 Selected jobs: {len(jobs)} (filtered from {original_count})")

    if args.list:
        logger.info("")
        logger.info("Available Jobs (post-filter):")
        for j in jobs:
            logger.info(
                f"- {j.id}: {j.name} | script={j.script} | out={j.output} | "
                f"epochs={j.supports_epochs} folds={j.supports_folds} sample={j.supports_sample} batch={j.supports_batch} outdir={j.use_output_dir} optional={j.optional}"
            )
        return 0

    if not jobs:
        logger.warning("No jobs selected after filtering. Exiting.")
        return 0

    base_output_dir = repo_root / args.base_output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = args.extra_arg

    # 6. Execute
    executed_jobs: List[TrainingJob] = []
    if args.parallel and len(jobs) > 1:
        logger.info("⚠️  Parallel mode enabled. Ensure sufficient resources.")
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
                    args.sample,
                    args.batch,
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
                args.sample,
                args.batch,
                extra_args,
                args.retries,
                args.dry_run,
                logger,
                logs_dir,
            )
            executed_jobs.append(result_job)

    # 7. Summary
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

    # 8. Exit code logic
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
