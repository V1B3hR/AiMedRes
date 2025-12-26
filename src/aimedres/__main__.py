#!/usr/bin/env python3
"""
DuetMind Adaptive System - Advanced Orchestrator

Enhancements:
- Subcommand-based CLI
- Structured result dataclasses
- Config (JSON/YAML + env overrides)
- Structured / JSON logging options
- Retry + timeout control
- Unified phase runner with timing
- Optional JSON machine-readable output
- Graceful signal handling
- Extensible plugin phase discovery (placeholder)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Local modules (new)
try:
    from config_loader import load_config_merged
    from phase_runner import PhaseDefinition, PhaseExecutionError, PhaseRunner
    from result_models import (
        ComprehensiveResultPayload,
        PhaseResult,
        ResultStatus,
        SimulationResultPayload,
        SystemResult,
        TrainingResultPayload,
    )
except ImportError:
    # Fallback minimal internal definitions if modules not yet added
    from dataclasses import dataclass, field

    class ResultStatus:
        SUCCESS = "success"
        FAILURE = "failure"

    @dataclass
    class PhaseResult:
        name: str
        status: str
        duration_sec: float
        details: Dict[str, Any] = field(default_factory=dict)
        error: Optional[str] = None

    @dataclass
    class SystemResult:
        status: str
        phases: List[PhaseResult]
        started_at: float
        finished_at: float
        meta: Dict[str, Any] = field(default_factory=dict)

    class PhaseExecutionError(Exception):
        pass

    def load_config_merged(path: Optional[str], env_prefix: str = "DUETMIND_") -> Dict[str, Any]:
        return {}

    class PhaseDefinition:
        def __init__(self, name: str, func, retries: int = 0):
            self.name = name
            self.func = func
            self.retries = retries

    class PhaseRunner:
        def __init__(self, logger: logging.Logger):
            self.logger = logger

        def run_phases(self, phases: Sequence[PhaseDefinition]) -> List[PhaseResult]:
            results = []
            for p in phases:
                start = time.time()
                try:
                    self.logger.info(f"Running phase: {p.name}")
                    details = p.func()
                    duration = time.time() - start
                    results.append(PhaseResult(p.name, ResultStatus.SUCCESS, duration, details))
                except Exception as e:
                    duration = time.time() - start
                    tb = traceback.format_exc()
                    self.logger.error(f"Phase {p.name} failed: {e}")
                    results.append(
                        PhaseResult(p.name, ResultStatus.FAILURE, duration, {}, error=str(e))
                    )
                    break
            return results


# -------------------------------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------------------------------


def configure_logging(verbose: bool, log_format: str) -> logging.Logger:
    logger = logging.getLogger("DuetMindMain")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    # Avoid duplicate handlers if reconfigured
    logger.handlers[:] = [handler]
    logger.propagate = False
    return logger


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


# -------------------------------------------------------------------------------------------------
# Custom Exceptions
# -------------------------------------------------------------------------------------------------


class TrainingError(Exception):
    pass


class SimulationError(Exception):
    pass


class ComprehensiveError(Exception):
    pass


# -------------------------------------------------------------------------------------------------
# Phase Implementations
# -------------------------------------------------------------------------------------------------


def phase_training(
    timeout: int, training_script: str, python_executable: str, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Runs the external comprehensive training script with a timeout.
    """
    import subprocess

    start_t = time.time()
    if not Path(training_script).exists():
        raise TrainingError(f"Training script not found: {training_script}")

    cmd = [
        python_executable,
        training_script,
        "--mode",
        "comprehensive",
    ]
    logger.debug(f"Executing training subprocess: {' '.join(cmd)} (timeout={timeout}s)")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise TrainingError(f"Training timed out after {timeout}s")

    if result.returncode != 0:
        logger.debug(f"Training stderr:\n{result.stderr}")
        raise TrainingError(f"Training failed (rc={result.returncode})")

    # Example placeholder metrics
    duration = time.time() - start_t
    return {
        "return_code": result.returncode,
        "stdout_sample": result.stdout[:5000],
        "duration_sec": duration,
        "accuracy": _extract_metric(result.stdout, "accuracy"),
        "loss": _extract_metric(result.stdout, "loss"),
    }


def phase_simulation(logger: logging.Logger) -> Dict[str, Any]:
    """
    Imports labyrinth simulation and executes it.
    """
    try:
        from labyrinth_simulation import run_labyrinth_simulation
    except ImportError as e:
        raise SimulationError(f"Simulation module import failed: {e}")

    sim_result = run_labyrinth_simulation()
    # Expecting a dict result; validate basics
    total_steps = sim_result.get("total_steps")
    if total_steps is None:
        raise SimulationError("Simulation result missing total_steps")

    return {
        "total_steps": total_steps,
        "maze_master_interventions": sim_result.get("maze_master_interventions"),
        "raw": sim_result,
    }


def phase_comprehensive(logger: logging.Logger) -> Dict[str, Any]:
    """
    Runs comprehensive medical AI training + simulation integration test.
    """
    try:
        from comprehensive_training_simulation import MedicalAIComprehensiveSystem
    except ImportError as e:
        raise ComprehensiveError(f"Comprehensive system import failed: {e}")

    system = MedicalAIComprehensiveSystem()
    result = system.run_comprehensive_system()

    if result.get("status") != "success":
        raise ComprehensiveError(f"Comprehensive system failure in phase {result.get('phase')}")

    return {
        "training_accuracy": result["training_phase"]["accuracy"],
        "simulation_health_score": result["simulation_phase"]["health_score"],
        "labyrinth_components_imported": result["system_integration"][
            "labyrinth_components_imported"
        ],
        "raw": result,
    }


# -------------------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------------------


def _extract_metric(output: str, key: str) -> Optional[float]:
    # Naive example: find lines like "accuracy: 0.912"
    for line in output.splitlines():
        if key in line.lower():
            parts = line.replace("=", ":").split(":")
            if len(parts) >= 2:
                try:
                    return float(parts[1].strip().split()[0])
                except ValueError:
                    pass
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DuetMind Adaptive System Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=False)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    common_parent.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    common_parent.add_argument(
        "--log-format", choices=["text", "json"], default="text", help="Log format"
    )
    common_parent.add_argument(
        "--json-output", action="store_true", help="Emit final result as JSON"
    )
    common_parent.add_argument(
        "--strict", action="store_true", help="Exit non-zero if any phase fails"
    )
    common_parent.add_argument(
        "--retries", type=int, default=0, help="Retries for each phase on failure"
    )
    common_parent.add_argument(
        "--dry-run", action="store_true", help="Validate configuration only and exit"
    )

    p_train = sub.add_parser(
        "training", parents=[common_parent], help="Run comprehensive training only"
    )
    p_train.add_argument("--timeout", type=int, default=300, help="Training timeout seconds")
    p_train.add_argument("--script", default="full_training.py", help="Training script filename")

    p_sim = sub.add_parser("simulation", parents=[common_parent], help="Run simulation only")
    p_both = sub.add_parser("both", parents=[common_parent], help="Run training then simulation")
    p_both.add_argument("--timeout", type=int, default=300, help="Training timeout seconds")
    p_both.add_argument("--script", default="full_training.py", help="Training script filename")

    p_comp = sub.add_parser(
        "comprehensive",
        parents=[common_parent],
        help="Run comprehensive medical AI training + simulation integration",
    )
    p_interactive = sub.add_parser(
        "interactive", parents=[common_parent], help="Interactive mode menu"
    )

    # Default to interactive if no subcommand
    parser.set_defaults(command="interactive")

    return parser


# -------------------------------------------------------------------------------------------------
# Signal Handling
# -------------------------------------------------------------------------------------------------


class GracefulTerminator:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.terminated = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if not self.terminated:
            self.logger.warning(f"Received signal {signum}, attempting graceful shutdown...")
            self.terminated = True
        else:
            self.logger.error("Second interrupt received; forcing exit.")
            sys.exit(2)


# -------------------------------------------------------------------------------------------------
# Execution Coordinator
# -------------------------------------------------------------------------------------------------


def execute_command(args: argparse.Namespace, logger: logging.Logger) -> SystemResult:
    start = time.time()
    config = load_config_merged(args.config, env_prefix="DUETMIND_")

    logger.info(f"Starting command: {args.command}")
    if args.dry_run:
        logger.info("Dry run: configuration loaded successfully. Exiting.")
        return SystemResult(
            status=ResultStatus.SUCCESS,
            phases=[],
            started_at=start,
            finished_at=time.time(),
            meta={"dry_run": True, "config_keys": list(config.keys())},
        )

    runner = PhaseRunner(logger)

    # Build phases list
    phases: List[PhaseDefinition] = []
    py_exec = sys.executable

    if args.command == "training":
        phases.append(
            PhaseDefinition(
                "training",
                lambda: phase_training(
                    timeout=getattr(args, "timeout", 300),
                    training_script=getattr(args, "script", "full_training.py"),
                    python_executable=py_exec,
                    logger=logger,
                ),
                retries=args.retries,
            )
        )
    elif args.command == "simulation":
        phases.append(
            PhaseDefinition("simulation", lambda: phase_simulation(logger), retries=args.retries)
        )
    elif args.command == "both":
        phases.append(
            PhaseDefinition(
                "training",
                lambda: phase_training(
                    timeout=getattr(args, "timeout", 300),
                    training_script=getattr(args, "script", "full_training.py"),
                    python_executable=py_exec,
                    logger=logger,
                ),
                retries=args.retries,
            )
        )
        phases.append(
            PhaseDefinition("simulation", lambda: phase_simulation(logger), retries=args.retries)
        )
    elif args.command == "comprehensive":
        phases.append(
            PhaseDefinition(
                "comprehensive_system", lambda: phase_comprehensive(logger), retries=args.retries
            )
        )
    elif args.command == "interactive":
        return run_interactive_menu(logger, config, args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    # Execute
    phase_results = runner.run_phases(phases)
    finished = time.time()

    # Determine overall status
    overall_status = ResultStatus.SUCCESS
    for pr in phase_results:
        if pr.status != ResultStatus.SUCCESS:
            overall_status = ResultStatus.FAILURE
            break

    system_result = SystemResult(
        status=overall_status,
        phases=phase_results,
        started_at=start,
        finished_at=finished,
        meta={
            "config_applied": bool(config),
            "config_source": args.config,
            "retries": args.retries,
            "strict": args.strict,
            "command": args.command,
        },
    )
    return system_result


def run_interactive_menu(
    logger: logging.Logger, config: Dict[str, Any], args: argparse.Namespace
) -> SystemResult:
    logger.info("Entering interactive mode")
    menu = [
        ("Run comprehensive training", "training"),
        ("Run simulation", "simulation"),
        ("Run both training + simulation", "both"),
        ("Run comprehensive medical AI system", "comprehensive"),
        ("Exit", "exit"),
    ]

    while True:
        print("\n=== DuetMind Adaptive System (Interactive) ===")
        for idx, (label, _) in enumerate(menu, start=1):
            print(f"{idx}. {label}")
        choice = input("Select an option: ").strip()
        try:
            ci = int(choice) - 1
            if ci < 0 or ci >= len(menu):
                print("Invalid selection.")
                continue
            label, action = menu[ci]
            if action == "exit":
                print("Goodbye.")
                return SystemResult(
                    status=ResultStatus.SUCCESS,
                    phases=[],
                    started_at=time.time(),
                    finished_at=time.time(),
                    meta={"interactive_exit": True},
                )
            # Simulate recursive call with new command set
            setattr(args, "command", action)
            return execute_command(args, logger)
        except ValueError:
            print("Please enter a number.")


# -------------------------------------------------------------------------------------------------
# Output Helpers
# -------------------------------------------------------------------------------------------------


def emit_final_result(result: SystemResult, json_output: bool, logger: logging.Logger):
    if json_output:
        safe = {
            "status": result.status,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_sec": result.finished_at - result.started_at,
            "meta": result.meta,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "duration_sec": p.duration_sec,
                    "details": p.details,
                    "error": p.error,
                }
                for p in result.phases
            ],
        }
        print(json.dumps(safe, indent=2))
    else:
        logger.info("=== Execution Summary ===")
        logger.info(f"Overall Status: {result.status.upper()}")
        logger.info(f"Total Duration: {result.finished_at - result.started_at:.2f}s")
        for p in result.phases:
            if p.status == ResultStatus.SUCCESS:
                logger.info(f"[OK] {p.name} ({p.duration_sec:.2f}s)")
            else:
                logger.error(f"[FAIL] {p.name} ({p.duration_sec:.2f}s) - {p.error}")


# -------------------------------------------------------------------------------------------------
# Main Entry
# -------------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logger = configure_logging(args.verbose, args.log_format)
    terminator = GracefulTerminator(logger)

    try:
        system_result = execute_command(args, logger)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug("Traceback:\n" + traceback.format_exc())
        if getattr(args, "json_output", False):
            print(json.dumps({"status": "failure", "error": str(e)}, indent=2))
        return 1

    emit_final_result(system_result, getattr(args, "json_output", False), logger)

    if system_result.status != ResultStatus.SUCCESS and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
