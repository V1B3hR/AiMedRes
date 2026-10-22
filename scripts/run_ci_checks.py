"""
One-Command Production CI/CD Gateway Runner.
Conforms to Phase 3.4 & 3.5 of the Consolidated Roadmap.

Executes all verification checks locally:
1. Scientific & Security Unit Tests (pytest)
2. PPMI Benchmark Reproducibility Runner
3. CycloneDX 1.5 SBOM Generation
4. Kubernetes & Helm Sandbox Deployment Validation
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("aimedres.ci")


def run_step(step_name: str, command: list[str]) -> bool:
    """Execute a subprocess command, stream output, and report pass/fail."""
    print(f"\n[CI GATEWAY] >>> Executing: {step_name}...")
    start = time.time()
    try:
        res = subprocess.run(command, text=True, capture_output=True, check=False)
        duration = time.time() - start
        if res.returncode == 0:
            print(f"[CI GATEWAY] [PASS] {step_name} ({duration:.2f}s)")
            return True
        else:
            print(f"[CI GATEWAY] [FAIL] {step_name} ({duration:.2f}s) with exit code {res.returncode}")
            if res.stdout:
                print("--- stdout ---")
                print(res.stdout[-2000:])
            if res.stderr:
                print("--- stderr ---")
                print(res.stderr[-2000:])
            return False
    except Exception as e:
        print(f"[CI GATEWAY] [FAIL] {step_name} EXCEPTION: {e}")
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("================================================================================")
    print("                     AIMEDRES PRODUCTION CI/CD GATEWAY                         ")
    print("================================================================================")

    steps = [
        (
            "1. Core Unit Tests (Research, Security, Service, Open Datasets, Compliance, CI/CD)",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_clinical_and_compliance.py",
                "tests/unit/test_cicd_and_deployment.py",
                "tests/unit/test_security_and_architecture.py",
                "tests/unit/test_research_parkinsons.py",
                "tests/unit/test_ppmi_reproducibility.py",
                "tests/unit/test_open_access_datasets.py",
                "tests/unit/test_p1_components.py",
                "-q",
            ],
        ),
        (
            "2. PPMI Parkinson's Reproducibility Benchmark",
            [sys.executable, "scripts/reproduce_ppmi_benchmark.py"],
        ),
        (
            "3. CycloneDX 1.5 SBOM Generation",
            [sys.executable, "scripts/generate_sbom.py"],
        ),
        (
            "4. Kubernetes & Helm Sandbox Deployment Validation",
            [sys.executable, "scripts/validate_sandbox_deployment.py"],
        ),
    ]

    all_passed = True
    for name, cmd in steps:
        passed = run_step(name, cmd)
        if not passed:
            all_passed = False

    print("\n================================================================================")
    if all_passed:
        print(" [CI GATEWAY RESULT] ALL VERIFICATION GATES PASSED! MERGE READY. [PASS]")
    else:
        print(" [CI GATEWAY RESULT] GATEWAY FAILURES DETECTED! REVIEW LOGS ABOVE. [FAIL]")
    print("================================================================================")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
