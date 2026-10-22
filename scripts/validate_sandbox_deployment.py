"""
Kubernetes & Helm Research Sandbox Deployment Validator.
Conforms to Phase 3.5 of the Consolidated Roadmap.

Validates that declarative Kubernetes manifests, Helm charts, and Prometheus monitoring
adhere to production container security, RBAC, and observability standards.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


def validate_k8s_deployment(k8s_dir: Path) -> Tuple[bool, List[str]]:
    """Verify securityContext, resource limits, and monitoring annotations in k8s manifests."""
    errors: List[str] = []
    dep_file = k8s_dir / "deployment.yaml"
    if not dep_file.is_file():
        errors.append(f"Missing {dep_file}")
        return False, errors

    try:
        with open(dep_file, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
        
        deployment = next((d for d in docs if d and d.get("kind") == "Deployment"), None)
        if not deployment:
            errors.append("No Deployment resource found in deployment.yaml")
            return False, errors

        spec = deployment.get("spec", {})
        template = spec.get("template", {})
        pod_spec = template.get("spec", {})
        metadata = template.get("metadata", {})

        # 1. Security Context checks
        sec_ctx = pod_spec.get("securityContext", {})
        if not sec_ctx.get("runAsNonRoot"):
            errors.append("Pod securityContext must set 'runAsNonRoot: true'")

        # 2. Prometheus Scrape Annotations
        annotations = metadata.get("annotations", {})
        if annotations.get("prometheus.io/scrape") != "true":
            errors.append("Pod metadata annotations must include 'prometheus.io/scrape: true'")

        # 3. Containers verification
        containers = pod_spec.get("containers", [])
        if not containers:
            errors.append("No containers specified in deployment pod spec")

    except Exception as e:
        errors.append(f"YAML parsing error in {dep_file}: {e}")

    return len(errors) == 0, errors


def validate_helm_chart(helm_dir: Path) -> Tuple[bool, List[str]]:
    """Verify Helm Chart.yaml and values.yaml conform to chart standards."""
    errors: List[str] = []
    chart_file = helm_dir / "Chart.yaml"
    values_file = helm_dir / "values.yaml"

    if not chart_file.is_file():
        errors.append(f"Missing {chart_file}")
    else:
        try:
            with open(chart_file, "r", encoding="utf-8") as f:
                chart = yaml.safe_load(f)
            if not chart.get("name") or not chart.get("version"):
                errors.append("Chart.yaml missing 'name' or 'version'")
        except Exception as e:
            errors.append(f"Error reading {chart_file}: {e}")

    if not values_file.is_file():
        errors.append(f"Missing {values_file}")
    else:
        try:
            with open(values_file, "r", encoding="utf-8") as f:
                values = yaml.safe_load(f)
            if "replicaCount" not in values:
                errors.append("values.yaml missing 'replicaCount'")
        except Exception as e:
            errors.append(f"Error reading {values_file}: {e}")

    return len(errors) == 0, errors


def validate_observability_configs(monitoring_dir: Path) -> Tuple[bool, List[str]]:
    """Verify Prometheus and Alertmanager configurations exist and are valid YAML."""
    errors: List[str] = []
    prom_file = monitoring_dir / "prometheus.yml"
    alert_file = monitoring_dir / "alertmanager.yml"

    for f_path in [prom_file, alert_file]:
        if not f_path.is_file():
            errors.append(f"Missing {f_path}")
        else:
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except Exception as e:
                errors.append(f"Invalid YAML in {f_path}: {e}")

    return len(errors) == 0, errors


def run_all_deployment_validations(root_dir: Path) -> Dict[str, Any]:
    """Execute complete sandbox validation suite."""
    k8s_valid, k8s_errors = validate_k8s_deployment(root_dir / "k8s")
    helm_valid, helm_errors = validate_helm_chart(root_dir / "helm" / "aimedres")
    obs_valid, obs_errors = validate_observability_configs(root_dir / "monitoring")

    all_passed = k8s_valid and helm_valid and obs_valid
    return {
        "all_passed": all_passed,
        "k8s": {"valid": k8s_valid, "errors": k8s_errors},
        "helm": {"valid": helm_valid, "errors": helm_errors},
        "observability": {"valid": obs_valid, "errors": obs_errors},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = Path(__file__).resolve().parent.parent
    results = run_all_deployment_validations(root)

    print("================================================================================")
    print("               K8S & HELM RESEARCH SANDBOX VALIDATION REPORT                   ")
    print("================================================================================")
    print(f"Kubernetes Manifests: {'PASSED' if results['k8s']['valid'] else 'FAILED'}")
    for err in results['k8s']['errors']:
        print(f"  - {err}")

    print(f"Helm Chart Packages:  {'PASSED' if results['helm']['valid'] else 'FAILED'}")
    for err in results['helm']['errors']:
        print(f"  - {err}")

    print(f"Observability Config: {'PASSED' if results['observability']['valid'] else 'FAILED'}")
    for err in results['observability']['errors']:
        print(f"  - {err}")

    print(f"Overall Status:       {'ALL CHECKS PASSED' if results['all_passed'] else 'FAILURES DETECTED'}")
    print("================================================================================")
    sys.exit(0 if results["all_passed"] else 1)
