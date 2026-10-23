"""
Unit tests for Production CI/CD Gateways (Phase 3.4)
and Kubernetes / Helm Sandbox Deployment (Phase 3.5).
"""

import json
from pathlib import Path
import pytest
import yaml

from scripts.generate_sbom import generate_cyclonedx_sbom
from scripts.validate_sandbox_deployment import (
    run_all_deployment_validations,
    validate_helm_chart,
    validate_k8s_deployment,
    validate_observability_configs,
)
from scripts.run_ci_checks import run_step


class TestProductionCICDGateways:
    """Validate CycloneDX SBOM generation and CI gateway operations (Phase 3.4)."""

    def test_sbom_generation_schema(self, tmp_path):
        target_file = tmp_path / "cyclonedx_sbom.json"
        sbom = generate_cyclonedx_sbom(output_path=target_file)

        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.5"
        assert "serialNumber" in sbom
        assert sbom["metadata"]["component"]["name"] == "AiMedRes"
        assert sbom["metadata"]["component"]["licenses"][0]["license"]["id"] == "MIT"

        # Check components list
        components = sbom["components"]
        assert len(components) >= 5
        comp_names = [c["name"] for c in components]
        assert "python" in comp_names
        assert "scikit-learn" in comp_names
        assert "torch" in comp_names

        # Check written file
        assert target_file.exists()
        loaded = json.loads(target_file.read_text(encoding="utf-8"))
        assert loaded["bomFormat"] == "CycloneDX"

    def test_ci_step_runner_success(self):
        passed = run_step("Test Echo", ["python", "-c", "print('hello from test')"])
        assert passed is True

    def test_ci_step_runner_failure(self):
        passed = run_step("Test Failing Command", ["python", "-c", "import sys; sys.exit(2)"])
        assert passed is False


class TestSandboxDeploymentValidation:
    """Validate Kubernetes manifests and Helm chart sandboxes (Phase 3.5)."""

    def test_repository_deployment_validations(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        results = run_all_deployment_validations(root_dir)

        assert results["all_passed"] is True, f"Validation errors: {results}"
        assert results["k8s"]["valid"] is True
        assert results["helm"]["valid"] is True
        assert results["observability"]["valid"] is True

    def test_mlflow_compose_services_are_not_publicly_bound(self):
        """Keep the MLflow service behind the local host or a reverse proxy."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        compose_paths = (
            root_dir / "deployment" / "technical" / "docker-compose.yml",
            root_dir / "mlops" / "infra" / "docker-compose.yml",
        )

        for compose_path in compose_paths:
            compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
            ports = compose["services"]["mlflow"]["ports"]
            assert all(str(port).startswith("127.0.0.1:") for port in ports)

    def test_k8s_deployment_validator_flags_missing_run_as_non_root(self, tmp_path):
        k8s_dir = tmp_path / "k8s"
        k8s_dir.mkdir()
        insecure_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-app
spec:
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
    spec:
      securityContext:
        runAsNonRoot: false
      containers:
        - name: app
          image: test:latest
"""
        (k8s_dir / "deployment.yaml").write_text(insecure_deployment, encoding="utf-8")
        valid, errors = validate_k8s_deployment(k8s_dir)
        assert valid is False
        assert any("runAsNonRoot" in e for e in errors)
