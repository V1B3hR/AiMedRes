"""Regression checks for the 25 October 2026 Render security advisory response."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_render_node_runtime_is_pinned_to_a_patched_release():
    assert (_REPO_ROOT / ".node-version").read_text(encoding="utf-8").strip() == "24.22.1"

    workflow = (_REPO_ROOT / ".github" / "workflows" / "performance-validation.yml").read_text(
        encoding="utf-8"
    )
    assert 'node-version: "24.22.1"' in workflow


def test_python_dependency_surfaces_require_patched_certifi_bundle():
    for relative_path in (
        "requirements.txt",
        "pyproject.toml",
        "mlops/infra/mlflow_requirements.txt",
        "deployment/technical/mlflow-requirements.txt",
    ):
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "certifi>=2026.10.19" in text


def test_project_dockerfiles_refresh_certificate_store_on_rebuild():
    for relative_path in (
        "Dockerfile",
        "deployment/technical/Dockerfile",
        "mlops/infra/Dockerfile-imaging",
    ):
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "CA_CERTIFICATES_REFRESH=2026-10-25" in text
        assert "ca-certificates" in text


def test_default_mlops_compose_does_not_start_the_php_based_adminer_image():
    compose = (_REPO_ROOT / "mlops" / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "adminer:" not in compose
    assert "adminer:latest" not in compose


def test_live_advisory_verification_script_structure():
    script = (_REPO_ROOT / "scripts" / "verify_render_advisory_live.py").read_text(encoding="utf-8")
    assert "https://dv-g3b-emudhra.we-have-a-problem.com/" in script
    assert "https://dv-g3b-aaa-emudhra.we-have-a-problem.com/" in script
    assert "2026, 10, 19" in script
    assert "24, 22, 1" in script

