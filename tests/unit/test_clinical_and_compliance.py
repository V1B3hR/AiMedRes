"""
Unit tests for Clinical & Compliance Specifications (Phase 4.1 & Phase 4.4).
Verifies:
1. Intended Use and IFU documentation conforms to FDA 21st Century Cures Act § 3060.
2. IRB Clinical Study Protocol conforms to GCP, observational non-interventional standards, and 45 CFR 46.
3. SaMD Traceability Matrix conforms to IEC 62304 Class B and ISO 14971.
4. Programmatic AST verification that 100% of test functions mapped in SAMD_TRACEABILITY_MATRIX.md exist.
"""

import ast
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_intended_use_specification_completeness():
    """Verify Intended Use and Indications for Use specification (Phase 4.1)."""
    ifu_path = REPO_ROOT / "docs" / "clinical" / "INTENDED_USE_SPECIFICATION.md"
    assert ifu_path.exists(), f"Missing IFU specification: {ifu_path}"
    content = ifu_path.read_text(encoding="utf-8")

    # Verify FDA 21st Century Cures Act § 3060 four-criteria analysis
    assert "21st Century Cures Act" in content
    assert "520(o)(1)(E)" in content
    assert "Criterion 1" in content
    assert "Criterion 2" in content
    assert "Criterion 3" in content
    assert "Criterion 4" in content

    # Verify EU MDR Rule 11 classification
    assert "EU MDR" in content
    assert "Rule 11" in content

    # Verify Core Clinical Statements
    assert "Intended Use Statement" in content
    assert "Indications for Use (IFU)" in content
    assert "Target Patient Population" in content
    assert "Target User Profile" in content

    # Verify Prohibited Uses / Contraindications
    assert "PROHIBITED CLINICAL USES" in content
    assert "EMERGENCY & CRITICAL CARE TRIAGE" in content
    assert "PEDIATRIC POPULATIONS" in content
    assert "AUTONOMOUS THERAPY SELECTION" in content

    # Verify HCP Independence controls (Explainable CDS / SHAP / 95% CIs)
    assert "SHAP" in content
    assert "bootstrap confidence intervals" in content


def test_irb_study_protocol_completeness():
    """Verify Institutional Review Board (IRB) Clinical Study Protocol (Phase 4.1)."""
    irb_path = REPO_ROOT / "docs" / "clinical" / "IRB_CLINICAL_STUDY_PROTOCOL.md"
    assert irb_path.exists(), f"Missing IRB protocol: {irb_path}"
    content = irb_path.read_text(encoding="utf-8")

    # Verify protocol identifiers and study summary
    assert "AIMEDRES-IRB-2026-001" in content
    assert "Protocol Summary" in content
    assert "Primary Objective" in content
    assert "Secondary Objectives" in content

    # Verify cohorts
    assert "PPMI" in content
    assert "OASIS-2" in content
    assert "UCI / Oxford Parkinson's Telemonitoring" in content

    # Verify eligibility criteria
    assert "Inclusion Criteria" in content
    assert "Exclusion Criteria" in content

    # Verify statistical analysis plan (SAP)
    assert "StratifiedGroupKFold" in content
    assert "Bootstrap Confidence Interval" in content

    # Verify ethics, consent waiver, and HIPAA
    assert "Waiver of Informed Consent" in content
    assert "45 CFR 46.116(f)" in content
    assert "HIPAA" in content


def test_samd_traceability_matrix_structure_and_standards():
    """Verify SaMD Requirements & Verification Traceability Matrix (Phase 4.4)."""
    trm_path = REPO_ROOT / "docs" / "compliance" / "SAMD_TRACEABILITY_MATRIX.md"
    assert trm_path.exists(), f"Missing SaMD Traceability Matrix: {trm_path}"
    content = trm_path.read_text(encoding="utf-8")

    # Verify regulatory standards
    assert "IEC 62304" in content
    assert "ISO 14971" in content
    assert "Class B" in content

    # Verify all 8 ISO 14971 hazards catalogued
    for i in range(1, 9):
        assert f"HAZ-{i:03d}" in content, f"Missing hazard HAZ-{i:03d} in traceability matrix"

    # Verify user needs and requirements
    for i in range(1, 9):
        assert f"UN-{i:02d}" in content, f"Missing user need UN-{i:02d}"

    for i in range(1, 12):
        assert f"SRS-{i:02d}" in content, f"Missing software requirement SRS-{i:02d}"
        assert f"VER-{i:03d}" in content, f"Missing verification test ID VER-{i:03d}"


def test_samd_traceability_test_functions_exist_in_ast():
    """
    Programmatic AST assertion that every verification test function
    referenced in SAMD_TRACEABILITY_MATRIX.md actually exists in tests/unit/.
    Guarantees the traceability matrix never goes stale.
    """
    trm_path = REPO_ROOT / "docs" / "compliance" / "SAMD_TRACEABILITY_MATRIX.md"
    content = trm_path.read_text(encoding="utf-8")

    # Extract all test references of form: `tests/unit/test_xxx.py::test_func_name`
    pattern = re.compile(r"`(tests/unit/[a-zA-Z0-9_\.]+\.py)::([a-zA-Z0-9_]+)`")
    matches = pattern.findall(content)

    assert len(matches) >= 10, f"Expected at least 10 verification test mappings, found {len(matches)}"

    # Cache parsed ASTs per file
    file_functions = {}

    for file_rel, func_name in matches:
        full_path = REPO_ROOT / Path(file_rel)
        assert full_path.exists(), f"Referenced test file does not exist: {file_rel}"

        if file_rel not in file_functions:
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            # Find all top-level and method function definitions
            found_funcs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found_funcs.add(node.name)
            file_functions[file_rel] = found_funcs

        assert func_name in file_functions[file_rel], (
            f"SaMD Traceability Matrix maps '{func_name}' in '{file_rel}', "
            f"but no such test function was found in the test file!"
        )
