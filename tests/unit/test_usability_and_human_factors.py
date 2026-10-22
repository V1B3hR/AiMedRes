"""
Unit and Human Factors Verification Tests for Phase 4.3 (Clinician Usability).
Verifies:
1. Feature attribution extraction from trained pipelines (linear and ensemble).
2. Waterfall contribution sorting and direction classification.
3. Clinical domain concordance evaluation against MDS consensus criteria.
4. Clinician case review tracking and decision turnaround time calculation.
5. System Usability Scale (SUS) and NASA-TLX score calculations.
6. Real-time inference and attribution latency benchmark (< 200 ms SLO).
7. Explainability API integration and payload structure.
"""

import time
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from aimedres.clinical.usability import (
    FeatureAttributionEvaluator,
    ClinicianReviewSession,
    benchmark_clinical_turnaround_time,
)
from aimedres.api.explain_routes import ExplainabilityEngine


@pytest.fixture
def sample_trained_linear_pipeline():
    """Create a minimal trained scikit-learn pipeline for attribution tests."""
    X = pd.DataFrame({
        "AGE": [60, 65, 70, 75, 80],
        "PET_PUTAMEN_L": [2.5, 2.2, 1.8, 1.4, 1.1],
        "PET_CAUDATE_L": [2.8, 2.6, 2.1, 1.7, 1.3],
        "MMSE": [29, 28, 26, 24, 22],
    })
    y = np.array([0, 0, 1, 1, 1])

    pipeline = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def sample_trained_tree_pipeline():
    """Create a trained random forest pipeline."""
    X = pd.DataFrame({
        "age": [55, 62, 71, 74, 82],
        "Shimmer": [0.015, 0.022, 0.038, 0.045, 0.052],
        "NHR": [0.008, 0.012, 0.024, 0.031, 0.042],
        "HNR": [28.0, 25.0, 20.0, 17.0, 14.0],
    })
    y = np.array([0, 0, 1, 1, 1])

    pipeline = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


def test_linear_model_feature_attributions(sample_trained_linear_pipeline):
    """Verify signed log-odds feature attributions from linear pipeline."""
    patient = {
        "AGE": 78,
        "PET_PUTAMEN_L": 1.2,
        "PET_CAUDATE_L": 1.4,
        "MMSE": 23,
    }

    attributions = FeatureAttributionEvaluator.compute_attributions(
        sample_trained_linear_pipeline, patient
    )

    assert len(attributions) == 4
    # Verify sorted by absolute contribution descending
    for i in range(len(attributions) - 1):
        assert abs(attributions[i].contribution) >= abs(attributions[i + 1].contribution)

    # Verify attribution fields
    for attr in attributions:
        assert attr.feature in patient
        assert attr.direction in ["risk_increasing", "protective"]
        assert 0.0 <= attr.importance <= 1.0


def test_tree_model_feature_attributions(sample_trained_tree_pipeline):
    """Verify feature importances and direction from tree pipeline."""
    patient = {
        "age": 75,
        "Shimmer": 0.040,
        "NHR": 0.028,
        "HNR": 16.0,
    }

    attributions = FeatureAttributionEvaluator.compute_attributions(
        sample_trained_tree_pipeline, patient
    )

    assert len(attributions) == 4
    assert all(a.importance >= 0.0 for a in attributions)
    assert all(a.direction in ["risk_increasing", "protective"] for a in attributions)


def test_clinical_domain_concordance():
    """Verify evaluation of model feature contributions against clinical guidelines."""
    # Test concordant Parkinson's markers
    patient = {
        "PET_PUTAMEN_L": 1.1,
        "AGE": 76,
        "MMSE": 22,
    }

    evaluator = FeatureAttributionEvaluator()
    fallback_attrs = evaluator.compute_attributions(None, patient)

    concordance = evaluator.evaluate_clinical_concordance(fallback_attrs)

    assert concordance["total_domain_features_checked"] == 3
    assert concordance["concordance_rate"] >= 0.85
    assert concordance["is_clinically_acceptable"] is True
    assert len(concordance["details"]) == 3


def test_clinician_review_session_tracking():
    """Verify recording of clinician decision turnaround and sign-off."""
    session = ClinicianReviewSession(clinician_id="dr_hopkins_md", specialty="Movement Disorders")
    t0 = session.start_case_review("CASE-PPMI-9912")

    time.sleep(0.01)  # Simulate review time

    record = session.record_decision(
        case_id="CASE-PPMI-9912",
        start_time=t0,
        model_risk_level="high",
        clinician_decision="concordant",
        clinician_notes="DaTscan putaminal deficit consistent with Stage 2 PD progression.",
    )

    assert record["case_id"] == "CASE-PPMI-9912"
    assert record["clinician_id"] == "dr_hopkins_md"
    assert record["is_concordant"] is True
    assert record["turnaround_seconds"] > 0.005
    assert len(session.reviewed_cases) == 1


def test_sus_score_calculation():
    """Verify System Usability Scale (SUS) scoring arithmetic."""
    # Perfect score: all positive items 5, all negative items 1
    # Items: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Pos: [0, 2, 4, 6, 8] -> 5 -> 5-1=4 -> 5*4=20
    # Neg: [1, 3, 5, 7, 9] -> 1 -> 5-1=4 -> 5*4=20 -> Total 40 * 2.5 = 100
    perfect_responses = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
    sus_perfect = ClinicianReviewSession.calculate_sus_score(perfect_responses)
    assert sus_perfect == 100.0

    # Worst score: all positive items 1, all negative items 5 -> 0
    worst_responses = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
    sus_worst = ClinicianReviewSession.calculate_sus_score(worst_responses)
    assert sus_worst == 0.0

    # High usability response (> 80.0)
    good_responses = [4, 2, 5, 1, 4, 2, 5, 2, 4, 2]
    sus_good = ClinicianReviewSession.calculate_sus_score(good_responses)
    assert sus_good >= 80.0

    # Invalid length raises error
    with pytest.raises(ValueError):
        ClinicianReviewSession.calculate_sus_score([5, 1, 5])


def test_nasa_tlx_workload_calculation():
    """Verify raw NASA-TLX cognitive workload computation."""
    subscales = {
        "mental_demand": 25.0,
        "physical_demand": 10.0,
        "temporal_demand": 20.0,
        "performance": 15.0,
        "effort": 20.0,
        "frustration": 10.0,
    }
    workload = ClinicianReviewSession.calculate_nasa_tlx(subscales)
    assert workload == pytest.approx(16.67, 0.05)
    assert workload < 35.0  # Under regulatory threshold for cognitive overload


def test_clinical_turnaround_time_benchmark(sample_trained_linear_pipeline):
    """
    Verify sub-second latency SLO (< 200 ms) for end-to-end inference and attribution.
    """
    records = [
        {"AGE": int(np.random.normal(70, 5)), "PET_PUTAMEN_L": 1.5, "PET_CAUDATE_L": 1.8, "MMSE": 26}
        for _ in range(50)
    ]

    benchmark_res = benchmark_clinical_turnaround_time(
        model_or_pipeline=sample_trained_linear_pipeline,
        sample_records=records,
        max_acceptable_latency_ms=200.0,
    )

    assert benchmark_res["n_samples"] == 50
    assert benchmark_res["mean_latency_ms"] < 200.0
    assert benchmark_res["meets_clinical_slo"] is True


def test_explainability_api_integration():
    """Verify ExplainabilityEngine integration with domain features."""
    engine = ExplainabilityEngine()
    case = {"PET_PUTAMEN_L": 1.25, "PET_CAUDATE_L": 1.60, "AGE": 71}

    attributions = engine.compute_feature_attribution("pred-real-001", case)

    assert len(attributions) >= 3
    features_returned = [a["feature"] for a in attributions]
    assert "PET_PUTAMEN_L" in features_returned
    assert "AGE" in features_returned
    assert all("direction" in a for a in attributions)
