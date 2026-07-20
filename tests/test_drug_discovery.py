"""
Tests for the Drug Discovery and Clinical Trial Support module.
"""

import sys
import os

import pytest

# Import module directly to avoid triggering heavy package-level dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "drug_discovery",
    os.path.join(os.path.dirname(__file__), '..', 'src', 'aimedres', 'clinical', 'drug_discovery.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AdverseEvent = _mod.AdverseEvent
AdverseEventAnalyser = _mod.AdverseEventAnalyser
AdverseEventSeverity = _mod.AdverseEventSeverity
DrugCandidate = _mod.DrugCandidate
DrugCandidateScreener = _mod.DrugCandidateScreener
DrugDiscoveryService = _mod.DrugDiscoveryService
DrugInteractionChecker = _mod.DrugInteractionChecker
TrialDesigner = _mod.TrialDesigner
TrialOutcomePredictor = _mod.TrialOutcomePredictor
TrialPhase = _mod.TrialPhase
TrialStatus = _mod.TrialStatus
ClinicalTrialProtocol = _mod.ClinicalTrialProtocol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_candidate():
    return DrugCandidate(
        candidate_id="DC001",
        name="NovaDrug-A",
        mechanism_of_action="AChE inhibitor",
        target_indication="Alzheimer's disease",
        molecular_weight=350.0,
        bioavailability=0.75,
        half_life_hours=12.0,
        toxicity_score=0.15,
        efficacy_score=0.82,
        development_phase=TrialPhase.PHASE_2,
    )


@pytest.fixture
def poor_candidate():
    return DrugCandidate(
        candidate_id="DC002",
        name="ToxDrug-B",
        mechanism_of_action="unknown",
        target_indication="unknown",
        molecular_weight=800.0,
        bioavailability=0.05,
        half_life_hours=1.0,
        toxicity_score=0.85,
        efficacy_score=0.20,
        development_phase=TrialPhase.PHASE_1,
    )


@pytest.fixture
def designer():
    return TrialDesigner()


@pytest.fixture
def service():
    return DrugDiscoveryService()


# ---------------------------------------------------------------------------
# DrugCandidateScreener
# ---------------------------------------------------------------------------

class TestDrugCandidateScreener:
    def test_score_returns_composite(self, good_candidate):
        screener = DrugCandidateScreener()
        result = screener.score_candidate(good_candidate)
        assert "scores" in result
        assert "composite" in result["scores"]
        assert 0.0 <= result["scores"]["composite"] <= 1.0

    def test_good_candidate_scores_higher(self, good_candidate, poor_candidate):
        screener = DrugCandidateScreener()
        good_score = screener.score_candidate(good_candidate)["scores"]["composite"]
        poor_score = screener.score_candidate(poor_candidate)["scores"]["composite"]
        assert good_score > poor_score

    def test_advance_recommendation_for_good(self, good_candidate):
        screener = DrugCandidateScreener()
        result = screener.score_candidate(good_candidate)
        assert result["recommendation"] in ("advance", "further_evaluation")

    def test_deprioritise_recommendation_for_poor(self, poor_candidate):
        screener = DrugCandidateScreener()
        result = screener.score_candidate(poor_candidate)
        assert result["recommendation"] in ("deprioritise", "optimise")

    def test_rank_candidates_ordered(self, good_candidate, poor_candidate):
        screener = DrugCandidateScreener()
        ranked = screener.rank_candidates([poor_candidate, good_candidate])
        assert ranked[0]["scores"]["composite"] >= ranked[1]["scores"]["composite"]

    def test_lipinski_penalty_for_high_mw(self, good_candidate):
        screener = DrugCandidateScreener()
        good_candidate.molecular_weight = 600.0
        result = screener.score_candidate(good_candidate)
        assert result["scores"]["lipinski"] < 1.0


# ---------------------------------------------------------------------------
# TrialDesigner
# ---------------------------------------------------------------------------

class TestTrialDesigner:
    def test_sample_size_positive(self, designer):
        result = designer.calculate_sample_size(
            baseline_rate=0.20,
            expected_effect_size=0.10,
        )
        assert result["n_per_arm"] > 0
        assert result["total_n"] == result["n_per_arm"] * 2

    def test_larger_effect_requires_fewer_subjects(self, designer):
        small_effect = designer.calculate_sample_size(0.20, 0.05)
        large_effect = designer.calculate_sample_size(0.20, 0.15)
        assert small_effect["n_per_arm"] > large_effect["n_per_arm"]

    def test_invalid_inputs_raise(self, designer):
        with pytest.raises(ValueError):
            designer.calculate_sample_size(0.0, 0.1)
        with pytest.raises(ValueError):
            designer.calculate_sample_size(0.5, 0.0)

    def test_suggest_endpoints_alzheimer(self, designer):
        endpoints = designer.suggest_endpoints("Alzheimer's disease")
        assert any("adas" in e.lower() or "cognitive" in e.lower() for e in endpoints)

    def test_suggest_endpoints_cardiovascular(self, designer):
        endpoints = designer.suggest_endpoints("cardiovascular disease")
        assert any("mace" in e.lower() or "mortality" in e.lower() for e in endpoints)

    def test_suggest_endpoints_unknown_returns_defaults(self, designer):
        endpoints = designer.suggest_endpoints("rare genetic condition XYZ")
        assert len(endpoints) >= 1


# ---------------------------------------------------------------------------
# AdverseEventAnalyser
# ---------------------------------------------------------------------------

class TestAdverseEventAnalyser:
    def _make_events(self, analyser: AdverseEventAnalyser, trial_id="T1", n_drug=10, n_other=100, event_type="nausea"):
        for i in range(n_drug):
            analyser.add_event(AdverseEvent(
                event_id=f"AE-drug-{i}",
                trial_id=trial_id,
                participant_id=f"p{i}",
                event_type=event_type,
                severity=AdverseEventSeverity.MILD,
                onset_day=i + 1,
                resolution_day=i + 3,
                drug_related=True,
            ))
        for i in range(n_other):
            analyser.add_event(AdverseEvent(
                event_id=f"AE-other-{i}",
                trial_id="OTHER",
                participant_id=f"q{i}",
                event_type="headache",
                severity=AdverseEventSeverity.MILD,
                onset_day=1,
                resolution_day=2,
            ))

    def test_add_event_increases_count(self):
        analyser = AdverseEventAnalyser()
        ae = AdverseEvent("e1", "T1", "p1", "nausea", AdverseEventSeverity.MILD, 1, 3)
        analyser.add_event(ae)
        assert len(analyser._events) == 1

    def test_prr_computes_for_known_signal(self):
        analyser = AdverseEventAnalyser()
        self._make_events(analyser)
        result = analyser.compute_prr("T1", "nausea")
        assert result is not None
        assert result["prr"] > 1.0
        assert "cases" in result

    def test_incidence_rate_bounds(self):
        analyser = AdverseEventAnalyser()
        self._make_events(analyser, n_drug=5)
        rate = analyser.incidence_rate("T1", "nausea")
        assert 0.0 <= rate <= 1.0

    def test_signal_summary_returns_list(self):
        analyser = AdverseEventAnalyser()
        self._make_events(analyser)
        signals = analyser.get_signal_summary("T1")
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# DrugInteractionChecker
# ---------------------------------------------------------------------------

class TestDrugInteractionChecker:
    def test_register_and_detect_interaction(self, good_candidate):
        checker = DrugInteractionChecker()
        checker.register_interaction(
            "NovaDrug-A", "Warfarin",
            severity="major",
            mechanism="CYP2C9 inhibition",
            clinical_effect="increased bleeding risk",
            management="monitor INR",
        )
        candidate2 = DrugCandidate("DC003", "Warfarin", "anticoagulant", "thrombosis")
        interactions = checker.check_interactions([good_candidate, candidate2])
        assert len(interactions) >= 1
        assert interactions[0]["severity"] == "major"

    def test_no_interaction_returns_empty(self):
        checker = DrugInteractionChecker()
        c1 = DrugCandidate("DC010", "SafeDrug-X", "x", "x")
        c2 = DrugCandidate("DC011", "SafeDrug-Y", "y", "y")
        interactions = checker.check_interactions([c1, c2])
        assert interactions == []

    def test_candidate_declared_interactions_detected(self):
        checker = DrugInteractionChecker()
        c1 = DrugCandidate("DC012", "DrugA", "mech", "ind", known_interactions=["DrugB"])
        c2 = DrugCandidate("DC013", "DrugB", "mech", "ind")
        interactions = checker.check_interactions([c1, c2])
        assert len(interactions) >= 1

    def test_severity_sorting(self):
        checker = DrugInteractionChecker()
        checker.register_interaction("D1", "D2", "minor", "", "", "")
        checker.register_interaction("D1", "D3", "contraindicated", "", "", "")
        c1 = DrugCandidate("1", "D1", "", "")
        c2 = DrugCandidate("2", "D2", "", "")
        c3 = DrugCandidate("3", "D3", "", "")
        interactions = checker.check_interactions([c1, c2, c3])
        severities = [i["severity"] for i in interactions]
        assert severities[0] == "contraindicated"


# ---------------------------------------------------------------------------
# TrialOutcomePredictor
# ---------------------------------------------------------------------------

class TestTrialOutcomePredictor:
    def test_predict_returns_expected_keys(self, good_candidate):
        predictor = TrialOutcomePredictor()
        protocol = ClinicalTrialProtocol(
            trial_id="T-001",
            title="Test",
            phase=TrialPhase.PHASE_2,
            drug_candidate=good_candidate,
            primary_endpoint="HbA1c reduction",
        )
        result = predictor.predict_success_probability(good_candidate, protocol)
        assert "phase_success_probability" in result
        assert "overall_pts" in result
        assert 0.0 < result["phase_success_probability"] <= 1.0
        assert 0.0 < result["overall_pts"] <= 1.0

    def test_good_candidate_higher_pts(self, good_candidate, poor_candidate):
        predictor = TrialOutcomePredictor()
        protocol_good = ClinicalTrialProtocol("T1", "", TrialPhase.PHASE_2, good_candidate, "")
        protocol_poor = ClinicalTrialProtocol("T2", "", TrialPhase.PHASE_2, poor_candidate, "")
        pts_good = predictor.predict_success_probability(good_candidate, protocol_good)["overall_pts"]
        pts_poor = predictor.predict_success_probability(poor_candidate, protocol_poor)["overall_pts"]
        assert pts_good > pts_poor


# ---------------------------------------------------------------------------
# DrugDiscoveryService (integration)
# ---------------------------------------------------------------------------

class TestDrugDiscoveryService:
    def test_evaluate_pipeline(self, service, good_candidate, poor_candidate):
        result = service.evaluate_pipeline([good_candidate, poor_candidate])
        assert result["pipeline_size"] == 2
        assert "ranked_candidates" in result
        assert "portfolio_stats" in result

    def test_design_trial(self, service, good_candidate):
        result = service.design_trial(
            candidate=good_candidate,
            phase=TrialPhase.PHASE_2,
            indication="Alzheimer's disease",
            baseline_event_rate=0.30,
            expected_effect=0.12,
        )
        assert "protocol" in result
        assert "sample_size_analysis" in result
        assert "outcome_prediction" in result
        assert result["protocol"]["target_enrollment"] > 0

    def test_record_and_analyse_adverse_event(self, service):
        ae = AdverseEvent(
            event_id="AE001",
            trial_id="T-DRUG",
            participant_id="P001",
            event_type="rash",
            severity=AdverseEventSeverity.MODERATE,
            onset_day=7,
            resolution_day=14,
            drug_related=True,
        )
        service.record_adverse_event(ae)
        signals = service.analyse_safety_signals("T-DRUG")
        assert isinstance(signals, list)
