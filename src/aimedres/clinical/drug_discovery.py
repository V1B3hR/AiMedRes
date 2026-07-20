"""
Drug Discovery and Clinical Trial Support Module.

Provides AI-assisted workflows for:
- Drug candidate screening and prioritisation
- Clinical trial design and cohort selection
- Adverse event signal detection
- Drug-drug interaction analysis
- Trial outcome prediction

Note: This module is for research-assistance purposes only and does not
constitute medical or regulatory advice.  All outputs must be reviewed by
qualified medical professionals before clinical application.
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aimedres.clinical.drug_discovery")


# ---------------------------------------------------------------------------
# Domain enumerations and data classes
# ---------------------------------------------------------------------------

class TrialPhase(str, Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"
    OBSERVATIONAL = "observational"


class TrialStatus(str, Enum):
    PLANNING = "planning"
    RECRUITING = "recruiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class AdverseEventSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    FATAL = "fatal"


@dataclass
class DrugCandidate:
    """Represents a drug candidate under investigation."""
    candidate_id: str
    name: str
    mechanism_of_action: str
    target_indication: str
    molecular_weight: float = 0.0
    bioavailability: float = 0.0         # 0–1
    half_life_hours: float = 0.0
    toxicity_score: float = 0.0          # 0–1 (lower is safer)
    efficacy_score: float = 0.0          # 0–1 (higher is better)
    development_phase: TrialPhase = TrialPhase.PHASE_1
    known_interactions: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalTrialProtocol:
    """Describes a clinical trial protocol."""
    trial_id: str
    title: str
    phase: TrialPhase
    drug_candidate: DrugCandidate
    primary_endpoint: str
    secondary_endpoints: List[str] = field(default_factory=list)
    target_enrollment: int = 0
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    duration_weeks: int = 0
    status: TrialStatus = TrialStatus.PLANNING
    sites: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdverseEvent:
    """Records an adverse event observed during a trial."""
    event_id: str
    trial_id: str
    participant_id: str
    event_type: str
    severity: AdverseEventSeverity
    onset_day: int
    resolution_day: Optional[int]
    drug_related: bool = False
    description: str = ""
    reported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Drug candidate analysis
# ---------------------------------------------------------------------------

class DrugCandidateScreener:
    """
    Screens and ranks drug candidates using multi-criteria scoring.

    Combines efficacy, safety, pharmacokinetic, and developability scores
    into a composite drug-likeness index.
    """

    # Lipinski rule-of-five reference bounds
    _LIPINSKI_MW_MAX = 500.0
    _LIPINSKI_BIOAVAIL_MIN = 0.2

    def score_candidate(self, candidate: DrugCandidate) -> Dict[str, Any]:
        """
        Compute a composite score for a drug candidate.

        Returns a dict with sub-scores and an overall composite score.
        """
        lipinski = self._lipinski_score(candidate)
        pkpd = self._pkpd_score(candidate)
        safety = max(0.0, 1.0 - candidate.toxicity_score)
        efficacy = candidate.efficacy_score

        # Weighted composite (weights reflect pharmaceutical priority)
        weights = {"efficacy": 0.35, "safety": 0.35, "pkpd": 0.15, "lipinski": 0.15}
        composite = (
            weights["efficacy"] * efficacy
            + weights["safety"] * safety
            + weights["pkpd"] * pkpd
            + weights["lipinski"] * lipinski
        )

        return {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "scores": {
                "efficacy": round(efficacy, 3),
                "safety": round(safety, 3),
                "pkpd": round(pkpd, 3),
                "lipinski": round(lipinski, 3),
                "composite": round(composite, 3),
            },
            "interaction_count": len(candidate.known_interactions),
            "contraindication_count": len(candidate.contraindications),
            "development_phase": candidate.development_phase.value,
            "recommendation": self._recommend(composite, candidate),
        }

    def rank_candidates(self, candidates: List[DrugCandidate]) -> List[Dict[str, Any]]:
        """Return candidates sorted by composite score (descending)."""
        scored = [self.score_candidate(c) for c in candidates]
        return sorted(scored, key=lambda x: x["scores"]["composite"], reverse=True)

    # ------------------------------------------------------------------
    def _lipinski_score(self, candidate: DrugCandidate) -> float:
        """Rule-of-five compliance score (simplified)."""
        score = 1.0
        if candidate.molecular_weight > self._LIPINSKI_MW_MAX:
            score -= 0.3
        if candidate.bioavailability < self._LIPINSKI_BIOAVAIL_MIN:
            score -= 0.3
        return max(0.0, score)

    def _pkpd_score(self, candidate: DrugCandidate) -> float:
        """Pharmacokinetic / pharmacodynamic quality score."""
        score = 0.5
        # Prefer half-life 4–24 h for oral dosing convenience
        if 4.0 <= candidate.half_life_hours <= 24.0:
            score += 0.3
        elif candidate.half_life_hours > 0:
            score += 0.1
        if candidate.bioavailability >= 0.7:
            score += 0.2
        elif candidate.bioavailability >= 0.4:
            score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _recommend(composite: float, candidate: DrugCandidate) -> str:
        if composite >= 0.75:
            return "advance"
        if composite >= 0.55:
            return "further_evaluation"
        if composite >= 0.35:
            return "optimise"
        return "deprioritise"


# ---------------------------------------------------------------------------
# Trial design and cohort selection
# ---------------------------------------------------------------------------

class TrialDesigner:
    """
    Assists in clinical trial protocol design and statistical planning.
    """

    def calculate_sample_size(
        self,
        baseline_rate: float,
        expected_effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate required sample size for a two-proportion z-test.

        Args:
            baseline_rate: Control arm event rate (0–1)
            expected_effect_size: Absolute difference in event rate (0–1)
            alpha: Type I error rate
            power: Statistical power (1 – β)
            two_sided: Whether to use a two-sided test

        Returns:
            Sample size per arm and total, with assumptions.
        """
        if not (0 < baseline_rate < 1) or not (0 < expected_effect_size < 1):
            raise ValueError("baseline_rate and expected_effect_size must be in (0, 1)")
        treatment_rate = min(baseline_rate + expected_effect_size, 0.999)
        p_avg = (baseline_rate + treatment_rate) / 2.0

        z_alpha = self._z_from_p(alpha / (2 if two_sided else 1))
        z_beta = self._z_from_p(1 - power)

        # Standard formula for two-proportion z-test
        n = (
            (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
             + z_beta * math.sqrt(baseline_rate * (1 - baseline_rate)
                                  + treatment_rate * (1 - treatment_rate))) ** 2
            / (expected_effect_size ** 2)
        )
        n_per_arm = math.ceil(n)
        return {
            "n_per_arm": n_per_arm,
            "total_n": n_per_arm * 2,
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "effect_size": expected_effect_size,
            "alpha": alpha,
            "power": power,
            "two_sided": two_sided,
        }

    def suggest_endpoints(self, indication: str) -> List[str]:
        """Return indicative endpoint suggestions for common indications."""
        indication_lower = indication.lower()
        endpoint_library: Dict[str, List[str]] = {
            "cardiovascular": [
                "major adverse cardiovascular events (MACE)",
                "all-cause mortality",
                "LDL-C reduction from baseline",
                "blood pressure reduction",
                "6-minute walk distance",
            ],
            "alzheimer": [
                "cognitive decline (ADAS-Cog)",
                "activities of daily living (ADCS-ADL)",
                "global function (CDR-SB)",
                "amyloid PET burden reduction",
                "cerebrospinal fluid biomarkers",
            ],
            "diabetes": [
                "HbA1c reduction from baseline",
                "fasting plasma glucose",
                "time-in-range (CGM)",
                "body weight change",
                "MACE (cardiovascular secondary)",
            ],
            "oncology": [
                "progression-free survival (PFS)",
                "overall survival (OS)",
                "objective response rate (ORR)",
                "duration of response (DoR)",
                "patient-reported outcomes (PRO)",
            ],
            "parkinson": [
                "UPDRS motor score",
                "UPDRS total score",
                "time to motor complications",
                "levodopa equivalent dose",
                "quality of life (PDQ-39)",
            ],
        }
        for key, endpoints in endpoint_library.items():
            if key in indication_lower:
                return endpoints
        return [
            "primary efficacy endpoint (to be defined)",
            "safety and tolerability",
            "patient-reported outcomes",
        ]

    @staticmethod
    def _z_from_p(p: float) -> float:
        """Approximate inverse normal CDF (Rational approximation, |error|<4.5e-4)."""
        if p <= 0 or p >= 1:
            raise ValueError("p must be in (0, 1)")
        if p > 0.5:
            p = 1 - p
        t = math.sqrt(-2.0 * math.log(p))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        numerator = c[0] + c[1] * t + c[2] * t * t
        denominator = 1 + d[0] * t + d[1] * t * t + d[2] * t * t * t
        return t - numerator / denominator


# ---------------------------------------------------------------------------
# Adverse event signal detection
# ---------------------------------------------------------------------------

class AdverseEventAnalyser:
    """
    Detects safety signals from adverse event reports.

    Implements the proportional reporting ratio (PRR) and reporting odds
    ratio (ROR) pharmacovigilance methods for signal detection.
    """

    def __init__(self):
        self._events: List[AdverseEvent] = []

    def add_event(self, event: AdverseEvent) -> None:
        self._events.append(event)

    def compute_prr(
        self, drug_id: str, event_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compute proportional reporting ratio for a drug–event pair.

        PRR > 2 and χ² > 4 is a commonly used signal threshold.
        """
        drug_events = [e for e in self._events if e.trial_id == drug_id]
        total_events = len(self._events)
        drug_total = len(drug_events)

        if total_events == 0 or drug_total == 0:
            return None

        a = sum(1 for e in drug_events if e.event_type == event_type)           # drug + event
        b = drug_total - a                                                        # drug + other events
        c = sum(1 for e in self._events if e.event_type == event_type) - a      # other drugs + event
        d = total_events - a - b - c                                              # other drugs + other events

        if (a + c) == 0 or (b + d) == 0 or (a + b) == 0:
            return None

        prr = (a / (a + b)) / ((a + c) / (total_events))
        # Chi-squared (Yates correction)
        n = a + b + c + d
        if n == 0:
            return None
        chi2 = (abs(a * d - b * c) - n / 2) ** 2 * n / (
            (a + b) * (c + d) * (a + c) * (b + d)
        ) if all([(a + b), (c + d), (a + c), (b + d)]) else 0.0

        signal = prr >= 2.0 and chi2 >= 4.0 and a >= 3

        return {
            "drug_id": drug_id,
            "event_type": event_type,
            "cases": a,
            "prr": round(prr, 3),
            "chi2": round(chi2, 3),
            "signal_detected": signal,
            "severity_breakdown": self._severity_counts(drug_events, event_type),
        }

    def get_signal_summary(self, drug_id: str) -> List[Dict[str, Any]]:
        """Return all safety signals for a trial."""
        drug_events = [e for e in self._events if e.trial_id == drug_id]
        event_types = {e.event_type for e in drug_events}
        results = []
        for et in event_types:
            r = self.compute_prr(drug_id, et)
            if r:
                results.append(r)
        return sorted(results, key=lambda x: x["prr"], reverse=True)

    def incidence_rate(self, trial_id: str, event_type: str) -> float:
        """Return crude incidence rate for an event type in a trial."""
        trial_events = [e for e in self._events if e.trial_id == trial_id]
        if not trial_events:
            return 0.0
        target = sum(1 for e in trial_events if e.event_type == event_type)
        return target / len(trial_events)

    @staticmethod
    def _severity_counts(events: List[AdverseEvent], event_type: str) -> Dict[str, int]:
        target = [e for e in events if e.event_type == event_type]
        counts: Dict[str, int] = {}
        for e in target:
            counts[e.severity.value] = counts.get(e.severity.value, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Drug-drug interaction checker
# ---------------------------------------------------------------------------

class DrugInteractionChecker:
    """
    Identifies potential drug-drug interactions (DDIs).

    Uses a curated interaction knowledge base that can be extended at
    runtime via ``register_interaction``.
    """

    def __init__(self):
        self._interactions: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def register_interaction(
        self,
        drug_a: str,
        drug_b: str,
        severity: str,
        mechanism: str,
        clinical_effect: str,
        management: str,
    ) -> None:
        """Register a drug-drug interaction."""
        key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
        self._interactions[key] = {
            "drugs": [drug_a, drug_b],
            "severity": severity,
            "mechanism": mechanism,
            "clinical_effect": clinical_effect,
            "management": management,
        }

    def check_interactions(
        self, drug_candidates: List[DrugCandidate], concomitant_drugs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check for interactions among a set of drug candidates and concomitant medications.

        Returns list of detected interactions sorted by severity.
        """
        all_drug_names = [c.name.lower() for c in drug_candidates]
        if concomitant_drugs:
            all_drug_names += [d.lower() for d in concomitant_drugs]

        # Include candidate-declared interactions
        for candidate in drug_candidates:
            for known in candidate.known_interactions:
                key = tuple(sorted([candidate.name.lower(), known.lower()]))
                if key not in self._interactions:
                    self._interactions[key] = {
                        "drugs": [candidate.name, known],
                        "severity": "unknown",
                        "mechanism": "candidate-declared",
                        "clinical_effect": "monitor",
                        "management": "clinical_judgement",
                    }

        found: List[Dict[str, Any]] = []
        checked = set()
        for i, drug_a in enumerate(all_drug_names):
            for drug_b in all_drug_names[i + 1:]:
                key = tuple(sorted([drug_a, drug_b]))
                if key in checked:
                    continue
                checked.add(key)
                if key in self._interactions:
                    found.append(self._interactions[key])

        severity_order = {
            "contraindicated": 0,
            "major": 1,
            "moderate": 2,
            "minor": 3,
            "unknown": 4,
        }
        return sorted(found, key=lambda x: severity_order.get(x["severity"], 5))


# ---------------------------------------------------------------------------
# Trial outcome predictor
# ---------------------------------------------------------------------------

class TrialOutcomePredictor:
    """
    Predicts trial outcome probability based on drug candidate properties
    and historical phase transition rates.

    Uses published industry averages for phase transition probabilities
    (DiMasi et al.) adjusted by candidate-level quality scores.
    """

    # Industry average phase transition probabilities (2003–2011, DiMasi 2016)
    _BASE_TRANSITION_RATES: Dict[str, float] = {
        TrialPhase.PHASE_1.value: 0.66,   # P1 → P2
        TrialPhase.PHASE_2.value: 0.46,   # P2 → P3
        TrialPhase.PHASE_3.value: 0.67,   # P3 → NDA/BLA
        TrialPhase.PHASE_4.value: 0.95,   # Approval → post-market
    }

    def predict_success_probability(
        self,
        candidate: DrugCandidate,
        trial: ClinicalTrialProtocol,
        additional_factors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate phase-specific and overall success probability.

        Returns per-phase transition probabilities and an estimated
        probability-of-technical-success (PTS) to first approval.
        """
        base_rate = self._BASE_TRANSITION_RATES.get(trial.phase.value, 0.5)

        # Adjust for candidate quality
        efficacy_adj = (candidate.efficacy_score - 0.5) * 0.2
        safety_adj = (1.0 - candidate.toxicity_score - 0.5) * 0.15
        adjusted_rate = max(0.05, min(0.98, base_rate + efficacy_adj + safety_adj))

        # Apply external factors if provided
        if additional_factors:
            for _, adjustment in additional_factors.items():
                adjusted_rate = max(0.05, min(0.98, adjusted_rate + float(adjustment)))

        # PTS: product of all subsequent phase transition probabilities
        phases_remaining = self._phases_from(trial.phase)
        pts = adjusted_rate
        for phase in phases_remaining[1:]:
            pts *= self._BASE_TRANSITION_RATES.get(phase, 0.5)

        return {
            "candidate_id": candidate.candidate_id,
            "trial_id": trial.trial_id,
            "current_phase": trial.phase.value,
            "phase_success_probability": round(adjusted_rate, 3),
            "overall_pts": round(pts, 3),
            "phases_remaining": phases_remaining,
            "confidence": "low" if pts < 0.1 else ("medium" if pts < 0.25 else "high"),
            "key_drivers": {
                "efficacy_score": candidate.efficacy_score,
                "toxicity_score": candidate.toxicity_score,
                "base_rate": base_rate,
            },
        }

    @staticmethod
    def _phases_from(phase: TrialPhase) -> List[str]:
        all_phases = [
            TrialPhase.PHASE_1.value,
            TrialPhase.PHASE_2.value,
            TrialPhase.PHASE_3.value,
            TrialPhase.PHASE_4.value,
        ]
        try:
            idx = all_phases.index(phase.value)
            return all_phases[idx:]
        except ValueError:
            return all_phases


# ---------------------------------------------------------------------------
# Unified drug discovery service
# ---------------------------------------------------------------------------

class DrugDiscoveryService:
    """
    Unified entry-point for drug discovery and clinical trial support.

    Composes all sub-components into a single cohesive service layer.
    """

    def __init__(self):
        self.screener = DrugCandidateScreener()
        self.trial_designer = TrialDesigner()
        self.ae_analyser = AdverseEventAnalyser()
        self.interaction_checker = DrugInteractionChecker()
        self.outcome_predictor = TrialOutcomePredictor()

    # ------------------------------------------------------------------
    def evaluate_pipeline(
        self, candidates: List[DrugCandidate]
    ) -> Dict[str, Any]:
        """
        Evaluate an entire drug pipeline in one call.

        Returns ranked candidates, interaction matrix, and portfolio-level
        statistics.
        """
        ranked = self.screener.rank_candidates(candidates)
        interactions = self.interaction_checker.check_interactions(candidates)

        avg_composite = statistics.mean(
            r["scores"]["composite"] for r in ranked
        ) if ranked else 0.0

        return {
            "pipeline_size": len(candidates),
            "ranked_candidates": ranked,
            "drug_drug_interactions": interactions,
            "portfolio_stats": {
                "average_composite_score": round(avg_composite, 3),
                "advance_count": sum(
                    1 for r in ranked if r["recommendation"] == "advance"
                ),
                "deprioritise_count": sum(
                    1 for r in ranked if r["recommendation"] == "deprioritise"
                ),
                "high_interaction_risk": len(
                    [i for i in interactions if i["severity"] in ("contraindicated", "major")]
                ),
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    def design_trial(
        self,
        candidate: DrugCandidate,
        phase: TrialPhase,
        indication: str,
        baseline_event_rate: float,
        expected_effect: float,
    ) -> Dict[str, Any]:
        """
        Generate a trial design recommendation with sample size and endpoints.
        """
        sample_size = self.trial_designer.calculate_sample_size(
            baseline_rate=baseline_event_rate,
            expected_effect_size=expected_effect,
        )
        endpoints = self.trial_designer.suggest_endpoints(indication)

        protocol = ClinicalTrialProtocol(
            trial_id=f"TRIAL-{candidate.candidate_id}-{phase.value.upper()}",
            title=f"{candidate.name} {phase.value.replace('_', ' ').title()} Study in {indication}",
            phase=phase,
            drug_candidate=candidate,
            primary_endpoint=endpoints[0] if endpoints else "primary endpoint TBD",
            secondary_endpoints=endpoints[1:] if len(endpoints) > 1 else [],
            target_enrollment=sample_size["total_n"],
            duration_weeks={"phase_1": 12, "phase_2": 24, "phase_3": 52}.get(phase.value, 24),
            status=TrialStatus.PLANNING,
        )

        outcome_pred = self.outcome_predictor.predict_success_probability(candidate, protocol)

        return {
            "protocol": {
                "trial_id": protocol.trial_id,
                "title": protocol.title,
                "phase": protocol.phase.value,
                "target_enrollment": protocol.target_enrollment,
                "duration_weeks": protocol.duration_weeks,
                "primary_endpoint": protocol.primary_endpoint,
                "secondary_endpoints": protocol.secondary_endpoints,
            },
            "sample_size_analysis": sample_size,
            "outcome_prediction": outcome_pred,
            "designed_at": datetime.now(timezone.utc).isoformat(),
        }

    def analyse_safety_signals(self, trial_id: str) -> List[Dict[str, Any]]:
        """Return detected safety signals for a trial."""
        return self.ae_analyser.get_signal_summary(trial_id)

    def record_adverse_event(self, event: AdverseEvent) -> None:
        """Record an adverse event for downstream signal detection."""
        self.ae_analyser.add_event(event)
        if event.severity in (AdverseEventSeverity.SEVERE, AdverseEventSeverity.LIFE_THREATENING,
                               AdverseEventSeverity.FATAL):
            logger.warning(
                "Serious adverse event recorded: %s | trial=%s severity=%s",
                event.event_type,
                event.trial_id,
                event.severity.value,
            )
