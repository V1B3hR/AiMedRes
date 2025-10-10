"""
Advanced Memory Consolidation - Population Health Insights (P14)

Extends the existing memory consolidation system with population-level
analytics capabilities for extracting cohort-level insights and
strategic analytics from aggregated clinical data.

This module provides:
- Cohort aggregation and analysis
- Population health trend identification
- Risk stratification at population level
- Longitudinal outcome tracking
- Strategic analytics for healthcare planning
"""

import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import json
import statistics
import numpy as np

logger = logging.getLogger('aimedres.agent_memory.population_insights')


class CohortType(Enum):
    """Types of patient cohorts."""
    DISEASE_BASED = "disease_based"
    AGE_BASED = "age_based"
    GEOGRAPHIC = "geographic"
    RISK_BASED = "risk_based"
    TREATMENT_BASED = "treatment_based"
    OUTCOME_BASED = "outcome_based"


class TrendDirection(Enum):
    """Direction of population health trends."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    EMERGING = "emerging"


@dataclass
class Cohort:
    """Represents a patient cohort for analysis."""
    cohort_id: str
    name: str
    cohort_type: CohortType
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    patient_ids: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationMetrics:
    """Aggregated metrics for a population cohort."""
    cohort_id: str
    calculation_time: datetime
    population_size: int
    age_distribution: Dict[str, int]  # age_group -> count
    gender_distribution: Dict[str, int]
    condition_prevalence: Dict[str, float]  # condition -> prevalence rate
    avg_risk_scores: Dict[str, float]  # risk_type -> avg_score
    outcome_rates: Dict[str, float]  # outcome_type -> rate
    comorbidity_patterns: List[Tuple[List[str], int]]  # (conditions, count)
    treatment_patterns: Dict[str, int]  # treatment -> count
    longitudinal_data: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class HealthTrend:
    """Population health trend analysis."""
    trend_id: str
    cohort_id: str
    metric_name: str
    direction: TrendDirection
    magnitude: float  # Rate of change
    confidence: float  # 0-1
    time_period: Tuple[datetime, datetime]
    data_points: List[Tuple[datetime, float]]
    statistical_significance: bool
    recommendations: List[str]


@dataclass
class RiskStratification:
    """Population-level risk stratification."""
    cohort_id: str
    risk_category: str
    strata: Dict[str, Dict[str, Any]]  # risk_level -> patient_stats
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    risk_factors: List[Tuple[str, float]]  # (factor, weight)
    intervention_targets: List[str]


@dataclass
class OutcomeTracking:
    """Longitudinal outcome tracking for populations."""
    cohort_id: str
    outcome_type: str
    baseline_date: datetime
    follow_up_periods: List[int]  # Days since baseline
    outcome_data: Dict[int, Dict[str, Any]]  # period -> outcomes
    survival_rates: Dict[int, float]  # period -> rate
    readmission_rates: Dict[int, float]
    quality_of_life_scores: Dict[int, float]
    intervention_effectiveness: Dict[str, float]


class PopulationInsightsEngine:
    """
    Population health insights extraction engine.
    
    Provides cohort-level analytics and strategic insights
    from aggregated clinical memory data.
    """
    
    def __init__(self):
        """Initialize the population insights engine."""
        self.cohorts: Dict[str, Cohort] = {}
        self.patient_data: Dict[str, Dict[str, Any]] = {}
        self.metrics_cache: Dict[str, PopulationMetrics] = {}
        self.trends: Dict[str, HealthTrend] = {}
        
        logger.info("Population insights engine initialized")
    
    # === Cohort Management ===
    
    def create_cohort(
        self,
        name: str,
        cohort_type: CohortType,
        inclusion_criteria: Dict[str, Any],
        exclusion_criteria: Optional[Dict[str, Any]] = None
    ) -> Cohort:
        """
        Create a new patient cohort.
        
        Args:
            name: Cohort name
            cohort_type: Type of cohort
            inclusion_criteria: Criteria for inclusion
            exclusion_criteria: Criteria for exclusion
            
        Returns:
            Created Cohort
        """
        cohort_id = str(uuid.uuid4())
        
        cohort = Cohort(
            cohort_id=cohort_id,
            name=name,
            cohort_type=cohort_type,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria or {}
        )
        
        self.cohorts[cohort_id] = cohort
        
        logger.info(
            f"Cohort created: {name} ({cohort_type.value}) "
            f"with {len(inclusion_criteria)} criteria"
        )
        
        return cohort
    
    def add_patient_to_cohort(
        self,
        cohort_id: str,
        patient_id: str,
        patient_data: Dict[str, Any]
    ) -> bool:
        """
        Add a patient to a cohort.
        
        Args:
            cohort_id: Cohort identifier
            patient_id: Patient identifier
            patient_data: Patient clinical data
            
        Returns:
            True if added successfully
        """
        if cohort_id not in self.cohorts:
            logger.error(f"Cohort not found: {cohort_id}")
            return False
        
        cohort = self.cohorts[cohort_id]
        
        # Check inclusion criteria
        if not self._matches_criteria(patient_data, cohort.inclusion_criteria):
            return False
        
        # Check exclusion criteria
        if self._matches_criteria(patient_data, cohort.exclusion_criteria):
            return False
        
        # Add to cohort
        cohort.patient_ids.add(patient_id)
        cohort.last_updated = datetime.now()
        
        # Store patient data
        self.patient_data[patient_id] = patient_data
        
        # Invalidate metrics cache
        if cohort_id in self.metrics_cache:
            del self.metrics_cache[cohort_id]
        
        logger.debug(f"Patient {patient_id} added to cohort {cohort.name}")
        
        return True
    
    def _matches_criteria(
        self,
        patient_data: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if patient data matches given criteria."""
        # If criteria is empty, matches everything
        if not criteria:
            return True
        
        for key, expected_value in criteria.items():
            if key not in patient_data:
                return False
            
            actual_value = patient_data[key]
            
            # Handle range criteria
            if isinstance(expected_value, dict):
                if "min" in expected_value and actual_value < expected_value["min"]:
                    return False
                if "max" in expected_value and actual_value > expected_value["max"]:
                    return False
            # Handle list matching (for conditions, treatments, etc.)
            elif isinstance(expected_value, list):
                # If actual_value is also a list, check if any expected value is in actual
                if isinstance(actual_value, list):
                    if not any(val in actual_value for val in expected_value):
                        return False
                # If actual_value is a single value, check if it's in expected list
                else:
                    if actual_value not in expected_value:
                        return False
            # Handle exact match
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def populate_cohort_from_data(
        self,
        cohort_id: str,
        patients: List[Dict[str, Any]]
    ) -> int:
        """
        Populate a cohort from a list of patient records.
        
        Args:
            cohort_id: Cohort identifier
            patients: List of patient records
            
        Returns:
            Number of patients added
        """
        added = 0
        for patient in patients:
            patient_id = patient.get("patient_id", str(uuid.uuid4()))
            if self.add_patient_to_cohort(cohort_id, patient_id, patient):
                added += 1
        
        logger.info(f"Populated cohort with {added} patients")
        return added
    
    # === Population Metrics ===
    
    def calculate_population_metrics(
        self,
        cohort_id: str,
        use_cache: bool = True
    ) -> PopulationMetrics:
        """
        Calculate comprehensive population metrics for a cohort.
        
        Args:
            cohort_id: Cohort identifier
            use_cache: Whether to use cached metrics
            
        Returns:
            PopulationMetrics object
        """
        # Check cache
        if use_cache and cohort_id in self.metrics_cache:
            return self.metrics_cache[cohort_id]
        
        if cohort_id not in self.cohorts:
            raise ValueError(f"Cohort not found: {cohort_id}")
        
        cohort = self.cohorts[cohort_id]
        patient_ids = list(cohort.patient_ids)
        
        if not patient_ids:
            # Return empty metrics
            return PopulationMetrics(
                cohort_id=cohort_id,
                calculation_time=datetime.now(),
                population_size=0,
                age_distribution={},
                gender_distribution={},
                condition_prevalence={},
                avg_risk_scores={},
                outcome_rates={},
                comorbidity_patterns=[],
                treatment_patterns={}
            )
        
        # Get patient data
        patients = [self.patient_data[pid] for pid in patient_ids if pid in self.patient_data]
        
        # Age distribution
        age_groups = defaultdict(int)
        for patient in patients:
            age = patient.get("age", 0)
            if age < 18:
                age_groups["pediatric"] += 1
            elif age < 40:
                age_groups["young_adult"] += 1
            elif age < 65:
                age_groups["middle_adult"] += 1
            else:
                age_groups["senior"] += 1
        
        # Gender distribution
        gender_dist = Counter(patient.get("gender", "unknown") for patient in patients)
        
        # Condition prevalence
        all_conditions = []
        for patient in patients:
            conditions = patient.get("conditions", [])
            if isinstance(conditions, list):
                all_conditions.extend(conditions)
            elif isinstance(conditions, str):
                all_conditions.append(conditions)
        
        condition_counts = Counter(all_conditions)
        condition_prevalence = {
            condition: count / len(patients)
            for condition, count in condition_counts.items()
        }
        
        # Average risk scores
        risk_scores = defaultdict(list)
        for patient in patients:
            risks = patient.get("risk_scores", {})
            for risk_type, score in risks.items():
                risk_scores[risk_type].append(score)
        
        avg_risk_scores = {
            risk_type: statistics.mean(scores)
            for risk_type, scores in risk_scores.items()
            if scores
        }
        
        # Outcome rates
        outcome_types = ["improved", "stable", "declined"]
        outcome_counts = Counter(
            patient.get("outcome", "unknown") for patient in patients
        )
        outcome_rates = {
            outcome: outcome_counts.get(outcome, 0) / len(patients)
            for outcome in outcome_types
        }
        
        # Comorbidity patterns (top 5)
        comorbidity_groups = []
        for patient in patients:
            conditions = patient.get("conditions", [])
            if isinstance(conditions, list) and len(conditions) > 1:
                comorbidity_groups.append(tuple(sorted(conditions)))
        
        comorbidity_patterns = Counter(comorbidity_groups).most_common(5)
        comorbidity_patterns = [(list(pattern), count) for pattern, count in comorbidity_patterns]
        
        # Treatment patterns
        all_treatments = []
        for patient in patients:
            treatments = patient.get("treatments", [])
            if isinstance(treatments, list):
                all_treatments.extend(treatments)
            elif isinstance(treatments, str):
                all_treatments.append(treatments)
        
        treatment_patterns = dict(Counter(all_treatments).most_common(10))
        
        metrics = PopulationMetrics(
            cohort_id=cohort_id,
            calculation_time=datetime.now(),
            population_size=len(patients),
            age_distribution=dict(age_groups),
            gender_distribution=dict(gender_dist),
            condition_prevalence=condition_prevalence,
            avg_risk_scores=avg_risk_scores,
            outcome_rates=outcome_rates,
            comorbidity_patterns=comorbidity_patterns,
            treatment_patterns=treatment_patterns
        )
        
        # Cache metrics
        self.metrics_cache[cohort_id] = metrics
        
        logger.info(
            f"Population metrics calculated for cohort {cohort_id}: "
            f"{metrics.population_size} patients"
        )
        
        return metrics
    
    # === Trend Analysis ===
    
    def analyze_health_trends(
        self,
        cohort_id: str,
        metric_name: str,
        time_series_data: List[Tuple[datetime, float]],
        min_confidence: float = 0.7
    ) -> HealthTrend:
        """
        Analyze health trends in a population cohort.
        
        Args:
            cohort_id: Cohort identifier
            metric_name: Name of the metric being analyzed
            time_series_data: List of (timestamp, value) tuples
            min_confidence: Minimum confidence threshold
            
        Returns:
            HealthTrend object
        """
        if len(time_series_data) < 3:
            raise ValueError("Need at least 3 data points for trend analysis")
        
        # Sort by time
        time_series_data = sorted(time_series_data, key=lambda x: x[0])
        
        # Extract values
        times = [t for t, v in time_series_data]
        values = [v for t, v in time_series_data]
        
        # Calculate linear trend
        n = len(values)
        x = list(range(n))
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)
        
        # Calculate slope (trend magnitude)
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.IMPROVING if metric_name in ["quality", "survival", "recovery"] else TrendDirection.DECLINING
        else:
            direction = TrendDirection.DECLINING if metric_name in ["quality", "survival", "recovery"] else TrendDirection.IMPROVING
        
        # Calculate R-squared for confidence
        y_pred = [mean_y + slope * (i - mean_x) for i in range(n)]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - mean_y) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        confidence = max(0, min(1, r_squared))
        statistical_significance = confidence >= min_confidence
        
        # Generate recommendations
        recommendations = []
        if direction == TrendDirection.DECLINING:
            recommendations.append("investigate_root_causes")
            recommendations.append("implement_intervention")
        elif direction == TrendDirection.IMPROVING:
            recommendations.append("maintain_current_strategies")
            recommendations.append("document_best_practices")
        elif direction == TrendDirection.STABLE:
            recommendations.append("continue_monitoring")
        
        trend_id = str(uuid.uuid4())
        trend = HealthTrend(
            trend_id=trend_id,
            cohort_id=cohort_id,
            metric_name=metric_name,
            direction=direction,
            magnitude=abs(slope),
            confidence=confidence,
            time_period=(times[0], times[-1]),
            data_points=time_series_data,
            statistical_significance=statistical_significance,
            recommendations=recommendations
        )
        
        self.trends[trend_id] = trend
        
        logger.info(
            f"Health trend analyzed: {metric_name} - {direction.value} "
            f"(confidence={confidence:.2f})"
        )
        
        return trend
    
    # === Risk Stratification ===
    
    def stratify_population_risk(
        self,
        cohort_id: str,
        risk_category: str,
        risk_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> RiskStratification:
        """
        Stratify population by risk levels.
        
        Args:
            cohort_id: Cohort identifier
            risk_category: Category of risk (e.g., "cardiovascular", "readmission")
            risk_thresholds: Optional custom thresholds (low, medium, high)
            
        Returns:
            RiskStratification object
        """
        if cohort_id not in self.cohorts:
            raise ValueError(f"Cohort not found: {cohort_id}")
        
        # Default thresholds
        if risk_thresholds is None:
            risk_thresholds = {
                "low": (0.0, 0.3),
                "medium": (0.3, 0.7),
                "high": (0.7, 1.0)
            }
        
        cohort = self.cohorts[cohort_id]
        patients = [self.patient_data[pid] for pid in cohort.patient_ids if pid in self.patient_data]
        
        # Stratify patients
        strata = {"low": [], "medium": [], "high": []}
        
        for patient in patients:
            risk_scores = patient.get("risk_scores", {})
            risk_score = risk_scores.get(risk_category, 0.5)
            
            if risk_score < risk_thresholds["medium"][0]:
                strata["low"].append(patient)
            elif risk_score < risk_thresholds["high"][0]:
                strata["medium"].append(patient)
            else:
                strata["high"].append(patient)
        
        # Calculate statistics for each stratum
        strata_stats = {}
        for level, level_patients in strata.items():
            if level_patients:
                ages = [p.get("age", 0) for p in level_patients]
                strata_stats[level] = {
                    "count": len(level_patients),
                    "percentage": len(level_patients) / len(patients) * 100,
                    "avg_age": statistics.mean(ages),
                    "common_conditions": Counter(
                        cond for p in level_patients
                        for cond in p.get("conditions", [])
                    ).most_common(3)
                }
            else:
                strata_stats[level] = {
                    "count": 0,
                    "percentage": 0.0,
                    "avg_age": 0.0,
                    "common_conditions": []
                }
        
        # Identify key risk factors (simplified)
        risk_factors = [
            ("age_over_65", 0.3),
            ("multiple_comorbidities", 0.25),
            ("previous_events", 0.2),
            ("medication_non_adherence", 0.15),
            ("social_determinants", 0.1)
        ]
        
        # Generate intervention targets
        intervention_targets = []
        if strata_stats["high"]["count"] > 0:
            intervention_targets.append("intensive_case_management")
            intervention_targets.append("medication_optimization")
        if strata_stats["medium"]["count"] > 0:
            intervention_targets.append("patient_education")
            intervention_targets.append("regular_monitoring")
        
        stratification = RiskStratification(
            cohort_id=cohort_id,
            risk_category=risk_category,
            strata=strata_stats,
            high_risk_count=strata_stats["high"]["count"],
            medium_risk_count=strata_stats["medium"]["count"],
            low_risk_count=strata_stats["low"]["count"],
            risk_factors=risk_factors,
            intervention_targets=intervention_targets
        )
        
        logger.info(
            f"Population risk stratified: {risk_category} - "
            f"High={stratification.high_risk_count}, "
            f"Medium={stratification.medium_risk_count}, "
            f"Low={stratification.low_risk_count}"
        )
        
        return stratification
    
    # === Outcome Tracking ===
    
    def track_longitudinal_outcomes(
        self,
        cohort_id: str,
        outcome_type: str,
        baseline_date: datetime,
        follow_up_periods: List[int]  # Days
    ) -> OutcomeTracking:
        """
        Track longitudinal outcomes for a population.
        
        Args:
            cohort_id: Cohort identifier
            outcome_type: Type of outcome being tracked
            baseline_date: Baseline date for tracking
            follow_up_periods: List of follow-up periods in days
            
        Returns:
            OutcomeTracking object
        """
        if cohort_id not in self.cohorts:
            raise ValueError(f"Cohort not found: {cohort_id}")
        
        cohort = self.cohorts[cohort_id]
        patients = [self.patient_data[pid] for pid in cohort.patient_ids if pid in self.patient_data]
        
        outcome_data = {}
        survival_rates = {}
        readmission_rates = {}
        qol_scores = {}
        
        for period in follow_up_periods:
            # Simulate outcome data (in production, this would query real data)
            surviving = sum(1 for p in patients if p.get("survived_to_day", 365) >= period)
            survival_rate = surviving / len(patients) if patients else 0
            
            # Readmission rate (simulated)
            readmission_rate = max(0.05, 0.20 - (period / 365) * 0.10)
            
            # Quality of life score (simulated, 0-100)
            qol_score = min(100, 60 + (period / 365) * 20)
            
            outcome_data[period] = {
                "surviving_count": surviving,
                "survival_rate": survival_rate,
                "readmission_rate": readmission_rate,
                "qol_score": qol_score
            }
            
            survival_rates[period] = survival_rate
            readmission_rates[period] = readmission_rate
            qol_scores[period] = qol_score
        
        # Assess intervention effectiveness (simulated)
        intervention_effectiveness = {
            "medication_adherence": 0.78,
            "lifestyle_modification": 0.65,
            "regular_monitoring": 0.82,
            "patient_education": 0.71
        }
        
        tracking = OutcomeTracking(
            cohort_id=cohort_id,
            outcome_type=outcome_type,
            baseline_date=baseline_date,
            follow_up_periods=follow_up_periods,
            outcome_data=outcome_data,
            survival_rates=survival_rates,
            readmission_rates=readmission_rates,
            quality_of_life_scores=qol_scores,
            intervention_effectiveness=intervention_effectiveness
        )
        
        logger.info(
            f"Longitudinal outcomes tracked for {len(patients)} patients "
            f"over {len(follow_up_periods)} periods"
        )
        
        return tracking
    
    # === Strategic Analytics ===
    
    def generate_strategic_report(
        self,
        cohort_id: str,
        include_trends: bool = True,
        include_risk_stratification: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive strategic analytics report.
        
        Args:
            cohort_id: Cohort identifier
            include_trends: Whether to include trend analysis
            include_risk_stratification: Whether to include risk stratification
            
        Returns:
            Strategic analytics report
        """
        if cohort_id not in self.cohorts:
            raise ValueError(f"Cohort not found: {cohort_id}")
        
        cohort = self.cohorts[cohort_id]
        
        # Calculate population metrics
        metrics = self.calculate_population_metrics(cohort_id)
        
        report = {
            "cohort_id": cohort_id,
            "cohort_name": cohort.name,
            "cohort_type": cohort.cohort_type.value,
            "report_date": datetime.now().isoformat(),
            "population_overview": {
                "total_patients": metrics.population_size,
                "age_distribution": metrics.age_distribution,
                "gender_distribution": metrics.gender_distribution
            },
            "clinical_profile": {
                "condition_prevalence": metrics.condition_prevalence,
                "avg_risk_scores": metrics.avg_risk_scores,
                "outcome_rates": metrics.outcome_rates,
                "top_comorbidities": metrics.comorbidity_patterns[:5],
                "treatment_patterns": metrics.treatment_patterns
            }
        }
        
        # Add risk stratification
        if include_risk_stratification:
            try:
                risk_strat = self.stratify_population_risk(cohort_id, "overall")
                report["risk_stratification"] = {
                    "high_risk": risk_strat.high_risk_count,
                    "medium_risk": risk_strat.medium_risk_count,
                    "low_risk": risk_strat.low_risk_count,
                    "intervention_targets": risk_strat.intervention_targets
                }
            except Exception as e:
                logger.warning(f"Risk stratification failed: {e}")
        
        # Add strategic recommendations
        report["strategic_recommendations"] = self._generate_recommendations(metrics)
        
        logger.info(f"Strategic report generated for cohort {cohort_id}")
        
        return report
    
    def _generate_recommendations(self, metrics: PopulationMetrics) -> List[str]:
        """Generate strategic recommendations based on metrics."""
        recommendations = []
        
        # Check outcome rates
        if metrics.outcome_rates.get("declined", 0) > 0.2:
            recommendations.append("Investigate factors contributing to decline in outcomes")
        
        # Check comorbidity burden
        if len(metrics.comorbidity_patterns) > 3:
            recommendations.append("Implement comprehensive care coordination for complex patients")
        
        # Check risk scores
        avg_risks = list(metrics.avg_risk_scores.values())
        if avg_risks and statistics.mean(avg_risks) > 0.6:
            recommendations.append("Consider population-level preventive interventions")
        
        # Check treatment diversity
        if len(metrics.treatment_patterns) < 3:
            recommendations.append("Evaluate treatment adherence to guidelines")
        
        return recommendations


def create_population_insights_engine() -> PopulationInsightsEngine:
    """
    Factory function to create a PopulationInsightsEngine instance.
    
    Returns:
        Configured PopulationInsightsEngine
    """
    return PopulationInsightsEngine()
