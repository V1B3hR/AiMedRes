"""
Predictive Healthcare Analytics (P17)

Implements comprehensive predictive analytics capabilities with:
- Population disease trend forecasting
- Personalized prevention strategy engine
- Treatment response temporal analytics
- Resource allocation optimization algorithms

This module provides advanced predictive analytics for proactive healthcare
management, prevention strategies, and operational optimization.
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("aimedres.analytics.predictive_healthcare")


class TrendType(Enum):
    """Types of disease trends."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    SEASONAL = "seasonal"
    EPIDEMIC = "epidemic"


class PreventionStrategy(Enum):
    """Types of prevention strategies."""

    LIFESTYLE_MODIFICATION = "lifestyle_modification"
    EARLY_SCREENING = "early_screening"
    MEDICATION_PROPHYLAXIS = "medication_prophylaxis"
    BEHAVIORAL_INTERVENTION = "behavioral_intervention"
    ENVIRONMENTAL_MODIFICATION = "environmental_modification"
    COMBINED_APPROACH = "combined_approach"


class TreatmentResponseType(Enum):
    """Types of treatment responses."""

    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"
    ADVERSE_REACTION = "adverse_reaction"


class ResourceType(Enum):
    """Types of healthcare resources."""

    HOSPITAL_BEDS = "hospital_beds"
    ICU_BEDS = "icu_beds"
    STAFF_PHYSICIANS = "staff_physicians"
    STAFF_NURSES = "staff_nurses"
    MEDICAL_EQUIPMENT = "medical_equipment"
    PHARMACEUTICALS = "pharmaceuticals"
    DIAGNOSTIC_CAPACITY = "diagnostic_capacity"


@dataclass
class DiseaseTrendForecast:
    """Represents a disease trend forecast."""

    forecast_id: str
    disease_name: str
    region: str
    forecast_period_days: int
    trend_type: TrendType
    current_incidence: float  # cases per 100,000
    forecasted_incidence: List[Tuple[datetime, float]]  # (date, incidence)
    confidence_interval: Tuple[float, float]  # (lower, upper)
    key_factors: List[str]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalizedPrevention:
    """Represents a personalized prevention plan."""

    plan_id: str
    patient_id: str
    risk_conditions: List[str]
    prevention_strategies: List[PreventionStrategy]
    risk_reduction_estimate: float  # % reduction
    implementation_timeline: str
    monitoring_frequency: str
    expected_outcomes: Dict[str, Any]
    cost_effectiveness: float  # cost per QALY saved
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreatmentResponse:
    """Represents a treatment response record."""

    response_id: str
    patient_id: str
    treatment_name: str
    start_date: datetime
    assessment_date: datetime
    response_type: TreatmentResponseType
    response_score: float  # 0-1 scale
    biomarker_changes: Dict[str, float]
    symptom_changes: Dict[str, float]
    side_effects: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Represents a resource allocation plan."""

    allocation_id: str
    facility_id: str
    resource_type: ResourceType
    current_capacity: int
    current_utilization: float  # 0-1 scale
    forecasted_demand: List[Tuple[datetime, int]]  # (date, demand)
    recommended_allocation: int
    optimization_score: float
    cost_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveHealthcareEngine:
    """
    Core engine for predictive healthcare analytics.

    Provides comprehensive predictive capabilities for disease forecasting,
    personalized prevention, treatment response prediction, and resource optimization.
    """

    def __init__(self, forecast_horizon_days: int = 365, confidence_threshold: float = 0.7):
        """
        Initialize the predictive healthcare engine.

        Args:
            forecast_horizon_days: Default forecasting horizon
            confidence_threshold: Minimum confidence for predictions
        """
        self.forecast_horizon_days = forecast_horizon_days
        self.confidence_threshold = confidence_threshold

        # Storage
        self.disease_forecasts: Dict[str, DiseaseTrendForecast] = {}
        self.prevention_plans: Dict[str, PersonalizedPrevention] = {}
        self.treatment_responses: Dict[str, List[TreatmentResponse]] = defaultdict(list)
        self.resource_allocations: Dict[str, ResourceAllocation] = {}

        # Tracking
        self.forecasts_generated: int = 0
        self.prevention_plans_created: int = 0
        self.responses_tracked: int = 0
        self.allocations_optimized: int = 0

        # Performance
        self.forecast_accuracies: List[float] = []
        self.prevention_effectiveness: List[float] = []

        # Historical data (simulated)
        self._initialize_historical_data()

        logger.info(f"PredictiveHealthcareEngine initialized: horizon={forecast_horizon_days} days")

    def _initialize_historical_data(self):
        """Initialize simulated historical data for predictions."""
        self.historical_incidence = {}
        self.historical_outcomes = {}
        self.historical_utilization = {}

    # ==================== Disease Trend Forecasting ====================

    def forecast_disease_trend(
        self,
        disease_name: str,
        region: str,
        current_incidence: float,
        historical_data: Optional[List[float]] = None,
        forecast_days: Optional[int] = None,
    ) -> DiseaseTrendForecast:
        """
        Forecast disease trend using time series analysis.

        Args:
            disease_name: Name of disease
            region: Geographic region
            current_incidence: Current incidence rate (per 100,000)
            historical_data: Optional historical incidence data
            forecast_days: Number of days to forecast

        Returns:
            Disease trend forecast
        """
        forecast_id = str(uuid.uuid4())
        forecast_days = forecast_days or self.forecast_horizon_days

        # Generate or use historical data
        if historical_data is None:
            historical_data = self._generate_historical_trend(current_incidence, 365)

        # Analyze trend
        trend_type = self._analyze_trend_pattern(historical_data)

        # Generate forecast
        forecasted_incidence = self._generate_forecast(
            current_incidence, historical_data, forecast_days, trend_type
        )

        # Calculate confidence interval
        forecast_std = np.std([f[1] for f in forecasted_incidence])
        confidence_interval = (
            float(max(0, current_incidence - 1.96 * forecast_std)),
            float(current_incidence + 1.96 * forecast_std),
        )

        # Identify key factors
        key_factors = self._identify_key_factors(disease_name, trend_type)

        # Calculate confidence score
        confidence_score = self._calculate_forecast_confidence(
            historical_data, trend_type, len(historical_data)
        )

        forecast = DiseaseTrendForecast(
            forecast_id=forecast_id,
            disease_name=disease_name,
            region=region,
            forecast_period_days=forecast_days,
            trend_type=trend_type,
            current_incidence=current_incidence,
            forecasted_incidence=forecasted_incidence,
            confidence_interval=confidence_interval,
            key_factors=key_factors,
            confidence_score=confidence_score,
        )

        self.disease_forecasts[forecast_id] = forecast
        self.forecasts_generated += 1

        logger.info(
            f"Generated disease forecast: {forecast_id} for {disease_name} ({trend_type.value})"
        )
        return forecast

    def _generate_historical_trend(self, current_incidence: float, num_days: int) -> List[float]:
        """Generate simulated historical trend data."""
        # Create realistic trend with seasonality and noise
        trend = []
        for i in range(num_days):
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 365)
            noise = np.random.normal(0, 0.05)
            value = current_incidence * seasonal_factor * (1 + noise)
            trend.append(max(0, value))
        return trend

    def _analyze_trend_pattern(self, historical_data: List[float]) -> TrendType:
        """Analyze historical data to identify trend pattern."""
        if len(historical_data) < 10:
            return TrendType.STABLE

        # Calculate trend slope
        x = np.arange(len(historical_data))
        coeffs = np.polyfit(x, historical_data, 1)
        slope = coeffs[0]

        # Check for seasonality
        if len(historical_data) >= 365:
            # Simple seasonality check using autocorrelation
            mean_val = np.mean(historical_data)
            # Calculate variance of monthly means
            monthly_means = [
                np.mean(historical_data[i::30]) for i in range(min(12, len(historical_data) // 30))
            ]
            seasonal_variance = np.var(monthly_means) if len(monthly_means) > 1 else 0.0
            if seasonal_variance > 0.1 * mean_val**2:
                return TrendType.SEASONAL

        # Check for epidemic (rapid increase)
        recent_slope = np.polyfit(x[-30:], historical_data[-30:], 1)[0]
        if recent_slope > slope * 3 and recent_slope > 0.1:
            return TrendType.EPIDEMIC

        # Classify based on slope
        if abs(slope) < 0.01:
            return TrendType.STABLE
        elif slope > 0:
            return TrendType.INCREASING
        else:
            return TrendType.DECREASING

    def _generate_forecast(
        self,
        current_incidence: float,
        historical_data: List[float],
        forecast_days: int,
        trend_type: TrendType,
    ) -> List[Tuple[datetime, float]]:
        """Generate future incidence forecast."""
        forecasted = []
        base_date = datetime.now()

        # Calculate trend parameters
        if trend_type == TrendType.INCREASING:
            growth_rate = 0.001  # Daily growth rate
        elif trend_type == TrendType.DECREASING:
            growth_rate = -0.001
        elif trend_type == TrendType.EPIDEMIC:
            growth_rate = 0.01
        else:
            growth_rate = 0.0

        # Generate forecast
        for day in range(forecast_days):
            date = base_date + timedelta(days=day)

            # Base forecast
            value = current_incidence * (1 + growth_rate * day)

            # Add seasonality if applicable
            if trend_type == TrendType.SEASONAL:
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 365)
                value *= seasonal_factor

            # Add uncertainty
            value *= 1 + np.random.normal(0, 0.02)

            forecasted.append((date, max(0.0, float(value))))

        return forecasted

    def _identify_key_factors(self, disease_name: str, trend_type: TrendType) -> List[str]:
        """Identify key factors influencing disease trend."""
        factors = []

        # Disease-specific factors
        if "alzheimer" in disease_name.lower():
            factors.extend(["aging_population", "genetic_risk", "lifestyle_factors"])
        elif "cardiovascular" in disease_name.lower():
            factors.extend(["obesity_rates", "smoking_prevalence", "diet_quality"])
        elif "diabetes" in disease_name.lower():
            factors.extend(["obesity_epidemic", "sedentary_lifestyle", "dietary_patterns"])

        # Trend-specific factors
        if trend_type == TrendType.INCREASING:
            factors.append("population_aging")
        elif trend_type == TrendType.EPIDEMIC:
            factors.extend(["outbreak_event", "transmission_rate"])
        elif trend_type == TrendType.SEASONAL:
            factors.append("seasonal_patterns")

        return factors[:5]  # Return top 5

    def _calculate_forecast_confidence(
        self, historical_data: List[float], trend_type: TrendType, data_points: int
    ) -> float:
        """Calculate confidence score for forecast."""
        base_confidence = 0.5

        # More data = higher confidence
        if data_points >= 365:
            base_confidence += 0.2
        elif data_points >= 180:
            base_confidence += 0.1

        # Stable trends = higher confidence
        if trend_type in [TrendType.STABLE, TrendType.SEASONAL]:
            base_confidence += 0.2
        elif trend_type == TrendType.EPIDEMIC:
            base_confidence -= 0.1

        # Low variance = higher confidence
        if historical_data:
            variance = np.var(historical_data)
            mean = np.mean(historical_data)
            if variance < 0.1 * mean**2:
                base_confidence += 0.1

        return min(1.0, max(0.0, base_confidence))

    # ==================== Personalized Prevention ====================

    def create_prevention_plan(
        self,
        patient_id: str,
        risk_conditions: List[str],
        patient_risk_factors: Dict[str, float],
        patient_preferences: Optional[Dict[str, Any]] = None,
    ) -> PersonalizedPrevention:
        """
        Create personalized prevention strategy plan.

        Args:
            patient_id: Patient identifier
            risk_conditions: List of conditions patient is at risk for
            patient_risk_factors: Risk factor scores (0-1 scale)
            patient_preferences: Optional patient preferences

        Returns:
            Personalized prevention plan
        """
        plan_id = str(uuid.uuid4())

        # Select appropriate prevention strategies
        prevention_strategies = self._select_prevention_strategies(
            risk_conditions, patient_risk_factors, patient_preferences
        )

        # Estimate risk reduction
        risk_reduction_estimate = self._estimate_risk_reduction(
            risk_conditions, prevention_strategies, patient_risk_factors
        )

        # Create implementation timeline
        implementation_timeline = self._create_implementation_timeline(prevention_strategies)

        # Determine monitoring frequency
        monitoring_frequency = self._determine_monitoring_frequency(
            risk_conditions, risk_reduction_estimate
        )

        # Calculate expected outcomes
        expected_outcomes = self._calculate_expected_outcomes(
            risk_conditions, prevention_strategies, risk_reduction_estimate
        )

        # Calculate cost-effectiveness
        cost_effectiveness = self._calculate_cost_effectiveness(
            prevention_strategies, risk_reduction_estimate
        )

        plan = PersonalizedPrevention(
            plan_id=plan_id,
            patient_id=patient_id,
            risk_conditions=risk_conditions,
            prevention_strategies=prevention_strategies,
            risk_reduction_estimate=risk_reduction_estimate,
            implementation_timeline=implementation_timeline,
            monitoring_frequency=monitoring_frequency,
            expected_outcomes=expected_outcomes,
            cost_effectiveness=cost_effectiveness,
        )

        self.prevention_plans[plan_id] = plan
        self.prevention_plans_created += 1

        logger.info(
            f"Created prevention plan: {plan_id} with {len(prevention_strategies)} strategies"
        )
        return plan

    def _select_prevention_strategies(
        self,
        risk_conditions: List[str],
        risk_factors: Dict[str, float],
        preferences: Optional[Dict[str, Any]],
    ) -> List[PreventionStrategy]:
        """Select appropriate prevention strategies."""
        strategies = []

        # High modifiable risk factors -> lifestyle modification
        modifiable_factors = ["obesity", "smoking", "physical_inactivity"]
        if any(risk_factors.get(f, 0) > 0.6 for f in modifiable_factors):
            strategies.append(PreventionStrategy.LIFESTYLE_MODIFICATION)

        # Family history or genetic risk -> early screening
        if risk_factors.get("genetic_risk", 0) > 0.5:
            strategies.append(PreventionStrategy.EARLY_SCREENING)

        # Multiple risk factors -> combined approach
        high_risk_count = sum(1 for v in risk_factors.values() if v > 0.7)
        if high_risk_count >= 3:
            strategies.append(PreventionStrategy.COMBINED_APPROACH)

        # Disease-specific strategies
        if "cardiovascular" in str(risk_conditions).lower():
            strategies.append(PreventionStrategy.MEDICATION_PROPHYLAXIS)

        if "alzheimer" in str(risk_conditions).lower():
            strategies.append(PreventionStrategy.BEHAVIORAL_INTERVENTION)

        return list(set(strategies)) or [PreventionStrategy.EARLY_SCREENING]

    def _estimate_risk_reduction(
        self,
        risk_conditions: List[str],
        strategies: List[PreventionStrategy],
        risk_factors: Dict[str, float],
    ) -> float:
        """Estimate risk reduction from prevention strategies."""
        # Base reduction by strategy
        strategy_reductions = {
            PreventionStrategy.LIFESTYLE_MODIFICATION: 0.30,
            PreventionStrategy.EARLY_SCREENING: 0.20,
            PreventionStrategy.MEDICATION_PROPHYLAXIS: 0.40,
            PreventionStrategy.BEHAVIORAL_INTERVENTION: 0.25,
            PreventionStrategy.ENVIRONMENTAL_MODIFICATION: 0.15,
            PreventionStrategy.COMBINED_APPROACH: 0.50,
        }

        # Calculate combined reduction (not simply additive)
        total_reduction = 0.0
        for strategy in strategies:
            reduction = strategy_reductions.get(strategy, 0.2)
            # Diminishing returns for multiple strategies
            total_reduction += reduction * (1 - total_reduction)

        # Adjust for current risk level
        avg_risk = np.mean(list(risk_factors.values())) if risk_factors else 0.5
        adjusted_reduction = total_reduction * avg_risk

        return float(min(0.9, adjusted_reduction))  # Cap at 90% reduction

    def _create_implementation_timeline(self, strategies: List[PreventionStrategy]) -> str:
        """Create implementation timeline description."""
        if PreventionStrategy.COMBINED_APPROACH in strategies:
            return "Phased implementation over 6-12 months with quarterly reviews"
        elif len(strategies) >= 3:
            return "Gradual implementation over 3-6 months with monthly check-ins"
        else:
            return "Implementation within 1-3 months with bi-monthly follow-ups"

    def _determine_monitoring_frequency(
        self, risk_conditions: List[str], risk_reduction: float
    ) -> str:
        """Determine appropriate monitoring frequency."""
        if risk_reduction < 0.3 or len(risk_conditions) >= 3:
            return "Monthly monitoring recommended"
        elif risk_reduction < 0.5:
            return "Bi-monthly monitoring recommended"
        else:
            return "Quarterly monitoring recommended"

    def _calculate_expected_outcomes(
        self,
        risk_conditions: List[str],
        strategies: List[PreventionStrategy],
        risk_reduction: float,
    ) -> Dict[str, Any]:
        """Calculate expected outcomes from prevention plan."""
        return {
            "risk_reduction_percent": float(risk_reduction * 100),
            "estimated_delay_years": float(risk_reduction * 10),  # Simplified
            "quality_of_life_improvement": float(risk_reduction * 0.8),
            "conditions_prevented": len(risk_conditions),
            "strategies_implemented": len(strategies),
        }

    def _calculate_cost_effectiveness(
        self, strategies: List[PreventionStrategy], risk_reduction: float
    ) -> float:
        """Calculate cost-effectiveness (cost per QALY saved)."""
        # Simplified cost calculation
        strategy_costs = {
            PreventionStrategy.LIFESTYLE_MODIFICATION: 5000,
            PreventionStrategy.EARLY_SCREENING: 3000,
            PreventionStrategy.MEDICATION_PROPHYLAXIS: 8000,
            PreventionStrategy.BEHAVIORAL_INTERVENTION: 6000,
            PreventionStrategy.ENVIRONMENTAL_MODIFICATION: 2000,
            PreventionStrategy.COMBINED_APPROACH: 12000,
        }

        total_cost = sum(strategy_costs.get(s, 5000) for s in strategies)

        # Estimate QALYs saved
        qalys_saved = risk_reduction * 5  # Simplified assumption

        if qalys_saved > 0:
            return float(total_cost / qalys_saved)
        else:
            return float(total_cost)

    # ==================== Treatment Response Analytics ====================

    def record_treatment_response(
        self,
        patient_id: str,
        treatment_name: str,
        start_date: datetime,
        response_type: TreatmentResponseType,
        response_score: float,
        biomarker_changes: Optional[Dict[str, float]] = None,
        symptom_changes: Optional[Dict[str, float]] = None,
    ) -> TreatmentResponse:
        """
        Record treatment response for temporal analytics.

        Args:
            patient_id: Patient identifier
            treatment_name: Name of treatment
            start_date: Treatment start date
            response_type: Type of response observed
            response_score: Response score (0-1 scale)
            biomarker_changes: Optional biomarker changes
            symptom_changes: Optional symptom changes

        Returns:
            Treatment response record
        """
        response_id = str(uuid.uuid4())

        response = TreatmentResponse(
            response_id=response_id,
            patient_id=patient_id,
            treatment_name=treatment_name,
            start_date=start_date,
            assessment_date=datetime.now(),
            response_type=response_type,
            response_score=response_score,
            biomarker_changes=biomarker_changes or {},
            symptom_changes=symptom_changes or {},
            side_effects=self._identify_side_effects(response_type, response_score),
        )

        self.treatment_responses[patient_id].append(response)
        self.responses_tracked += 1

        logger.info(f"Recorded treatment response: {response_id} ({response_type.value})")
        return response

    def _identify_side_effects(
        self, response_type: TreatmentResponseType, response_score: float
    ) -> List[str]:
        """Identify potential side effects based on response."""
        side_effects = []

        if response_type == TreatmentResponseType.ADVERSE_REACTION:
            side_effects = ["severe_reaction", "treatment_discontinued"]
        elif response_score < 0.5:
            side_effects = ["mild_intolerance", "suboptimal_response"]

        return side_effects

    def analyze_treatment_trajectory(
        self, patient_id: str, treatment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze treatment response trajectory over time.

        Args:
            patient_id: Patient identifier
            treatment_name: Optional specific treatment to analyze

        Returns:
            Treatment trajectory analysis
        """
        analysis_id = str(uuid.uuid4())

        responses = self.treatment_responses.get(patient_id, [])

        if treatment_name:
            responses = [r for r in responses if r.treatment_name == treatment_name]

        if not responses:
            return {"error": "No treatment data found"}

        # Sort by assessment date
        responses.sort(key=lambda x: x.assessment_date)

        # Calculate trajectory metrics
        response_scores = [r.response_score for r in responses]
        trend = (
            "improving"
            if len(response_scores) > 1 and response_scores[-1] > response_scores[0]
            else "declining"
        )

        # Response type distribution
        response_distribution = defaultdict(int)
        for r in responses:
            response_distribution[r.response_type.value] += 1

        # Time to response
        if responses:
            time_to_response = (responses[0].assessment_date - responses[0].start_date).days
        else:
            time_to_response = 0

        return {
            "analysis_id": analysis_id,
            "patient_id": patient_id,
            "num_assessments": len(responses),
            "trajectory_trend": trend,
            "current_response_score": float(response_scores[-1]) if response_scores else 0.0,
            "average_response_score": float(np.mean(response_scores)) if response_scores else 0.0,
            "response_distribution": dict(response_distribution),
            "time_to_response_days": time_to_response,
            "side_effects_reported": sum(len(r.side_effects) for r in responses),
        }

    def predict_treatment_outcome(
        self, patient_id: str, treatment_name: str, patient_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict treatment outcome based on patient profile and historical data.

        Args:
            patient_id: Patient identifier
            treatment_name: Name of treatment
            patient_profile: Patient characteristics

        Returns:
            Treatment outcome prediction
        """
        prediction_id = str(uuid.uuid4())

        # Analyze similar patients (simplified)
        similar_responses = []
        for pid, responses in self.treatment_responses.items():
            treatment_responses = [r for r in responses if r.treatment_name == treatment_name]
            if treatment_responses:
                similar_responses.extend(treatment_responses)

        # Calculate predicted outcome
        if similar_responses:
            avg_score = np.mean([r.response_score for r in similar_responses])
            success_rate = sum(
                1
                for r in similar_responses
                if r.response_type
                in [TreatmentResponseType.COMPLETE_RESPONSE, TreatmentResponseType.PARTIAL_RESPONSE]
            ) / len(similar_responses)
        else:
            avg_score = 0.7  # Default assumption
            success_rate = 0.6

        # Adjust for patient-specific factors
        age_factor = patient_profile.get("age", 50) / 100
        comorbidity_factor = 1 - (patient_profile.get("comorbidity_count", 0) * 0.1)

        adjusted_score = avg_score * (1 - age_factor * 0.2) * comorbidity_factor
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        return {
            "prediction_id": prediction_id,
            "patient_id": patient_id,
            "treatment_name": treatment_name,
            "predicted_response_score": float(adjusted_score),
            "predicted_success_rate": float(success_rate * comorbidity_factor),
            "confidence": float(len(similar_responses) / 100) if similar_responses else 0.5,
            "based_on_cases": len(similar_responses),
            "estimated_time_to_response_days": int(30 + np.random.randint(0, 60)),
        }

    # ==================== Resource Allocation Optimization ====================

    def optimize_resource_allocation(
        self,
        facility_id: str,
        resource_type: ResourceType,
        current_capacity: int,
        current_utilization: float,
        forecasted_demand: Optional[List[Tuple[datetime, int]]] = None,
    ) -> ResourceAllocation:
        """
        Optimize resource allocation based on demand forecasting.

        Args:
            facility_id: Facility identifier
            resource_type: Type of resource
            current_capacity: Current resource capacity
            current_utilization: Current utilization rate (0-1)
            forecasted_demand: Optional demand forecast

        Returns:
            Resource allocation plan
        """
        allocation_id = str(uuid.uuid4())

        # Generate demand forecast if not provided
        if forecasted_demand is None:
            forecasted_demand = self._forecast_resource_demand(
                resource_type, current_capacity, current_utilization
            )

        # Calculate recommended allocation
        recommended_allocation = self._calculate_optimal_capacity(
            current_capacity, current_utilization, forecasted_demand
        )

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            current_capacity, recommended_allocation, forecasted_demand
        )

        # Estimate cost impact
        cost_impact = self._estimate_cost_impact(
            resource_type, current_capacity, recommended_allocation
        )

        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            facility_id=facility_id,
            resource_type=resource_type,
            current_capacity=current_capacity,
            current_utilization=current_utilization,
            forecasted_demand=forecasted_demand,
            recommended_allocation=recommended_allocation,
            optimization_score=optimization_score,
            cost_impact=cost_impact,
        )

        self.resource_allocations[allocation_id] = allocation
        self.allocations_optimized += 1

        logger.info(f"Optimized resource allocation: {allocation_id} ({resource_type.value})")
        return allocation

    def _forecast_resource_demand(
        self, resource_type: ResourceType, current_capacity: int, current_utilization: float
    ) -> List[Tuple[datetime, int]]:
        """Forecast resource demand."""
        forecast = []
        base_date = datetime.now()
        current_demand = int(current_capacity * current_utilization)

        for day in range(90):  # 90-day forecast
            date = base_date + timedelta(days=day)

            # Add trend (slight growth)
            trend_factor = 1 + (day / 365) * 0.1

            # Add weekly seasonality
            day_of_week = date.weekday()
            weekly_factor = 1.2 if day_of_week < 5 else 0.8  # Weekday vs weekend

            # Add noise
            noise_factor = 1 + np.random.normal(0, 0.05)

            demand = int(current_demand * trend_factor * weekly_factor * noise_factor)
            forecast.append((date, max(0, demand)))

        return forecast

    def _calculate_optimal_capacity(
        self,
        current_capacity: int,
        current_utilization: float,
        forecasted_demand: List[Tuple[datetime, int]],
    ) -> int:
        """Calculate optimal capacity to meet forecasted demand."""
        # Get peak demand
        peak_demand = max(d for _, d in forecasted_demand)

        # Target 85% utilization at peak
        optimal_capacity = int(peak_demand / 0.85)

        # Don't recommend reducing capacity below 80% of current
        min_capacity = int(current_capacity * 0.8)

        # Don't recommend more than 150% of current capacity
        max_capacity = int(current_capacity * 1.5)

        return max(min_capacity, min(max_capacity, optimal_capacity))

    def _calculate_optimization_score(
        self,
        current_capacity: int,
        recommended_capacity: int,
        forecasted_demand: List[Tuple[datetime, int]],
    ) -> float:
        """Calculate optimization score (0-1 scale)."""
        # Check how well recommended capacity meets demand
        demands = [d for _, d in forecasted_demand]
        avg_demand = np.mean(demands)

        # Calculate expected utilization
        expected_utilization = avg_demand / recommended_capacity if recommended_capacity > 0 else 0

        # Optimal utilization is around 75-85%
        if 0.75 <= expected_utilization <= 0.85:
            utilization_score = 1.0
        elif expected_utilization < 0.75:
            utilization_score = expected_utilization / 0.75
        else:
            utilization_score = 0.85 / expected_utilization

        # Efficiency score (minimizing excess capacity)
        efficiency_score = 1.0 - abs(recommended_capacity - avg_demand) / recommended_capacity

        # Combined score
        return float((utilization_score + efficiency_score) / 2)

    def _estimate_cost_impact(
        self, resource_type: ResourceType, current_capacity: int, recommended_capacity: int
    ) -> float:
        """Estimate cost impact of capacity change."""
        # Unit costs (simplified)
        unit_costs = {
            ResourceType.HOSPITAL_BEDS: 50000,
            ResourceType.ICU_BEDS: 150000,
            ResourceType.STAFF_PHYSICIANS: 200000,
            ResourceType.STAFF_NURSES: 80000,
            ResourceType.MEDICAL_EQUIPMENT: 100000,
            ResourceType.PHARMACEUTICALS: 10000,
            ResourceType.DIAGNOSTIC_CAPACITY: 75000,
        }

        unit_cost = unit_costs.get(resource_type, 50000)
        capacity_change = recommended_capacity - current_capacity

        return float(capacity_change * unit_cost)

    # ==================== Statistics and Reporting ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        avg_forecast_accuracy = (
            np.mean(self.forecast_accuracies) if self.forecast_accuracies else 0.0
        )
        avg_prevention_effectiveness = (
            np.mean(self.prevention_effectiveness) if self.prevention_effectiveness else 0.0
        )

        return {
            "forecasts_generated": self.forecasts_generated,
            "prevention_plans_created": self.prevention_plans_created,
            "treatment_responses_tracked": self.responses_tracked,
            "resource_allocations_optimized": self.allocations_optimized,
            "average_forecast_accuracy": float(avg_forecast_accuracy),
            "average_prevention_effectiveness": float(avg_prevention_effectiveness),
            "forecast_horizon_days": self.forecast_horizon_days,
            "confidence_threshold": self.confidence_threshold,
        }


def create_predictive_healthcare_engine(
    forecast_horizon_days: int = 365, confidence_threshold: float = 0.7
) -> PredictiveHealthcareEngine:
    """
    Factory function to create a predictive healthcare engine.

    Args:
        forecast_horizon_days: Default forecasting horizon
        confidence_threshold: Minimum confidence for predictions

    Returns:
        Configured PredictiveHealthcareEngine instance
    """
    return PredictiveHealthcareEngine(
        forecast_horizon_days=forecast_horizon_days, confidence_threshold=confidence_threshold
    )
