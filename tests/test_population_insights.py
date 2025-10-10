"""
Test Suite for Advanced Memory Consolidation - Population Health Insights (P14)

Tests for:
- Cohort aggregation and analysis
- Population health trend identification
- Risk stratification at population level
- Longitudinal outcome tracking
- Strategic analytics
"""

import pytest
from datetime import datetime, timedelta

from aimedres.agent_memory.population_insights import (
    PopulationInsightsEngine,
    create_population_insights_engine,
    CohortType,
    TrendDirection,
    Cohort
)


class TestCohortManagement:
    """Tests for cohort creation and management."""
    
    def test_engine_initialization(self):
        """Test population insights engine initialization."""
        engine = create_population_insights_engine()
        assert engine is not None
        assert len(engine.cohorts) == 0
    
    def test_create_cohort(self):
        """Test creating a patient cohort."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Diabetes Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={"conditions": ["diabetes"]},
            exclusion_criteria={"age": {"max": 18}}
        )
        
        assert cohort.cohort_id is not None
        assert cohort.name == "Diabetes Cohort"
        assert cohort.cohort_type == CohortType.DISEASE_BASED
        assert len(cohort.patient_ids) == 0
    
    def test_add_patient_to_cohort(self):
        """Test adding patients to a cohort."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Hypertension Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={"conditions": ["hypertension"]},
            exclusion_criteria={}
        )
        
        patient_data = {
            "patient_id": "patient_001",
            "age": 65,
            "gender": "male",
            "conditions": ["hypertension"],
            "risk_scores": {"cardiovascular": 0.65}
        }
        
        success = engine.add_patient_to_cohort(
            cohort.cohort_id,
            "patient_001",
            patient_data
        )
        
        assert success is True
        assert "patient_001" in cohort.patient_ids
    
    def test_patient_inclusion_criteria(self):
        """Test patient inclusion criteria matching."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Senior Cohort",
            cohort_type=CohortType.AGE_BASED,
            inclusion_criteria={"age": {"min": 65}},
            exclusion_criteria={}
        )
        
        # Patient who meets criteria
        patient_match = {
            "patient_id": "patient_senior",
            "age": 70,
            "gender": "female",
            "conditions": []
        }
        
        # Patient who doesn't meet criteria
        patient_no_match = {
            "patient_id": "patient_young",
            "age": 45,
            "gender": "male",
            "conditions": []
        }
        
        success1 = engine.add_patient_to_cohort(
            cohort.cohort_id,
            "patient_senior",
            patient_match
        )
        
        success2 = engine.add_patient_to_cohort(
            cohort.cohort_id,
            "patient_young",
            patient_no_match
        )
        
        assert success1 is True
        assert success2 is False
        assert len(cohort.patient_ids) == 1
    
    def test_patient_exclusion_criteria(self):
        """Test patient exclusion criteria."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Adult Non-Pregnant Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={"age": {"min": 18}},
            exclusion_criteria={"conditions": ["pregnancy"]}
        )
        
        # Patient excluded due to pregnancy
        patient_excluded = {
            "patient_id": "patient_excluded",
            "age": 30,
            "conditions": ["pregnancy"],
            "gender": "female"
        }
        
        success = engine.add_patient_to_cohort(
            cohort.cohort_id,
            "patient_excluded",
            patient_excluded
        )
        
        assert success is False
    
    def test_populate_cohort_from_data(self):
        """Test populating cohort from patient list."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Cardiovascular Risk Cohort",
            cohort_type=CohortType.RISK_BASED,
            inclusion_criteria={"conditions": ["hypertension", "diabetes"]},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 60 + i,
                "gender": "male" if i % 2 == 0 else "female",
                "conditions": ["hypertension"] if i % 2 == 0 else ["diabetes"],
                "risk_scores": {"cardiovascular": 0.5 + (i * 0.05)}
            }
            for i in range(20)
        ]
        
        added = engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        assert added == 20
        assert len(cohort.patient_ids) == 20


class TestPopulationMetrics:
    """Tests for population-level metrics calculation."""
    
    def setup_test_cohort(self, engine):
        """Helper to set up a test cohort with patients."""
        cohort = engine.create_cohort(
            name="Test Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 50 + (i % 40),
                "gender": "male" if i % 2 == 0 else "female",
                "conditions": ["diabetes"] if i % 3 == 0 else ["hypertension"],
                "treatments": ["metformin"] if i % 3 == 0 else ["lisinopril"],
                "risk_scores": {"cardiovascular": 0.3 + (i % 10) * 0.07},
                "outcome": "improved" if i % 5 != 0 else "stable"
            }
            for i in range(100)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        return cohort
    
    def test_calculate_population_metrics(self):
        """Test calculating comprehensive population metrics."""
        engine = create_population_insights_engine()
        cohort = self.setup_test_cohort(engine)
        
        metrics = engine.calculate_population_metrics(cohort.cohort_id)
        
        assert metrics.cohort_id == cohort.cohort_id
        assert metrics.population_size == 100
        assert len(metrics.age_distribution) > 0
        assert len(metrics.gender_distribution) > 0
        assert len(metrics.condition_prevalence) > 0
    
    def test_age_distribution(self):
        """Test age distribution calculation."""
        engine = create_population_insights_engine()
        cohort = self.setup_test_cohort(engine)
        
        metrics = engine.calculate_population_metrics(cohort.cohort_id)
        
        assert "pediatric" in metrics.age_distribution or "young_adult" in metrics.age_distribution
        assert sum(metrics.age_distribution.values()) == 100
    
    def test_condition_prevalence(self):
        """Test condition prevalence calculation."""
        engine = create_population_insights_engine()
        cohort = self.setup_test_cohort(engine)
        
        metrics = engine.calculate_population_metrics(cohort.cohort_id)
        
        assert "diabetes" in metrics.condition_prevalence
        assert "hypertension" in metrics.condition_prevalence
        assert 0 <= metrics.condition_prevalence["diabetes"] <= 1
        assert 0 <= metrics.condition_prevalence["hypertension"] <= 1
    
    def test_treatment_patterns(self):
        """Test treatment pattern analysis."""
        engine = create_population_insights_engine()
        cohort = self.setup_test_cohort(engine)
        
        metrics = engine.calculate_population_metrics(cohort.cohort_id)
        
        assert len(metrics.treatment_patterns) > 0
        assert "metformin" in metrics.treatment_patterns or "lisinopril" in metrics.treatment_patterns
    
    def test_metrics_caching(self):
        """Test that metrics are cached."""
        engine = create_population_insights_engine()
        cohort = self.setup_test_cohort(engine)
        
        # Calculate once
        metrics1 = engine.calculate_population_metrics(cohort.cohort_id, use_cache=False)
        
        # Should be cached
        assert cohort.cohort_id in engine.metrics_cache
        
        # Calculate again with caching
        metrics2 = engine.calculate_population_metrics(cohort.cohort_id, use_cache=True)
        
        # Should be same object
        assert metrics1.calculation_time == metrics2.calculation_time


class TestHealthTrendAnalysis:
    """Tests for health trend analysis."""
    
    def test_analyze_improving_trend(self):
        """Test detection of improving health trends."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Trend Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Create time series data showing improvement
        base_date = datetime.now() - timedelta(days=365)
        time_series = [
            (base_date + timedelta(days=30*i), 0.5 + i * 0.05)
            for i in range(12)
        ]
        
        trend = engine.analyze_health_trends(
            cohort_id=cohort.cohort_id,
            metric_name="quality",
            time_series_data=time_series
        )
        
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.magnitude > 0
        assert 0 <= trend.confidence <= 1
    
    def test_analyze_declining_trend(self):
        """Test detection of declining health trends."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Decline Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Create time series data showing decline
        base_date = datetime.now() - timedelta(days=180)
        time_series = [
            (base_date + timedelta(days=30*i), 0.8 - i * 0.1)
            for i in range(6)
        ]
        
        trend = engine.analyze_health_trends(
            cohort_id=cohort.cohort_id,
            metric_name="survival",
            time_series_data=time_series
        )
        
        assert trend.direction == TrendDirection.DECLINING
        assert trend.magnitude > 0
    
    def test_analyze_stable_trend(self):
        """Test detection of stable trends."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Stable Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Create time series data showing stability
        base_date = datetime.now() - timedelta(days=90)
        time_series = [
            (base_date + timedelta(days=15*i), 0.75 + (i % 2) * 0.01)
            for i in range(6)
        ]
        
        trend = engine.analyze_health_trends(
            cohort_id=cohort.cohort_id,
            metric_name="recovery",
            time_series_data=time_series
        )
        
        assert trend.direction == TrendDirection.STABLE
    
    def test_trend_statistical_significance(self):
        """Test statistical significance assessment."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Significance Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Strong trend
        base_date = datetime.now() - timedelta(days=365)
        time_series = [
            (base_date + timedelta(days=30*i), 0.3 + i * 0.06)
            for i in range(12)
        ]
        
        trend = engine.analyze_health_trends(
            cohort_id=cohort.cohort_id,
            metric_name="improvement",
            time_series_data=time_series,
            min_confidence=0.7
        )
        
        assert trend.confidence >= 0.7
        assert trend.statistical_significance is True
    
    def test_trend_recommendations(self):
        """Test trend-based recommendations."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Recommendation Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Declining trend
        base_date = datetime.now() - timedelta(days=180)
        time_series = [
            (base_date + timedelta(days=30*i), 0.85 - i * 0.12)
            for i in range(6)
        ]
        
        trend = engine.analyze_health_trends(
            cohort_id=cohort.cohort_id,
            metric_name="quality",
            time_series_data=time_series
        )
        
        assert len(trend.recommendations) > 0
        assert any("investigate" in rec or "intervention" in rec for rec in trend.recommendations)


class TestRiskStratification:
    """Tests for population-level risk stratification."""
    
    def test_stratify_population_risk(self):
        """Test risk stratification of a population."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Risk Stratification Cohort",
            cohort_type=CohortType.RISK_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Add patients with varying risk levels
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 55 + i,
                "gender": "male" if i % 2 == 0 else "female",
                "conditions": ["diabetes", "hypertension"] if i > 15 else ["hypertension"],
                "risk_scores": {"cardiovascular": i * 0.05}
            }
            for i in range(30)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        stratification = engine.stratify_population_risk(
            cohort_id=cohort.cohort_id,
            risk_category="cardiovascular"
        )
        
        assert stratification.cohort_id == cohort.cohort_id
        assert stratification.high_risk_count >= 0
        assert stratification.medium_risk_count >= 0
        assert stratification.low_risk_count >= 0
        assert (stratification.high_risk_count +
                stratification.medium_risk_count +
                stratification.low_risk_count) == 30
    
    def test_risk_strata_statistics(self):
        """Test statistics for each risk stratum."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Strata Stats Cohort",
            cohort_type=CohortType.RISK_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 50 + i * 2,
                "gender": "male",
                "conditions": ["condition_a"],
                "risk_scores": {"test_risk": 0.1 + i * 0.05}
            }
            for i in range(20)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        stratification = engine.stratify_population_risk(
            cohort_id=cohort.cohort_id,
            risk_category="test_risk"
        )
        
        assert "low" in stratification.strata
        assert "medium" in stratification.strata
        assert "high" in stratification.strata
        
        for level, stats in stratification.strata.items():
            assert "count" in stats
            assert "percentage" in stats
    
    def test_intervention_targets(self):
        """Test generation of intervention targets."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Intervention Cohort",
            cohort_type=CohortType.RISK_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Create high-risk population
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 70 + i,
                "gender": "male",
                "conditions": ["diabetes", "hypertension", "ckd"],
                "risk_scores": {"overall": 0.8 + i * 0.01}
            }
            for i in range(15)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        stratification = engine.stratify_population_risk(
            cohort_id=cohort.cohort_id,
            risk_category="overall"
        )
        
        assert len(stratification.intervention_targets) > 0


class TestLongitudinalOutcomeTracking:
    """Tests for longitudinal outcome tracking."""
    
    def test_track_longitudinal_outcomes(self):
        """Test tracking outcomes over time."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Longitudinal Cohort",
            cohort_type=CohortType.OUTCOME_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Add patients
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 65,
                "gender": "male" if i % 2 == 0 else "female",
                "conditions": ["condition_a"],
                "survived_to_day": 300 + i * 5
            }
            for i in range(50)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        baseline = datetime.now() - timedelta(days=365)
        follow_ups = [30, 90, 180, 365]
        
        tracking = engine.track_longitudinal_outcomes(
            cohort_id=cohort.cohort_id,
            outcome_type="survival",
            baseline_date=baseline,
            follow_up_periods=follow_ups
        )
        
        assert tracking.cohort_id == cohort.cohort_id
        assert tracking.outcome_type == "survival"
        assert len(tracking.outcome_data) == len(follow_ups)
        assert len(tracking.survival_rates) == len(follow_ups)
    
    def test_survival_rates(self):
        """Test survival rate calculation."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Survival Cohort",
            cohort_type=CohortType.OUTCOME_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 70,
                "gender": "male",
                "survived_to_day": 365
            }
            for i in range(100)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        tracking = engine.track_longitudinal_outcomes(
            cohort_id=cohort.cohort_id,
            outcome_type="survival",
            baseline_date=datetime.now() - timedelta(days=365),
            follow_up_periods=[30, 90, 180, 365]
        )
        
        for period, rate in tracking.survival_rates.items():
            assert 0 <= rate <= 1
    
    def test_intervention_effectiveness(self):
        """Test intervention effectiveness tracking."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Intervention Cohort",
            cohort_type=CohortType.TREATMENT_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {"patient_id": f"patient_{i}", "age": 60}
            for i in range(50)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        tracking = engine.track_longitudinal_outcomes(
            cohort_id=cohort.cohort_id,
            outcome_type="treatment_response",
            baseline_date=datetime.now() - timedelta(days=180),
            follow_up_periods=[30, 90, 180]
        )
        
        assert len(tracking.intervention_effectiveness) > 0
        for intervention, effectiveness in tracking.intervention_effectiveness.items():
            assert 0 <= effectiveness <= 1


class TestStrategicAnalytics:
    """Tests for strategic analytics and reporting."""
    
    def test_generate_strategic_report(self):
        """Test generation of strategic analytics report."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Strategic Report Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 55 + (i % 30),
                "gender": "male" if i % 2 == 0 else "female",
                "conditions": ["diabetes", "hypertension"] if i % 3 == 0 else ["diabetes"],
                "treatments": ["metformin", "lisinopril"],
                "risk_scores": {"cardiovascular": 0.4 + (i % 10) * 0.05},
                "outcome": "improved" if i % 4 != 0 else "declined"
            }
            for i in range(100)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        report = engine.generate_strategic_report(
            cohort_id=cohort.cohort_id,
            include_trends=False,
            include_risk_stratification=True
        )
        
        assert "cohort_id" in report
        assert "population_overview" in report
        assert "clinical_profile" in report
        assert "strategic_recommendations" in report
        assert "risk_stratification" in report
    
    def test_strategic_recommendations(self):
        """Test generation of strategic recommendations."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Recommendations Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        # Create a population with concerning trends
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 65,
                "gender": "male",
                "conditions": ["diabetes", "hypertension", "ckd"],
                "treatments": ["metformin"],
                "risk_scores": {"overall": 0.75},
                "outcome": "declined"
            }
            for i in range(50)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        report = engine.generate_strategic_report(cohort_id=cohort.cohort_id)
        
        recommendations = report["strategic_recommendations"]
        assert len(recommendations) > 0
        assert any("outcome" in rec.lower() or "intervention" in rec.lower() 
                  for rec in recommendations)
    
    def test_comprehensive_report_structure(self):
        """Test that comprehensive report has all expected sections."""
        engine = create_population_insights_engine()
        
        cohort = engine.create_cohort(
            name="Comprehensive Cohort",
            cohort_type=CohortType.DISEASE_BASED,
            inclusion_criteria={},
            exclusion_criteria={}
        )
        
        patients = [
            {
                "patient_id": f"patient_{i}",
                "age": 60,
                "gender": "male",
                "conditions": ["diabetes"],
                "risk_scores": {"cardiovascular": 0.5}
            }
            for i in range(25)
        ]
        
        engine.populate_cohort_from_data(cohort.cohort_id, patients)
        
        report = engine.generate_strategic_report(cohort_id=cohort.cohort_id)
        
        # Check main sections
        assert "cohort_name" in report
        assert "cohort_type" in report
        assert "report_date" in report
        assert "population_overview" in report
        assert "clinical_profile" in report
        
        # Check population overview
        overview = report["population_overview"]
        assert "total_patients" in overview
        assert "age_distribution" in overview
        assert "gender_distribution" in overview
        
        # Check clinical profile
        profile = report["clinical_profile"]
        assert "condition_prevalence" in profile
        assert "outcome_rates" in profile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
