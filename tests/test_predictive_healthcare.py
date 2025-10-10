"""
Tests for Predictive Healthcare Analytics (P17)
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.analytics.predictive_healthcare import (
    create_predictive_healthcare_engine,
    PredictiveHealthcareEngine,
    TrendType,
    PreventionStrategy,
    TreatmentResponseType,
    ResourceType
)


class TestPredictiveHealthcareEngine:
    """Test suite for Predictive Healthcare Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test predictive healthcare engine."""
        return create_predictive_healthcare_engine(forecast_horizon_days=365, confidence_threshold=0.7)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.forecast_horizon_days == 365
        assert engine.confidence_threshold == 0.7
    
    def test_forecast_disease_trend(self, engine):
        """Test disease trend forecasting."""
        forecast = engine.forecast_disease_trend(
            disease_name="Alzheimers Disease",
            region="Northeast",
            current_incidence=50.0
        )
        
        assert forecast is not None
        assert forecast.disease_name == "Alzheimers Disease"
        assert forecast.trend_type in TrendType
        assert len(forecast.forecasted_incidence) > 0
        assert forecast.confidence_score > 0
    
    def test_forecast_with_historical_data(self, engine):
        """Test forecasting with custom historical data."""
        historical = [45.0 + i * 0.5 for i in range(365)]
        
        forecast = engine.forecast_disease_trend(
            disease_name="Diabetes",
            region="Midwest",
            current_incidence=50.0,
            historical_data=historical,
            forecast_days=180
        )
        
        assert forecast is not None
        assert forecast.forecast_period_days == 180
    
    def test_create_prevention_plan(self, engine):
        """Test creating personalized prevention plan."""
        plan = engine.create_prevention_plan(
            patient_id="patient_001",
            risk_conditions=["cardiovascular", "diabetes"],
            patient_risk_factors={'obesity': 0.7, 'smoking': 0.8, 'physical_inactivity': 0.6}
        )
        
        assert plan is not None
        assert len(plan.prevention_strategies) > 0
        assert 0.0 <= plan.risk_reduction_estimate <= 1.0
        assert plan.cost_effectiveness > 0
    
    def test_prevention_plan_genetic_risk(self, engine):
        """Test prevention plan with genetic risk factors."""
        plan = engine.create_prevention_plan(
            patient_id="patient_002",
            risk_conditions=["alzheimers"],
            patient_risk_factors={'genetic_risk': 0.8, 'age': 0.6}
        )
        
        assert plan is not None
        assert PreventionStrategy.EARLY_SCREENING in plan.prevention_strategies
    
    def test_record_treatment_response(self, engine):
        """Test recording treatment response."""
        response = engine.record_treatment_response(
            patient_id="patient_001",
            treatment_name="Donepezil",
            start_date=datetime.now() - timedelta(days=90),
            response_type=TreatmentResponseType.PARTIAL_RESPONSE,
            response_score=0.65,
            biomarker_changes={'mmse': 2.0},
            symptom_changes={'memory': 0.2}
        )
        
        assert response is not None
        assert response.treatment_name == "Donepezil"
        assert response.response_type == TreatmentResponseType.PARTIAL_RESPONSE
        assert engine.responses_tracked == 1
    
    def test_analyze_treatment_trajectory(self, engine):
        """Test analyzing treatment trajectory."""
        # Record multiple responses
        for i in range(3):
            engine.record_treatment_response(
                patient_id="patient_001",
                treatment_name="Donepezil",
                start_date=datetime.now() - timedelta(days=90),
                response_type=TreatmentResponseType.PARTIAL_RESPONSE,
                response_score=0.5 + i * 0.1
            )
        
        analysis = engine.analyze_treatment_trajectory(
            patient_id="patient_001",
            treatment_name="Donepezil"
        )
        
        assert analysis is not None
        assert analysis['num_assessments'] == 3
        assert 'trajectory_trend' in analysis
    
    def test_predict_treatment_outcome(self, engine):
        """Test predicting treatment outcome."""
        prediction = engine.predict_treatment_outcome(
            patient_id="patient_002",
            treatment_name="Donepezil",
            patient_profile={'age': 72, 'comorbidity_count': 1}
        )
        
        assert prediction is not None
        assert 'predicted_response_score' in prediction
        assert 'confidence' in prediction
    
    def test_optimize_resource_allocation(self, engine):
        """Test resource allocation optimization."""
        allocation = engine.optimize_resource_allocation(
            facility_id="hospital_001",
            resource_type=ResourceType.HOSPITAL_BEDS,
            current_capacity=100,
            current_utilization=0.75
        )
        
        assert allocation is not None
        assert allocation.resource_type == ResourceType.HOSPITAL_BEDS
        assert allocation.recommended_allocation > 0
        assert 0.0 <= allocation.optimization_score <= 1.0
    
    def test_resource_allocation_with_forecast(self, engine):
        """Test resource allocation with custom demand forecast."""
        forecast = [(datetime.now() + timedelta(days=i), 80 + i) for i in range(90)]
        
        allocation = engine.optimize_resource_allocation(
            facility_id="hospital_001",
            resource_type=ResourceType.ICU_BEDS,
            current_capacity=20,
            current_utilization=0.85,
            forecasted_demand=forecast
        )
        
        assert allocation is not None
        assert len(allocation.forecasted_demand) == 90
    
    def test_get_statistics(self, engine):
        """Test getting engine statistics."""
        # Generate some activity
        engine.forecast_disease_trend("Diabetes", "Region A", 40.0)
        engine.create_prevention_plan("patient_001", ["diabetes"], {'obesity': 0.7})
        
        stats = engine.get_statistics()
        
        assert stats is not None
        assert 'forecasts_generated' in stats
        assert stats['forecasts_generated'] >= 1
        assert stats['prevention_plans_created'] >= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
