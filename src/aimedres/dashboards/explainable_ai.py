#!/usr/bin/env python3
"""
Explainable AI Dashboard for Clinical Decision Support

Provides clinician-friendly interfaces for understanding AI model decisions:
- Interactive model interpretation tools
- Feature importance visualization
- Decision pathway explanations
- Uncertainty quantification
- Comparative analysis tools

This module creates interpretable visualizations and explanations for medical AI models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

# For visualization (optional imports with fallbacks)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger("ExplainableAIDashboard")


@dataclass
class FeatureImportance:
    """Feature importance data structure"""
    feature_name: str
    importance_score: float
    direction: str  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    confidence: float
    clinical_meaning: str
    normal_range: Optional[Tuple[float, float]] = None
    patient_value: Optional[float] = None


@dataclass
class DecisionExplanation:
    """Complete decision explanation structure"""
    model_prediction: str
    confidence_score: float
    primary_factors: List[FeatureImportance]
    contributing_factors: List[FeatureImportance]
    protective_factors: List[FeatureImportance]
    uncertainty_factors: List[str]
    alternative_scenarios: List[Dict[str, Any]]
    clinical_recommendations: List[str]


class ModelExplainer(ABC):
    """Abstract base class for model explainers"""
    
    @abstractmethod
    def explain_prediction(self, patient_data: Dict[str, Any], 
                         model_output: Dict[str, Any]) -> DecisionExplanation:
        """Generate explanation for a specific prediction"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, patient_data: Dict[str, Any]) -> List[FeatureImportance]:
        """Get feature importance for the patient"""
        pass


class AlzheimerExplainer(ModelExplainer):
    """Explainer specifically for Alzheimer's disease models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_definitions = self._load_feature_definitions()
        
    def explain_prediction(self, patient_data: Dict[str, Any], 
                         model_output: Dict[str, Any]) -> DecisionExplanation:
        """Generate comprehensive explanation for Alzheimer's prediction"""
        
        # Get feature importance
        feature_importance = self.get_feature_importance(patient_data)
        
        # Categorize factors
        primary_factors = [f for f in feature_importance if f.importance_score > 0.3]
        contributing_factors = [f for f in feature_importance if 0.1 < f.importance_score <= 0.3]
        protective_factors = [f for f in feature_importance if f.direction == 'NEGATIVE']
        
        # Generate uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(patient_data)
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(patient_data, model_output)
        
        # Generate clinical recommendations
        clinical_recommendations = self._generate_clinical_recommendations(patient_data, model_output)
        
        return DecisionExplanation(
            model_prediction=model_output.get('prediction', 'Unknown'),
            confidence_score=model_output.get('confidence', 0.0),
            primary_factors=primary_factors,
            contributing_factors=contributing_factors,
            protective_factors=protective_factors,
            uncertainty_factors=uncertainty_factors,
            alternative_scenarios=alternative_scenarios,
            clinical_recommendations=clinical_recommendations
        )
    
    def get_feature_importance(self, patient_data: Dict[str, Any]) -> List[FeatureImportance]:
        """Calculate feature importance for Alzheimer's assessment"""
        features = []
        
        # Age analysis
        age = patient_data.get('Age', 65)
        age_importance = min(0.4, (age - 50) / 100) if age > 50 else 0.0
        features.append(FeatureImportance(
            feature_name='Age',
            importance_score=age_importance,
            direction='POSITIVE' if age > 75 else 'NEUTRAL',
            confidence=0.95,
            clinical_meaning='Advanced age is the strongest risk factor for Alzheimer\'s disease',
            normal_range=(50, 90),
            patient_value=age
        ))
        
        # MMSE analysis
        mmse = patient_data.get('MMSE', 30)
        mmse_importance = max(0.0, (30 - mmse) / 30 * 0.5)
        features.append(FeatureImportance(
            feature_name='MMSE',
            importance_score=mmse_importance,
            direction='NEGATIVE' if mmse < 24 else 'NEUTRAL',
            confidence=0.9,
            clinical_meaning='Mini-Mental State Examination score indicates cognitive function',
            normal_range=(24, 30),
            patient_value=mmse
        ))
        
        # CDR analysis
        cdr = patient_data.get('CDR', 0)
        cdr_importance = cdr * 0.3 if cdr > 0 else 0.0
        features.append(FeatureImportance(
            feature_name='CDR',
            importance_score=cdr_importance,
            direction='POSITIVE' if cdr > 0 else 'NEUTRAL',
            confidence=0.85,
            clinical_meaning='Clinical Dementia Rating measures dementia severity',
            normal_range=(0, 0),
            patient_value=cdr
        ))
        
        # Brain volume analysis
        nwbv = patient_data.get('nWBV', 0.8)
        nwbv_importance = max(0.0, (0.8 - nwbv) * 0.4) if nwbv < 0.8 else 0.0
        features.append(FeatureImportance(
            feature_name='nWBV',
            importance_score=nwbv_importance,
            direction='NEGATIVE' if nwbv < 0.75 else 'NEUTRAL',
            confidence=0.75,
            clinical_meaning='Normalized Whole Brain Volume indicates brain atrophy',
            normal_range=(0.75, 0.85),
            patient_value=nwbv
        ))
        
        # Education analysis (protective factor)
        education = patient_data.get('EDUC', 12)
        education_importance = max(0.0, (education - 12) / 20 * 0.2) if education > 12 else 0.0
        features.append(FeatureImportance(
            feature_name='Education',
            importance_score=education_importance,
            direction='NEGATIVE',  # Protective
            confidence=0.7,
            clinical_meaning='Higher education provides cognitive reserve protection',
            normal_range=(12, 20),
            patient_value=education
        ))
        
        return features
    
    def _load_feature_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load clinical definitions for features"""
        return {
            'Age': {
                'unit': 'years',
                'normal_range': (50, 90),
                'risk_thresholds': {'high': 75, 'medium': 65},
                'clinical_significance': 'Primary risk factor for dementia'
            },
            'MMSE': {
                'unit': 'score',
                'normal_range': (24, 30),
                'risk_thresholds': {'severe': 10, 'moderate': 18, 'mild': 24},
                'clinical_significance': 'Cognitive function assessment'
            },
            'CDR': {
                'unit': 'score',
                'normal_range': (0, 0),
                'risk_thresholds': {'severe': 2, 'moderate': 1, 'mild': 0.5},
                'clinical_significance': 'Dementia severity rating'
            },
            'nWBV': {
                'unit': 'ratio',
                'normal_range': (0.75, 0.85),
                'risk_thresholds': {'high': 0.70, 'medium': 0.75},
                'clinical_significance': 'Brain volume measurement'
            }
        }
    
    def _identify_uncertainty_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identify factors that contribute to prediction uncertainty"""
        uncertainty_factors = []
        
        # Missing data
        required_fields = ['Age', 'MMSE', 'CDR', 'EDUC', 'nWBV']
        missing_fields = [field for field in required_fields if field not in patient_data]
        if missing_fields:
            uncertainty_factors.append(f"Missing clinical data: {', '.join(missing_fields)}")
        
        # Conflicting indicators
        mmse = patient_data.get('MMSE', 30)
        cdr = patient_data.get('CDR', 0)
        if mmse > 24 and cdr > 0:
            uncertainty_factors.append("Conflicting cognitive assessments (MMSE vs CDR)")
        
        # Atypical values
        age = patient_data.get('Age', 65)
        if age < 50:
            uncertainty_factors.append("Unusually young age for typical Alzheimer's onset")
        
        return uncertainty_factors
    
    def _generate_alternative_scenarios(self, patient_data: Dict[str, Any], 
                                      model_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate what-if scenarios for the patient"""
        scenarios = []
        
        # Scenario 1: Improved cognitive function
        improved_data = patient_data.copy()
        current_mmse = improved_data.get('MMSE', 30)
        improved_data['MMSE'] = min(30, current_mmse + 3)
        scenarios.append({
            'name': 'Improved Cognitive Function',
            'description': f'If MMSE improved from {current_mmse} to {improved_data["MMSE"]}',
            'changes': {'MMSE': improved_data['MMSE']},
            'expected_impact': 'Risk reduction of approximately 15-25%'
        })
        
        # Scenario 2: Early intervention
        if patient_data.get('CDR', 0) == 0.5:
            scenarios.append({
                'name': 'Early Intervention Success',
                'description': 'If cognitive training and lifestyle interventions are successful',
                'changes': {'CDR': 0, 'MMSE': min(30, current_mmse + 2)},
                'expected_impact': 'Potential to slow progression significantly'
            })
        
        return scenarios
    
    def _generate_clinical_recommendations(self, patient_data: Dict[str, Any], 
                                         model_output: Dict[str, Any]) -> List[str]:
        """Generate specific clinical recommendations"""
        recommendations = []
        
        risk_score = model_output.get('risk_score', 0.0)
        
        if risk_score > 0.7:
            recommendations.extend([
                'Immediate referral to memory disorders specialist',
                'Comprehensive neuropsychological testing',
                'Brain MRI with volumetric analysis',
                'Consider biomarker testing (CSF or PET)'
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                'Regular cognitive monitoring (3-6 months)',
                'Cognitive training program enrollment',
                'Lifestyle modification counseling',
                'Family education and support'
            ])
        else:
            recommendations.extend([
                'Annual cognitive screening',
                'Lifestyle optimization guidance',
                'Cardiovascular risk management'
            ])
        
        # Specific recommendations based on patient factors
        mmse = patient_data.get('MMSE', 30)
        if mmse < 24:
            recommendations.append('Detailed neuropsychological evaluation needed')
        
        age = patient_data.get('Age', 65)
        if age > 80:
            recommendations.append('Consider fall risk assessment and home safety evaluation')
        
        return recommendations


class DashboardGenerator:
    """
    Generates clinical dashboard components for explainable AI.
    
    Creates clinician-friendly visualizations and summaries.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainers = {
            'alzheimer': AlzheimerExplainer(config)
        }
    
    def generate_patient_dashboard(self, patient_data: Dict[str, Any], 
                                 assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete patient dashboard data"""
        
        dashboard_data = {
            'patient_info': self._format_patient_info(patient_data),
            'risk_summary': self._generate_risk_summary(assessments),
            'explanations': {},
            'visualizations': {},
            'recommendations': [],
            'generated_at': datetime.now().isoformat()
        }
        
        # Generate explanations for each condition
        for condition, assessment in assessments.items():
            if condition in self.explainers:
                explanation = self.explainers[condition].explain_prediction(
                    patient_data, assessment.__dict__
                )
                dashboard_data['explanations'][condition] = explanation
                
                # Generate visualizations
                dashboard_data['visualizations'][condition] = self._generate_visualizations(
                    explanation, condition
                )
        
        # Aggregate recommendations
        for condition, explanation in dashboard_data['explanations'].items():
            dashboard_data['recommendations'].extend(explanation.clinical_recommendations)
        
        return dashboard_data
    
    def _format_patient_info(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format patient information for display"""
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'age': patient_data.get('Age', 'Unknown'),
            'gender': 'Female' if patient_data.get('M/F') == 0 else 'Male' if patient_data.get('M/F') == 1 else 'Unknown',
            'education_years': patient_data.get('EDUC', 'Unknown'),
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    
    def _generate_risk_summary(self, assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk summary across all conditions"""
        risk_summary = {
            'total_conditions': len(assessments),
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'overall_status': 'Low Risk'
        }
        
        for condition, assessment in assessments.items():
            if hasattr(assessment, 'risk_level'):
                if assessment.risk_level == 'HIGH':
                    risk_summary['high_risk_count'] += 1
                elif assessment.risk_level == 'MEDIUM':
                    risk_summary['medium_risk_count'] += 1
                else:
                    risk_summary['low_risk_count'] += 1
        
        # Determine overall status
        if risk_summary['high_risk_count'] > 0:
            risk_summary['overall_status'] = 'High Risk'
        elif risk_summary['medium_risk_count'] > 0:
            risk_summary['overall_status'] = 'Medium Risk'
        
        return risk_summary
    
    def _generate_visualizations(self, explanation: DecisionExplanation, 
                               condition: str) -> Dict[str, Any]:
        """Generate visualization data for the explanation"""
        
        visualizations = {
            'feature_importance_chart': self._create_feature_importance_data(explanation),
            'confidence_gauge': self._create_confidence_gauge_data(explanation),
            'risk_factors_breakdown': self._create_risk_breakdown_data(explanation)
        }
        
        return visualizations
    
    def _create_feature_importance_data(self, explanation: DecisionExplanation) -> Dict[str, Any]:
        """Create data for feature importance visualization"""
        
        all_factors = (explanation.primary_factors + 
                      explanation.contributing_factors + 
                      explanation.protective_factors)
        
        # Sort by importance score
        all_factors.sort(key=lambda x: x.importance_score, reverse=True)
        
        return {
            'chart_type': 'horizontal_bar',
            'title': 'Feature Importance Analysis',
            'data': [
                {
                    'feature': factor.feature_name,
                    'importance': factor.importance_score,
                    'direction': factor.direction,
                    'clinical_meaning': factor.clinical_meaning,
                    'patient_value': factor.patient_value,
                    'normal_range': factor.normal_range
                }
                for factor in all_factors[:10]  # Top 10 features
            ]
        }
    
    def _create_confidence_gauge_data(self, explanation: DecisionExplanation) -> Dict[str, Any]:
        """Create data for confidence gauge visualization"""
        
        confidence_score = explanation.confidence_score
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = 'High'
            color = 'green'
        elif confidence_score >= 0.6:
            confidence_level = 'Medium'
            color = 'yellow'
        else:
            confidence_level = 'Low'
            color = 'red'
        
        return {
            'chart_type': 'gauge',
            'title': 'Model Confidence',
            'value': confidence_score,
            'level': confidence_level,
            'color': color,
            'uncertainty_factors': explanation.uncertainty_factors
        }
    
    def _create_risk_breakdown_data(self, explanation: DecisionExplanation) -> Dict[str, Any]:
        """Create data for risk factors breakdown"""
        
        return {
            'chart_type': 'stacked_bar',
            'title': 'Risk Factors Breakdown',
            'categories': {
                'Primary Risk Factors': len(explanation.primary_factors),
                'Contributing Factors': len(explanation.contributing_factors),
                'Protective Factors': len(explanation.protective_factors)
            },
            'details': {
                'primary': [f.feature_name for f in explanation.primary_factors],
                'contributing': [f.feature_name for f in explanation.contributing_factors],
                'protective': [f.feature_name for f in explanation.protective_factors]
            }
        }
    
    def export_dashboard_html(self, dashboard_data: Dict[str, Any]) -> str:
        """Export dashboard as HTML for web display"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Decision Support Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .high-risk {{ background-color: #ffe6e6; }}
                .medium-risk {{ background-color: #fff2e6; }}
                .low-risk {{ background-color: #e6f7e6; }}
                .factor {{ margin: 5px 0; padding: 5px; background-color: #f9f9f9; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical Decision Support Dashboard</h1>
                <p><strong>Patient ID:</strong> {dashboard_data['patient_info']['patient_id']}</p>
                <p><strong>Assessment Date:</strong> {dashboard_data['patient_info']['assessment_date']}</p>
                <p><strong>Overall Status:</strong> {dashboard_data['risk_summary']['overall_status']}</p>
            </div>
            
            <div class="section">
                <h2>Risk Summary</h2>
                <p>High Risk Conditions: {dashboard_data['risk_summary']['high_risk_count']}</p>
                <p>Medium Risk Conditions: {dashboard_data['risk_summary']['medium_risk_count']}</p>
                <p>Low Risk Conditions: {dashboard_data['risk_summary']['low_risk_count']}</p>
            </div>
            
            <div class="section">
                <h2>Clinical Recommendations</h2>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in dashboard_data['recommendations'][:10]])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Detailed Explanations</h2>
                {self._generate_explanation_html(dashboard_data['explanations'])}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_explanation_html(self, explanations: Dict[str, DecisionExplanation]) -> str:
        """Generate HTML for detailed explanations"""
        
        html_parts = []
        
        for condition, explanation in explanations.items():
            html_parts.append(f"""
                <h3>{condition.title()} Assessment</h3>
                <p><strong>Prediction:</strong> {explanation.model_prediction}</p>
                <p><strong>Confidence:</strong> {explanation.confidence_score:.2%}</p>
                
                <h4>Primary Risk Factors</h4>
                {''.join([f'<div class="factor"><strong>{f.feature_name}:</strong> {f.clinical_meaning}</div>' 
                         for f in explanation.primary_factors])}
                
                <h4>Protective Factors</h4>
                {''.join([f'<div class="factor"><strong>{f.feature_name}:</strong> {f.clinical_meaning}</div>' 
                         for f in explanation.protective_factors])}
                
                <h4>Clinical Recommendations</h4>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in explanation.clinical_recommendations])}
                </ul>
            """)
        
        return ''.join(html_parts)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {'model_type': 'alzheimer'}
    
    # Create dashboard generator
    dashboard = DashboardGenerator(config)
    
    # Example patient data
    patient_data = {
        'patient_id': 'PATIENT_001',
        'Age': 78,
        'M/F': 0,
        'MMSE': 22,
        'CDR': 0.5,
        'EDUC': 14,
        'nWBV': 0.72
    }
    
    # Mock assessment data
    from aimedres.clinical.decision_support import RiskAssessment
    
    mock_assessment = RiskAssessment(
        patient_id='PATIENT_001',
        risk_level='MEDIUM',
        risk_score=0.65,
        condition='alzheimer',
        assessment_date=datetime.now(),
        interventions=['cognitive_training', 'regular_monitoring'],
        confidence=0.8,
        explanation={}
    )
    
    assessments = {'alzheimer': mock_assessment}
    
    # Generate dashboard
    dashboard_data = dashboard.generate_patient_dashboard(patient_data, assessments)
    
    # Print summary
    print("=== Explainable AI Dashboard Generated ===")
    print(f"Patient: {dashboard_data['patient_info']['patient_id']}")
    print(f"Overall Status: {dashboard_data['risk_summary']['overall_status']}")
    print(f"Explanations Generated: {list(dashboard_data['explanations'].keys())}")
    
    # Show feature importance for Alzheimer's
    if 'alzheimer' in dashboard_data['explanations']:
        explanation = dashboard_data['explanations']['alzheimer']
        print(f"\nAlzheimer's Assessment:")
        print(f"Prediction: {explanation.model_prediction}")
        print(f"Confidence: {explanation.confidence_score:.2%}")
        print(f"Primary Factors: {[f.feature_name for f in explanation.primary_factors]}")
        print(f"Recommendations: {explanation.clinical_recommendations[:3]}")