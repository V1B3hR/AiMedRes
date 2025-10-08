#!/usr/bin/env python3
"""
Clinical Decision Support System for AiMedRes

A comprehensive system for clinical decision making that includes:
- Risk stratification models for early intervention
- Explainable AI components for clinician interpretation
- EHR integration capabilities
- Regulatory compliance features

This module serves as the core orchestrator for clinical decision support workflows.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json
from pathlib import Path

# Import existing components
from aimedres.agents.specialized_medical_agents import MedicalKnowledgeAgent
from secure_medical_processor import SecureMedicalDataProcessor

logger = logging.getLogger("ClinicalDecisionSupport")


@dataclass
class RiskAssessment:
    """Risk assessment result structure"""
    patient_id: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    risk_score: float  # 0.0 to 1.0
    condition: str
    assessment_date: datetime
    interventions: List[str]
    confidence: float
    explanation: Dict[str, Any]
    next_assessment_date: Optional[datetime] = None


@dataclass
class InterventionRecommendation:
    """Clinical intervention recommendation"""
    intervention_id: str
    intervention_type: str
    priority: str  # 'URGENT', 'HIGH', 'MEDIUM', 'LOW'
    description: str
    rationale: str
    expected_outcome: str
    contraindications: List[str]
    monitoring_requirements: List[str]
    estimated_timeline: str


class RiskStratificationEngine:
    """
    Advanced risk stratification engine for multiple medical conditions.
    
    Features:
    - Multi-condition risk assessment
    - Temporal progression tracking
    - Personalized intervention recommendations
    - Early intervention triggers
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.risk_thresholds = {
            'alzheimer': {'low': 0.2, 'medium': 0.5, 'high': 0.8},
            'cardiovascular': {'low': 0.25, 'medium': 0.55, 'high': 0.75},
            'diabetes': {'low': 0.3, 'medium': 0.6, 'high': 0.85},
            'stroke': {'low': 0.15, 'medium': 0.45, 'high': 0.7},
            'parkinsons': {'low': 0.2, 'medium': 0.5, 'high': 0.75},
            'als': {'low': 0.25, 'medium': 0.6, 'high': 0.8}
        }
        self.intervention_database = self._load_intervention_database()
        
    def assess_risk(self, patient_data: Dict[str, Any], condition: str) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for a specific condition.
        
        Args:
            patient_data: Patient clinical data
            condition: Medical condition to assess
            
        Returns:
            RiskAssessment object with detailed results
        """
        # Calculate base risk score
        risk_score = self._calculate_risk_score(patient_data, condition)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score, condition)
        
        # Generate interventions
        interventions = self._recommend_interventions(risk_score, risk_level, condition, patient_data)
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(patient_data, condition)
        
        # Generate explanation
        explanation = self._generate_explanation(patient_data, condition, risk_score)
        
        # Calculate next assessment date
        next_assessment = self._calculate_next_assessment_date(risk_level, condition)
        
        return RiskAssessment(
            patient_id=patient_data.get('patient_id', 'unknown'),
            risk_level=risk_level,
            risk_score=risk_score,
            condition=condition,
            assessment_date=datetime.now(),
            interventions=interventions,
            confidence=confidence,
            explanation=explanation,
            next_assessment_date=next_assessment
        )
    
    def _calculate_risk_score(self, patient_data: Dict[str, Any], condition: str) -> float:
        """Calculate condition-specific risk score."""
        if condition == 'alzheimer':
            return self._calculate_alzheimer_risk(patient_data)
        elif condition == 'cardiovascular':
            return self._calculate_cardiovascular_risk(patient_data)
        elif condition == 'diabetes':
            return self._calculate_diabetes_risk(patient_data)
        elif condition == 'stroke':
            return self._calculate_stroke_risk(patient_data)
        elif condition == 'parkinsons':
            return self._calculate_parkinsons_risk(patient_data)
        elif condition == 'als':
            return self._calculate_als_risk(patient_data)
        else:
            # Generic risk calculation
            return self._calculate_generic_risk(patient_data)
    
    def _calculate_alzheimer_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate Alzheimer's disease risk using established factors."""
        risk_score = 0.0
        
        # Age factor (strongest predictor)
        age = patient_data.get('Age', 65)
        if age >= 85:
            risk_score += 0.4
        elif age >= 75:
            risk_score += 0.3
        elif age >= 65:
            risk_score += 0.2
        
        # MMSE score (cognitive assessment)
        mmse = patient_data.get('MMSE', 30)
        if mmse < 20:
            risk_score += 0.3
        elif mmse < 24:
            risk_score += 0.2
        elif mmse < 27:
            risk_score += 0.1
        
        # CDR (Clinical Dementia Rating)
        cdr = patient_data.get('CDR', 0)
        if cdr >= 1.0:
            risk_score += 0.25
        elif cdr >= 0.5:
            risk_score += 0.15
        
        # Brain volume indicators
        nwbv = patient_data.get('nWBV', 0.8)  # Normalized Whole Brain Volume
        if nwbv < 0.7:
            risk_score += 0.15
        elif nwbv < 0.75:
            risk_score += 0.1
        
        # Education level (protective factor)
        education = patient_data.get('EDUC', 12)
        if education < 8:
            risk_score += 0.1
        elif education >= 16:
            risk_score -= 0.05  # Protective effect
        
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_cardiovascular_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate cardiovascular disease risk."""
        risk_score = 0.0
        
        # Age and gender
        age = patient_data.get('Age', 50)
        gender = patient_data.get('M/F', 1)  # 1 = Male, 0 = Female
        
        if gender == 1:  # Male
            if age >= 45:
                risk_score += 0.2
        else:  # Female
            if age >= 55:
                risk_score += 0.2
        
        # Hypertension
        if patient_data.get('hypertension', False):
            risk_score += 0.25
        
        # Diabetes
        if patient_data.get('diabetes', False):
            risk_score += 0.2
        
        # Smoking
        if patient_data.get('smoking', False):
            risk_score += 0.2
        
        # Cholesterol
        cholesterol = patient_data.get('cholesterol', 200)
        if cholesterol > 240:
            risk_score += 0.15
        elif cholesterol > 200:
            risk_score += 0.1
        
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_diabetes_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate diabetes risk."""
        risk_score = 0.0
        
        # BMI
        bmi = patient_data.get('BMI', 25)
        if bmi >= 30:
            risk_score += 0.3
        elif bmi >= 25:
            risk_score += 0.2
        
        # Age
        age = patient_data.get('Age', 40)
        if age >= 45:
            risk_score += 0.2
        
        # Family history
        if patient_data.get('family_history_diabetes', False):
            risk_score += 0.25
        
        # Hypertension
        if patient_data.get('hypertension', False):
            risk_score += 0.15
        
        # Physical activity
        if not patient_data.get('regular_exercise', True):
            risk_score += 0.1
        
        return min(1.0, max(0.0, risk_score))
    
    def _calculate_stroke_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate stroke risk."""
        risk_score = 0.0
        
        # Age
        age = patient_data.get('Age', 50)
        if age >= 75:
            risk_score += 0.3
        elif age >= 65:
            risk_score += 0.2
        elif age >= 55:
            risk_score += 0.1
        
        # Hypertension
        if patient_data.get('hypertension', False):
            risk_score += 0.3
        
        # Atrial fibrillation
        if patient_data.get('atrial_fibrillation', False):
            risk_score += 0.25
        
        # Diabetes
        if patient_data.get('diabetes', False):
            risk_score += 0.15
        
        # Previous stroke/TIA
        if patient_data.get('previous_stroke', False):
            risk_score += 0.2
        
        return min(1.0, max(0.0, risk_score))

    def _calculate_parkinsons_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate Parkinson's disease risk."""
        risk_score = 0.0

        # Age
        age = patient_data.get('Age', 60)
        if age >= 70:
            risk_score += 0.3
        elif age >= 60:
            risk_score += 0.2

        # Gender (higher risk in males)
        if patient_data.get('M/F', 1) == 1:
            risk_score += 0.1

        # Family history
        if patient_data.get('family_history_parkinsons', False):
            risk_score += 0.25

        # Motor symptoms (UPDRS)
        updrs = patient_data.get('UPDRS_motor_score', 0)
        if updrs > 20:
            risk_score += 0.3
        elif updrs > 10:
            risk_score += 0.2

        # Non-motor symptoms
        if patient_data.get('smell_test_score', 10) < 5:
            risk_score += 0.15
        if patient_data.get('REM_sleep_disorder', False):
            risk_score += 0.1

        return min(1.0, max(0.0, risk_score))

    def _calculate_als_risk(self, patient_data: Dict[str, Any]) -> float:
        """Calculate Amyotrophic Lateral Sclerosis (ALS) risk."""
        risk_score = 0.0

        # Age
        age = patient_data.get('Age', 55)
        if 60 <= age <= 75:
            risk_score += 0.25
        
        # Gender (slightly higher in males)
        if patient_data.get('M/F', 1) == 1:
            risk_score += 0.1
            
        # Family history
        if patient_data.get('family_history_als', False):
            risk_score += 0.4  # Strong predictor

        # Functional rating (ALSFRS-R)
        alsfrs = patient_data.get('ALSFRS_R_score', 48)
        if alsfrs < 40:
            risk_score += 0.3
        elif alsfrs < 44:
            risk_score += 0.2
        
        # Respiratory function
        fvc = patient_data.get('fvc_percent', 100)
        if fvc < 70:
            risk_score += 0.2
        elif fvc < 80:
            risk_score += 0.1

        return min(1.0, max(0.0, risk_score))
    
    def _calculate_generic_risk(self, patient_data: Dict[str, Any]) -> float:
        """Generic risk calculation for unknown conditions."""
        # Simple age-based risk
        age = patient_data.get('Age', 50)
        return min(1.0, max(0.0, (age - 30) / 100))
    
    def _determine_risk_level(self, risk_score: float, condition: str) -> str:
        """Determine risk level based on score and condition-specific thresholds."""
        thresholds = self.risk_thresholds.get(condition, self.risk_thresholds['alzheimer'])
        
        if risk_score >= thresholds['high']:
            return 'HIGH'
        elif risk_score >= thresholds['medium']:
            return 'MEDIUM'
        elif risk_score >= thresholds['low']:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _recommend_interventions(self, risk_score: float, risk_level: str, 
                               condition: str, patient_data: Dict[str, Any]) -> List[str]:
        """Recommend appropriate interventions based on risk assessment."""
        interventions = []
        
        # Get condition-specific interventions
        condition_interventions = self.intervention_database.get(condition, {})
        risk_interventions = condition_interventions.get(risk_level.lower(), [])
        
        # Add personalized interventions based on patient data
        if condition == 'alzheimer':
            if patient_data.get('MMSE', 30) < 24:
                interventions.append('cognitive_assessment_referral')
            if patient_data.get('Age', 0) > 75:
                interventions.append('fall_prevention_program')
        
        elif condition == 'cardiovascular':
            if patient_data.get('smoking', False):
                interventions.append('smoking_cessation_program')
            if patient_data.get('BMI', 25) > 30:
                interventions.append('weight_management_program')
        
        # Combine general and specific interventions
        interventions.extend(risk_interventions)
        
        return list(set(interventions))  # Remove duplicates
    
    def _calculate_confidence(self, patient_data: Dict[str, Any], condition: str) -> float:
        """Calculate confidence based on data completeness and quality."""
        required_fields = {
            'alzheimer': ['Age', 'MMSE', 'CDR', 'EDUC'],
            'cardiovascular': ['Age', 'M/F', 'hypertension', 'diabetes'],
            'diabetes': ['Age', 'BMI', 'family_history_diabetes'],
            'stroke': ['Age', 'hypertension', 'diabetes'],
            'parkinsons': ['Age', 'M/F', 'UPDRS_motor_score', 'family_history_parkinsons'],
            'als': ['Age', 'M/F', 'ALSFRS_R_score', 'fvc_percent', 'family_history_als']
        }
        
        fields = required_fields.get(condition, ['Age'])
        available_fields = sum(1 for field in fields if field in patient_data and pd.notna(patient_data[field]))
        
        return available_fields / len(fields)
    
    def _generate_explanation(self, patient_data: Dict[str, Any], 
                            condition: str, risk_score: float) -> Dict[str, Any]:
        """Generate detailed explanation for the risk assessment."""
        explanation = {
            'primary_factors': [],
            'protective_factors': [],
            'modifiable_factors': [],
            'monitoring_indicators': []
        }
        
        if condition == 'alzheimer':
            age = patient_data.get('Age', 65)
            if age > 75:
                explanation['primary_factors'].append(f'Advanced age ({age} years)')
            
            mmse = patient_data.get('MMSE', 30)
            if mmse < 24:
                explanation['primary_factors'].append(f'Cognitive impairment (MMSE: {mmse})')
            
            education = patient_data.get('EDUC', 12)
            if education >= 16:
                explanation['protective_factors'].append(f'High education level ({education} years)')
            
            explanation['modifiable_factors'] = ['Physical exercise', 'Cognitive training', 'Social engagement']
            explanation['monitoring_indicators'] = ['MMSE score', 'CDR rating', 'Brain imaging']

        if condition == 'parkinsons':
            age = patient_data.get('Age', 60)
            if age > 60:
                explanation['primary_factors'].append(f'Age over 60 ({age} years)')
            if patient_data.get('family_history_parkinsons', False):
                explanation['primary_factors'].append('Family history of Parkinson\'s')
            updrs = patient_data.get('UPDRS_motor_score', 0)
            if updrs > 10:
                explanation['primary_factors'].append(f'Motor symptoms (UPDRS: {updrs})')
            explanation['modifiable_factors'] = ['Regular exercise', 'Physical therapy']
            explanation['monitoring_indicators'] = ['UPDRS score', 'Non-motor symptom progression']

        if condition == 'als':
            if patient_data.get('family_history_als', False):
                explanation['primary_factors'].append('Family history of ALS')
            alsfrs = patient_data.get('ALSFRS_R_score', 48)
            if alsfrs < 44:
                explanation['primary_factors'].append(f'Functional impairment (ALSFRS-R: {alsfrs})')
            fvc = patient_data.get('fvc_percent', 100)
            if fvc < 80:
                explanation['primary_factors'].append(f'Reduced respiratory function (FVC: {fvc}%)')
            explanation['modifiable_factors'] = ['Nutritional support', 'Respiratory support']
            explanation['monitoring_indicators'] = ['ALSFRS-R score', 'Forced Vital Capacity (FVC)', 'Muscle strength']
        
        return explanation
    
    def _calculate_next_assessment_date(self, risk_level: str, condition: str) -> datetime:
        """Calculate recommended next assessment date."""
        assessment_intervals = {
            'HIGH': timedelta(days=90),      # 3 months
            'MEDIUM': timedelta(days=180),   # 6 months
            'LOW': timedelta(days=365),      # 1 year
            'MINIMAL': timedelta(days=730)   # 2 years
        }
        
        interval = assessment_intervals.get(risk_level, timedelta(days=365))
        return datetime.now() + interval
    
    def _load_intervention_database(self) -> Dict[str, Dict[str, List[str]]]:
        """Load intervention recommendations database."""
        return {
            'alzheimer': {
                'high': ['neurologist_referral', 'cognitive_assessment', 'medication_review', 'safety_assessment'],
                'medium': ['cognitive_training', 'regular_monitoring', 'lifestyle_counseling'],
                'low': ['annual_screening', 'lifestyle_optimization', 'education']
            },
            'cardiovascular': {
                'high': ['cardiology_referral', 'medication_optimization', 'lifestyle_intervention'],
                'medium': ['risk_factor_modification', 'regular_monitoring'],
                'low': ['lifestyle_counseling', 'annual_screening']
            },
            'diabetes': {
                'high': ['endocrinology_referral', 'intensive_monitoring', 'medication_adjustment'],
                'medium': ['lifestyle_modification', 'regular_screening'],
                'low': ['lifestyle_counseling', 'annual_screening']
            },
            'stroke': {
                'high': ['neurology_referral', 'anticoagulation_assessment', 'intensive_monitoring'],
                'medium': ['risk_factor_management', 'regular_monitoring'],
                'low': ['lifestyle_modification', 'annual_screening']
            },
            'parkinsons': {
                'high': ['neurology_referral_movement_disorder_specialist', 'medication_initiation_review', 'physical_occupational_therapy'],
                'medium': ['regular_neurology_follow_up', 'symptom_monitoring', 'exercise_program'],
                'low': ['patient_education', 'annual_monitoring']
            },
            'als': {
                'high': ['multidisciplinary_als_clinic_referral', 'respiratory_support_assessment', 'nutritional_support'],
                'medium': ['regular_neurology_follow_up', 'symptom_management', 'physical_therapy'],
                'low': ['patient_family_education', 'genetic_counseling_if_familial']
            }
        }


class ClinicalDecisionSupportSystem:
    """
    Main Clinical Decision Support System orchestrator.
    
    Coordinates risk assessment, intervention recommendations,
    and clinical workflow integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_engine = RiskStratificationEngine(config)
        self.medical_processor = SecureMedicalDataProcessor(config)
        self.assessment_history = []
        
    def comprehensive_assessment(self, patient_data: Dict[str, Any], 
                               conditions: List[str] = None) -> Dict[str, RiskAssessment]:
        """
        Perform comprehensive risk assessment for multiple conditions.
        
        Args:
            patient_data: Patient clinical data
            conditions: List of conditions to assess (default: common conditions)
            
        Returns:
            Dictionary mapping conditions to risk assessments
        """
        if conditions is None:
            conditions = ['alzheimer', 'cardiovascular', 'diabetes', 'stroke', 'parkinsons', 'als']
        
        assessments = {}
        
        for condition in conditions:
            try:
                assessment = self.risk_engine.assess_risk(patient_data, condition)
                assessments[condition] = assessment
                
                # Store in history
                self.assessment_history.append(assessment)
                
            except Exception as e:
                logger.error(f"Assessment failed for condition {condition}: {e}")
                continue
        
        return assessments
    
    def get_priority_interventions(self, assessments: Dict[str, RiskAssessment]) -> List[str]:
        """Get prioritized list of interventions across all conditions."""
        all_interventions = []
        
        # Collect interventions by risk level
        high_risk_interventions = []
        medium_risk_interventions = []
        low_risk_interventions = []
        
        for condition, assessment in assessments.items():
            if assessment.risk_level == 'HIGH':
                high_risk_interventions.extend(assessment.interventions)
            elif assessment.risk_level == 'MEDIUM':
                medium_risk_interventions.extend(assessment.interventions)
            else:
                low_risk_interventions.extend(assessment.interventions)
        
        # Prioritize by risk level
        all_interventions.extend(list(set(high_risk_interventions)))
        all_interventions.extend(list(set(medium_risk_interventions)))
        all_interventions.extend(list(set(low_risk_interventions)))
        
        return all_interventions
    
    def generate_clinical_summary(self, assessments: Dict[str, RiskAssessment]) -> Dict[str, Any]:
        """Generate comprehensive clinical summary."""
        # Calculate overall risk
        risk_scores = [assessment.risk_score for assessment in assessments.values()]
        overall_risk = np.mean(risk_scores) if risk_scores else 0.0
        
        # Count by risk level
        risk_counts = {}
        for assessment in assessments.values():
            risk_counts[assessment.risk_level] = risk_counts.get(assessment.risk_level, 0) + 1
        
        # Get priority interventions
        priority_interventions = self.get_priority_interventions(assessments)
        
        return {
            'patient_id': list(assessments.values())[0].patient_id if assessments else 'unknown',
            'assessment_date': datetime.now().isoformat(),
            'overall_risk_score': overall_risk,
            'risk_level_distribution': risk_counts,
            'conditions_assessed': list(assessments.keys()),
            'high_risk_conditions': [
                condition for condition, assessment in assessments.items() 
                if assessment.risk_level == 'HIGH'
            ],
            'priority_interventions': priority_interventions[:10],  # Top 10
            'next_assessment_date': min(
                assessment.next_assessment_date for assessment in assessments.values()
                if assessment.next_assessment_date
            ) if assessments else None,
            'confidence_scores': {
                condition: assessment.confidence 
                for condition, assessment in assessments.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'master_password': 'test_password',
        'risk_thresholds': {},
        'intervention_database': {}
    }
    
    # Create CDSS
    cdss = ClinicalDecisionSupportSystem(config)
    
    # Example patient data
    patient_data = {
        'patient_id': 'PATIENT_001',
        'Age': 78,
        'M/F': 0,  # Female
        'MMSE': 22,
        'CDR': 0.5,
        'EDUC': 14,
        'nWBV': 0.72,
        'hypertension': True,
        'diabetes': False,
        'BMI': 28.5,
        'smoking': False,
        'UPDRS_motor_score': 15,
        'family_history_parkinsons': False,
        'ALSFRS_R_score': 46,
        'fvc_percent': 85,
        'family_history_als': True
    }
    
    # Perform comprehensive assessment
    assessments = cdss.comprehensive_assessment(patient_data)
    
    # Generate summary
    summary = cdss.generate_clinical_summary(assessments)
    
    # Print results
    print("=== Clinical Decision Support Assessment ===")
    print(f"Patient ID: {summary['patient_id']}")
    print(f"Overall Risk Score: {summary['overall_risk_score']:.3f}")
    print(f"High Risk Conditions: {summary['high_risk_conditions']}")
    print(f"Priority Interventions: {summary['priority_interventions'][:5]}")
    
    for condition, assessment in assessments.items():
        print(f"\n{condition.upper()} Assessment:")
        print(f"  Risk Level: {assessment.risk_level}")
        print(f"  Risk Score: {assessment.risk_score:.3f}")
        print(f"  Confidence: {assessment.confidence:.3f}")
        print(f"  Interventions: {assessment.interventions[:3]}")
