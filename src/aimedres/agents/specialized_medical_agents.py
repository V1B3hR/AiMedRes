#!/usr/bin/env python3
"""
Specialized Medical Agents for Enhanced Multi-Agent Medical Simulation
Implements specialized agent roles with domain-specific expertise and consensus mechanisms
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import base components - create our own MedicalKnowledgeAgent to avoid import issues
from aimedres.core.cognitive_engine import UnifiedAdaptiveAgent

logger = logging.getLogger("SpecializedMedicalAgents")


class MedicalKnowledgeAgent(UnifiedAdaptiveAgent):
    """Enhanced agent with medical reasoning capabilities - base class"""

    def __init__(self, name, cognitive_profile, alive_node, resource_room, medical_model=None):
        # Pass cognitive_profile as style to parent class
        super().__init__(name, cognitive_profile, alive_node, resource_room)
        self.medical_model = medical_model
        self.cognitive_profile = cognitive_profile  # Store separately for medical reasoning
        self.medical_knowledge = {}
        self.patient_assessments = []

    def medical_reasoning(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform medical reasoning using trained model or heuristics"""

        # If no medical model, use heuristic reasoning
        if self.medical_model is None:
            return self._heuristic_medical_reasoning(patient_data)

        try:
            # Convert patient data to format expected by model
            feature_cols = ["M/F", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

            # Create feature vector
            features = []
            for col in feature_cols:
                if col in patient_data:
                    features.append(patient_data[col])
                else:
                    # Use default values for missing features
                    defaults = {
                        "M/F": 1,
                        "Age": 70,
                        "EDUC": 12,
                        "SES": 2,
                        "MMSE": 24,
                        "CDR": 0.5,
                        "eTIV": 1500,
                        "nWBV": 0.75,
                        "ASF": 1.2,
                    }
                    features.append(defaults.get(col, 0))

            # Make prediction (simulate model prediction)
            feature_array = np.array(features).reshape(1, -1)

            # Simple heuristic for prediction
            risk_score = 0
            if "MMSE" in patient_data and patient_data["MMSE"] < 24:
                risk_score += 0.3
            if "CDR" in patient_data and patient_data["CDR"] > 0:
                risk_score += 0.4
            if "Age" in patient_data and patient_data["Age"] > 75:
                risk_score += 0.2
            if "nWBV" in patient_data and patient_data["nWBV"] < 0.75:
                risk_score += 0.3

            prediction = "Demented" if risk_score > 0.5 else "Nondemented"
            confidence = min(0.9, max(0.1, abs(risk_score - 0.5) + 0.5))

            assessment = {
                "prediction": prediction,
                "confidence": confidence,
                "risk_score": risk_score,
                "reasoning": self._generate_medical_reasoning(
                    patient_data, {"prediction": prediction}
                ),
            }

            return assessment

        except Exception as e:
            logger.warning(f"Medical reasoning failed: {e}")
            return {"error": str(e), "reasoning": "Assessment failed due to technical error"}

    def _heuristic_medical_reasoning(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic medical reasoning when no model is available"""

        # Simple rule-based reasoning
        risk_factors = 0
        reasoning_points = []

        if patient_data.get("Age", 0) > 75:
            risk_factors += 1
            reasoning_points.append("Advanced age is a risk factor")

        if patient_data.get("MMSE", 30) < 24:
            risk_factors += 2
            reasoning_points.append("MMSE score indicates cognitive impairment")

        if patient_data.get("CDR", 0) > 0:
            risk_factors += 2
            reasoning_points.append("CDR score indicates functional decline")

        prediction = "Demented" if risk_factors >= 2 else "Nondemented"
        confidence = 0.6 + min(0.3, risk_factors * 0.1)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "reasoning": (
                "; ".join(reasoning_points)
                if reasoning_points
                else "No significant risk factors identified"
            ),
        }

    def _generate_medical_reasoning(
        self, patient_data: Dict[str, Any], assessment: Dict[str, Any]
    ) -> str:
        """Generate reasoning based on cognitive profile"""

        reasoning_style = (
            "analytical" if self.cognitive_profile.get("analytical", 0.5) > 0.7 else "intuitive"
        )

        if reasoning_style == "analytical":
            return f"Systematic analysis of patient data suggests {assessment['prediction']} based on objective measures"
        else:
            return f"Clinical intuition and pattern recognition indicate {assessment['prediction']} likelihood"


logger = logging.getLogger("SpecializedMedicalAgents")


@dataclass
class ConsensusMetrics:
    """Metrics for measuring consensus quality"""

    agreement_score: float  # 0.0 to 1.0
    confidence_weighted_score: float  # Weighted by individual confidences
    diversity_index: float  # Measures diversity of opinions
    risk_assessment: str  # "LOW", "MEDIUM", "HIGH"


class SpecializedMedicalAgent(MedicalKnowledgeAgent):
    """Base class for specialized medical agents"""

    def __init__(
        self,
        name: str,
        cognitive_profile: Dict[str, float],
        alive_node,
        resource_room,
        medical_model=None,
        specialization: str = "general",
    ):
        super().__init__(name, cognitive_profile, alive_node, resource_room, medical_model)
        self.specialization = specialization
        self.expertise_areas = []
        self.case_history = []
        self.learning_rate = 0.1
        self.expertise_confidence_boost = 0.2  # Boost confidence in area of expertise

    @abstractmethod
    def get_specialized_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get assessment based on specialization"""
        pass

    @abstractmethod
    def interpret_specialized_findings(self, findings: Dict[str, Any]) -> Dict[str, str]:
        """Interpret findings from specialist perspective"""
        pass

    def learn_from_case(self, case_data: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Learn from case outcomes to improve future assessments"""
        learning_case = {
            "timestamp": datetime.now().isoformat(),
            "case_data": case_data,
            "my_assessment": outcome.get("my_assessment"),
            "consensus_outcome": outcome.get("consensus"),
            "accuracy": outcome.get("accuracy", 0.5),
        }

        self.case_history.append(learning_case)

        # Adjust confidence based on accuracy
        if outcome.get("accuracy", 0.5) > 0.8:
            self.expertise_confidence_boost = min(
                0.3, self.expertise_confidence_boost + self.learning_rate * 0.1
            )
        elif outcome.get("accuracy", 0.5) < 0.5:
            self.expertise_confidence_boost = max(
                0.1, self.expertise_confidence_boost - self.learning_rate * 0.05
            )

    def get_expertise_weight(self, case_type: str) -> float:
        """Get expertise weight for this case type"""
        if case_type in self.expertise_areas:
            return 1.0 + self.expertise_confidence_boost
        return 1.0


class RadiologistAgent(SpecializedMedicalAgent):
    """Specialized agent for radiological assessments"""

    def __init__(
        self,
        name: str,
        cognitive_profile: Dict[str, float],
        alive_node,
        resource_room,
        medical_model=None,
    ):
        super().__init__(
            name, cognitive_profile, alive_node, resource_room, medical_model, "radiology"
        )
        self.expertise_areas = [
            "imaging",
            "brain_scans",
            "structural_analysis",
            "volume_measurements",
        ]

    def get_specialized_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Radiological assessment focusing on imaging data"""
        assessment = self.medical_reasoning(patient_data)

        # Add radiological specific insights
        imaging_features = {}
        if "eTIV" in patient_data:
            imaging_features["total_intracranial_volume"] = {
                "value": patient_data["eTIV"],
                "interpretation": self._interpret_etiv(patient_data["eTIV"]),
            }

        if "nWBV" in patient_data:
            imaging_features["normalized_whole_brain_volume"] = {
                "value": patient_data["nWBV"],
                "interpretation": self._interpret_nwbv(patient_data["nWBV"]),
            }

        if "ASF" in patient_data:
            imaging_features["atlas_scaling_factor"] = {
                "value": patient_data["ASF"],
                "interpretation": self._interpret_asf(patient_data["ASF"]),
            }

        assessment["radiological_findings"] = imaging_features
        assessment["imaging_risk_factors"] = self._assess_imaging_risk_factors(imaging_features)

        return assessment

    def interpret_specialized_findings(self, findings: Dict[str, Any]) -> Dict[str, str]:
        """Interpret findings from radiological perspective"""
        interpretations = {}

        if "radiological_findings" in findings:
            for feature, data in findings["radiological_findings"].items():
                interpretations[f"radiology_{feature}"] = data.get("interpretation", "Normal range")

        return interpretations

    def _interpret_etiv(self, etiv_value: float) -> str:
        """Interpret eTIV (Estimated Total Intracranial Volume)"""
        if etiv_value < 1200:
            return "Below average total intracranial volume - may indicate developmental factors"
        elif etiv_value > 1800:
            return "Above average total intracranial volume"
        return "Total intracranial volume within normal range"

    def _interpret_nwbv(self, nwbv_value: float) -> str:
        """Interpret nWBV (Normalized Whole Brain Volume)"""
        if nwbv_value < 0.7:
            return "Significantly reduced brain volume - concerning for atrophy"
        elif nwbv_value < 0.75:
            return "Mildly reduced brain volume - possible early atrophy"
        elif nwbv_value > 0.85:
            return "Well-preserved brain volume"
        return "Brain volume within normal range"

    def _interpret_asf(self, asf_value: float) -> str:
        """Interpret ASF (Atlas Scaling Factor)"""
        if asf_value < 1.0:
            return "Smaller than atlas reference - consider individual variation"
        elif asf_value > 1.4:
            return "Larger than atlas reference - consider individual variation"
        return "Atlas scaling factor within expected range"

    def _assess_imaging_risk_factors(self, imaging_features: Dict[str, Any]) -> List[str]:
        """Assess imaging-based risk factors"""
        risk_factors = []

        if "normalized_whole_brain_volume" in imaging_features:
            nwbv = imaging_features["normalized_whole_brain_volume"]["value"]
            if nwbv < 0.7:
                risk_factors.append("Severe brain atrophy detected")
            elif nwbv < 0.75:
                risk_factors.append("Mild brain atrophy present")

        return risk_factors


class NeurologistAgent(SpecializedMedicalAgent):
    """Specialized agent for neurological assessments"""

    def __init__(
        self,
        name: str,
        cognitive_profile: Dict[str, float],
        alive_node,
        resource_room,
        medical_model=None,
    ):
        super().__init__(
            name, cognitive_profile, alive_node, resource_room, medical_model, "neurology"
        )
        self.expertise_areas = [
            "cognitive_assessment",
            "mmse",
            "cdr",
            "dementia_staging",
            "neurological_examination",
        ]

    def get_specialized_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Neurological assessment focusing on cognitive measures"""
        assessment = self.medical_reasoning(patient_data)

        # Add neurological specific insights
        cognitive_profile = {}

        if "MMSE" in patient_data:
            cognitive_profile["mmse_assessment"] = {
                "score": patient_data["MMSE"],
                "interpretation": self._interpret_mmse(patient_data["MMSE"]),
                "severity": self._mmse_severity(patient_data["MMSE"]),
            }

        if "CDR" in patient_data:
            cognitive_profile["cdr_assessment"] = {
                "score": patient_data["CDR"],
                "interpretation": self._interpret_cdr(patient_data["CDR"]),
                "stage": self._cdr_stage(patient_data["CDR"]),
            }

        # Assess cognitive-imaging correlation
        if "MMSE" in patient_data and "nWBV" in patient_data:
            cognitive_profile["cognitive_imaging_correlation"] = (
                self._assess_cognitive_imaging_correlation(
                    patient_data["MMSE"], patient_data["nWBV"]
                )
            )

        assessment["neurological_findings"] = cognitive_profile
        assessment["cognitive_risk_factors"] = self._assess_cognitive_risk_factors(
            cognitive_profile
        )

        return assessment

    def interpret_specialized_findings(self, findings: Dict[str, Any]) -> Dict[str, str]:
        """Interpret findings from neurological perspective"""
        interpretations = {}

        if "neurological_findings" in findings:
            for assessment_type, data in findings["neurological_findings"].items():
                if isinstance(data, dict) and "interpretation" in data:
                    interpretations[f"neurology_{assessment_type}"] = data["interpretation"]

        return interpretations

    def _interpret_mmse(self, mmse_score: float) -> str:
        """Interpret MMSE score"""
        if mmse_score >= 24:
            return "Cognitive function within normal limits"
        elif mmse_score >= 18:
            return "Mild cognitive impairment detected"
        elif mmse_score >= 10:
            return "Moderate cognitive impairment"
        else:
            return "Severe cognitive impairment"

    def _mmse_severity(self, mmse_score: float) -> str:
        """Determine MMSE severity category"""
        if mmse_score >= 24:
            return "Normal"
        elif mmse_score >= 18:
            return "Mild"
        elif mmse_score >= 10:
            return "Moderate"
        else:
            return "Severe"

    def _interpret_cdr(self, cdr_score: float) -> str:
        """Interpret CDR score"""
        if cdr_score == 0.0:
            return "No dementia - cognitive function normal"
        elif cdr_score == 0.5:
            return "Very mild dementia - questionable cognitive decline"
        elif cdr_score == 1.0:
            return "Mild dementia - clear cognitive decline"
        elif cdr_score == 2.0:
            return "Moderate dementia - significant cognitive impairment"
        elif cdr_score >= 3.0:
            return "Severe dementia - profound cognitive impairment"
        else:
            return "CDR score interpretation unclear"

    def _cdr_stage(self, cdr_score: float) -> str:
        """Determine CDR stage"""
        if cdr_score == 0.0:
            return "Normal"
        elif cdr_score == 0.5:
            return "Very Mild"
        elif cdr_score == 1.0:
            return "Mild"
        elif cdr_score == 2.0:
            return "Moderate"
        elif cdr_score >= 3.0:
            return "Severe"
        else:
            return "Indeterminate"

    def _assess_cognitive_imaging_correlation(self, mmse: float, nwbv: float) -> str:
        """Assess correlation between cognitive and imaging findings"""
        # Simple correlation assessment
        expected_nwbv = 0.85 - (30 - mmse) * 0.02  # Rough correlation

        if abs(nwbv - expected_nwbv) < 0.05:
            return "Cognitive performance correlates well with brain volume"
        elif nwbv < expected_nwbv - 0.05:
            return "Brain atrophy more severe than cognitive symptoms suggest"
        else:
            return "Cognitive symptoms more severe than brain volume suggests"

    def _assess_cognitive_risk_factors(self, cognitive_profile: Dict[str, Any]) -> List[str]:
        """Assess cognitive risk factors"""
        risk_factors = []

        if "mmse_assessment" in cognitive_profile:
            severity = cognitive_profile["mmse_assessment"].get("severity", "Normal")
            if severity in ["Moderate", "Severe"]:
                risk_factors.append(f"Significant cognitive impairment ({severity})")

        if "cdr_assessment" in cognitive_profile:
            stage = cognitive_profile["cdr_assessment"].get("stage", "Normal")
            if stage != "Normal":
                risk_factors.append(f"Dementia stage: {stage}")

        return risk_factors


class PsychiatristAgent(SpecializedMedicalAgent):
    """Specialized agent for psychiatric and behavioral assessments"""

    def __init__(
        self,
        name: str,
        cognitive_profile: Dict[str, float],
        alive_node,
        resource_room,
        medical_model=None,
    ):
        super().__init__(
            name, cognitive_profile, alive_node, resource_room, medical_model, "psychiatry"
        )
        self.expertise_areas = [
            "behavioral_changes",
            "mood_assessment",
            "psychiatric_comorbidities",
            "functional_assessment",
        ]

    def get_specialized_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Psychiatric assessment focusing on behavioral and functional aspects"""
        assessment = self.medical_reasoning(patient_data)

        # Add psychiatric specific insights
        behavioral_profile = {}

        # Assess functional decline based on CDR and education
        if "CDR" in patient_data and "EDUC" in patient_data:
            behavioral_profile["functional_assessment"] = {
                "cdr_score": patient_data["CDR"],
                "education_years": patient_data["EDUC"],
                "functional_impact": self._assess_functional_impact(
                    patient_data["CDR"], patient_data["EDUC"]
                ),
            }

        # Assess psychosocial factors
        if "SES" in patient_data:
            behavioral_profile["psychosocial_factors"] = {
                "socioeconomic_status": patient_data["SES"],
                "risk_assessment": self._assess_psychosocial_risk(patient_data["SES"]),
            }

        # Age-related considerations
        if "Age" in patient_data:
            behavioral_profile["age_related_factors"] = {
                "age": patient_data["Age"],
                "age_risk": self._assess_age_related_risk(patient_data["Age"]),
            }

        assessment["psychiatric_findings"] = behavioral_profile
        assessment["behavioral_risk_factors"] = self._assess_behavioral_risk_factors(
            behavioral_profile
        )

        return assessment

    def interpret_specialized_findings(self, findings: Dict[str, Any]) -> Dict[str, str]:
        """Interpret findings from psychiatric perspective"""
        interpretations = {}

        if "psychiatric_findings" in findings:
            for assessment_type, data in findings["psychiatric_findings"].items():
                if isinstance(data, dict):
                    if "functional_impact" in data:
                        interpretations[f"psychiatry_functional"] = data["functional_impact"]
                    if "risk_assessment" in data:
                        interpretations[f"psychiatry_psychosocial"] = data["risk_assessment"]
                    if "age_risk" in data:
                        interpretations[f"psychiatry_age_risk"] = data["age_risk"]

        return interpretations

    def _assess_functional_impact(self, cdr_score: float, education_years: int) -> str:
        """Assess functional impact considering education level"""
        # Higher education may provide cognitive reserve
        cognitive_reserve_factor = min(1.0, education_years / 16.0)

        if cdr_score == 0.0:
            return "No functional impairment detected"
        elif cdr_score == 0.5:
            if cognitive_reserve_factor > 0.8:
                return "Very mild functional changes, good cognitive reserve may be protective"
            else:
                return "Very mild functional changes, limited cognitive reserve"
        elif cdr_score >= 1.0:
            return f"Significant functional impairment (CDR {cdr_score}), impacts daily activities"
        else:
            return "Functional status assessment inconclusive"

    def _assess_psychosocial_risk(self, ses_score: float) -> str:
        """Assess psychosocial risk factors"""
        if ses_score <= 2:
            return "Higher socioeconomic status - potential protective factors"
        elif ses_score <= 3:
            return "Moderate socioeconomic status - mixed risk profile"
        else:
            return "Lower socioeconomic status - potential additional risk factors"

    def _assess_age_related_risk(self, age: float) -> str:
        """Assess age-related risk factors"""
        if age < 65:
            return "Below typical dementia age - early onset considerations"
        elif age < 75:
            return "Early elderly - age-appropriate risk assessment"
        elif age < 85:
            return "Advanced age - increased baseline risk"
        else:
            return "Very advanced age - high baseline risk, consider multiple factors"

    def _assess_behavioral_risk_factors(self, behavioral_profile: Dict[str, Any]) -> List[str]:
        """Assess behavioral and psychiatric risk factors"""
        risk_factors = []

        if "functional_assessment" in behavioral_profile:
            functional_data = behavioral_profile["functional_assessment"]
            if functional_data.get("cdr_score", 0) >= 0.5:
                risk_factors.append("Functional decline detected")

        if "psychosocial_factors" in behavioral_profile:
            ses_data = behavioral_profile["psychosocial_factors"]
            if ses_data.get("socioeconomic_status", 1) >= 4:
                risk_factors.append("Lower socioeconomic status - additional stressors")

        if "age_related_factors" in behavioral_profile:
            age_data = behavioral_profile["age_related_factors"]
            if age_data.get("age", 65) >= 85:
                risk_factors.append("Advanced age - multiple comorbidity risk")

        return risk_factors


class SafetyValidator:
    """Safety validation system for multi-agent medical decisions"""

    def __init__(self):
        self.safety_rules = self._initialize_safety_rules()
        self.risk_thresholds = {"high_risk": 0.8, "medium_risk": 0.6, "low_risk": 0.3}

    def _initialize_safety_rules(self) -> List[Dict[str, Any]]:
        """Initialize medical safety rules"""
        return [
            {
                "name": "high_confidence_disagreement",
                "condition": lambda assessments: self._check_high_confidence_disagreement(
                    assessments
                ),
                "action": "flag_for_review",
                "severity": "high",
            },
            {
                "name": "extreme_age_outlier",
                "condition": lambda patient_data: patient_data.get("Age", 70) > 95
                or patient_data.get("Age", 70) < 18,
                "action": "require_specialist_review",
                "severity": "medium",
            },
            {
                "name": "missing_critical_data",
                "condition": lambda patient_data: self._check_missing_critical_data(patient_data),
                "action": "request_additional_data",
                "severity": "medium",
            },
        ]

    def validate_consensus(
        self, patient_data: Dict[str, Any], assessments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate consensus for safety issues"""
        safety_flags = []

        for rule in self.safety_rules:
            if (
                rule["condition"](assessments)
                if "assessments" in rule["condition"].__code__.co_varnames
                else rule["condition"](patient_data)
            ):
                safety_flags.append(
                    {
                        "rule": rule["name"],
                        "action": rule["action"],
                        "severity": rule["severity"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return {
            "safe_to_proceed": len([f for f in safety_flags if f["severity"] == "high"]) == 0,
            "safety_flags": safety_flags,
            "recommended_actions": [f["action"] for f in safety_flags],
        }

    def _check_high_confidence_disagreement(self, assessments: List[Dict[str, Any]]) -> bool:
        """Check for high confidence disagreement between agents"""
        if len(assessments) < 2:
            return False

        predictions = [a.get("prediction") for a in assessments if a.get("confidence", 0) > 0.8]
        if len(predictions) < 2:
            return False

        unique_predictions = set(predictions)
        return len(unique_predictions) > 1

    def _check_missing_critical_data(self, patient_data: Dict[str, Any]) -> bool:
        """Check for missing critical medical data"""
        critical_fields = ["Age", "MMSE", "CDR"]
        missing_fields = [field for field in critical_fields if field not in patient_data]
        return len(missing_fields) > 0


class ExplainabilityEngine:
    """Engine for generating explainable AI outputs for medical decisions"""

    def __init__(self):
        self.explanation_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize explanation templates"""
        return {
            "consensus_explanation": """
            Based on analysis from {num_specialists} specialist agents:
            - {specialist_summary}
            - Consensus reached with {agreement_level}% agreement
            - Primary factors: {key_factors}
            - Confidence level: {confidence_level}
            """,
            "disagreement_explanation": """
            Specialists show disagreement in their assessments:
            - {disagreement_details}
            - Recommended action: {recommended_action}
            """,
            "risk_explanation": """
            Risk assessment based on:
            {risk_factors_detail}
            Overall risk level: {risk_level}
            """,
        }

    def generate_consensus_explanation(self, consensus_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of consensus decision"""

        assessments = consensus_result.get("individual_assessments", [])
        metrics = consensus_result.get("consensus_metrics", {})

        # Calculate agreement level
        agreement_score = metrics.get("agreement_score", 0) * 100

        # Summarize specialist opinions
        specialist_summary = self._summarize_specialist_opinions(assessments)

        # Extract key factors
        key_factors = self._extract_key_factors(assessments)

        # Generate explanation
        explanation = {
            "summary": f"Consensus: {consensus_result.get('consensus_prediction', 'Unknown')}",
            "detailed_explanation": self.explanation_templates["consensus_explanation"].format(
                num_specialists=len(assessments),
                specialist_summary=specialist_summary,
                agreement_level=f"{agreement_score:.1f}",
                key_factors=", ".join(key_factors[:3]),  # Top 3 factors
                confidence_level=f"{consensus_result.get('consensus_confidence', 0):.2f}",
            ),
            "key_evidence": key_factors,
            "specialist_contributions": self._detail_specialist_contributions(assessments),
            "uncertainty_factors": self._identify_uncertainty_factors(metrics),
        }

        return explanation

    def _summarize_specialist_opinions(self, assessments: List[Dict[str, Any]]) -> str:
        """Summarize specialist opinions"""
        if not assessments:
            return "No specialist assessments available"

        opinions = {}
        for assessment in assessments:
            specialist = assessment.get("specialization", "unknown")
            prediction = assessment.get("prediction", "unknown")
            confidence = assessment.get("confidence", 0)

            if specialist not in opinions:
                opinions[specialist] = []
            opinions[specialist].append(f"{prediction} (confidence: {confidence:.2f})")

        return "; ".join([f"{spec}: {', '.join(preds)}" for spec, preds in opinions.items()])

    def _extract_key_factors(self, assessments: List[Dict[str, Any]]) -> List[str]:
        """Extract key factors from assessments"""
        all_factors = []
        for assessment in assessments:
            risk_factors = assessment.get("risk_factors", [])
            all_factors.extend(risk_factors)

        # Count frequency and return most common
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

        return sorted(factor_counts.keys(), key=factor_counts.get, reverse=True)

    def _detail_specialist_contributions(
        self, assessments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detail individual specialist contributions"""
        contributions = []
        for assessment in assessments:
            contribution = {
                "specialist": assessment.get("specialization", "unknown"),
                "prediction": assessment.get("prediction", "unknown"),
                "confidence": assessment.get("confidence", 0),
                "key_findings": assessment.get("risk_factors", [])[:2],  # Top 2 findings
                "reasoning": assessment.get("reasoning", "Not provided"),
            }
            contributions.append(contribution)
        return contributions

    def _identify_uncertainty_factors(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to uncertainty"""
        uncertainty_factors = []

        if metrics.get("diversity_index", 0) > 0.5:
            uncertainty_factors.append("High diversity in specialist opinions")

        if metrics.get("agreement_score", 1.0) < 0.7:
            uncertainty_factors.append("Low agreement between specialists")

        if metrics.get("confidence_weighted_score", 1.0) < 0.6:
            uncertainty_factors.append("Low overall confidence in assessments")

        return uncertainty_factors


class AgentCommunicationProtocol:
    """Protocol for secure agent-to-agent communication"""

    def __init__(self):
        self.communication_log = []
        self.security_validator = self._initialize_security_validator()

    def _initialize_security_validator(self):
        """Initialize security validation for communications"""
        return {
            "max_message_size": 10000,  # bytes
            "allowed_message_types": [
                "assessment_request",
                "assessment_response",
                "consensus_update",
                "clarification_request",
            ],
            "encryption_required": True,
        }

    def send_message(
        self, sender_agent: str, receiver_agent: str, message_type: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send secure message between agents"""

        # Validate message
        validation_result = self._validate_message(message_type, content)
        if not validation_result["valid"]:
            return {"status": "error", "message": validation_result["error"]}

        # Create communication record
        communication = {
            "id": str(len(self.communication_log) + 1),
            "timestamp": datetime.now().isoformat(),
            "sender": sender_agent,
            "receiver": receiver_agent,
            "message_type": message_type,
            "content": content,
            "status": "sent",
        }

        self.communication_log.append(communication)

        return {"status": "success", "communication_id": communication["id"]}

    def _validate_message(self, message_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate message security and format"""

        # Check message type
        if message_type not in self.security_validator["allowed_message_types"]:
            return {"valid": False, "error": f"Unauthorized message type: {message_type}"}

        # Check message size
        content_size = len(str(content).encode("utf-8"))
        if content_size > self.security_validator["max_message_size"]:
            return {"valid": False, "error": "Message too large"}

        # Check for potential security issues
        if self._contains_sensitive_data(content):
            return {"valid": False, "error": "Contains potentially sensitive data"}

        return {"valid": True}

    def _contains_sensitive_data(self, content: Dict[str, Any]) -> bool:
        """Check if content contains sensitive medical data that shouldn't be transmitted"""
        sensitive_patterns = ["ssn", "social_security", "patient_id", "mrn"]
        content_str = str(content).lower()

        return any(pattern in content_str for pattern in sensitive_patterns)

    def get_communication_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get communication history for an agent"""
        return [
            comm
            for comm in self.communication_log
            if comm["sender"] == agent_name or comm["receiver"] == agent_name
        ]


class ConsensusManager:
    """Manages consensus building between specialized agents with enhanced communication and safety"""

    def __init__(self):
        self.consensus_history = []
        self.agent_communications = []
        self.safety_validator = SafetyValidator()
        self.explainability_engine = ExplainabilityEngine()
        self.communication_protocol = AgentCommunicationProtocol()

    def build_consensus(
        self, agents: List[SpecializedMedicalAgent], patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus from multiple specialized agent assessments with enhanced safety and explainability"""

        # Validate safety before proceeding
        safety_check = self.safety_validator.validate_consensus(patient_data, [])
        if not safety_check["safe_to_proceed"]:
            return {
                "status": "safety_hold",
                "safety_flags": safety_check["safety_flags"],
                "recommended_actions": safety_check["recommended_actions"],
            }

        # Enable agent-to-agent communication for complex cases
        assessments = self._conduct_collaborative_assessment(agents, patient_data)

        if not assessments:
            return {"error": "No valid assessments obtained"}

        # Validate assessments with safety rules
        post_assessment_safety = self.safety_validator.validate_consensus(patient_data, assessments)

        # Extract predictions and confidences
        predictions = []
        confidences = []
        weights = []

        for assessment in assessments:
            if "prediction" in assessment and "confidence" in assessment:
                pred = 1 if assessment["prediction"] == "Demented" else 0
                predictions.append(pred)
                confidences.append(assessment["confidence"])
                weights.append(assessment["expertise_weight"])

        if not predictions:
            return {"error": "No predictions found in assessments"}

        # Calculate weighted consensus
        weighted_predictions = np.array(predictions) * np.array(weights)
        total_weight = sum(weights)
        consensus_score = (
            sum(weighted_predictions) / total_weight if total_weight > 0 else np.mean(predictions)
        )

        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(predictions, confidences, weights)

        # Determine final prediction
        consensus_prediction = "Demented" if consensus_score >= 0.5 else "Nondemented"
        consensus_confidence = np.mean(confidences) * (1 - consensus_metrics.diversity_index * 0.2)

        # Build comprehensive result
        consensus_result = {
            "patient_data": patient_data,
            "individual_assessments": assessments,
            "consensus_prediction": consensus_prediction,
            "consensus_confidence": consensus_confidence,
            "consensus_score": consensus_score,
            "consensus_metrics": {
                "agreement_score": consensus_metrics.agreement_score,
                "confidence_weighted_score": consensus_metrics.confidence_weighted_score,
                "diversity_index": consensus_metrics.diversity_index,
                "risk_assessment": consensus_metrics.risk_assessment,
            },
            "specialist_insights": self._compile_specialist_insights(assessments),
            "timestamp": datetime.now().isoformat(),
            "safety_validation": post_assessment_safety,
            "communication_log": self.communication_protocol.communication_log,
        }

        # Generate explainable AI output
        explanation = self.explainability_engine.generate_consensus_explanation(consensus_result)
        consensus_result["explanation"] = explanation

        # Store for learning
        self.consensus_history.append(consensus_result)

        return consensus_result

    def _conduct_collaborative_assessment(
        self, agents: List[SpecializedMedicalAgent], patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Conduct collaborative assessment with agent-to-agent communication"""
        assessments = []

        # First round: individual assessments
        for agent in agents:
            try:
                assessment = agent.get_specialized_assessment(patient_data)
                assessment["agent_name"] = agent.name
                assessment["specialization"] = agent.specialization
                assessment["expertise_weight"] = agent.get_expertise_weight("cognitive_assessment")
                assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Assessment failed for {agent.name}: {e}")
                continue

        # Second round: peer review and clarification if needed
        if len(assessments) > 1:
            assessments = self._enable_peer_review(agents, assessments, patient_data)

        return assessments

    def _enable_peer_review(
        self,
        agents: List[SpecializedMedicalAgent],
        initial_assessments: List[Dict[str, Any]],
        patient_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Enable peer review between agents for complex cases"""

        # Check if peer review is needed (high disagreement or low confidence)
        needs_review = self._assess_need_for_peer_review(initial_assessments)

        if not needs_review:
            return initial_assessments

        logger.info("Enabling peer review due to disagreement or low confidence")

        # Enable agent communications
        for i, agent in enumerate(agents):
            for j, other_agent in enumerate(agents):
                if i != j:
                    # Send assessment for peer review
                    message_content = {
                        "patient_summary": self._create_patient_summary(patient_data),
                        "peer_assessment": (
                            initial_assessments[j] if j < len(initial_assessments) else None
                        ),
                        "request_type": "peer_review",
                    }

                    self.communication_protocol.send_message(
                        sender_agent=agent.name,
                        receiver_agent=other_agent.name,
                        message_type="assessment_request",
                        content=message_content,
                    )

        # Simulate peer review responses (in a real system, agents would respond)
        refined_assessments = []
        for assessment in initial_assessments:
            # Simulate refinement based on peer input
            refined_assessment = assessment.copy()

            # Adjust confidence based on peer agreement
            peer_predictions = [
                a["prediction"]
                for a in initial_assessments
                if a["agent_name"] != assessment["agent_name"]
            ]
            if peer_predictions:
                agreement_ratio = sum(
                    1 for p in peer_predictions if p == assessment["prediction"]
                ) / len(peer_predictions)
                refined_assessment["confidence"] *= 0.7 + 0.3 * agreement_ratio  # Adjust confidence
                refined_assessment["peer_reviewed"] = True
                refined_assessment["peer_agreement_ratio"] = agreement_ratio

            refined_assessments.append(refined_assessment)

        return refined_assessments

    def _assess_need_for_peer_review(self, assessments: List[Dict[str, Any]]) -> bool:
        """Assess if peer review is needed"""
        if len(assessments) < 2:
            return False

        # Check for disagreement
        predictions = [a["prediction"] for a in assessments]
        unique_predictions = set(predictions)
        has_disagreement = len(unique_predictions) > 1

        # Check for low confidence
        avg_confidence = np.mean([a.get("confidence", 0) for a in assessments])
        low_confidence = avg_confidence < 0.7

        return has_disagreement or low_confidence

    def _create_patient_summary(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create anonymized patient summary for peer review"""
        # Remove any identifying information and create summary
        summary = {
            "demographics": {
                "age_range": self._categorize_age(patient_data.get("Age", 70)),
                "gender": patient_data.get("M/F", 1),
            },
            "cognitive_scores": {
                k: v
                for k, v in patient_data.items()
                if k in ["MMSE", "CDR", "eTIV", "nWBV", "ASF", "EDUC", "SES"]
            },
        }
        return summary

    def _categorize_age(self, age: int) -> str:
        """Categorize age for privacy"""
        if age < 65:
            return "under_65"
        elif age < 75:
            return "65_to_75"
        elif age < 85:
            return "75_to_85"
        else:
            return "over_85"

    def _calculate_consensus_metrics(
        self, predictions: List[int], confidences: List[float], weights: List[float]
    ) -> ConsensusMetrics:
        """Calculate consensus quality metrics"""

        # Agreement score (how similar predictions are)
        agreement_score = 1.0 - np.std(predictions)

        # Confidence weighted score
        weighted_confidences = np.array(confidences) * np.array(weights)
        confidence_weighted_score = np.sum(weighted_confidences) / np.sum(weights)

        # Diversity index (measure of opinion diversity)
        diversity_index = np.std(predictions) if len(predictions) > 1 else 0.0

        # Risk assessment based on agreement and confidence
        if agreement_score > 0.8 and confidence_weighted_score > 0.7:
            risk_assessment = "LOW"
        elif agreement_score > 0.6 and confidence_weighted_score > 0.5:
            risk_assessment = "MEDIUM"
        else:
            risk_assessment = "HIGH"

        return ConsensusMetrics(
            agreement_score=agreement_score,
            confidence_weighted_score=confidence_weighted_score,
            diversity_index=diversity_index,
            risk_assessment=risk_assessment,
        )

    def _compile_specialist_insights(
        self, assessments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Compile insights from all specialists"""
        insights = {}

        for assessment in assessments:
            specialization = assessment.get("specialization", "unknown")
            agent_name = assessment.get("agent_name", "unknown")

            # Extract specialized findings
            specialist_findings = {}
            for key in ["radiological_findings", "neurological_findings", "psychiatric_findings"]:
                if key in assessment:
                    specialist_findings[key] = assessment[key]

            # Extract risk factors
            risk_factors = []
            for key in [
                "imaging_risk_factors",
                "cognitive_risk_factors",
                "behavioral_risk_factors",
            ]:
                if key in assessment and assessment[key]:
                    risk_factors.extend(assessment[key])

            insights[specialization] = {
                "agent": agent_name,
                "findings": specialist_findings,
                "risk_factors": risk_factors,
                "confidence": assessment.get("confidence", 0.5),
            }

        return insights


def create_specialized_medical_team(
    alive_node, resource_room, medical_model=None
) -> List[SpecializedMedicalAgent]:
    """Create a team of specialized medical agents"""

    agents = []

    # Radiologist with high analytical skills
    radiologist = RadiologistAgent(
        name="Dr_Radiologist_Alpha",
        cognitive_profile={"analytical": 0.9, "logical": 0.85, "creative": 0.6, "empathy": 0.7},
        alive_node=alive_node,
        resource_room=resource_room,
        medical_model=medical_model,
    )
    agents.append(radiologist)

    # Neurologist with high logical reasoning
    neurologist = NeurologistAgent(
        name="Dr_Neurologist_Beta",
        cognitive_profile={"analytical": 0.85, "logical": 0.9, "creative": 0.7, "empathy": 0.8},
        alive_node=alive_node,
        resource_room=resource_room,
        medical_model=medical_model,
    )
    agents.append(neurologist)

    # Psychiatrist with high empathy and creativity
    psychiatrist = PsychiatristAgent(
        name="Dr_Psychiatrist_Gamma",
        cognitive_profile={"analytical": 0.75, "logical": 0.8, "creative": 0.9, "empathy": 0.95},
        alive_node=alive_node,
        resource_room=resource_room,
        medical_model=medical_model,
    )
    agents.append(psychiatrist)

    return agents


def run_multi_step_diagnostic_simulation(
    agents: List[SpecializedMedicalAgent],
    consensus_manager: ConsensusManager,
    patient_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run multi-step diagnostic simulation with learning"""

    results = []

    for i, case in enumerate(patient_cases):
        logger.info(f"Processing case {i+1}/{len(patient_cases)}")

        # Step 1: Initial specialized assessments
        consensus_result = consensus_manager.build_consensus(agents, case)

        # Step 2: Learning phase (simulate outcome feedback)
        simulated_accuracy = np.random.uniform(0.6, 0.9)  # Simulate diagnostic accuracy

        learning_outcome = {
            "consensus": consensus_result,
            "accuracy": simulated_accuracy,
            "my_assessment": consensus_result.get("consensus_prediction"),
        }

        # Step 3: Each agent learns from the case
        for agent in agents:
            agent.learn_from_case(case, learning_outcome)

        # Add learning metrics to result
        consensus_result["learning_outcome"] = learning_outcome
        consensus_result["case_number"] = i + 1

        results.append(consensus_result)

    return results


def create_test_case() -> Dict[str, Any]:
    """Create a test medical case for simulation"""
    np.random.seed()  # Use random seed for variety

    return {
        "M/F": np.random.randint(0, 2),
        "Age": np.random.randint(60, 90),
        "EDUC": np.random.randint(8, 20),
        "SES": np.random.randint(1, 5),
        "MMSE": np.random.randint(15, 30),
        "CDR": np.random.choice([0.0, 0.5, 1.0, 2.0]),
        "eTIV": np.random.randint(1200, 1800),
        "nWBV": np.random.uniform(0.6, 0.9),
        "ASF": np.random.uniform(1.0, 1.5),
    }


if __name__ == "__main__":
    # Example usage
    from labyrinth_adaptive import AliveLoopNode, ResourceRoom

    # Create test environment
    alive_node = AliveLoopNode((0, 0), (0.5, 0), 15.0, node_id=1)
    resource_room = ResourceRoom()

    # Create specialized team
    medical_team = create_specialized_medical_team(alive_node, resource_room)
    consensus_manager = ConsensusManager()

    # Generate test cases
    test_cases = [create_test_case() for _ in range(5)]

    # Run simulation
    results = run_multi_step_diagnostic_simulation(medical_team, consensus_manager, test_cases)

    # Print results summary
    for i, result in enumerate(results):
        print(f"\nCase {i+1} Results:")
        print(
            f"  Consensus: {result['consensus_prediction']} (confidence: {result['consensus_confidence']:.3f})"
        )
        print(f"  Agreement Score: {result['consensus_metrics']['agreement_score']:.3f}")
        print(f"  Risk Assessment: {result['consensus_metrics']['risk_assessment']}")
        print(f"  Learning Accuracy: {result['learning_outcome']['accuracy']:.3f}")
