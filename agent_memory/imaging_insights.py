#!/usr/bin/env python3
"""
Imaging Insight Summarizer for DuetMind Adaptive Agent Memory System.

This module provides functionality to analyze imaging data and generate 
structured insights that can be stored in agent memory with type "imaging_insight".
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ImagingInsight:
    """Structured imaging insight data"""
    insight_id: str
    patient_id: Optional[str]
    modality: str  # "MRI", "CT", "PET", etc.
    acquisition_date: Optional[str]
    key_findings: List[str]
    quantitative_measures: Dict[str, float]
    clinical_significance: str
    confidence_score: float
    processing_pipeline: str
    generated_at: str
    
    def to_memory_dict(self) -> Dict[str, Any]:
        """Convert to format suitable for agent memory storage"""
        return {
            "content": self.summarize_for_memory(),
            "type": "imaging_insight",
            "importance": self.calculate_importance(),
            "metadata": asdict(self),
            "created_at": self.generated_at
        }
    
    def summarize_for_memory(self) -> str:
        """Create a natural language summary for memory storage"""
        summary = f"Imaging Analysis ({self.modality}"
        if self.acquisition_date:
            summary += f", {self.acquisition_date}"
        summary += f"): {self.clinical_significance}"
        
        if self.key_findings:
            summary += f" Key findings: {'; '.join(self.key_findings[:3])}"
        
        if self.quantitative_measures:
            measures = [f"{k}={v:.2f}" for k, v in list(self.quantitative_measures.items())[:2]]
            summary += f" Measures: {', '.join(measures)}"
        
        summary += f" (Confidence: {self.confidence_score:.2f})"
        return summary
    
    def calculate_importance(self) -> float:
        """Calculate importance score for memory ranking"""
        base_importance = self.confidence_score
        
        # Boost importance for certain findings
        critical_keywords = ["abnormal", "lesion", "tumor", "atrophy", "severe", "significant"]
        for keyword in critical_keywords:
            if any(keyword.lower() in finding.lower() for finding in self.key_findings):
                base_importance += 0.1
                
        # Boost for quantitative measures with extreme values
        if self.quantitative_measures:
            for value in self.quantitative_measures.values():
                if isinstance(value, (int, float)) and (value > 95 or value < 5):  # Percentile-based
                    base_importance += 0.05
        
        return min(base_importance, 1.0)


class ImagingInsightSummarizer:
    """
    Analyzes imaging data and generates structured insights for agent memory.
    """
    
    def __init__(self, memory_store=None):
        self.memory_store = memory_store
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def analyze_brain_mri_features(self, features: Dict[str, Any], 
                                   patient_id: Optional[str] = None) -> ImagingInsight:
        """
        Analyze brain MRI radiomics/volumetric features and generate insights.
        
        Args:
            features: Dictionary of extracted features (volumes, intensities, etc.)
            patient_id: Optional patient identifier
            
        Returns:
            ImagingInsight object
        """
        insight_id = f"brain_mri_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract key findings based on feature values
        key_findings = []
        quantitative_measures = {}
        
        # Analyze brain volumes if available
        if 'total_brain_volume_mm3' in features:
            volume = features['total_brain_volume_mm3']
            quantitative_measures['brain_volume_mm3'] = volume
            
            # Normal brain volume range approximately 1200-1600 cm³ (1.2-1.6M mm³)
            if volume < 1200000:
                key_findings.append("Reduced total brain volume suggesting possible atrophy")
            elif volume > 1700000:
                key_findings.append("Increased total brain volume")
            else:
                key_findings.append("Total brain volume within normal range")
                
        # Analyze gray matter if available
        if 'gray_matter_volume_mm3' in features:
            gm_volume = features['gray_matter_volume_mm3']
            quantitative_measures['gray_matter_volume_mm3'] = gm_volume
            
            if gm_volume < 500000:
                key_findings.append("Reduced gray matter volume")
                
        # Analyze image quality metrics
        if 'qc_snr_basic' in features:
            snr = features['qc_snr_basic']
            quantitative_measures['snr'] = snr
            
            if snr < 10:
                key_findings.append("Poor image quality (low SNR)")
            elif snr > 20:
                key_findings.append("Good image quality")
                
        # Analyze motion artifacts
        if 'qc_motion_score' in features:
            motion = features['qc_motion_score']
            quantitative_measures['motion_score'] = motion
            
            if motion > 0.5:
                key_findings.append("Significant motion artifacts detected")
                
        # Generate clinical significance assessment
        clinical_significance = self._assess_clinical_significance(key_findings, quantitative_measures)
        
        # Calculate confidence based on feature completeness and quality
        confidence = self._calculate_confidence(features, quantitative_measures)
        
        return ImagingInsight(
            insight_id=insight_id,
            patient_id=patient_id,
            modality="MRI",
            acquisition_date=features.get('acquisition_date'),
            key_findings=key_findings,
            quantitative_measures=quantitative_measures,
            clinical_significance=clinical_significance,
            confidence_score=confidence,
            processing_pipeline="radiomics_extraction_v1",
            generated_at=datetime.now().isoformat()
        )
    
    def analyze_prediction_results(self, predictions: Dict[str, Any], 
                                   features: Dict[str, Any],
                                   patient_id: Optional[str] = None) -> ImagingInsight:
        """
        Analyze ML model predictions on imaging data and generate insights.
        
        Args:
            predictions: Model prediction results
            features: Original imaging features
            patient_id: Optional patient identifier
            
        Returns:
            ImagingInsight object
        """
        insight_id = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        key_findings = []
        quantitative_measures = {}
        
        # Extract prediction information
        if 'prediction' in predictions:
            pred_class = predictions['prediction']
            key_findings.append(f"Model prediction: {pred_class}")
            
        if 'confidence' in predictions or 'max_probability' in predictions:
            conf = predictions.get('confidence', predictions.get('max_probability', 0))
            quantitative_measures['prediction_confidence'] = conf
            
            if conf > 0.8:
                key_findings.append("High confidence prediction")
            elif conf < 0.6:
                key_findings.append("Low confidence prediction - requires review")
                
        # Analyze class probabilities if available
        if 'class_probabilities' in predictions:
            probs = predictions['class_probabilities']
            if isinstance(probs, dict):
                for class_name, prob in probs.items():
                    quantitative_measures[f'prob_{class_name}'] = prob
                    
        # Clinical significance based on prediction
        clinical_significance = self._assess_prediction_significance(predictions, key_findings)
        
        # Confidence based on model performance and feature quality
        confidence = predictions.get('confidence', predictions.get('max_probability', 0.5))
        
        return ImagingInsight(
            insight_id=insight_id,
            patient_id=patient_id,
            modality=features.get('modality', 'Unknown'),
            acquisition_date=features.get('acquisition_date'),
            key_findings=key_findings,
            quantitative_measures=quantitative_measures,
            clinical_significance=clinical_significance,
            confidence_score=confidence,
            processing_pipeline="ml_prediction_v1",
            generated_at=datetime.now().isoformat()
        )
    
    def store_insight_in_memory(self, insight: ImagingInsight, agent_id: str = "default"):
        """
        Store the imaging insight in agent memory system.
        
        Args:
            insight: ImagingInsight to store
            agent_id: Agent identifier for memory storage
        """
        if not self.memory_store:
            self.logger.warning("No memory store configured, cannot save insight")
            return
            
        try:
            memory_data = insight.to_memory_dict()
            self.memory_store.store_memory(
                agent_id=agent_id,
                **memory_data
            )
            self.logger.info(f"Stored imaging insight {insight.insight_id} in memory")
        except Exception as e:
            self.logger.error(f"Failed to store insight in memory: {e}")
    
    def _assess_clinical_significance(self, findings: List[str], 
                                     measures: Dict[str, float]) -> str:
        """Assess overall clinical significance of findings"""
        
        critical_terms = ["reduced", "atrophy", "lesion", "abnormal", "poor quality"]
        critical_count = sum(1 for finding in findings 
                           if any(term in finding.lower() for term in critical_terms))
        
        if critical_count >= 2:
            return "Multiple concerning findings requiring clinical attention"
        elif critical_count == 1:
            return "Single finding noted, recommend clinical correlation"
        else:
            return "No significant abnormalities detected in automated analysis"
    
    def _assess_prediction_significance(self, predictions: Dict[str, Any], 
                                       findings: List[str]) -> str:
        """Assess clinical significance of model predictions"""
        
        pred = predictions.get('prediction', '').lower()
        conf = predictions.get('confidence', predictions.get('max_probability', 0))
        
        if 'positive' in pred or 'abnormal' in pred or 'disease' in pred:
            if conf > 0.8:
                return "High-confidence positive prediction requiring clinical review"
            else:
                return "Positive prediction with moderate confidence, clinical correlation advised"
        else:
            if conf > 0.8:
                return "High-confidence normal prediction"
            else:
                return "Normal prediction with moderate confidence"
    
    def _calculate_confidence(self, features: Dict[str, Any], 
                             measures: Dict[str, float]) -> float:
        """Calculate overall confidence in the analysis"""
        
        base_confidence = 0.5
        
        # Boost for having key measurements
        key_features = ['total_brain_volume_mm3', 'gray_matter_volume_mm3', 'qc_snr_basic']
        available_features = sum(1 for key in key_features if key in features)
        base_confidence += (available_features / len(key_features)) * 0.3
        
        # Reduce for poor quality indicators
        if 'qc_snr_basic' in features and features['qc_snr_basic'] < 10:
            base_confidence -= 0.1
            
        if 'qc_motion_score' in features and features['qc_motion_score'] > 0.5:
            base_confidence -= 0.1
            
        return max(0.1, min(base_confidence, 0.95))


# Integration functions for easy use in existing systems
def create_imaging_insight_from_features(features: Dict[str, Any], 
                                        patient_id: Optional[str] = None,
                                        memory_store=None) -> ImagingInsight:
    """
    Convenience function to create imaging insight from extracted features.
    """
    summarizer = ImagingInsightSummarizer(memory_store)
    return summarizer.analyze_brain_mri_features(features, patient_id)


def create_imaging_insight_from_predictions(predictions: Dict[str, Any],
                                           features: Dict[str, Any],
                                           patient_id: Optional[str] = None,
                                           memory_store=None) -> ImagingInsight:
    """
    Convenience function to create imaging insight from model predictions.
    """
    summarizer = ImagingInsightSummarizer(memory_store)
    return summarizer.analyze_prediction_results(predictions, features, patient_id)