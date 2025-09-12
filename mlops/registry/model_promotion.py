#!/usr/bin/env python3
"""
Model Promotion Automation Gate for DuetMind Adaptive MLOps.
Automated model promotion based on accuracy and drift thresholds with rollback capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import yaml
import json
import os
from pathlib import Path

# MLOps imports
from mlflow import MlflowClient
import mlflow
from ..drift.evidently_drift_monitor import DriftMonitor, ModelDriftMonitor
from ..audit.event_chain import AuditEventChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""
    min_accuracy: float = 0.85
    max_drift_score: float = 0.1
    min_precision: float = 0.80
    min_recall: float = 0.80
    max_performance_degradation: float = 0.05
    require_manual_approval: bool = False
    

@dataclass
class PromotionResult:
    """Result of a model promotion attempt."""
    promoted: bool
    model_version: str
    promotion_stage: str
    criteria_met: Dict[str, bool]
    metrics: Dict[str, float]
    drift_results: Optional[Dict[str, Any]]
    audit_event_id: Optional[str]
    rollback_available: bool
    message: str


class ModelPromotionGate:
    """
    Automated model promotion system with accuracy and drift validation.
    """
    
    def __init__(self, mlflow_tracking_uri: str, audit_chain: Optional[AuditEventChain] = None,
                 criteria: Optional[PromotionCriteria] = None):
        """
        Initialize the model promotion gate.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            audit_chain: Audit event chain for logging
            criteria: Promotion criteria (uses defaults if None)
        """
        self.mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        self.audit_chain = audit_chain
        self.criteria = criteria or PromotionCriteria()
        
        logger.info("Initialized ModelPromotionGate")
    
    def evaluate_model_for_promotion(self, model_name: str, candidate_version: str,
                                   reference_data: Optional[Any] = None, 
                                   current_data: Optional[Any] = None) -> PromotionResult:
        """
        Evaluate a model version for promotion to production.
        
        Args:
            model_name: Name of the registered model
            candidate_version: Version number to evaluate
            reference_data: Reference dataset for drift detection
            current_data: Current dataset for drift detection
            
        Returns:
            PromotionResult with evaluation details
        """
        try:
            # Get model version details
            model_version = self.mlflow_client.get_model_version(model_name, candidate_version)
            
            # Get model metrics
            run_id = model_version.run_id
            run = self.mlflow_client.get_run(run_id)
            metrics = run.data.metrics
            
            # Initialize criteria check results
            criteria_met = {}
            
            # Check accuracy threshold
            accuracy = metrics.get('accuracy', 0.0)
            criteria_met['accuracy'] = accuracy >= self.criteria.min_accuracy
            
            # Check precision threshold
            precision = metrics.get('precision', 0.0)
            criteria_met['precision'] = precision >= self.criteria.min_precision
            
            # Check recall threshold  
            recall = metrics.get('recall', 0.0)
            criteria_met['recall'] = recall >= self.criteria.min_recall
            
            # Perform drift detection if data provided
            drift_results = None
            if reference_data is not None and current_data is not None:
                drift_results = self._check_data_drift(reference_data, current_data)
                criteria_met['drift'] = not drift_results.get('overall_drift_detected', True)
            else:
                criteria_met['drift'] = True  # Skip drift check if no data provided
            
            # Check performance degradation against production model
            performance_degradation_ok = self._check_performance_degradation(model_name, metrics)
            criteria_met['performance_degradation'] = performance_degradation_ok
            
            # Determine if promotion should proceed
            all_criteria_met = all(criteria_met.values())
            
            # Check for manual approval requirement
            if self.criteria.require_manual_approval and all_criteria_met:
                return self._create_promotion_result(
                    promoted=False,
                    model_version=candidate_version,
                    promotion_stage="pending_approval",
                    criteria_met=criteria_met,
                    metrics=metrics,
                    drift_results=drift_results,
                    message="All automated criteria met. Manual approval required."
                )
            
            if all_criteria_met:
                # Promote the model
                return self._promote_model(model_name, candidate_version, metrics, criteria_met, drift_results)
            else:
                # Promotion failed
                failed_criteria = [k for k, v in criteria_met.items() if not v]
                return self._create_promotion_result(
                    promoted=False,
                    model_version=candidate_version,
                    promotion_stage="staging",
                    criteria_met=criteria_met,
                    metrics=metrics,
                    drift_results=drift_results,
                    message=f"Promotion failed. Criteria not met: {', '.join(failed_criteria)}"
                )
                
        except Exception as e:
            logger.error(f"Error evaluating model for promotion: {e}")
            return self._create_promotion_result(
                promoted=False,
                model_version=candidate_version,
                promotion_stage="error",
                criteria_met={},
                metrics={},
                drift_results=None,
                message=f"Error during evaluation: {str(e)}"
            )
    
    def _check_data_drift(self, reference_data: Any, current_data: Any) -> Dict[str, Any]:
        """
        Check for data drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Drift detection results
        """
        try:
            drift_monitor = DriftMonitor(reference_data, drift_threshold=self.criteria.max_drift_score)
            return drift_monitor.detect_data_drift(current_data, generate_report=False)
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {'overall_drift_detected': True, 'error': str(e)}
    
    def _check_performance_degradation(self, model_name: str, current_metrics: Dict[str, float]) -> bool:
        """
        Check if performance has degraded compared to production model.
        
        Args:
            model_name: Name of the model
            current_metrics: Metrics of the candidate model
            
        Returns:
            True if performance degradation is within acceptable limits
        """
        try:
            # Get current production model metrics
            prod_versions = self.mlflow_client.get_latest_versions(model_name, stages=["Production"])
            
            if not prod_versions:
                # No production model exists, so no degradation to check
                return True
            
            prod_version = prod_versions[0]
            prod_run = self.mlflow_client.get_run(prod_version.run_id)
            prod_metrics = prod_run.data.metrics
            
            # Check key metrics for degradation
            key_metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
            
            for metric in key_metrics:
                if metric in prod_metrics and metric in current_metrics:
                    prod_value = prod_metrics[metric]
                    current_value = current_metrics[metric]
                    
                    if prod_value > 0:
                        degradation = (prod_value - current_value) / prod_value
                        if degradation > self.criteria.max_performance_degradation:
                            logger.warning(f"Performance degradation in {metric}: {degradation:.3f} > {self.criteria.max_performance_degradation}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
            return False
    
    def _promote_model(self, model_name: str, version: str, metrics: Dict[str, float],
                      criteria_met: Dict[str, bool], drift_results: Optional[Dict[str, Any]]) -> PromotionResult:
        """
        Promote a model to production.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            metrics: Model metrics
            criteria_met: Criteria evaluation results
            drift_results: Drift detection results
            
        Returns:
            PromotionResult
        """
        try:
            # Archive current production model if exists
            self._archive_current_production(model_name)
            
            # Promote to production
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Log audit event
            audit_event_id = None
            if self.audit_chain:
                audit_event_id = self.audit_chain.log_event(
                    event_type="model_promoted",
                    entity_type="model",
                    entity_id=f"{model_name}_v{version}",
                    event_data={
                        'model_name': model_name,
                        'version': version,
                        'stage': 'Production',
                        'metrics': metrics,
                        'criteria_met': criteria_met,
                        'drift_detected': drift_results.get('overall_drift_detected') if drift_results else False,
                        'promotion_timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    user_id="promotion_system"
                )
            
            return self._create_promotion_result(
                promoted=True,
                model_version=version,
                promotion_stage="Production",
                criteria_met=criteria_met,
                metrics=metrics,
                drift_results=drift_results,
                audit_event_id=audit_event_id,
                message=f"Model {model_name} v{version} promoted to Production successfully"
            )
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return self._create_promotion_result(
                promoted=False,
                model_version=version,
                promotion_stage="error",
                criteria_met=criteria_met,
                metrics=metrics,
                drift_results=drift_results,
                message=f"Error during promotion: {str(e)}"
            )
    
    def _archive_current_production(self, model_name: str):
        """Archive the current production model version."""
        try:
            prod_versions = self.mlflow_client.get_latest_versions(model_name, stages=["Production"])
            
            for version in prod_versions:
                # Log audit event for archival
                if self.audit_chain:
                    self.audit_chain.log_event(
                        event_type="model_archived",
                        entity_type="model",
                        entity_id=f"{model_name}_v{version.version}",
                        event_data={
                            'model_name': model_name,
                            'version': version.version,
                            'previous_stage': 'Production',
                            'new_stage': 'Archived',
                            'archive_reason': 'replaced_by_new_production',
                            'archive_timestamp': datetime.now(timezone.utc).isoformat()
                        },
                        user_id="promotion_system"
                    )
                
                logger.info(f"Archived production model {model_name} v{version.version}")
                
        except Exception as e:
            logger.warning(f"Error archiving current production model: {e}")
    
    def _create_promotion_result(self, promoted: bool, model_version: str, promotion_stage: str,
                               criteria_met: Dict[str, bool], metrics: Dict[str, float],
                               drift_results: Optional[Dict[str, Any]], message: str,
                               audit_event_id: Optional[str] = None) -> PromotionResult:
        """Create a PromotionResult object."""
        return PromotionResult(
            promoted=promoted,
            model_version=model_version,
            promotion_stage=promotion_stage,
            criteria_met=criteria_met,
            metrics=metrics,
            drift_results=drift_results,
            audit_event_id=audit_event_id,
            rollback_available=promoted,  # Rollback only available if promotion succeeded
            message=message
        )
    
    def rollback_model(self, model_name: str, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the model
            target_version: Specific version to rollback to (None for previous production)
            
        Returns:
            Rollback result
        """
        try:
            if target_version:
                # Rollback to specific version
                rollback_version = target_version
            else:
                # Get the most recent archived version
                archived_versions = self.mlflow_client.get_latest_versions(model_name, stages=["Archived"])
                
                if not archived_versions:
                    return {
                        'success': False,
                        'message': "No archived versions available for rollback"
                    }
                
                # Sort by version number and get the latest
                archived_versions.sort(key=lambda v: int(v.version), reverse=True)
                rollback_version = archived_versions[0].version
            
            # Promote the rollback version to production
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=rollback_version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Log audit event
            if self.audit_chain:
                self.audit_chain.log_event(
                    event_type="model_rollback",
                    entity_type="model",
                    entity_id=f"{model_name}_v{rollback_version}",
                    event_data={
                        'model_name': model_name,
                        'rollback_version': rollback_version,
                        'rollback_reason': 'manual_rollback',
                        'rollback_timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    user_id="rollback_system"
                )
            
            return {
                'success': True,
                'rollback_version': rollback_version,
                'message': f"Successfully rolled back {model_name} to version {rollback_version}"
            }
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return {
                'success': False,
                'message': f"Rollback failed: {str(e)}"
            }
    
    def get_promotion_history(self, model_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get promotion history for a model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of events to return
            
        Returns:
            List of promotion events
        """
        if not self.audit_chain:
            return []
        
        # Get audit trail for the model
        trail = self.audit_chain.get_entity_audit_trail("model", f"{model_name}_*", limit=limit * 2)
        
        # Filter for promotion-related events
        promotion_events = []
        for event in trail:
            if event['event_type'] in ['model_promoted', 'model_archived', 'model_rollback']:
                promotion_events.append(event)
                
                if len(promotion_events) >= limit:
                    break
        
        return promotion_events


def demo_model_promotion():
    """
    Demonstrate the model promotion system.
    """
    logger.info("Starting model promotion system demonstration...")
    
    try:
        # Initialize components (using local SQLite for demo)
        audit_chain = AuditEventChain("sqlite:///promotion_demo.db")
        
        # Set up criteria
        criteria = PromotionCriteria(
            min_accuracy=0.85,
            max_drift_score=0.1,
            min_precision=0.80,
            min_recall=0.75,
            max_performance_degradation=0.05
        )
        
        promotion_gate = ModelPromotionGate(
            mlflow_tracking_uri="sqlite:///mlflow_demo.db",
            audit_chain=audit_chain,
            criteria=criteria
        )
        
        # Simulate model evaluation (would normally use real MLflow models)
        print("\n=== Model Promotion System Demo ===")
        print("Note: This is a demonstration with simulated data")
        print("In production, this would evaluate real MLflow models")
        
        print("\nPromotion Criteria:")
        print(f"  - Min accuracy: {criteria.min_accuracy}")
        print(f"  - Max drift score: {criteria.max_drift_score}")
        print(f"  - Min precision: {criteria.min_precision}")
        print(f"  - Min recall: {criteria.min_recall}")
        print(f"  - Max performance degradation: {criteria.max_performance_degradation}")
        
        # Simulate a successful promotion
        print("\nSimulating model promotion evaluation...")
        print("✓ Model meets accuracy threshold: 0.94 >= 0.85")
        print("✓ Model meets precision threshold: 0.91 >= 0.80")
        print("✓ Model meets recall threshold: 0.88 >= 0.75")
        print("✓ No significant drift detected")
        print("✓ Performance degradation within limits")
        
        print("\n✅ Model promotion would succeed!")
        
        # Log some demo audit events
        audit_event_id = audit_chain.log_event(
            event_type="model_promoted",
            entity_type="model", 
            entity_id="demo_alzheimer_classifier_v2.0",
            event_data={
                'model_name': 'demo_alzheimer_classifier',
                'version': '2.0',
                'stage': 'Production',
                'metrics': {'accuracy': 0.94, 'precision': 0.91, 'recall': 0.88},
                'promotion_reason': 'automated_criteria_met'
            },
            user_id="demo_promotion_system"
        )
        
        print(f"\nAudit event logged: {audit_event_id}")
        
        # Verify audit chain
        verification = audit_chain.verify_chain_integrity()
        print(f"Audit chain integrity: {'VALID' if verification['chain_valid'] else 'INVALID'}")
        
    except Exception as e:
        logger.error(f"Error in promotion demo: {e}")


if __name__ == "__main__":
    demo_model_promotion()