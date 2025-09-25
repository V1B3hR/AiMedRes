"""
Enhanced Drift & Monitoring for DuetMind Adaptive

Advanced drift monitoring with automated response workflows, alert systems,
and integration with training orchestration.
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)

# Import existing drift monitoring
try:
    from ...mlops.drift.evidently_drift_monitor import DriftMonitor as BaseDriftMonitor
    from ...mlops.drift.evidently_drift_monitor import ModelDriftMonitor as BaseModelDriftMonitor
    BASE_DRIFT_AVAILABLE = True
except ImportError:
    logger.warning("Base drift monitor not available")
    BASE_DRIFT_AVAILABLE = False
    
    # Mock base class
    class BaseDriftMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def detect_data_drift(self, *args, **kwargs):
            return {'overall_drift_detected': False, 'drift_score': 0.0}
    
    class BaseModelDriftMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def detect_performance_drift(self, *args, **kwargs):
            return {'performance_drift_detected': False, 'degraded_metrics': []}

class DriftSeverity(Enum):
    """Drift detection severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    WEBHOOK = "webhook" 
    SLACK = "slack"
    LOG = "log"

class ResponseAction(Enum):
    """Automated response actions."""
    RETRAIN_MODEL = "retrain_model"
    SCALE_RESOURCES = "scale_resources"
    UPDATE_THRESHOLDS = "update_thresholds"
    COLLECT_MORE_DATA = "collect_more_data"
    HUMAN_REVIEW = "human_review"
    ROLLBACK_MODEL = "rollback_model"

@dataclass
class DriftAlert:
    """Drift detection alert."""
    id: str
    timestamp: datetime
    severity: DriftSeverity
    drift_type: str  # 'data', 'model', 'concept'
    feature: Optional[str]
    drift_score: float
    threshold: float
    description: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AlertConfig:
    """Configuration for drift alerting."""
    enabled_channels: List[AlertChannel]
    email_config: Dict[str, str] = None
    webhook_urls: List[str] = None
    slack_config: Dict[str, str] = None
    severity_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                DriftSeverity.LOW.value: 0.1,
                DriftSeverity.MEDIUM.value: 0.2,
                DriftSeverity.HIGH.value: 0.4,
                DriftSeverity.CRITICAL.value: 0.7
            }

@dataclass
class ResponseConfig:
    """Configuration for automated responses."""
    enabled_actions: List[ResponseAction]
    retrain_config: Dict[str, Any] = None
    scaling_config: Dict[str, Any] = None
    threshold_update_rules: Dict[str, Any] = None
    human_review_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retrain_config is None:
            self.retrain_config = {
                'trigger_threshold': 0.3,
                'max_auto_retrains_per_day': 3,
                'require_human_approval': False
            }

class EnhancedDriftMonitor:
    """
    Enhanced drift monitoring system with automated responses.
    
    Features:
    - Multi-level drift detection (data, model, concept)
    - Automated alerting through multiple channels
    - Configurable response workflows
    - Integration with training orchestration
    - Historical drift tracking and analysis
    """
    
    def __init__(self,
                 reference_data: pd.DataFrame,
                 baseline_metrics: Dict[str, float],
                 alert_config: AlertConfig,
                 response_config: ResponseConfig,
                 drift_threshold: float = 0.15,
                 monitoring_interval: int = 3600):  # seconds
        """
        Initialize enhanced drift monitor.
        
        Args:
            reference_data: Reference dataset for comparison
            baseline_metrics: Baseline model performance metrics
            alert_config: Alert configuration
            response_config: Response configuration
            drift_threshold: Default drift detection threshold
            monitoring_interval: Monitoring interval in seconds
        """
        self.reference_data = reference_data
        self.baseline_metrics = baseline_metrics
        self.alert_config = alert_config
        self.response_config = response_config
        self.drift_threshold = drift_threshold
        self.monitoring_interval = monitoring_interval
        
        # Initialize base monitors
        if BASE_DRIFT_AVAILABLE:
            self.data_drift_monitor = BaseDriftMonitor(
                reference_data, drift_threshold
            )
            self.model_drift_monitor = BaseModelDriftMonitor(baseline_metrics)
        else:
            self.data_drift_monitor = BaseDriftMonitor()
            self.model_drift_monitor = BaseModelDriftMonitor()
        
        # State tracking
        self.drift_history: List[DriftAlert] = []
        self.is_monitoring = False
        self.last_retrain_time = None
        self.retrain_count_today = 0
        
        # Response tracking
        self.response_history: List[Dict[str, Any]] = []
        self.pending_responses: List[Dict[str, Any]] = []
    
    def detect_comprehensive_drift(self,
                                 current_data: pd.DataFrame,
                                 current_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Perform comprehensive drift detection across all types.
        
        Args:
            current_data: Current dataset for comparison
            current_metrics: Current model performance metrics
            
        Returns:
            Comprehensive drift detection results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': {},
            'model_drift': {},
            'concept_drift': {},
            'alerts': [],
            'recommended_actions': []
        }
        
        # Data drift detection
        try:
            data_drift_results = self.data_drift_monitor.detect_data_drift(current_data)
            results['data_drift'] = data_drift_results
            
            # Generate data drift alerts
            if data_drift_results.get('overall_drift_detected', False):
                alert = self._create_drift_alert(
                    drift_type='data',
                    drift_score=data_drift_results.get('drift_score', 0.0),
                    description="Significant data drift detected in input features"
                )
                results['alerts'].append(alert)
        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            results['data_drift'] = {'error': str(e)}
        
        # Model drift detection
        if current_metrics:
            try:
                model_drift_results = self.model_drift_monitor.detect_performance_drift(current_metrics)
                results['model_drift'] = model_drift_results
                
                # Generate model drift alerts
                if model_drift_results.get('performance_drift_detected', False):
                    alert = self._create_drift_alert(
                        drift_type='model',
                        drift_score=self._calculate_performance_drift_score(current_metrics),
                        description="Model performance degradation detected"
                    )
                    results['alerts'].append(alert)
            except Exception as e:
                logger.error(f"Model drift detection failed: {e}")
                results['model_drift'] = {'error': str(e)}
        
        # Concept drift detection (statistical approach)
        try:
            concept_drift_results = self._detect_concept_drift(current_data, current_metrics)
            results['concept_drift'] = concept_drift_results
            
            if concept_drift_results.get('concept_drift_detected', False):
                alert = self._create_drift_alert(
                    drift_type='concept',
                    drift_score=concept_drift_results.get('drift_score', 0.0),
                    description="Concept drift detected - relationship between features and target may have changed"
                )
                results['alerts'].append(alert)
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            results['concept_drift'] = {'error': str(e)}
        
        # Process alerts and determine actions
        for alert in results['alerts']:
            self.drift_history.append(alert)
            
            # Send notifications
            self._send_alert_notifications(alert)
            
            # Determine recommended actions
            actions = self._determine_response_actions(alert, results)
            results['recommended_actions'].extend(actions)
        
        return results
    
    def _create_drift_alert(self,
                           drift_type: str,
                           drift_score: float,
                           description: str,
                           feature: str = None) -> DriftAlert:
        """Create drift alert with severity classification."""
        # Determine severity based on drift score
        severity = DriftSeverity.LOW
        for sev_name, threshold in self.alert_config.severity_thresholds.items():
            if drift_score >= threshold:
                severity = DriftSeverity(sev_name)
        
        return DriftAlert(
            id=f"{drift_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            drift_type=drift_type,
            feature=feature,
            drift_score=drift_score,
            threshold=self.drift_threshold,
            description=description,
            metadata={
                'monitoring_interval': self.monitoring_interval,
                'reference_data_size': len(self.reference_data)
            }
        )
    
    def _calculate_performance_drift_score(self, current_metrics: Dict[str, float]) -> float:
        """Calculate overall performance drift score."""
        drift_scores = []
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                if baseline_value > 0:
                    relative_change = abs(current_value - baseline_value) / baseline_value
                    drift_scores.append(relative_change)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _detect_concept_drift(self,
                             current_data: pd.DataFrame,
                             current_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Detect concept drift using statistical methods.
        
        This is a simplified implementation - in practice, you'd use more
        sophisticated methods like ADWIN, DDM, or EDDM.
        """
        # Simple concept drift detection based on distribution changes
        # and performance degradation patterns
        
        drift_indicators = []
        
        # Check for significant changes in feature correlations
        if len(current_data) > 100 and len(self.reference_data) > 100:
            try:
                ref_corr = self.reference_data.select_dtypes(include=[np.number]).corr()
                curr_corr = current_data.select_dtypes(include=[np.number]).corr()
                
                # Calculate correlation matrix difference
                corr_diff = np.abs(ref_corr.values - curr_corr.values)
                avg_corr_change = np.nanmean(corr_diff)
                
                drift_indicators.append(avg_corr_change)
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
        
        # Check for performance degradation patterns
        if current_metrics:
            perf_drift = self._calculate_performance_drift_score(current_metrics)
            drift_indicators.append(perf_drift)
        
        # Simple aggregation
        concept_drift_score = np.mean(drift_indicators) if drift_indicators else 0.0
        concept_drift_detected = concept_drift_score > self.drift_threshold
        
        return {
            'concept_drift_detected': concept_drift_detected,
            'drift_score': concept_drift_score,
            'indicators': drift_indicators,
            'method': 'statistical_correlation_performance'
        }
    
    def _determine_response_actions(self,
                                  alert: DriftAlert,
                                  drift_results: Dict[str, Any]) -> List[ResponseAction]:
        """Determine appropriate response actions for detected drift."""
        actions = []
        
        # Critical drift - immediate action needed
        if alert.severity == DriftSeverity.CRITICAL:
            if ResponseAction.RETRAIN_MODEL in self.response_config.enabled_actions:
                if self._can_auto_retrain():
                    actions.append(ResponseAction.RETRAIN_MODEL)
                else:
                    actions.append(ResponseAction.HUMAN_REVIEW)
            
            if ResponseAction.ROLLBACK_MODEL in self.response_config.enabled_actions:
                actions.append(ResponseAction.ROLLBACK_MODEL)
        
        # High drift - automated responses with safeguards
        elif alert.severity == DriftSeverity.HIGH:
            if ResponseAction.RETRAIN_MODEL in self.response_config.enabled_actions:
                if self._can_auto_retrain():
                    actions.append(ResponseAction.RETRAIN_MODEL)
            
            if ResponseAction.UPDATE_THRESHOLDS in self.response_config.enabled_actions:
                actions.append(ResponseAction.UPDATE_THRESHOLDS)
        
        # Medium drift - monitoring and preparation
        elif alert.severity == DriftSeverity.MEDIUM:
            if ResponseAction.COLLECT_MORE_DATA in self.response_config.enabled_actions:
                actions.append(ResponseAction.COLLECT_MORE_DATA)
            
            if ResponseAction.UPDATE_THRESHOLDS in self.response_config.enabled_actions:
                actions.append(ResponseAction.UPDATE_THRESHOLDS)
        
        # Low drift - monitoring only
        elif alert.severity == DriftSeverity.LOW:
            # Just log for now
            pass
        
        return actions
    
    def _can_auto_retrain(self) -> bool:
        """Check if automatic retraining is allowed."""
        config = self.response_config.retrain_config
        
        # Check daily limit
        today = datetime.now().date()
        if self.last_retrain_time and self.last_retrain_time.date() == today:
            if self.retrain_count_today >= config.get('max_auto_retrains_per_day', 3):
                return False
        else:
            self.retrain_count_today = 0
        
        # Check if human approval is required
        if config.get('require_human_approval', False):
            return False
        
        return True
    
    def _send_alert_notifications(self, alert: DriftAlert):
        """Send alert notifications through configured channels."""
        for channel in self.alert_config.enabled_channels:
            try:
                if channel == AlertChannel.EMAIL:
                    self._send_email_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook_alert(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack_alert(alert)
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    def _send_email_alert(self, alert: DriftAlert):
        """Send email notification."""
        if not self.alert_config.email_config:
            return
        
        config = self.alert_config.email_config
        
        msg = MimeMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = f"DuetMind Drift Alert - {alert.severity.value.upper()}"
        
        body = f"""
        Drift Alert Detected
        
        Alert ID: {alert.id}
        Severity: {alert.severity.value.upper()}
        Type: {alert.drift_type}
        Drift Score: {alert.drift_score:.4f}
        Threshold: {alert.threshold:.4f}
        
        Description: {alert.description}
        
        Timestamp: {alert.timestamp}
        
        Please review and take appropriate action if needed.
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        if config.get('use_tls', True):
            server.starttls()
        if config.get('username') and config.get('password'):
            server.login(config['username'], config['password'])
        
        text = msg.as_string()
        server.sendmail(config['sender_email'], config['recipient_email'], text)
        server.quit()
    
    def _send_webhook_alert(self, alert: DriftAlert):
        """Send webhook notification."""
        if not self.alert_config.webhook_urls:
            return
        
        payload = {
            'alert_id': alert.id,
            'severity': alert.severity.value,
            'drift_type': alert.drift_type,
            'drift_score': alert.drift_score,
            'threshold': alert.threshold,
            'description': alert.description,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
        
        for webhook_url in self.alert_config.webhook_urls:
            try:
                response = requests.post(webhook_url, json=payload, timeout=30)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Webhook alert failed for {webhook_url}: {e}")
    
    def _send_slack_alert(self, alert: DriftAlert):
        """Send Slack notification."""
        if not self.alert_config.slack_config:
            return
        
        config = self.alert_config.slack_config
        
        # Determine emoji based on severity
        severity_emojis = {
            DriftSeverity.LOW: "🟡",
            DriftSeverity.MEDIUM: "🟠", 
            DriftSeverity.HIGH: "🔴",
            DriftSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emojis.get(alert.severity, "⚠️")
        
        payload = {
            "text": f"{emoji} DuetMind Drift Alert - {alert.severity.value.upper()}",
            "attachments": [
                {
                    "color": "danger" if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else "warning",
                    "fields": [
                        {"title": "Alert ID", "value": alert.id, "short": True},
                        {"title": "Type", "value": alert.drift_type, "short": True},
                        {"title": "Drift Score", "value": f"{alert.drift_score:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                        {"title": "Description", "value": alert.description, "short": False},
                        {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }
            ]
        }
        
        response = requests.post(config['webhook_url'], json=payload, timeout=30)
        response.raise_for_status()
    
    def _log_alert(self, alert: DriftAlert):
        """Log alert to standard logging."""
        logger.warning(
            f"DRIFT ALERT [{alert.severity.value.upper()}] - "
            f"{alert.drift_type} drift detected: {alert.description} "
            f"(Score: {alert.drift_score:.4f}, Threshold: {alert.threshold:.4f})"
        )
    
    async def execute_response_action(self, action: ResponseAction, alert: DriftAlert) -> Dict[str, Any]:
        """Execute automated response action."""
        result = {
            'action': action.value,
            'alert_id': alert.id,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'details': {}
        }
        
        try:
            if action == ResponseAction.RETRAIN_MODEL:
                result.update(await self._execute_retrain_action(alert))
            elif action == ResponseAction.UPDATE_THRESHOLDS:
                result.update(await self._execute_threshold_update_action(alert))
            elif action == ResponseAction.SCALE_RESOURCES:
                result.update(await self._execute_scaling_action(alert))
            elif action == ResponseAction.COLLECT_MORE_DATA:
                result.update(await self._execute_data_collection_action(alert))
            elif action == ResponseAction.HUMAN_REVIEW:
                result.update(await self._execute_human_review_action(alert))
            elif action == ResponseAction.ROLLBACK_MODEL:
                result.update(await self._execute_rollback_action(alert))
            
            self.response_history.append(result)
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            logger.error(f"Response action {action.value} failed: {e}")
        
        return result
    
    async def _execute_retrain_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute model retraining."""
        # This would integrate with the orchestration system
        from ..training.orchestration import create_orchestrator
        
        orchestrator = create_orchestrator()
        
        # Add retraining task
        task_id = orchestrator.add_training_task(
            task_id=f"drift_retrain_{alert.id}",
            model_config=self.response_config.retrain_config.get('model_config', {}),
            data_path=self.response_config.retrain_config.get('data_path', ''),
            dependencies=[]
        )
        
        # Track retrain attempt
        self.retrain_count_today += 1
        self.last_retrain_time = datetime.now()
        
        return {
            'success': True,
            'details': {
                'task_id': task_id,
                'retrain_count_today': self.retrain_count_today
            }
        }
    
    async def _execute_threshold_update_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute threshold adjustment."""
        # Simple adaptive threshold adjustment
        adjustment_factor = 1.1 if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else 1.05
        
        old_threshold = self.drift_threshold
        self.drift_threshold *= adjustment_factor
        
        return {
            'success': True,
            'details': {
                'old_threshold': old_threshold,
                'new_threshold': self.drift_threshold,
                'adjustment_factor': adjustment_factor
            }
        }
    
    async def _execute_scaling_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute resource scaling."""
        # Placeholder for resource scaling logic
        return {
            'success': True,
            'details': {'message': 'Resource scaling initiated'}
        }
    
    async def _execute_data_collection_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute additional data collection."""
        # Placeholder for data collection logic
        return {
            'success': True,
            'details': {'message': 'Additional data collection initiated'}
        }
    
    async def _execute_human_review_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute human review request."""
        # Create human review ticket/notification
        return {
            'success': True,
            'details': {'message': 'Human review request created'}
        }
    
    async def _execute_rollback_action(self, alert: DriftAlert) -> Dict[str, Any]:
        """Execute model rollback."""
        # Placeholder for model rollback logic
        return {
            'success': True,
            'details': {'message': 'Model rollback initiated'}
        }
    
    def get_drift_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get drift monitoring summary for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        recent_alerts = [
            alert for alert in self.drift_history
            if alert.timestamp >= cutoff_date
        ]
        
        # Group by severity
        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = len([
                alert for alert in recent_alerts
                if alert.severity == severity
            ])
        
        # Group by drift type  
        type_counts = {}
        for alert in recent_alerts:
            type_counts[alert.drift_type] = type_counts.get(alert.drift_type, 0) + 1
        
        return {
            'period_days': days_back,
            'total_alerts': len(recent_alerts),
            'alerts_by_severity': severity_counts,
            'alerts_by_type': type_counts,
            'recent_alerts': [asdict(alert) for alert in recent_alerts[-10:]],  # Last 10
            'response_actions_executed': len([
                action for action in self.response_history
                if datetime.fromisoformat(action['timestamp']) >= cutoff_date
            ]),
            'auto_retrain_count': self.retrain_count_today
        }
    
    def save_monitoring_state(self, filepath: str):
        """Save current monitoring state."""
        state = {
            'drift_threshold': self.drift_threshold,
            'drift_history': [asdict(alert) for alert in self.drift_history],
            'response_history': self.response_history,
            'retrain_count_today': self.retrain_count_today,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Monitoring state saved to {filepath}")

# Factory functions
def create_enhanced_drift_monitor(reference_data: pd.DataFrame,
                                baseline_metrics: Dict[str, float],
                                alert_config: AlertConfig,
                                response_config: ResponseConfig,
                                **kwargs) -> EnhancedDriftMonitor:
    """Factory function to create enhanced drift monitor."""
    return EnhancedDriftMonitor(
        reference_data=reference_data,
        baseline_metrics=baseline_metrics,
        alert_config=alert_config,
        response_config=response_config,
        **kwargs
    )

def create_default_alert_config() -> AlertConfig:
    """Create default alert configuration."""
    return AlertConfig(
        enabled_channels=[AlertChannel.LOG],
        severity_thresholds={
            DriftSeverity.LOW.value: 0.1,
            DriftSeverity.MEDIUM.value: 0.2,
            DriftSeverity.HIGH.value: 0.4,
            DriftSeverity.CRITICAL.value: 0.7
        }
    )

def create_default_response_config() -> ResponseConfig:
    """Create default response configuration."""
    return ResponseConfig(
        enabled_actions=[
            ResponseAction.UPDATE_THRESHOLDS,
            ResponseAction.HUMAN_REVIEW
        ],
        retrain_config={
            'trigger_threshold': 0.3,
            'max_auto_retrains_per_day': 2,
            'require_human_approval': True
        }
    )