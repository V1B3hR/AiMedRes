#!/usr/bin/env python3
"""
Production Model Monitoring for DuetMind Adaptive MLOps.
Real-time monitoring of model performance, data drift, and system health.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import sqlite3
from threading import Thread
import time
import redis
from contextlib import contextmanager

from ..drift.evidently_drift_monitor import DriftMonitor
from ..audit.event_chain import AuditEventChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Metrics for production monitoring."""
    timestamp: datetime
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    drift_score: float
    data_quality_score: float
    prediction_count: int
    feature_drift_detected: bool
    concept_drift_detected: bool


@dataclass 
class AlertConfig:
    """Configuration for monitoring alerts."""
    critical_accuracy_drop: float = 0.05  # 5% drop
    warning_accuracy_drop: float = 0.02   # 2% drop
    critical_error_rate: float = 0.02      # 2% error rate
    warning_error_rate: float = 0.01       # 1% error rate
    critical_latency_ms: float = 500.0     # 500ms
    warning_latency_ms: float = 200.0      # 200ms
    drift_threshold: float = 0.1           # 10% drift
    data_quality_threshold: float = 0.95   # 95% quality


class ProductionMonitor:
    """Production monitoring system for model performance and health."""
    
    def __init__(self, 
                 model_name: str,
                 mlflow_tracking_uri: str = "sqlite:///mlflow.db",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 alert_config: Optional[AlertConfig] = None):
        """Initialize production monitor."""
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.alert_config = alert_config or AlertConfig()
        
        # Setup MLflow client
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Setup Redis for real-time metrics (optional)
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connected for real-time metrics")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using local storage.")
        
        # Initialize monitoring storage
        self.monitoring_db = "monitoring.db"
        self._init_monitoring_db()
        
        # Initialize drift monitor
        self.drift_monitor = None
        
        # Monitoring state
        self.is_monitoring = False
        self.baseline_metrics = None
        
        logger.info(f"Production monitor initialized for model: {model_name}")
    
    def _init_monitoring_db(self):
        """Initialize monitoring database."""
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    latency_ms REAL,
                    throughput_rps REAL,
                    error_rate REAL,
                    drift_score REAL,
                    data_quality_score REAL,
                    prediction_count INTEGER,
                    feature_drift_detected BOOLEAN,
                    concept_drift_detected BOOLEAN,
                    raw_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metrics TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    def set_baseline_metrics(self, baseline_data: pd.DataFrame, 
                           baseline_labels: pd.Series,
                           model_version: str = "baseline"):
        """Set baseline metrics for comparison."""
        logger.info("Setting baseline metrics for monitoring")
        
        # Initialize drift monitor with baseline data
        self.drift_monitor = DriftMonitor(baseline_data)
        
        # Store baseline metrics
        self.baseline_metrics = {
            'model_version': model_version,
            'data_size': len(baseline_data),
            'feature_count': len(baseline_data.columns),
            'label_distribution': baseline_labels.value_counts().to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in Redis if available
        if self.redis_client:
            try:
                baseline_dict = {k: str(v) for k, v in self.baseline_metrics.items()}
                self.redis_client.hset("monitoring:baseline", mapping=baseline_dict)
            except Exception as e:
                logger.warning(f"Failed to store baseline in Redis: {e}")
                self.redis_client = None  # Disable Redis for future operations
        
        logger.info(f"Baseline metrics set for {len(baseline_data)} samples with {len(baseline_data.columns)} features")
    
    def log_prediction_batch(self, 
                           features: pd.DataFrame,
                           predictions: np.ndarray,
                           true_labels: Optional[np.ndarray] = None,
                           model_version: str = "current",
                           latency_ms: float = 0.0) -> Dict[str, Any]:
        """Log a batch of predictions for monitoring."""
        timestamp = datetime.now()
        
        # Calculate basic metrics
        prediction_count = len(predictions)
        error_rate = 0.0
        
        # Performance metrics (if true labels available)
        accuracy = precision = recall = f1_score = 0.0
        if true_labels is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as f1
            try:
                accuracy = accuracy_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
                recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
                f1_score = f1(true_labels, predictions, average='weighted', zero_division=0)
            except Exception as e:
                logger.warning(f"Error calculating performance metrics: {e}")
        
        # Drift detection
        drift_score = 0.0
        feature_drift_detected = False
        concept_drift_detected = False
        
        if self.drift_monitor and len(features) > 0:
            try:
                drift_results = self.drift_monitor.detect_data_drift(features)
                drift_score = drift_results.get('drift_score', 0.0)
                feature_drift_detected = drift_results.get('overall_drift_detected', False)
                
                # Simple concept drift detection (performance degradation)
                if self.baseline_metrics and accuracy > 0:
                    # For concept drift, we'd need historical performance data
                    # Simplified: check if current accuracy is significantly lower
                    concept_drift_detected = accuracy < 0.8  # Simplified threshold
                    
            except Exception as e:
                logger.warning(f"Error in drift detection: {e}")
        
        # Data quality score (simplified)
        data_quality_score = self._calculate_data_quality(features)
        
        # Create monitoring metrics
        metrics = MonitoringMetrics(
            timestamp=timestamp,
            model_version=model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=latency_ms,
            throughput_rps=prediction_count / max(latency_ms / 1000, 0.001),  # Rough estimate
            error_rate=error_rate,
            drift_score=drift_score,
            data_quality_score=data_quality_score,
            prediction_count=prediction_count,
            feature_drift_detected=feature_drift_detected,
            concept_drift_detected=concept_drift_detected
        )
        
        # Store metrics
        self._store_metrics(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update Redis with latest metrics
        if self.redis_client:
            try:
                metrics_dict = {
                    'timestamp': metrics.timestamp.isoformat(),
                    'model_version': metrics.model_version,
                    'accuracy': str(metrics.accuracy),
                    'precision': str(metrics.precision),
                    'recall': str(metrics.recall),
                    'f1_score': str(metrics.f1_score),
                    'latency_ms': str(metrics.latency_ms),
                    'throughput_rps': str(metrics.throughput_rps),
                    'error_rate': str(metrics.error_rate),
                    'drift_score': str(metrics.drift_score),
                    'data_quality_score': str(metrics.data_quality_score),
                    'prediction_count': str(metrics.prediction_count),
                    'feature_drift_detected': str(metrics.feature_drift_detected),
                    'concept_drift_detected': str(metrics.concept_drift_detected)
                }
                self.redis_client.hset("monitoring:latest", mapping=metrics_dict)
            except Exception as e:
                logger.warning(f"Failed to update Redis metrics: {e}")
                self.redis_client = None  # Disable Redis for future operations
        
        logger.info(f"Logged {prediction_count} predictions - Accuracy: {accuracy:.3f}, Drift: {drift_score:.3f}")
        
        return {
            'metrics': asdict(metrics),
            'alerts_triggered': self._get_recent_alerts(minutes=1)
        }
    
    def _calculate_data_quality(self, features: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if len(features) == 0:
            return 0.0
        
        # Simple data quality metrics
        missing_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
        duplicate_ratio = features.duplicated().sum() / len(features)
        
        # Quality score (1.0 = perfect, 0.0 = terrible)
        quality_score = 1.0 - (missing_ratio + duplicate_ratio)
        return max(0.0, min(1.0, quality_score))
    
    def _store_metrics(self, metrics: MonitoringMetrics):
        """Store metrics in database."""
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                INSERT INTO monitoring_metrics (
                    timestamp, model_version, accuracy, precision, recall, f1_score,
                    latency_ms, throughput_rps, error_rate, drift_score, data_quality_score,
                    prediction_count, feature_drift_detected, concept_drift_detected, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.model_version,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.latency_ms,
                metrics.throughput_rps,
                metrics.error_rate,
                metrics.drift_score,
                metrics.data_quality_score,
                metrics.prediction_count,
                metrics.feature_drift_detected,
                metrics.concept_drift_detected,
                json.dumps({
                    'timestamp': metrics.timestamp.isoformat(),
                    'model_version': metrics.model_version,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'latency_ms': metrics.latency_ms,
                    'throughput_rps': metrics.throughput_rps,
                    'error_rate': metrics.error_rate,
                    'drift_score': metrics.drift_score,
                    'data_quality_score': metrics.data_quality_score,
                    'prediction_count': metrics.prediction_count,
                    'feature_drift_detected': metrics.feature_drift_detected,
                    'concept_drift_detected': metrics.concept_drift_detected
                })
            ))
            conn.commit()
    
    def _check_alerts(self, metrics: MonitoringMetrics):
        """Check if metrics trigger any alerts."""
        alerts = []
        
        # Accuracy alerts
        if self.baseline_metrics and metrics.accuracy > 0:
            baseline_accuracy = 0.85  # Default baseline
            accuracy_drop = baseline_accuracy - metrics.accuracy
            
            if accuracy_drop > self.alert_config.critical_accuracy_drop:
                alerts.append(('CRITICAL', f'Model accuracy dropped by {accuracy_drop:.3f}'))
            elif accuracy_drop > self.alert_config.warning_accuracy_drop:
                alerts.append(('WARNING', f'Model accuracy dropped by {accuracy_drop:.3f}'))
        
        # Error rate alerts
        if metrics.error_rate > self.alert_config.critical_error_rate:
            alerts.append(('CRITICAL', f'High error rate: {metrics.error_rate:.3f}'))
        elif metrics.error_rate > self.alert_config.warning_error_rate:
            alerts.append(('WARNING', f'Elevated error rate: {metrics.error_rate:.3f}'))
        
        # Latency alerts
        if metrics.latency_ms > self.alert_config.critical_latency_ms:
            alerts.append(('CRITICAL', f'High latency: {metrics.latency_ms:.1f}ms'))
        elif metrics.latency_ms > self.alert_config.warning_latency_ms:
            alerts.append(('WARNING', f'Elevated latency: {metrics.latency_ms:.1f}ms'))
        
        # Drift alerts
        if metrics.feature_drift_detected:
            alerts.append(('WARNING', f'Feature drift detected: {metrics.drift_score:.3f}'))
        
        if metrics.concept_drift_detected:
            alerts.append(('CRITICAL', 'Concept drift detected - model retraining recommended'))
        
        # Data quality alerts
        if metrics.data_quality_score < self.alert_config.data_quality_threshold:
            alerts.append(('WARNING', f'Data quality issues: {metrics.data_quality_score:.3f}'))
        
        # Store alerts
        for severity, message in alerts:
            self._store_alert(severity, message, metrics)
            logger.warning(f"ALERT [{severity}]: {message}")
    
    def _store_alert(self, severity: str, message: str, metrics: MonitoringMetrics):
        """Store alert in database."""
        with sqlite3.connect(self.monitoring_db) as conn:
            conn.execute("""
                INSERT INTO monitoring_alerts (timestamp, alert_type, severity, message, metrics)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                'MONITORING',
                severity,
                message,
                json.dumps({
                    'timestamp': metrics.timestamp.isoformat(),
                    'model_version': metrics.model_version,
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'latency_ms': metrics.latency_ms,
                    'throughput_rps': metrics.throughput_rps,
                    'error_rate': metrics.error_rate,
                    'drift_score': metrics.drift_score,
                    'data_quality_score': metrics.data_quality_score,
                    'prediction_count': metrics.prediction_count,
                    'feature_drift_detected': metrics.feature_drift_detected,
                    'concept_drift_detected': metrics.concept_drift_detected
                })
            ))
            conn.commit()
    
    def _get_recent_alerts(self, minutes: int = 60) -> List[Dict]:
        """Get recent alerts."""
        cutoff_time = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        
        with sqlite3.connect(self.monitoring_db) as conn:
            cursor = conn.execute("""
                SELECT timestamp, alert_type, severity, message 
                FROM monitoring_alerts 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            return [
                {
                    'timestamp': row[0],
                    'alert_type': row[1],
                    'severity': row[2],
                    'message': row[3]
                }
                for row in cursor.fetchall()
            ]
    
    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get monitoring summary for the specified time period."""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.monitoring_db) as conn:
            # Get metrics summary
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(accuracy) as avg_accuracy,
                    AVG(latency_ms) as avg_latency,
                    AVG(error_rate) as avg_error_rate,
                    AVG(drift_score) as avg_drift_score,
                    SUM(CASE WHEN feature_drift_detected THEN 1 ELSE 0 END) as drift_incidents,
                    AVG(data_quality_score) as avg_data_quality
                FROM monitoring_metrics 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            metrics_row = cursor.fetchone()
            
            # Get alerts summary
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM monitoring_alerts 
                WHERE timestamp > ?
                GROUP BY severity
            """, (cutoff_time,))
            
            alerts_summary = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            'period_hours': hours,
            'total_predictions': metrics_row[0] or 0,
            'avg_accuracy': metrics_row[1] or 0.0,
            'avg_latency_ms': metrics_row[2] or 0.0,
            'avg_error_rate': metrics_row[3] or 0.0,
            'avg_drift_score': metrics_row[4] or 0.0,
            'drift_incidents': metrics_row[5] or 0,
            'avg_data_quality': metrics_row[6] or 0.0,
            'alerts': alerts_summary,
            'status': self._get_overall_status(metrics_row, alerts_summary)
        }
    
    def _get_overall_status(self, metrics_row: tuple, alerts_summary: Dict[str, int]) -> str:
        """Determine overall system status."""
        critical_alerts = alerts_summary.get('CRITICAL', 0)
        warning_alerts = alerts_summary.get('WARNING', 0)
        
        if critical_alerts > 0:
            return 'CRITICAL'
        elif warning_alerts > 5:  # Threshold for too many warnings
            return 'DEGRADED'
        elif metrics_row[1] and metrics_row[1] < 0.8:  # Low accuracy
            return 'DEGRADED'
        else:
            return 'HEALTHY'
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring (for real-time systems)."""
        self.is_monitoring = True
        
        def monitoring_loop():
            while self.is_monitoring:
                try:
                    # Generate monitoring report
                    summary = self.get_monitoring_summary(hours=1)
                    logger.info(f"Monitoring status: {summary['status']} - {summary['total_predictions']} predictions")
                    
                    # Update Redis with summary
                    if self.redis_client:
                        try:
                            summary_dict = {k: str(v) for k, v in summary.items() if not isinstance(v, dict)}
                            self.redis_client.hset("monitoring:summary", mapping=summary_dict)
                        except Exception as e:
                            logger.warning(f"Failed to update Redis summary: {e}")
                            self.redis_client = None  # Disable Redis
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait longer on error
        
        monitoring_thread = Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        logger.info(f"Started continuous monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        logger.info("Stopped continuous monitoring")


def create_production_monitor(model_name: str, 
                            baseline_data: Optional[pd.DataFrame] = None,
                            baseline_labels: Optional[pd.Series] = None) -> ProductionMonitor:
    """Factory function to create a production monitor."""
    monitor = ProductionMonitor(model_name)
    
    if baseline_data is not None and baseline_labels is not None:
        monitor.set_baseline_metrics(baseline_data, baseline_labels)
    
    return monitor


if __name__ == "__main__":
    # Demo usage
    print("Production Monitor Demo")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    })
    
    sample_labels = pd.Series(np.random.choice([0, 1], 100))
    sample_predictions = np.random.choice([0, 1], 50)
    
    # Create monitor
    monitor = create_production_monitor(
        "alzheimer_classifier",
        baseline_data=sample_data,
        baseline_labels=sample_labels
    )
    
    # Log some predictions
    current_data = sample_data.iloc[:50]
    result = monitor.log_prediction_batch(
        features=current_data,
        predictions=sample_predictions,
        true_labels=sample_labels.iloc[:50].values,
        latency_ms=150.0
    )
    
    print("Logged predictions:", result['metrics']['prediction_count'])
    
    # Get monitoring summary
    summary = monitor.get_monitoring_summary()
    print(f"Monitoring Summary: {summary['status']} status, {summary['total_predictions']} predictions")