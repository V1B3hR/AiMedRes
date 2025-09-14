#!/usr/bin/env python3
"""
A/B Testing Infrastructure for DuetMind Adaptive MLOps.
Enables controlled model deployment and performance comparison.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
import random
import sqlite3
import hashlib
from enum import Enum
import mlflow
from mlflow import MlflowClient
from scipy import stats

from ..audit.event_chain import AuditEventChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class ABTestConfig:
    """Configuration for A/B test experiment."""
    experiment_name: str
    model_a_version: str  # Control model
    model_b_version: str  # Treatment model
    traffic_split: float = 0.5  # Fraction going to model B
    duration_days: int = 7
    min_samples_per_variant: int = 1000
    significance_level: float = 0.05
    statistical_power: float = 0.8
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = None
    segment_criteria: Dict[str, Any] = None  # User segment filtering
    ramping_strategy: str = "immediate"  # immediate, gradual, canary


@dataclass
class ABTestResult:
    """Result of A/B test analysis."""
    experiment_name: str
    model_a_version: str
    model_b_version: str
    samples_a: int
    samples_b: int
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, bool]
    practical_significance: Dict[str, bool]
    winner: Optional[str]  # "model_a", "model_b", or None
    recommendation: str
    
    
@dataclass
class PredictionRecord:
    """Record of a prediction for A/B testing."""
    timestamp: datetime
    user_id: str
    model_version: str
    features: Dict[str, Any]
    prediction: Any
    true_label: Optional[Any]
    latency_ms: float
    variant: str  # "control" or "treatment"
    experiment_name: str


class UserSegmenter:
    """Handles user segmentation for A/B testing."""
    
    def __init__(self, segment_criteria: Optional[Dict[str, Any]] = None):
        self.segment_criteria = segment_criteria or {}
    
    def should_include_user(self, user_id: str, features: Dict[str, Any]) -> bool:
        """Check if user should be included in the experiment."""
        if not self.segment_criteria:
            return True
        
        # Implement segment criteria checking
        for criterion, value in self.segment_criteria.items():
            if criterion in features:
                feature_value = features[criterion]
                
                if isinstance(value, dict):
                    # Range criteria
                    if "min" in value and feature_value < value["min"]:
                        return False
                    if "max" in value and feature_value > value["max"]:
                        return False
                elif isinstance(value, list):
                    # Include list
                    if feature_value not in value:
                        return False
                else:
                    # Exact match
                    if feature_value != value:
                        return False
        
        return True
    
    def assign_variant(self, user_id: str, traffic_split: float) -> str:
        """Assign user to control or treatment variant."""
        # Use deterministic hash-based assignment for consistency
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        hash_value = int(user_hash[:8], 16) / (2**32)  # Convert to 0-1 range
        
        return "treatment" if hash_value < traffic_split else "control"


class ABTestingManager:
    """A/B testing manager for model deployment and evaluation."""
    
    def __init__(self, 
                 mlflow_tracking_uri: str = "sqlite:///mlflow.db",
                 experiment_db: str = "ab_experiments.db"):
        """Initialize A/B testing manager."""
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_db = experiment_db
        
        # Setup MLflow client
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize experiment database
        self._init_experiment_db()
        
        # Initialize audit chain
        self.audit_chain = AuditEventChain("sqlite:///ab_testing_audit.db")
        
        # Active experiments
        self.active_experiments = {}
        self._load_active_experiments()
        
        logger.info("A/B Testing Manager initialized")
    
    def _init_experiment_db(self):
        """Initialize A/B testing database."""
        with sqlite3.connect(self.experiment_db) as conn:
            # Experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT UNIQUE NOT NULL,
                    model_a_version TEXT NOT NULL,
                    model_b_version TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    results TEXT
                )
            """)
            
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    features TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    true_label TEXT,
                    latency_ms REAL,
                    correct BOOLEAN
                )
            """)
            
            # Metrics aggregation table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def _load_active_experiments(self):
        """Load active experiments from database."""
        with sqlite3.connect(self.experiment_db) as conn:
            cursor = conn.execute("""
                SELECT experiment_name, config FROM ab_experiments 
                WHERE status IN ('running', 'paused')
            """)
            
            for row in cursor.fetchall():
                experiment_name, config_json = row
                config = ABTestConfig(**json.loads(config_json))
                self.active_experiments[experiment_name] = {
                    'config': config,
                    'segmenter': UserSegmenter(config.segment_criteria)
                }
        
        logger.info(f"Loaded {len(self.active_experiments)} active experiments")
    
    def create_experiment(self, config: ABTestConfig) -> str:
        """Create a new A/B test experiment."""
        logger.info(f"Creating A/B test experiment: {config.experiment_name}")
        
        # Validate configuration
        self._validate_config(config)
        
        # Create experiment record
        with sqlite3.connect(self.experiment_db) as conn:
            try:
                conn.execute("""
                    INSERT INTO ab_experiments (
                        experiment_name, model_a_version, model_b_version,
                        config, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    config.experiment_name,
                    config.model_a_version,
                    config.model_b_version,
                    json.dumps(asdict(config)),
                    ExperimentStatus.DRAFT.value,
                    datetime.now().isoformat()
                ))
                conn.commit()
                
                # Log audit event
                audit_event_id = self.audit_chain.log_event(
                    event_type="ab_experiment_created",
                    entity_type="experiment",
                    entity_id=config.experiment_name,
                    event_data={
                        'experiment_name': config.experiment_name,
                        'model_a': config.model_a_version,
                        'model_b': config.model_b_version,
                        'traffic_split': config.traffic_split
                    },
                    user_id="system"
                )
                
                logger.info(f"Created experiment: {config.experiment_name}")
                return config.experiment_name
                
            except sqlite3.IntegrityError:
                raise ValueError(f"Experiment {config.experiment_name} already exists")
    
    def _validate_config(self, config: ABTestConfig):
        """Validate experiment configuration."""
        if config.traffic_split < 0 or config.traffic_split > 1:
            raise ValueError("Traffic split must be between 0 and 1")
        
        if config.duration_days <= 0:
            raise ValueError("Duration must be positive")
        
        if config.min_samples_per_variant <= 0:
            raise ValueError("Minimum samples per variant must be positive")
        
        # Check if models exist in MLflow
        try:
            # This is a simplified check - in practice you'd verify model versions exist
            pass
        except Exception as e:
            logger.warning(f"Could not validate model versions: {e}")
    
    def start_experiment(self, experiment_name: str) -> bool:
        """Start an A/B test experiment."""
        logger.info(f"Starting A/B test experiment: {experiment_name}")
        
        with sqlite3.connect(self.experiment_db) as conn:
            # Check experiment exists and is in draft status
            cursor = conn.execute("""
                SELECT config FROM ab_experiments 
                WHERE experiment_name = ? AND status = ?
            """, (experiment_name, ExperimentStatus.DRAFT.value))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment {experiment_name} not found or not in draft status")
            
            # Load configuration
            config = ABTestConfig(**json.loads(row[0]))
            
            # Update experiment status
            conn.execute("""
                UPDATE ab_experiments 
                SET status = ?, started_at = ?
                WHERE experiment_name = ?
            """, (ExperimentStatus.RUNNING.value, datetime.now().isoformat(), experiment_name))
            
            conn.commit()
            
            # Add to active experiments
            self.active_experiments[experiment_name] = {
                'config': config,
                'segmenter': UserSegmenter(config.segment_criteria)
            }
            
            # Log audit event
            self.audit_chain.log_event(
                event_type="ab_experiment_started",
                entity_type="experiment",
                entity_id=experiment_name,
                event_data={'experiment_name': experiment_name},
                user_id="system"
            )
            
            logger.info(f"Started experiment: {experiment_name}")
            return True
    
    def make_prediction(self, 
                       experiment_name: str,
                       user_id: str,
                       features: Dict[str, Any],
                       models: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction through A/B test framework."""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} is not active")
        
        experiment = self.active_experiments[experiment_name]
        config = experiment['config']
        segmenter = experiment['segmenter']
        
        # Check if user should be included
        if not segmenter.should_include_user(user_id, features):
            # Use default model (model A)
            variant = "control"
            model_version = config.model_a_version
        else:
            # Assign variant
            variant = segmenter.assign_variant(user_id, config.traffic_split)
            model_version = config.model_b_version if variant == "treatment" else config.model_a_version
        
        # Get model and make prediction
        start_time = datetime.now()
        
        if model_version in models:
            model = models[model_version]
            # Convert features to appropriate format for model
            feature_array = self._prepare_features_for_model(features)
            prediction = model.predict(feature_array)[0] if hasattr(model, 'predict') else 0
        else:
            # Fallback or error handling
            logger.warning(f"Model {model_version} not available, using default")
            prediction = 0  # Default prediction
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Record prediction
        prediction_record = PredictionRecord(
            timestamp=datetime.now(),
            user_id=user_id,
            model_version=model_version,
            features=features,
            prediction=prediction,
            true_label=None,  # Will be updated later
            latency_ms=latency_ms,
            variant=variant,
            experiment_name=experiment_name
        )
        
        self._store_prediction(prediction_record)
        
        return {
            'prediction': prediction,
            'model_version': model_version,
            'variant': variant,
            'latency_ms': latency_ms
        }
    
    def _prepare_features_for_model(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction (simplified)."""
        # This is a simplified implementation
        # In practice, you'd need proper feature preprocessing
        feature_values = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                feature_values.append(value)
            else:
                feature_values.append(0)  # Handle non-numeric features
        
        return np.array(feature_values).reshape(1, -1)
    
    def _store_prediction(self, record: PredictionRecord):
        """Store prediction record in database."""
        with sqlite3.connect(self.experiment_db) as conn:
            conn.execute("""
                INSERT INTO ab_predictions (
                    timestamp, experiment_name, user_id, model_version,
                    variant, features, prediction, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp.isoformat(),
                record.experiment_name,
                record.user_id,
                record.model_version,
                record.variant,
                json.dumps(record.features),
                json.dumps(record.prediction),
                record.latency_ms
            ))
            conn.commit()
    
    def update_prediction_outcomes(self, 
                                 experiment_name: str,
                                 user_predictions: List[Tuple[str, Any]]):
        """Update predictions with true outcomes."""
        with sqlite3.connect(self.experiment_db) as conn:
            for user_id, true_label in user_predictions:
                # Find recent prediction for this user
                cursor = conn.execute("""
                    SELECT id, prediction FROM ab_predictions 
                    WHERE experiment_name = ? AND user_id = ? AND true_label IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                """, (experiment_name, user_id))
                
                row = cursor.fetchone()
                if row:
                    prediction_id, prediction_json = row
                    prediction = json.loads(prediction_json)
                    
                    # Determine if prediction was correct
                    correct = (prediction == true_label)
                    
                    # Update record
                    conn.execute("""
                        UPDATE ab_predictions 
                        SET true_label = ?, correct = ?
                        WHERE id = ?
                    """, (json.dumps(true_label), correct, prediction_id))
            
            conn.commit()
    
    def analyze_experiment(self, experiment_name: str) -> ABTestResult:
        """Analyze A/B test results."""
        logger.info(f"Analyzing experiment: {experiment_name}")
        
        if experiment_name not in self.active_experiments:
            # Load experiment configuration
            with sqlite3.connect(self.experiment_db) as conn:
                cursor = conn.execute("""
                    SELECT config FROM ab_experiments 
                    WHERE experiment_name = ?
                """, (experiment_name,))
                
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Experiment {experiment_name} not found")
                
                config = ABTestConfig(**json.loads(row[0]))
        else:
            config = self.active_experiments[experiment_name]['config']
        
        # Get prediction data
        with sqlite3.connect(self.experiment_db) as conn:
            cursor = conn.execute("""
                SELECT variant, correct, latency_ms, prediction, true_label
                FROM ab_predictions 
                WHERE experiment_name = ? AND true_label IS NOT NULL
            """, (experiment_name,))
            
            predictions = cursor.fetchall()
        
        if not predictions:
            raise ValueError(f"No labeled data available for experiment {experiment_name}")
        
        # Separate by variant
        control_data = [p for p in predictions if p[0] == "control"]
        treatment_data = [p for p in predictions if p[0] == "treatment"]
        
        # Calculate metrics for each variant
        metrics_a = self._calculate_variant_metrics(control_data)
        metrics_b = self._calculate_variant_metrics(treatment_data)
        
        # Statistical testing
        p_values = {}
        confidence_intervals = {}
        statistical_significance = {}
        practical_significance = {}
        
        # Test primary metric (accuracy)
        if len(control_data) > 0 and len(treatment_data) > 0:
            control_correct = [p[1] for p in control_data]
            treatment_correct = [p[1] for p in treatment_data]
            
            # Chi-square test for accuracy
            control_success = sum(control_correct)
            control_total = len(control_correct)
            treatment_success = sum(treatment_correct)
            treatment_total = len(treatment_correct)
            
            if control_total > 0 and treatment_total > 0:
                # Two-proportion z-test
                p1 = control_success / control_total
                p2 = treatment_success / treatment_total
                
                pooled_p = (control_success + treatment_success) / (control_total + treatment_total)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_total + 1/treatment_total))
                
                if se > 0:
                    z_score = (p2 - p1) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    
                    p_values['accuracy'] = p_value
                    statistical_significance['accuracy'] = p_value < config.significance_level
                    
                    # Confidence interval for difference
                    margin_error = stats.norm.ppf(1 - config.significance_level/2) * se
                    ci_lower = (p2 - p1) - margin_error
                    ci_upper = (p2 - p1) + margin_error
                    confidence_intervals['accuracy'] = (ci_lower, ci_upper)
                    
                    # Practical significance (effect size > 1%)
                    practical_significance['accuracy'] = abs(p2 - p1) > 0.01
        
        # Test latency
        control_latency = [p[2] for p in control_data if p[2] is not None]
        treatment_latency = [p[2] for p in treatment_data if p[2] is not None]
        
        if len(control_latency) > 10 and len(treatment_latency) > 10:
            t_stat, p_value = stats.ttest_ind(control_latency, treatment_latency)
            p_values['latency'] = p_value
            statistical_significance['latency'] = p_value < config.significance_level
            practical_significance['latency'] = abs(metrics_b['avg_latency'] - metrics_a['avg_latency']) > 50  # 50ms threshold
        
        # Determine winner
        winner = None
        recommendation = "Continue testing - insufficient evidence"
        
        if statistical_significance.get('accuracy', False):
            if metrics_b['accuracy'] > metrics_a['accuracy']:
                if practical_significance.get('accuracy', False):
                    winner = "model_b"
                    recommendation = "Deploy Model B - statistically and practically significant improvement"
                else:
                    recommendation = "Model B wins statistically but improvement is small"
            else:
                winner = "model_a"
                recommendation = "Keep Model A - Model B performs significantly worse"
        
        return ABTestResult(
            experiment_name=experiment_name,
            model_a_version=config.model_a_version,
            model_b_version=config.model_b_version,
            samples_a=len(control_data),
            samples_b=len(treatment_data),
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            winner=winner,
            recommendation=recommendation
        )
    
    def _calculate_variant_metrics(self, variant_data: List[Tuple]) -> Dict[str, float]:
        """Calculate metrics for a variant."""
        if not variant_data:
            return {'accuracy': 0.0, 'avg_latency': 0.0, 'sample_count': 0}
        
        correct_predictions = [p[1] for p in variant_data if p[1] is not None]
        latencies = [p[2] for p in variant_data if p[2] is not None]
        
        return {
            'accuracy': sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0.0,
            'avg_latency': np.mean(latencies) if latencies else 0.0,
            'sample_count': len(variant_data)
        }
    
    def stop_experiment(self, experiment_name: str, reason: str = "Manual stop") -> ABTestResult:
        """Stop an A/B test experiment."""
        logger.info(f"Stopping experiment: {experiment_name}")
        
        # Analyze results before stopping
        results = self.analyze_experiment(experiment_name)
        
        # Update experiment status
        with sqlite3.connect(self.experiment_db) as conn:
            conn.execute("""
                UPDATE ab_experiments 
                SET status = ?, ended_at = ?, results = ?
                WHERE experiment_name = ?
            """, (
                ExperimentStatus.COMPLETED.value,
                datetime.now().isoformat(),
                json.dumps(asdict(results)),
                experiment_name
            ))
            conn.commit()
        
        # Remove from active experiments
        if experiment_name in self.active_experiments:
            del self.active_experiments[experiment_name]
        
        # Log audit event
        self.audit_chain.log_event(
            event_type="ab_experiment_stopped",
            entity_type="experiment",
            entity_id=experiment_name,
            event_data={
                'experiment_name': experiment_name,
                'reason': reason,
                'winner': results.winner,
                'recommendation': results.recommendation
            },
            user_id="system"
        )
        
        logger.info(f"Stopped experiment: {experiment_name} - Winner: {results.winner}")
        return results
    
    def get_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        with sqlite3.connect(self.experiment_db) as conn:
            cursor = conn.execute("""
                SELECT status, created_at, started_at, ended_at FROM ab_experiments 
                WHERE experiment_name = ?
            """, (experiment_name,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment {experiment_name} not found")
            
            status, created_at, started_at, ended_at = row
            
            # Get sample counts
            cursor = conn.execute("""
                SELECT variant, COUNT(*) FROM ab_predictions 
                WHERE experiment_name = ?
                GROUP BY variant
            """, (experiment_name,))
            
            sample_counts = dict(cursor.fetchall())
        
        return {
            'experiment_name': experiment_name,
            'status': status,
            'created_at': created_at,
            'started_at': started_at,
            'ended_at': ended_at,
            'sample_counts': sample_counts,
            'is_active': experiment_name in self.active_experiments
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        with sqlite3.connect(self.experiment_db) as conn:
            cursor = conn.execute("""
                SELECT experiment_name, status, model_a_version, model_b_version,
                       created_at, started_at, ended_at
                FROM ab_experiments 
                ORDER BY created_at DESC
            """)
            
            columns = ['experiment_name', 'status', 'model_a_version', 'model_b_version',
                      'created_at', 'started_at', 'ended_at']
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


if __name__ == "__main__":
    # Demo usage
    print("A/B Testing Manager Demo")
    
    # Create A/B testing manager
    ab_manager = ABTestingManager()
    
    # Create experiment configuration
    config = ABTestConfig(
        experiment_name="alzheimer_model_v2_test",
        model_a_version="v1.0.0",
        model_b_version="v2.0.0",
        traffic_split=0.3,  # 30% to new model
        duration_days=7,
        min_samples_per_variant=100
    )
    
    # Create and start experiment
    experiment_name = ab_manager.create_experiment(config)
    ab_manager.start_experiment(experiment_name)
    
    # Simulate some predictions
    print("Simulating predictions...")
    
    # Mock models for demo
    class MockModel:
        def __init__(self, accuracy):
            self.accuracy = accuracy
        
        def predict(self, X):
            # Simulate predictions with given accuracy
            return [1 if random.random() < self.accuracy else 0 for _ in range(len(X))]
    
    models = {
        "v1.0.0": MockModel(0.85),
        "v2.0.0": MockModel(0.88)
    }
    
    # Generate predictions
    predictions_to_update = []
    
    for i in range(200):
        user_id = f"user_{i}"
        features = {
            'age': random.randint(60, 90),
            'mmse_score': random.randint(10, 30),
            'education': random.randint(8, 20)
        }
        
        result = ab_manager.make_prediction(experiment_name, user_id, features, models)
        
        # Simulate true outcome (with some noise)
        true_label = 1 if random.random() < 0.8 else 0
        predictions_to_update.append((user_id, true_label))
    
    # Update with true outcomes
    ab_manager.update_prediction_outcomes(experiment_name, predictions_to_update)
    
    # Analyze results
    results = ab_manager.analyze_experiment(experiment_name)
    
    print(f"\nExperiment Results:")
    print(f"Model A (Control): {results.samples_a} samples, {results.metrics_a['accuracy']:.3f} accuracy")
    print(f"Model B (Treatment): {results.samples_b} samples, {results.metrics_b['accuracy']:.3f} accuracy")
    print(f"Statistical Significance: {results.statistical_significance}")
    print(f"Winner: {results.winner}")
    print(f"Recommendation: {results.recommendation}")
    
    # Stop experiment
    final_results = ab_manager.stop_experiment(experiment_name)
    
    print("\nDemo completed!")