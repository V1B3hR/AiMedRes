#!/usr/bin/env python3
"""
Data-Driven Automated Retraining Trigger for DuetMind Adaptive MLOps.
Automatically triggers model retraining based on data arrival, drift, and performance metrics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import yaml
import mlflow
from mlflow import MlflowClient
import hashlib
import sqlite3
from threading import Thread, Event
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .production_monitor import ProductionMonitor
from ..drift.evidently_drift_monitor import DriftMonitor
from ..audit.event_chain import AuditEventChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrainingTriggerConfig:
    """Configuration for automated retraining triggers."""
    # Data-based triggers
    min_new_samples: int = 1000
    max_days_without_retrain: int = 7
    new_data_check_interval: int = 3600  # Check every hour
    
    # Performance-based triggers  
    accuracy_degradation_threshold: float = 0.05
    drift_score_threshold: float = 0.1
    error_rate_threshold: float = 0.02
    
    # Data quality triggers
    data_quality_threshold: float = 0.95
    missing_data_threshold: float = 0.1
    
    # Retraining constraints
    min_hours_between_retrains: int = 6
    max_retrains_per_day: int = 4
    require_manual_approval: bool = False
    
    # File system monitoring
    data_directories: List[str] = None
    watch_file_patterns: List[str] = None


@dataclass 
class RetrainingEvent:
    """Record of a retraining event."""
    timestamp: datetime
    trigger_type: str
    trigger_reason: str
    data_size: int
    previous_model_version: str
    new_model_version: Optional[str]
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]]
    success: bool
    error_message: Optional[str]
    audit_event_id: Optional[str]


class DataWatcher(FileSystemEventHandler):
    """File system watcher for new data arrival."""
    
    def __init__(self, trigger_callback: Callable, file_patterns: List[str] = None):
        self.trigger_callback = trigger_callback
        self.file_patterns = file_patterns or ['.csv', '.parquet', '.json']
        self.last_trigger = datetime.now()
        self.min_trigger_interval = timedelta(minutes=30)  # Avoid rapid triggers
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._check_and_trigger(event.src_path, 'file_created')
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._check_and_trigger(event.src_path, 'file_modified')
    
    def _check_and_trigger(self, file_path: str, event_type: str):
        """Check if file matches patterns and trigger if needed."""
        # Check if enough time has passed since last trigger
        if datetime.now() - self.last_trigger < self.min_trigger_interval:
            return
        
        # Check if file matches patterns
        if any(pattern in file_path for pattern in self.file_patterns):
            logger.info(f"Data file {event_type}: {file_path}")
            
            # Check file size (avoid triggering on empty files)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 1024:  # At least 1KB
                    self.trigger_callback('data_arrival', f'{event_type}: {file_path}')
                    self.last_trigger = datetime.now()
            except OSError:
                logger.warning(f"Could not check size of {file_path}")


class DataDrivenRetrainingTrigger:
    """Automated retraining trigger based on data and performance metrics."""
    
    def __init__(self, 
                 model_name: str,
                 config: Optional[RetrainingTriggerConfig] = None,
                 production_monitor: Optional[ProductionMonitor] = None,
                 mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize automated retraining trigger."""
        self.model_name = model_name
        self.config = config or RetrainingTriggerConfig()
        self.production_monitor = production_monitor
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        # Setup MLflow client
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize tracking database
        self.trigger_db = "retraining_triggers.db"
        self._init_trigger_db()
        
        # Initialize audit chain
        self.audit_chain = AuditEventChain("sqlite:///retraining_audit.db")
        
        # File system monitoring
        self.observer = None
        self.data_watcher = None
        
        # Trigger state
        self.is_active = False
        self.stop_event = Event()
        self.last_retrain_time = None
        self.retrains_today = 0
        
        # Load last retrain time
        self._load_last_retrain_time()
        
        logger.info(f"Retraining trigger initialized for model: {model_name}")
    
    def _init_trigger_db(self):
        """Initialize trigger tracking database."""
        with sqlite3.connect(self.trigger_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retraining_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    data_size INTEGER,
                    previous_model_version TEXT,
                    new_model_version TEXT,
                    performance_before TEXT,
                    performance_after TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    audit_event_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trigger_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def _load_last_retrain_time(self):
        """Load last retrain time from database."""
        with sqlite3.connect(self.trigger_db) as conn:
            cursor = conn.execute("""
                SELECT value FROM trigger_state WHERE key = 'last_retrain_time'
            """)
            row = cursor.fetchone()
            
            if row:
                self.last_retrain_time = datetime.fromisoformat(row[0])
                
                # Count retrains today
                today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM retraining_events 
                    WHERE timestamp > ? AND success = 1
                """, (today_start.isoformat(),))
                
                self.retrains_today = cursor.fetchone()[0]
    
    def _update_trigger_state(self, key: str, value: str):
        """Update trigger state in database."""
        with sqlite3.connect(self.trigger_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trigger_state (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))
            conn.commit()
    
    def start_monitoring(self):
        """Start automated monitoring for retraining triggers."""
        if self.is_active:
            logger.warning("Trigger monitoring is already active")
            return
        
        self.is_active = True
        self.stop_event.clear()
        
        # Start file system monitoring
        if self.config.data_directories:
            self._start_file_monitoring()
        
        # Start periodic checks
        monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Started automated retraining trigger monitoring")
    
    def stop_monitoring(self):
        """Stop automated monitoring."""
        self.is_active = False
        self.stop_event.set()
        
        # Stop file system monitoring
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        logger.info("Stopped automated retraining trigger monitoring")
    
    def _start_file_monitoring(self):
        """Start file system monitoring for new data."""
        self.data_watcher = DataWatcher(
            trigger_callback=self._handle_trigger,
            file_patterns=self.config.watch_file_patterns
        )
        
        self.observer = Observer()
        
        for directory in self.config.data_directories:
            if os.path.exists(directory):
                self.observer.schedule(self.data_watcher, directory, recursive=True)
                logger.info(f"Watching directory for new data: {directory}")
        
        self.observer.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop for periodic checks."""
        while self.is_active and not self.stop_event.wait(self.config.new_data_check_interval):
            try:
                # Check various trigger conditions
                self._check_time_based_trigger()
                self._check_performance_based_trigger()
                self._check_data_quality_trigger()
                
                # Reset daily counter if needed
                self._reset_daily_counter()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _check_time_based_trigger(self):
        """Check if retraining should be triggered based on time."""
        if not self.last_retrain_time:
            return
        
        days_since_retrain = (datetime.now() - self.last_retrain_time).days
        
        if days_since_retrain >= self.config.max_days_without_retrain:
            self._handle_trigger(
                'time_based',
                f'Maximum days without retraining exceeded: {days_since_retrain} days'
            )
    
    def _check_performance_based_trigger(self):
        """Check if retraining should be triggered based on performance."""
        if not self.production_monitor:
            return
        
        # Get recent performance metrics
        summary = self.production_monitor.get_monitoring_summary(hours=24)
        
        # Check accuracy degradation
        if summary['avg_accuracy'] > 0:
            baseline_accuracy = 0.85  # Could be loaded from baseline metrics
            accuracy_drop = baseline_accuracy - summary['avg_accuracy']
            
            if accuracy_drop > self.config.accuracy_degradation_threshold:
                self._handle_trigger(
                    'performance_degradation',
                    f'Accuracy dropped by {accuracy_drop:.3f} (threshold: {self.config.accuracy_degradation_threshold})'
                )
                return
        
        # Check drift score
        if summary['avg_drift_score'] > self.config.drift_score_threshold:
            self._handle_trigger(
                'data_drift',
                f'High drift score: {summary["avg_drift_score"]:.3f} (threshold: {self.config.drift_score_threshold})'
            )
            return
        
        # Check error rate
        if summary['avg_error_rate'] > self.config.error_rate_threshold:
            self._handle_trigger(
                'high_error_rate',
                f'High error rate: {summary["avg_error_rate"]:.3f} (threshold: {self.config.error_rate_threshold})'
            )
            return
    
    def _check_data_quality_trigger(self):
        """Check if retraining should be triggered based on data quality."""
        if not self.production_monitor:
            return
        
        summary = self.production_monitor.get_monitoring_summary(hours=24)
        
        if summary['avg_data_quality'] < self.config.data_quality_threshold:
            self._handle_trigger(
                'data_quality',
                f'Low data quality: {summary["avg_data_quality"]:.3f} (threshold: {self.config.data_quality_threshold})'
            )
    
    def _check_new_data_volume(self) -> bool:
        """Check if enough new data has arrived for retraining."""
        # This is a simplified implementation
        # In practice, you'd check your data pipeline for new samples
        
        # Check data directories for new files
        new_samples = 0
        
        if self.config.data_directories:
            for directory in self.config.data_directories:
                if os.path.exists(directory):
                    cutoff_time = self.last_retrain_time or (datetime.now() - timedelta(days=1))
                    
                    for file_path in Path(directory).rglob("*.csv"):
                        if file_path.stat().st_mtime > cutoff_time.timestamp():
                            # Estimate samples (simplified)
                            try:
                                df = pd.read_csv(file_path)
                                new_samples += len(df)
                            except Exception:
                                continue
        
        return new_samples >= self.config.min_new_samples
    
    def _handle_trigger(self, trigger_type: str, reason: str):
        """Handle a retraining trigger."""
        logger.info(f"Retraining trigger: {trigger_type} - {reason}")
        
        # Check constraints
        if not self._can_retrain():
            logger.info("Retraining skipped due to constraints")
            return
        
        # Record trigger attempt
        event = RetrainingEvent(
            timestamp=datetime.now(),
            trigger_type=trigger_type,
            trigger_reason=reason,
            data_size=0,  # Will be updated
            previous_model_version="current",
            new_model_version=None,
            performance_before={},
            performance_after=None,
            success=False,
            error_message=None,
            audit_event_id=None
        )
        
        # Log audit event
        audit_event_id = self.audit_chain.log_event(
            event_type="retraining_triggered",
            user_id="system",
            details={
                'trigger_type': trigger_type,
                'reason': reason,
                'model_name': self.model_name
            }
        )
        event.audit_event_id = audit_event_id
        
        try:
            # Get current performance if monitor available
            if self.production_monitor:
                summary = self.production_monitor.get_monitoring_summary(hours=24)
                event.performance_before = {
                    'accuracy': summary['avg_accuracy'],
                    'error_rate': summary['avg_error_rate'],
                    'drift_score': summary['avg_drift_score']
                }
            
            # Trigger retraining
            success = self._trigger_retraining(event)
            event.success = success
            
            if success:
                self.last_retrain_time = datetime.now()
                self.retrains_today += 1
                self._update_trigger_state('last_retrain_time', self.last_retrain_time.isoformat())
                logger.info(f"Retraining triggered successfully: {trigger_type}")
            
        except Exception as e:
            error_msg = f"Error triggering retraining: {e}"
            logger.error(error_msg)
            event.error_message = error_msg
        
        # Store event
        self._store_retraining_event(event)
    
    def _can_retrain(self) -> bool:
        """Check if retraining is allowed based on constraints."""
        now = datetime.now()
        
        # Check minimum time between retrains
        if self.last_retrain_time:
            hours_since_retrain = (now - self.last_retrain_time).total_seconds() / 3600
            if hours_since_retrain < self.config.min_hours_between_retrains:
                logger.info(f"Too soon since last retrain: {hours_since_retrain:.1f} hours")
                return False
        
        # Check daily retrain limit
        if self.retrains_today >= self.config.max_retrains_per_day:
            logger.info(f"Daily retrain limit reached: {self.retrains_today}")
            return False
        
        # Check manual approval requirement
        if self.config.require_manual_approval:
            logger.info("Manual approval required for retraining")
            return False
        
        return True
    
    def _trigger_retraining(self, event: RetrainingEvent) -> bool:
        """Actually trigger the retraining process."""
        try:
            # In a real implementation, this would:
            # 1. Trigger the CI/CD pipeline
            # 2. Or call the training pipeline directly
            # 3. Or send a message to a queue
            
            # For this demo, we'll simulate by calling the training script
            training_script = "mlops/pipelines/train_model.py"
            
            if os.path.exists(training_script):
                # Run training pipeline
                result = subprocess.run([
                    "python", training_script
                ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                
                if result.returncode == 0:
                    logger.info("Training pipeline completed successfully")
                    
                    # Get new model version (simplified)
                    event.new_model_version = f"auto_{int(datetime.now().timestamp())}"
                    
                    # Update performance metrics if monitor available
                    if self.production_monitor:
                        time.sleep(5)  # Give time for metrics to update
                        summary = self.production_monitor.get_monitoring_summary(hours=1)
                        event.performance_after = {
                            'accuracy': summary['avg_accuracy'],
                            'error_rate': summary['avg_error_rate'],
                            'drift_score': summary['avg_drift_score']
                        }
                    
                    return True
                else:
                    event.error_message = result.stderr
                    return False
            else:
                # Alternative: trigger via GitHub Actions API or other CI/CD
                logger.info("Would trigger retraining via CI/CD pipeline")
                return True  # Simulate success
                
        except subprocess.TimeoutExpired:
            event.error_message = "Training pipeline timeout"
            return False
        except Exception as e:
            event.error_message = str(e)
            return False
    
    def _store_retraining_event(self, event: RetrainingEvent):
        """Store retraining event in database."""
        with sqlite3.connect(self.trigger_db) as conn:
            conn.execute("""
                INSERT INTO retraining_events (
                    timestamp, trigger_type, trigger_reason, data_size,
                    previous_model_version, new_model_version,
                    performance_before, performance_after,
                    success, error_message, audit_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp.isoformat(),
                event.trigger_type,
                event.trigger_reason,
                event.data_size,
                event.previous_model_version,
                event.new_model_version,
                json.dumps(event.performance_before),
                json.dumps(event.performance_after) if event.performance_after else None,
                event.success,
                event.error_message,
                event.audit_event_id
            ))
            conn.commit()
    
    def _reset_daily_counter(self):
        """Reset daily retrain counter if it's a new day."""
        now = datetime.now()
        if self.last_retrain_time and self.last_retrain_time.date() < now.date():
            self.retrains_today = 0
    
    def get_trigger_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get retraining trigger history."""
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.trigger_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM retraining_events 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def force_retrain(self, reason: str = "Manual trigger") -> bool:
        """Force immediate retraining regardless of constraints."""
        logger.info(f"Force retraining requested: {reason}")
        
        # Temporarily disable constraints
        original_min_hours = self.config.min_hours_between_retrains
        original_max_retrains = self.config.max_retrains_per_day
        original_approval = self.config.require_manual_approval
        
        self.config.min_hours_between_retrains = 0
        self.config.max_retrains_per_day = 100
        self.config.require_manual_approval = False
        
        try:
            self._handle_trigger('manual', reason)
            return True
        finally:
            # Restore original constraints
            self.config.min_hours_between_retrains = original_min_hours
            self.config.max_retrains_per_day = original_max_retrains
            self.config.require_manual_approval = original_approval


def create_data_trigger(model_name: str,
                       config: Optional[RetrainingTriggerConfig] = None,
                       production_monitor: Optional[ProductionMonitor] = None) -> DataDrivenRetrainingTrigger:
    """Factory function to create a data-driven retraining trigger."""
    return DataDrivenRetrainingTrigger(
        model_name=model_name,
        config=config,
        production_monitor=production_monitor
    )


if __name__ == "__main__":
    # Demo usage
    print("Data-Driven Retraining Trigger Demo")
    
    # Create configuration
    config = RetrainingTriggerConfig(
        min_new_samples=100,
        max_days_without_retrain=3,
        accuracy_degradation_threshold=0.03,
        data_directories=["data/raw", "data/processed"],
        watch_file_patterns=[".csv", ".parquet"]
    )
    
    # Create trigger
    trigger = create_data_trigger(
        model_name="alzheimer_classifier",
        config=config
    )
    
    # Start monitoring (in practice, this would run continuously)
    trigger.start_monitoring()
    
    print("Monitoring started. Waiting for triggers...")
    
    # Simulate for a short time
    time.sleep(5)
    
    # Force a retrain to demonstrate
    trigger.force_retrain("Demo trigger")
    
    # Get history
    history = trigger.get_trigger_history(days=1)
    print(f"Trigger history: {len(history)} events")
    
    # Stop monitoring
    trigger.stop_monitoring()
    print("Demo completed")