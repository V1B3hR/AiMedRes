#!/usr/bin/env python3
"""
Remote Training Manager for DuetMind Adaptive System

Provides secure, remote model training capabilities with:
- Authenticated API endpoints for training management
- Encrypted model parameter exchange
- Real-time training progress monitoring
- Secure distributed training coordination
- Medical data privacy protection
"""

import os
import json
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Core training and security modules
from training import AlzheimerTrainer
from security.auth import SecureAuthManager
from security.encryption import DataEncryption
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger('duetmind.remote_training')
logger.setLevel(logging.INFO)

class TrainingStatus(Enum):
    """Training job status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingJob:
    """Training job data structure."""
    job_id: str
    user_id: str
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    config: Dict[str, Any] = None
    results: Dict[str, Any] = None
    error_message: Optional[str] = None
    encrypted_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        data['status'] = data['status'].value
        return data

class RemoteTrainingManager:
    """
    Secure remote training manager for distributed model training.
    
    Features:
    - Secure API-based training job management
    - Real-time progress monitoring
    - Encrypted model parameter exchange  
    - Multi-user training isolation
    - Medical data privacy protection
    - Training resource management
    """
    
    def __init__(self, auth_manager: SecureAuthManager, encryption: DataEncryption):
        self.auth_manager = auth_manager
        self.encryption = encryption
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.active_trainings: Dict[str, threading.Thread] = {}
        
        # Training configuration
        self.max_concurrent_jobs = 3
        self.job_timeout_hours = 24
        self.cleanup_interval_hours = 1
        
        # Start background cleanup task
        self._start_cleanup_task()
        
        logger.info("Remote Training Manager initialized")
    
    def submit_training_job(self, user_id: str, training_config: Dict[str, Any]) -> str:
        """
        Submit a new training job.
        
        Args:
            user_id: User submitting the job
            training_config: Training configuration parameters
            
        Returns:
            Job ID for tracking
        """
        # Validate configuration
        self._validate_training_config(training_config)
        
        # Check resource limits
        user_jobs = self._get_user_active_jobs(user_id)
        if len(user_jobs) >= 2:  # Max 2 jobs per user
            raise ValueError("Maximum concurrent jobs per user exceeded")
        
        if len(self.active_trainings) >= self.max_concurrent_jobs:
            raise ValueError("System at maximum capacity. Please try again later.")
        
        # Create job
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            user_id=user_id,
            status=TrainingStatus.PENDING,
            created_at=datetime.now(),
            config=training_config
        )
        
        self.training_jobs[job_id] = job
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._run_training_job,
            args=(job_id,),
            daemon=True
        )
        training_thread.start()
        self.active_trainings[job_id] = training_thread
        
        logger.info(f"Training job {job_id} submitted for user {user_id}")
        return job_id
    
    def get_job_status(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get training job status.
        
        Args:
            job_id: Job identifier
            user_id: User requesting status
            
        Returns:
            Job status information
        """
        if job_id not in self.training_jobs:
            raise ValueError("Job not found")
        
        job = self.training_jobs[job_id]
        
        # Check user authorization
        if job.user_id != user_id:
            raise PermissionError("Unauthorized access to job")
        
        return job.to_dict()
    
    def list_user_jobs(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List training jobs for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        user_jobs = [
            job for job in self.training_jobs.values() 
            if job.user_id == user_id
        ]
        
        # Sort by creation time (newest first)
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return [job.to_dict() for job in user_jobs[:limit]]
    
    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """
        Cancel a training job.
        
        Args:
            job_id: Job identifier
            user_id: User requesting cancellation
            
        Returns:
            True if cancelled successfully
        """
        if job_id not in self.training_jobs:
            raise ValueError("Job not found")
        
        job = self.training_jobs[job_id]
        
        # Check user authorization
        if job.user_id != user_id:
            raise PermissionError("Unauthorized access to job")
        
        if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
            raise ValueError("Cannot cancel completed job")
        
        # Mark as cancelled
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now()
        
        # Clean up active training
        if job_id in self.active_trainings:
            # Note: Python threads cannot be forcefully terminated
            # The training loop should check for cancellation
            logger.info(f"Job {job_id} marked for cancellation")
        
        return True
    
    def get_encrypted_model(self, job_id: str, user_id: str) -> str:
        """
        Get encrypted trained model.
        
        Args:
            job_id: Job identifier
            user_id: User requesting model
            
        Returns:
            Encrypted model data
        """
        if job_id not in self.training_jobs:
            raise ValueError("Job not found")
        
        job = self.training_jobs[job_id]
        
        # Check user authorization
        if job.user_id != user_id:
            raise PermissionError("Unauthorized access to job")
        
        if job.status != TrainingStatus.COMPLETED:
            raise ValueError("Job not completed successfully")
        
        if not job.encrypted_model:
            raise ValueError("No model available for this job")
        
        return job.encrypted_model
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system-wide training status.
        
        Returns:
            System status information
        """
        active_jobs = len(self.active_trainings)
        pending_jobs = len([j for j in self.training_jobs.values() if j.status == TrainingStatus.PENDING])
        completed_jobs = len([j for j in self.training_jobs.values() if j.status == TrainingStatus.COMPLETED])
        failed_jobs = len([j for j in self.training_jobs.values() if j.status == TrainingStatus.FAILED])
        
        return {
            "active_jobs": active_jobs,
            "pending_jobs": pending_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "capacity_available": self.max_concurrent_jobs - active_jobs,
            "system_healthy": active_jobs <= self.max_concurrent_jobs
        }
    
    def _validate_training_config(self, config: Dict[str, Any]):
        """Validate training configuration parameters."""
        required_fields = ['model_type', 'dataset_source']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate model type
        valid_models = ['alzheimer_classifier', 'adaptive_neural_net']
        if config['model_type'] not in valid_models:
            raise ValueError(f"Invalid model type. Must be one of: {valid_models}")
        
        # Validate dataset source
        valid_sources = ['kaggle_alzheimer', 'synthetic_test_data', 'user_provided']
        if config['dataset_source'] not in valid_sources:
            raise ValueError(f"Invalid dataset source. Must be one of: {valid_sources}")
    
    def _get_user_active_jobs(self, user_id: str) -> List[TrainingJob]:
        """Get active jobs for a user."""
        return [
            job for job in self.training_jobs.values() 
            if job.user_id == user_id and job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]
        ]
    
    def _run_training_job(self, job_id: str):
        """
        Execute training job in background thread.
        
        Args:
            job_id: Job identifier
        """
        job = self.training_jobs[job_id]
        
        try:
            # Update status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            logger.info(f"Starting training job {job_id}")
            
            # Initialize trainer
            trainer = AlzheimerTrainer()
            
            # Load and preprocess data based on config
            job.progress = 0.1
            data = self._load_training_data(job.config)
            
            job.progress = 0.3
            X, y = trainer.preprocess_data(data)
            
            # Train model
            job.progress = 0.5
            model, metrics = trainer.train_model(X, y)
            
            job.progress = 0.8
            
            # Encrypt and store model
            model_data = {
                'model_params': trainer.get_model_parameters(),
                'feature_columns': trainer.feature_columns,
                'metrics': metrics,
                'config': job.config
            }
            
            encrypted_model = self.encryption.encrypt_model_parameters(model_data)
            job.encrypted_model = encrypted_model
            
            # Store results
            job.results = {
                'training_accuracy': metrics.get('accuracy', 0.0),
                'model_type': job.config['model_type'],
                'training_samples': len(data),
                'feature_count': len(trainer.feature_columns),
                'training_duration_seconds': (datetime.now() - job.started_at).total_seconds()
            }
            
            job.progress = 1.0
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Training job {job_id} failed: {e}")
        
        finally:
            # Clean up active training tracking
            if job_id in self.active_trainings:
                del self.active_trainings[job_id]
    
    def _load_training_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load training data based on configuration."""
        source = config['dataset_source']
        
        if source == 'synthetic_test_data':
            from files.dataset.create_test_data import create_test_alzheimer_data
            return create_test_alzheimer_data()
        
        elif source == 'kaggle_alzheimer':
            # Use existing Kaggle loader functionality
            try:
                import kagglehub
                # This is a simplified example - in production, you'd handle the full kagglehub integration
                from files.dataset.create_test_data import create_test_alzheimer_data
                logger.warning("Using synthetic data as Kaggle integration requires API setup")
                return create_test_alzheimer_data()
            except ImportError:
                logger.warning("Kagglehub not available, using synthetic data")
                from files.dataset.create_test_data import create_test_alzheimer_data
                return create_test_alzheimer_data()
        
        elif source == 'user_provided':
            # Handle user-provided data (would require additional security validation)
            raise NotImplementedError("User-provided data source not yet implemented")
        
        else:
            raise ValueError(f"Unknown dataset source: {source}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task for old jobs."""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_jobs()
                    time.sleep(self.cleanup_interval_hours * 3600)
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_jobs(self):
        """Remove old completed jobs to manage memory."""
        cutoff_time = datetime.now() - timedelta(hours=self.job_timeout_hours)
        
        jobs_to_remove = [
            job_id for job_id, job in self.training_jobs.items()
            if job.completed_at and job.completed_at < cutoff_time
        ]
        
        for job_id in jobs_to_remove:
            del self.training_jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")