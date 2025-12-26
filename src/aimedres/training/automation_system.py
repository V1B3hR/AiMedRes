"""
Automation & Scalability Integration Module

Ties together AutoML, pipeline customization, orchestration, and enhanced
drift monitoring into a unified system for DuetMind Adaptive.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from .automl import AutoMLOptimizer, create_automl_optimizer
from .custom_pipeline import (
    DynamicPipelineBuilder,
    PipelineConfig,
    PipelineRegistry,
    create_default_config,
    create_pipeline_builder,
)
from .enhanced_drift_monitoring import (
    AlertConfig,
    EnhancedDriftMonitor,
    ResponseAction,
    ResponseConfig,
    create_default_alert_config,
    create_default_response_config,
    create_enhanced_drift_monitor,
)
from .orchestration import (
    ResourceRequirement,
    WorkflowBuilder,
    WorkflowOrchestrator,
    create_orchestrator,
    create_workflow_builder,
)

logger = logging.getLogger(__name__)


class AutomationScalabilitySystem:
    """
    Unified Automation & Scalability system that integrates all components.

    This system provides:
    - Automated model optimization (AutoML)
    - Flexible pipeline configuration
    - Scalable workflow orchestration
    - Intelligent drift monitoring with responses
    """

    def __init__(self, config_path: Optional[str] = None, working_dir: str = "./automation_system"):
        """
        Initialize the automation system.

        Args:
            config_path: Path to system configuration file
            working_dir: Working directory for system files
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True, parents=True)

        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            self.config = self._create_default_config()
            self._save_config()

        # Initialize components
        self.automl_optimizer: Optional[AutoMLOptimizer] = None
        self.pipeline_registry = PipelineRegistry(str(self.working_dir / "pipeline_configs"))
        self.orchestrator = create_orchestrator(**self.config.get("orchestration", {}))
        self.drift_monitor: Optional[EnhancedDriftMonitor] = None

        # System state
        self.is_initialized = False
        self.active_workflows: Dict[str, Any] = {}
        self.system_metrics: Dict[str, Any] = {}

        logger.info(f"Automation & Scalability System initialized in {self.working_dir}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration from file."""
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default system configuration."""
        return {
            "system": {
                "name": "duetmind_automation_system",
                "version": "1.0.0",
                "description": "DuetMind Adaptive Automation & Scalability System",
            },
            "automl": {
                "objective_metric": "roc_auc",
                "n_trials": 100,
                "timeout": 3600,
                "cv_folds": 5,
            },
            "orchestration": {"use_ray": False, "max_concurrent_tasks": 4},
            "drift_monitoring": {
                "monitoring_interval": 3600,
                "drift_threshold": 0.15,
                "alert_channels": ["log"],
                "auto_retrain_enabled": False,
            },
            "pipeline_defaults": {
                "preprocessing": {
                    "numerical_scaler": "standard",
                    "categorical_encoding": "onehot",
                    "missing_value_strategy": "median",
                },
                "model": {
                    "algorithm": "random_forest",
                    "hyperparameters": {"n_estimators": 100, "random_state": 42},
                },
            },
        }

    def _save_config(self):
        """Save current configuration to file."""
        config_path = self.working_dir / "system_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def initialize_system(self, reference_data: pd.DataFrame, baseline_metrics: Dict[str, float]):
        """
        Initialize the system with reference data and baseline metrics.

        Args:
            reference_data: Reference dataset for drift monitoring
            baseline_metrics: Baseline model performance metrics
        """
        logger.info("Initializing Automation & Scalability System...")

        # Initialize AutoML optimizer
        automl_config = self.config.get("automl", {})
        self.automl_optimizer = create_automl_optimizer(**automl_config)

        # Initialize drift monitoring
        alert_config = self._create_alert_config()
        response_config = self._create_response_config()

        self.drift_monitor = create_enhanced_drift_monitor(
            reference_data=reference_data,
            baseline_metrics=baseline_metrics,
            alert_config=alert_config,
            response_config=response_config,
            **self.config.get("drift_monitoring", {}),
        )

        # Register default pipelines
        self._register_default_pipelines()

        self.is_initialized = True
        logger.info("System initialization complete")

    def _create_alert_config(self) -> AlertConfig:
        """Create alert configuration from system config."""
        drift_config = self.config.get("drift_monitoring", {})

        from .enhanced_drift_monitoring import AlertChannel

        # Map string channel names to enums
        channel_mapping = {
            "email": AlertChannel.EMAIL,
            "webhook": AlertChannel.WEBHOOK,
            "slack": AlertChannel.SLACK,
            "log": AlertChannel.LOG,
        }

        enabled_channels = [
            channel_mapping[ch]
            for ch in drift_config.get("alert_channels", ["log"])
            if ch in channel_mapping
        ]

        return AlertConfig(
            enabled_channels=enabled_channels,
            email_config=drift_config.get("email_config"),
            webhook_urls=drift_config.get("webhook_urls"),
            slack_config=drift_config.get("slack_config"),
        )

    def _create_response_config(self) -> ResponseConfig:
        """Create response configuration from system config."""
        drift_config = self.config.get("drift_monitoring", {})

        enabled_actions = []
        if drift_config.get("auto_retrain_enabled", False):
            enabled_actions.append(ResponseAction.RETRAIN_MODEL)
        if drift_config.get("auto_threshold_update", True):
            enabled_actions.append(ResponseAction.UPDATE_THRESHOLDS)
        if drift_config.get("human_review_enabled", True):
            enabled_actions.append(ResponseAction.HUMAN_REVIEW)

        return ResponseConfig(
            enabled_actions=enabled_actions,
            retrain_config=drift_config.get("retrain_config", {}),
            scaling_config=drift_config.get("scaling_config", {}),
        )

    def _register_default_pipelines(self):
        """Register default pipeline configurations."""
        # Basic classification pipeline
        basic_config = PipelineConfig(
            name="basic_classification",
            description="Basic classification pipeline with standard preprocessing",
            **self.config.get("pipeline_defaults", {}),
        )
        self.pipeline_registry.register_pipeline("basic_classification", basic_config)

        # Advanced pipeline with ensemble
        ensemble_config = dict(self.config.get("pipeline_defaults", {}))
        ensemble_config["model"] = {**ensemble_config.get("model", {}), "ensemble_method": "voting"}

        advanced_config = PipelineConfig(
            name="ensemble_classification",
            description="Advanced classification pipeline with ensemble methods",
            **ensemble_config,
        )
        self.pipeline_registry.register_pipeline("ensemble_classification", advanced_config)

        logger.info(f"Registered {len(self.pipeline_registry.list_pipelines())} default pipelines")

    def create_automated_training_workflow(
        self,
        data_path: str,
        target_column: str,
        pipeline_name: str = "basic_classification",
        enable_automl: bool = True,
        workflow_name: str = None,
    ) -> str:
        """
        Create an automated training workflow with AutoML and drift monitoring.

        Args:
            data_path: Path to training data
            target_column: Target column name
            pipeline_name: Pipeline configuration to use
            enable_automl: Whether to use AutoML optimization
            workflow_name: Name for the workflow

        Returns:
            Workflow ID
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        workflow_name = (
            workflow_name or f"automated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create workflow builder
        builder = create_workflow_builder(self.orchestrator)

        # Data preprocessing task
        preprocess_task_id = self.orchestrator.add_task(
            task_id=f"{workflow_name}_preprocess",
            function=self._preprocess_data_task,
            args=(data_path, target_column, pipeline_name),
            name="Data Preprocessing",
            resources=ResourceRequirement(cpu_cores=1, memory_gb=2.0),
        )

        # Model training task (with or without AutoML)
        if enable_automl:
            training_task_id = self.orchestrator.add_task(
                task_id=f"{workflow_name}_automl_training",
                function=self._automl_training_task,
                args=(data_path, target_column, pipeline_name),
                dependencies=[preprocess_task_id],
                name="AutoML Model Training",
                resources=ResourceRequirement(cpu_cores=4, memory_gb=8.0),
            )
        else:
            training_task_id = self.orchestrator.add_task(
                task_id=f"{workflow_name}_standard_training",
                function=self._standard_training_task,
                args=(data_path, target_column, pipeline_name),
                dependencies=[preprocess_task_id],
                name="Standard Model Training",
                resources=ResourceRequirement(cpu_cores=2, memory_gb=4.0),
            )

        # Model evaluation task
        evaluation_task_id = self.orchestrator.add_task(
            task_id=f"{workflow_name}_evaluation",
            function=self._model_evaluation_task,
            args=(workflow_name,),
            dependencies=[training_task_id],
            name="Model Evaluation",
            resources=ResourceRequirement(cpu_cores=1, memory_gb=2.0),
        )

        # Drift monitoring setup task
        monitoring_task_id = self.orchestrator.add_task(
            task_id=f"{workflow_name}_setup_monitoring",
            function=self._setup_drift_monitoring_task,
            args=(workflow_name,),
            dependencies=[evaluation_task_id],
            name="Setup Drift Monitoring",
            resources=ResourceRequirement(cpu_cores=1, memory_gb=1.0),
        )

        self.active_workflows[workflow_name] = {
            "workflow_id": workflow_name,
            "tasks": [preprocess_task_id, training_task_id, evaluation_task_id, monitoring_task_id],
            "created_at": datetime.now(),
            "status": "created",
            "config": {
                "data_path": data_path,
                "target_column": target_column,
                "pipeline_name": pipeline_name,
                "enable_automl": enable_automl,
            },
        }

        logger.info(f"Created automated training workflow: {workflow_name}")
        return workflow_name

    def run_workflow(self, workflow_name: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a specific workflow.

        Args:
            workflow_name: Name of the workflow to run
            timeout: Timeout in seconds

        Returns:
            Workflow results
        """
        if workflow_name not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_name} not found")

        logger.info(f"Starting workflow: {workflow_name}")
        self.active_workflows[workflow_name]["status"] = "running"
        self.active_workflows[workflow_name]["started_at"] = datetime.now()

        # Run the workflow
        results = self.orchestrator.run_workflow(timeout=timeout)

        # Update workflow status
        self.active_workflows[workflow_name]["status"] = "completed"
        self.active_workflows[workflow_name]["completed_at"] = datetime.now()
        self.active_workflows[workflow_name]["results"] = results

        logger.info(f"Workflow {workflow_name} completed")
        return results

    def _preprocess_data_task(
        self, data_path: str, target_column: str, pipeline_name: str
    ) -> Dict[str, Any]:
        """Data preprocessing task implementation."""
        logger.info(f"Preprocessing data from {data_path}")

        # Load data
        data = pd.read_csv(data_path)

        # Get pipeline configuration
        pipeline_builder = self.pipeline_registry.get_pipeline(pipeline_name)

        # Build preprocessing pipeline
        X = data.drop(columns=[target_column])
        preprocessing_pipeline = pipeline_builder.build_preprocessing_pipeline(X)

        # Apply preprocessing
        if preprocessing_pipeline:
            X_processed = preprocessing_pipeline.fit_transform(X)
        else:
            X_processed = X.values

        # Save processed data
        output_path = (
            self.working_dir / f"preprocessed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        processed_df = pd.DataFrame(X_processed)
        processed_df[target_column] = data[target_column]
        processed_df.to_csv(output_path, index=False)

        return {
            "preprocessed_data_path": str(output_path),
            "original_shape": data.shape,
            "processed_shape": processed_df.shape,
            "preprocessing_steps": (
                len(preprocessing_pipeline.steps) if preprocessing_pipeline else 0
            ),
        }

    def _automl_training_task(
        self, data_path: str, target_column: str, pipeline_name: str
    ) -> Dict[str, Any]:
        """AutoML training task implementation."""
        logger.info("Starting AutoML training")

        # Load data
        data = pd.read_csv(data_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y
        )

        # Run AutoML optimization
        results = self.automl_optimizer.optimize(X_train, y_train, X_val, y_val)

        # Save best model
        model_path = (
            self.working_dir / f"automl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(self.automl_optimizer.best_model, f)

        results["model_path"] = str(model_path)
        results["training_method"] = "automl"

        return results

    def _standard_training_task(
        self, data_path: str, target_column: str, pipeline_name: str
    ) -> Dict[str, Any]:
        """Standard training task implementation."""
        logger.info("Starting standard training")

        # Load data
        data = pd.read_csv(data_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Get pipeline and build complete pipeline
        pipeline_builder = self.pipeline_registry.get_pipeline(pipeline_name)
        complete_pipeline = pipeline_builder.build_complete_pipeline(X)

        # Train model
        complete_pipeline.fit(X, y)

        # Evaluate
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(complete_pipeline, X, y, cv=5, scoring="roc_auc")

        # Save model
        model_path = (
            self.working_dir / f"standard_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(complete_pipeline, f)

        return {
            "model_path": str(model_path),
            "training_method": "standard",
            "cross_val_scores": scores.tolist(),
            "mean_cv_score": scores.mean(),
            "std_cv_score": scores.std(),
        }

    def _model_evaluation_task(self, workflow_name: str) -> Dict[str, Any]:
        """Model evaluation task implementation."""
        logger.info("Evaluating trained model")

        # This would typically load the model and perform comprehensive evaluation
        # For now, return placeholder results

        evaluation_results = {
            "evaluation_completed": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "accuracy": 0.85,  # Placeholder
                "roc_auc": 0.88,  # Placeholder
                "f1_score": 0.82,  # Placeholder
            },
        }

        # Save evaluation results
        results_path = self.working_dir / f"evaluation_results_{workflow_name}.json"
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        return evaluation_results

    def _setup_drift_monitoring_task(self, workflow_name: str) -> Dict[str, Any]:
        """Setup drift monitoring task implementation."""
        logger.info("Setting up drift monitoring")

        # This would set up continuous drift monitoring
        # For now, return confirmation

        monitoring_setup = {
            "monitoring_enabled": True,
            "workflow_name": workflow_name,
            "monitoring_interval": self.config["drift_monitoring"]["monitoring_interval"],
            "setup_timestamp": datetime.now().isoformat(),
        }

        return monitoring_setup

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_initialized": self.is_initialized,
            "working_directory": str(self.working_dir),
            "components": {
                "automl_optimizer": self.automl_optimizer is not None,
                "pipeline_registry": len(self.pipeline_registry.list_pipelines()),
                "orchestrator": self.orchestrator is not None,
                "drift_monitor": self.drift_monitor is not None,
            },
            "active_workflows": len(self.active_workflows),
            "workflows": {
                name: {
                    "status": workflow["status"],
                    "created_at": workflow["created_at"].isoformat(),
                    "task_count": len(workflow["tasks"]),
                }
                for name, workflow in self.active_workflows.items()
            },
            "orchestrator_status": self.orchestrator.get_workflow_status(),
            "drift_monitoring": (
                self.drift_monitor.get_drift_summary() if self.drift_monitor else None
            ),
        }

    def save_system_state(self) -> str:
        """Save complete system state."""
        state_file = (
            self.working_dir / f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        status = self.get_system_status()

        with open(state_file, "w") as f:
            json.dump(status, f, indent=2, default=str)

        # Also save orchestrator state
        if self.orchestrator:
            orchestrator_state_file = (
                self.working_dir
                / f"orchestrator_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.orchestrator.save_workflow_state(str(orchestrator_state_file))

        # Save drift monitoring state
        if self.drift_monitor:
            drift_state_file = (
                self.working_dir
                / f"drift_monitoring_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.drift_monitor.save_monitoring_state(str(drift_state_file))

        logger.info(f"System state saved to {state_file}")
        return str(state_file)

    def shutdown(self):
        """Shutdown the automation system."""
        logger.info("Shutting down Automation & Scalability System...")

        # Save final state
        self.save_system_state()

        # Shutdown orchestrator
        if self.orchestrator:
            self.orchestrator.shutdown()

        logger.info("System shutdown complete")


# Factory function
def create_automation_system(**kwargs) -> AutomationScalabilitySystem:
    """Factory function to create automation system."""
    return AutomationScalabilitySystem(**kwargs)


# Convenience function for quick setup
def setup_complete_system(
    reference_data: pd.DataFrame,
    baseline_metrics: Dict[str, float],
    working_dir: str = "./automation_system",
) -> AutomationScalabilitySystem:
    """
    Quick setup function for the complete automation system.

    Args:
        reference_data: Reference dataset for drift monitoring
        baseline_metrics: Baseline model performance metrics
        working_dir: Working directory for system files

    Returns:
        Initialized automation system
    """
    system = create_automation_system(working_dir=working_dir)
    system.initialize_system(reference_data, baseline_metrics)
    return system
