"""
Scalable Orchestration for DuetMind Adaptive

Implements workflow orchestration and resource management for distributed
training and inference using Ray and custom schedulers.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import uuid

logger = logging.getLogger(__name__)

# Check for Ray availability
try:
    import ray
    from ray import workflow
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Using local orchestration only.")

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceType(Enum):
    """Resource types for task execution."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"

@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0

@dataclass
class Task:
    """Orchestration task definition."""
    id: str
    name: str
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    dependencies: List[str] = None
    resources: ResourceRequirement = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    priority: int = 0  # Higher numbers = higher priority
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.resources is None:
            self.resources = ResourceRequirement()

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0

class WorkflowOrchestrator:
    """
    Main orchestrator for managing distributed workflows.
    
    Features:
    - Task dependency management
    - Resource allocation and scheduling
    - Fault tolerance with retries
    - Progress monitoring
    - Both Ray-based and local execution
    """
    
    def __init__(self,
                 use_ray: bool = None,
                 max_concurrent_tasks: int = 4,
                 resource_pool: Optional[Dict[ResourceType, float]] = None):
        """
        Initialize workflow orchestrator.
        
        Args:
            use_ray: Whether to use Ray (auto-detect if None)
            max_concurrent_tasks: Maximum concurrent tasks for local execution
            resource_pool: Available resources for scheduling
        """
        self.use_ray = use_ray if use_ray is not None else RAY_AVAILABLE
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize resource pool
        if resource_pool is None:
            import psutil
            self.resource_pool = {
                ResourceType.CPU: psutil.cpu_count(),
                ResourceType.MEMORY: psutil.virtual_memory().total / (1024**3),  # GB
                ResourceType.GPU: 0  # Will be detected if Ray is available
            }
        else:
            self.resource_pool = resource_pool
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.running_tasks: Dict[str, Any] = {}
        
        # Resource tracking
        self.allocated_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
            ResourceType.GPU: 0.0
        }
        
        # Execution state
        self.is_running = False
        self.executor = None
        
        # Initialize Ray if requested
        if self.use_ray and RAY_AVAILABLE:
            self._init_ray()
        elif self.use_ray and not RAY_AVAILABLE:
            logger.warning("Ray requested but not available. Falling back to local execution.")
            self.use_ray = False
    
    def _init_ray(self):
        """Initialize Ray cluster."""
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Update resource pool with Ray cluster info
            cluster_resources = ray.cluster_resources()
            self.resource_pool[ResourceType.CPU] = cluster_resources.get('CPU', 1)
            self.resource_pool[ResourceType.MEMORY] = cluster_resources.get('memory', 1e9) / 1e9
            self.resource_pool[ResourceType.GPU] = cluster_resources.get('GPU', 0)
            
            logger.info(f"Ray initialized. Resources: {self.resource_pool}")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            self.use_ray = False
    
    def add_task(self,
                 task_id: str,
                 function: Callable,
                 args: tuple = (),
                 kwargs: dict = None,
                 dependencies: List[str] = None,
                 resources: ResourceRequirement = None,
                 **task_options) -> str:
        """
        Add a task to the workflow.
        
        Args:
            task_id: Unique task identifier
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            dependencies: List of task IDs this task depends on
            resources: Resource requirements
            **task_options: Additional task options
            
        Returns:
            Task ID
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        task = Task(
            id=task_id,
            name=task_options.get('name', task_id),
            function=function,
            args=args,
            kwargs=kwargs or {},
            dependencies=dependencies or [],
            resources=resources or ResourceRequirement(),
            **{k: v for k, v in task_options.items() if k != 'name'}
        )
        
        self.tasks[task_id] = task
        self.task_dependencies[task_id] = task.dependencies.copy()
        
        logger.info(f"Added task: {task_id} with dependencies: {task.dependencies}")
        return task_id
    
    def add_training_task(self,
                         task_id: str,
                         model_config: Dict[str, Any],
                         data_path: str,
                         dependencies: List[str] = None,
                         resources: ResourceRequirement = None) -> str:
        """
        Add a specialized training task.
        
        Args:
            task_id: Task identifier
            model_config: Model configuration
            data_path: Path to training data
            dependencies: Task dependencies
            resources: Resource requirements
            
        Returns:
            Task ID
        """
        from .automl import run_automl_pipeline
        
        # Default resources for training
        if resources is None:
            resources = ResourceRequirement(
                cpu_cores=2,
                memory_gb=4.0,
                gpu_count=0
            )
        
        return self.add_task(
            task_id=task_id,
            function=self._training_wrapper,
            args=(model_config, data_path),
            dependencies=dependencies,
            resources=resources,
            name=f"Training: {task_id}"
        )
    
    def _training_wrapper(self, model_config: Dict[str, Any], data_path: str) -> Dict[str, Any]:
        """Wrapper function for training tasks."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from .automl import run_automl_pipeline
        
        # Load data
        data = pd.read_csv(data_path)
        target_col = model_config.get('target_column', 'target')
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Run AutoML
        automl_config = model_config.get('automl_config', {})
        results = run_automl_pipeline(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            automl_config
        )
        
        return results
    
    def can_allocate_resources(self, requirements: ResourceRequirement) -> bool:
        """Check if required resources can be allocated."""
        return (
            self.allocated_resources[ResourceType.CPU] + requirements.cpu_cores <= self.resource_pool[ResourceType.CPU] and
            self.allocated_resources[ResourceType.MEMORY] + requirements.memory_gb <= self.resource_pool[ResourceType.MEMORY] and
            self.allocated_resources[ResourceType.GPU] + requirements.gpu_count <= self.resource_pool[ResourceType.GPU]
        )
    
    def allocate_resources(self, requirements: ResourceRequirement):
        """Allocate resources for task execution."""
        self.allocated_resources[ResourceType.CPU] += requirements.cpu_cores
        self.allocated_resources[ResourceType.MEMORY] += requirements.memory_gb
        self.allocated_resources[ResourceType.GPU] += requirements.gpu_count
    
    def deallocate_resources(self, requirements: ResourceRequirement):
        """Deallocate resources after task completion."""
        self.allocated_resources[ResourceType.CPU] -= requirements.cpu_cores
        self.allocated_resources[ResourceType.MEMORY] -= requirements.memory_gb
        self.allocated_resources[ResourceType.GPU] -= requirements.gpu_count
    
    def get_ready_tasks(self) -> List[str]:
        """Get list of tasks ready for execution (dependencies satisfied)."""
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            # Skip if already processed
            if task_id in self.task_results or task_id in self.running_tasks:
                continue
            
            # Check if all dependencies are completed
            dependencies_satisfied = all(
                dep_id in self.task_results and 
                self.task_results[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if dependencies_satisfied and self.can_allocate_resources(task.resources):
                ready_tasks.append(task_id)
        
        # Sort by priority (higher first)
        ready_tasks.sort(key=lambda tid: -self.tasks[tid].priority)
        return ready_tasks
    
    def execute_task_ray(self, task: Task) -> Any:
        """Execute task using Ray."""
        @ray.remote(**asdict(task.resources))
        def ray_task_wrapper():
            return task.function(*task.args, **task.kwargs)
        
        return ray_task_wrapper.remote()
    
    def execute_task_local(self, task: Task) -> TaskResult:
        """Execute task locally."""
        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Execute function
            output = task.function(*task.args, **task.kwargs)
            
            result.status = TaskStatus.COMPLETED
            result.result = output
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()
            
            logger.error(f"Task {task.id} failed: {e}")
        
        return result
    
    def run_workflow(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """
        Execute the complete workflow.
        
        Args:
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary of task results
        """
        logger.info("Starting workflow execution...")
        self.is_running = True
        start_time = datetime.now()
        
        try:
            if self.use_ray:
                return self._run_workflow_ray(timeout)
            else:
                return self._run_workflow_local(timeout)
        finally:
            self.is_running = False
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Workflow completed in {execution_time:.2f}s")
    
    def _run_workflow_ray(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Run workflow using Ray."""
        ray_futures = {}
        
        while len(self.task_results) < len(self.tasks):
            # Get ready tasks
            ready_tasks = self.get_ready_tasks()
            
            # Submit ready tasks
            for task_id in ready_tasks:
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    task = self.tasks[task_id]
                    self.allocate_resources(task.resources)
                    
                    future = self.execute_task_ray(task)
                    ray_futures[task_id] = future
                    self.running_tasks[task_id] = future
                    
                    logger.info(f"Submitted task: {task_id}")
            
            # Check for completed tasks
            completed_tasks = []
            for task_id, future in ray_futures.items():
                if task_id not in self.running_tasks:
                    continue
                    
                if ray.get([future], timeout=0.1):  # Non-blocking check
                    try:
                        result = ray.get(future)
                        task_result = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.COMPLETED,
                            result=result
                        )
                    except Exception as e:
                        task_result = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.FAILED,
                            error=str(e)
                        )
                    
                    self.task_results[task_id] = task_result
                    self.deallocate_resources(self.tasks[task_id].resources)
                    completed_tasks.append(task_id)
                    
                    logger.info(f"Task {task_id} completed: {task_result.status}")
            
            # Remove completed tasks from running
            for task_id in completed_tasks:
                self.running_tasks.pop(task_id, None)
            
            # Brief pause to prevent busy waiting
            time.sleep(0.1)
            
            # Check timeout
            if timeout and (datetime.now() - start_time).total_seconds() > timeout:
                logger.warning("Workflow timeout reached")
                break
        
        return self.task_results
    
    def _run_workflow_local(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Run workflow using local ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            start_time = datetime.now()
            
            while len(self.task_results) < len(self.tasks):
                # Get ready tasks
                ready_tasks = self.get_ready_tasks()
                
                # Submit ready tasks
                for task_id in ready_tasks:
                    if len(self.running_tasks) < self.max_concurrent_tasks:
                        task = self.tasks[task_id]
                        self.allocate_resources(task.resources)
                        
                        future = executor.submit(self.execute_task_local, task)
                        self.running_tasks[task_id] = future
                        
                        logger.info(f"Submitted task: {task_id}")
                
                # Check for completed tasks
                completed_tasks = []
                for task_id, future in list(self.running_tasks.items()):
                    if future.done():
                        try:
                            task_result = future.result()
                            self.task_results[task_id] = task_result
                        except Exception as e:
                            task_result = TaskResult(
                                task_id=task_id,
                                status=TaskStatus.FAILED,
                                error=str(e)
                            )
                            self.task_results[task_id] = task_result
                        
                        self.deallocate_resources(self.tasks[task_id].resources)
                        completed_tasks.append(task_id)
                        
                        logger.info(f"Task {task_id} completed: {task_result.status}")
                
                # Remove completed tasks
                for task_id in completed_tasks:
                    self.running_tasks.pop(task_id)
                
                # Brief pause
                time.sleep(0.1)
                
                # Check timeout
                if timeout and (datetime.now() - start_time).total_seconds() > timeout:
                    logger.warning("Workflow timeout reached")
                    break
        
        return self.task_results
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        total_tasks = len(self.tasks)
        completed_tasks = len([r for r in self.task_results.values() if r.status == TaskStatus.COMPLETED])
        failed_tasks = len([r for r in self.task_results.values() if r.status == TaskStatus.FAILED])
        running_tasks = len(self.running_tasks)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'running_tasks': running_tasks,
            'pending_tasks': total_tasks - completed_tasks - failed_tasks - running_tasks,
            'progress_percent': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'resource_utilization': {
                resource_type.value: {
                    'allocated': allocated,
                    'total': self.resource_pool[resource_type],
                    'utilization_percent': (allocated / self.resource_pool[resource_type] * 100) 
                    if self.resource_pool[resource_type] > 0 else 0
                }
                for resource_type, allocated in self.allocated_resources.items()
            }
        }
    
    def save_workflow_state(self, filepath: str):
        """Save current workflow state to file."""
        state = {
            'tasks': {tid: asdict(task) for tid, task in self.tasks.items()},
            'task_results': {tid: asdict(result) for tid, result in self.task_results.items()},
            'workflow_status': self.get_workflow_status(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Workflow state saved to {filepath}")
    
    def shutdown(self):
        """Shutdown orchestrator and clean up resources."""
        self.is_running = False
        
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Orchestrator shutdown complete")

class WorkflowBuilder:
    """Builder pattern for creating complex workflows."""
    
    def __init__(self, orchestrator: WorkflowOrchestrator = None):
        self.orchestrator = orchestrator or WorkflowOrchestrator()
        self.task_counter = 0
    
    def add_data_preprocessing_stage(self,
                                   data_paths: List[str],
                                   preprocessing_config: Dict[str, Any]) -> List[str]:
        """Add data preprocessing stage."""
        task_ids = []
        
        for i, data_path in enumerate(data_paths):
            task_id = f"preprocess_data_{i}_{self.task_counter}"
            self.orchestrator.add_task(
                task_id=task_id,
                function=self._preprocess_data,
                args=(data_path, preprocessing_config),
                name=f"Preprocess {Path(data_path).name}"
            )
            task_ids.append(task_id)
            self.task_counter += 1
        
        return task_ids
    
    def add_model_training_stage(self,
                               model_configs: List[Dict[str, Any]],
                               data_task_ids: List[str]) -> List[str]:
        """Add model training stage."""
        task_ids = []
        
        for i, model_config in enumerate(model_configs):
            task_id = f"train_model_{i}_{self.task_counter}"
            
            resources = ResourceRequirement(
                cpu_cores=model_config.get('cpu_cores', 2),
                memory_gb=model_config.get('memory_gb', 4.0),
                gpu_count=model_config.get('gpu_count', 0)
            )
            
            self.orchestrator.add_training_task(
                task_id=task_id,
                model_config=model_config,
                data_path="",  # Will be provided by dependency
                dependencies=data_task_ids,
                resources=resources
            )
            
            task_ids.append(task_id)
            self.task_counter += 1
        
        return task_ids
    
    def add_model_evaluation_stage(self,
                                 evaluation_config: Dict[str, Any],
                                 training_task_ids: List[str]) -> str:
        """Add model evaluation stage."""
        task_id = f"evaluate_models_{self.task_counter}"
        
        self.orchestrator.add_task(
            task_id=task_id,
            function=self._evaluate_models,
            args=(evaluation_config,),
            dependencies=training_task_ids,
            name="Model Evaluation"
        )
        
        self.task_counter += 1
        return task_id
    
    def _preprocess_data(self, data_path: str, config: Dict[str, Any]) -> str:
        """Data preprocessing function."""
        # Placeholder implementation
        logger.info(f"Preprocessing data from {data_path}")
        return f"preprocessed_{Path(data_path).stem}.csv"
    
    def _evaluate_models(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Model evaluation function."""
        # Placeholder implementation
        logger.info("Evaluating trained models")
        return {"evaluation_complete": True}
    
    def build_complete_pipeline(self,
                              data_paths: List[str],
                              model_configs: List[Dict[str, Any]],
                              preprocessing_config: Dict[str, Any] = None,
                              evaluation_config: Dict[str, Any] = None) -> WorkflowOrchestrator:
        """Build a complete ML pipeline."""
        # Add preprocessing stage
        if preprocessing_config is None:
            preprocessing_config = {"scaling": "standard", "encoding": "onehot"}
        
        preprocess_tasks = self.add_data_preprocessing_stage(data_paths, preprocessing_config)
        
        # Add training stage
        training_tasks = self.add_model_training_stage(model_configs, preprocess_tasks)
        
        # Add evaluation stage
        if evaluation_config is None:
            evaluation_config = {"metrics": ["accuracy", "roc_auc", "f1"]}
        
        evaluation_task = self.add_model_evaluation_stage(evaluation_config, training_tasks)
        
        logger.info(f"Built complete pipeline with {len(self.orchestrator.tasks)} tasks")
        return self.orchestrator

# Factory functions
def create_orchestrator(**kwargs) -> WorkflowOrchestrator:
    """Factory function to create workflow orchestrator."""
    return WorkflowOrchestrator(**kwargs)

def create_workflow_builder(orchestrator: WorkflowOrchestrator = None) -> WorkflowBuilder:
    """Factory function to create workflow builder."""
    return WorkflowBuilder(orchestrator)