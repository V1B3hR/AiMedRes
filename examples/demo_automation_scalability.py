#!/usr/bin/env python3
"""
Automation & Scalability Demo for DuetMind Adaptive

Demonstrates the complete automation and scalability features including:
- AutoML Integration with hyperparameter optimization
- Pipeline Customization with dynamic configurations
- Scalable Orchestration with workflow management
- Enhanced Drift Monitoring with automated responses

This script shows all four components working together.
"""

import sys
import os
import tempfile
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a sample medical dataset for demonstration."""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(65, 15, n_samples)
    age = np.clip(age, 18, 95)
    
    mmse = np.random.normal(24, 6, n_samples)
    mmse = np.clip(mmse, 0, 30)
    
    education = np.random.normal(14, 4, n_samples)
    education = np.clip(education, 6, 20)
    
    ses = np.random.uniform(1, 5, n_samples)
    
    # Create some categorical features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    smoking = np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Generate target with some logic
    # Higher age, lower MMSE, certain smoking patterns increase risk
    risk_score = (
        (age - 65) * 0.02 +
        (24 - mmse) * 0.05 +
        (smoking == 'Current').astype(int) * 0.3 +
        (gender == 'Male').astype(int) * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    diagnosis = (risk_score > 0.5).astype(int)
    
    return pd.DataFrame({
        'age': age,
        'mmse': mmse,
        'education': education,
        'ses': ses,
        'gender': gender,
        'smoking': smoking,
        'diagnosis': diagnosis
    })

def demonstrate_automl_integration():
    """Demonstrate AutoML integration capabilities."""
    print("\n" + "="*80)
    print("🤖 AUTOML INTEGRATION DEMONSTRATION")
    print("="*80)
    
    try:
        from src.duetmind_adaptive.training.automl import create_automl_optimizer
        
        print("Creating sample dataset...")
        data = create_sample_dataset(500)  # Smaller for demo
        
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']
        
        # Prepare data (simple preprocessing for demo)
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded.values, y.values, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Initializing AutoML optimizer...")
        automl_config = {
            'objective_metric': 'roc_auc',
            'n_trials': 20,  # Reduced for demo
            'timeout': 300,  # 5 minutes
            'cv_folds': 3
        }
        
        optimizer = create_automl_optimizer(**automl_config)
        
        print("Running AutoML optimization...")
        print("This may take a few minutes...")
        
        results = optimizer.optimize(
            X_train, y_train, X_val, y_val,
            algorithms=['random_forest', 'logistic_regression']
        )
        
        print("\n🎯 AutoML Results:")
        print(f"Best Score: {results['best_score']:.4f}")
        print(f"Best Algorithm: {results['best_algorithm']}")
        print(f"Number of Trials: {results['n_trials']}")
        print(f"Optimization Time: {results['optimization_time']:.2f} seconds")
        
        if 'param_importance' in results and results['param_importance']:
            print("\nTop 3 Most Important Parameters:")
            sorted_params = sorted(results['param_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for param, importance in sorted_params[:3]:
                print(f"  {param}: {importance:.4f}")
        
        print("✅ AutoML Integration demonstration completed!")
        
    except ImportError as e:
        print(f"❌ AutoML demonstration failed - missing dependencies: {e}")
        print("Install optuna: pip install optuna")
    except Exception as e:
        print(f"❌ AutoML demonstration failed: {e}")
        logger.exception("AutoML demo failed")

def demonstrate_pipeline_customization():
    """Demonstrate pipeline customization capabilities."""
    print("\n" + "="*80)
    print("🔧 PIPELINE CUSTOMIZATION DEMONSTRATION")
    print("="*80)
    
    try:
        from src.duetmind_adaptive.training.custom_pipeline import (
            create_pipeline_builder, PipelineRegistry, PipelineConfig,
            PreprocessingConfig, ModelConfig
        )
        
        print("Creating sample dataset...")
        data = create_sample_dataset(300)
        
        print("Creating custom pipeline configurations...")
        
        # Create different pipeline configurations
        configs = {
            "basic_rf": PipelineConfig(
                name="basic_random_forest",
                description="Basic Random Forest pipeline",
                preprocessing=PreprocessingConfig(
                    numerical_scaler='standard',
                    categorical_encoding='onehot',
                    missing_value_strategy='median'
                ),
                model=ModelConfig(
                    algorithm='random_forest',
                    hyperparameters={'n_estimators': 50, 'random_state': 42}
                )
            ),
            "advanced_lr": PipelineConfig(
                name="advanced_logistic_regression",
                description="Advanced Logistic Regression with feature selection",
                preprocessing=PreprocessingConfig(
                    numerical_scaler='robust',
                    categorical_encoding='onehot',
                    missing_value_strategy='mean',
                    feature_selection='selectkbest'
                ),
                model=ModelConfig(
                    algorithm='logistic_regression',
                    hyperparameters={'C': 1.0, 'max_iter': 1000, 'random_state': 42}
                )
            )
        }
        
        # Create temporary directory for pipeline registry
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = PipelineRegistry(temp_dir)
            
            # Register pipelines
            for name, config in configs.items():
                registry.register_pipeline(name, config)
                print(f"✓ Registered pipeline: {name}")
            
            print(f"\nRegistered pipelines: {registry.list_pipelines()}")
            
            # Demonstrate pipeline building and execution
            for pipeline_name in registry.list_pipelines():
                print(f"\n--- Testing pipeline: {pipeline_name} ---")
                
                try:
                    builder = registry.get_pipeline(pipeline_name)
                    
                    X = data.drop(columns=['diagnosis'])
                    y = data['diagnosis']
                    
                    # Build and test pipeline
                    complete_pipeline = builder.build_complete_pipeline(X)
                    
                    print(f"Pipeline steps: {len(complete_pipeline.steps)}")
                    for i, (step_name, step) in enumerate(complete_pipeline.steps):
                        print(f"  {i+1}. {step_name}: {type(step).__name__}")
                    
                    # Quick evaluation
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(complete_pipeline, X, y, cv=3, scoring='accuracy')
                    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                    
                except Exception as e:
                    print(f"❌ Pipeline {pipeline_name} failed: {e}")
        
        print("\n✅ Pipeline Customization demonstration completed!")
        
    except Exception as e:
        print(f"❌ Pipeline customization demonstration failed: {e}")
        logger.exception("Pipeline customization demo failed")

def demonstrate_scalable_orchestration():
    """Demonstrate scalable orchestration capabilities."""
    print("\n" + "="*80)
    print("🚀 SCALABLE ORCHESTRATION DEMONSTRATION")
    print("="*80)
    
    try:
        from src.duetmind_adaptive.training.orchestration import (
            create_orchestrator, create_workflow_builder, ResourceRequirement
        )
        
        print("Creating workflow orchestrator...")
        orchestrator = create_orchestrator(
            use_ray=False,  # Use local execution for demo
            max_concurrent_tasks=2
        )
        
        print("Building demo workflow...")
        
        def demo_data_prep_task(dataset_name: str):
            """Demo data preparation task."""
            time.sleep(2)  # Simulate processing time
            return f"Preprocessed data for {dataset_name}"
        
        def demo_training_task(preprocessed_data: str, model_type: str):
            """Demo training task."""
            time.sleep(3)  # Simulate training time
            accuracy = np.random.uniform(0.75, 0.95)
            return {
                'model_type': model_type,
                'accuracy': accuracy,
                'data': preprocessed_data
            }
        
        def demo_evaluation_task(model_results: dict):
            """Demo evaluation task."""
            time.sleep(1)  # Simulate evaluation time
            return f"Evaluated {model_results['model_type']} with accuracy {model_results['accuracy']:.3f}"
        
        # Add tasks to orchestrator
        datasets = ['medical_data', 'imaging_data']
        models = ['random_forest', 'neural_network']
        
        # Data preprocessing tasks
        prep_task_ids = []
        for i, dataset in enumerate(datasets):
            task_id = f"prep_{dataset}"
            orchestrator.add_task(
                task_id=task_id,
                function=demo_data_prep_task,
                args=(dataset,),
                resources=ResourceRequirement(cpu_cores=1, memory_gb=2.0),
                name=f"Preprocess {dataset}"
            )
            prep_task_ids.append(task_id)
        
        # Training tasks (depend on preprocessing)
        training_task_ids = []
        for i, model in enumerate(models):
            task_id = f"train_{model}"
            orchestrator.add_task(
                task_id=task_id,
                function=demo_training_task,
                args=("preprocessed_data", model),
                dependencies=[prep_task_ids[i % len(prep_task_ids)]],
                resources=ResourceRequirement(cpu_cores=2, memory_gb=4.0),
                name=f"Train {model}"
            )
            training_task_ids.append(task_id)
        
        # Evaluation task (depends on all training)
        evaluation_task_id = "evaluate_all"
        orchestrator.add_task(
            task_id=evaluation_task_id,
            function=demo_evaluation_task,
            args=({"model_type": "ensemble", "accuracy": 0.9},),
            dependencies=training_task_ids,
            resources=ResourceRequirement(cpu_cores=1, memory_gb=1.0),
            name="Evaluate All Models"
        )
        
        print(f"Created workflow with {len(orchestrator.tasks)} tasks")
        
        # Show workflow status before execution
        status = orchestrator.get_workflow_status()
        print(f"Workflow status: {status['pending_tasks']} pending, {status['total_tasks']} total")
        
        print("\nExecuting workflow...")
        start_time = time.time()
        
        # Run workflow
        results = orchestrator.run_workflow(timeout=60)
        
        execution_time = time.time() - start_time
        
        print(f"\nWorkflow completed in {execution_time:.2f} seconds")
        print(f"Task Results:")
        for task_id, result in results.items():
            print(f"  {task_id}: {result.status.value}")
            if result.status.value == 'completed' and result.result:
                print(f"    Result: {result.result}")
        
        # Show final status
        final_status = orchestrator.get_workflow_status()
        print(f"\nFinal Status:")
        print(f"  Completed: {final_status['completed_tasks']}")
        print(f"  Failed: {final_status['failed_tasks']}")
        print(f"  Progress: {final_status['progress_percent']:.1f}%")
        
        print("✅ Scalable Orchestration demonstration completed!")
        
    except Exception as e:
        print(f"❌ Orchestration demonstration failed: {e}")
        logger.exception("Orchestration demo failed")

def demonstrate_enhanced_drift_monitoring():
    """Demonstrate enhanced drift monitoring capabilities."""
    print("\n" + "="*80)
    print("📊 ENHANCED DRIFT MONITORING DEMONSTRATION")
    print("="*80)
    
    try:
        from src.duetmind_adaptive.training.enhanced_drift_monitoring import (
            create_enhanced_drift_monitor, create_default_alert_config,
            create_default_response_config, DriftSeverity, ResponseAction
        )
        
        print("Creating reference and current datasets...")
        
        # Reference dataset
        reference_data = create_sample_dataset(500)
        reference_data = pd.get_dummies(reference_data, drop_first=True)
        
        # Current dataset with drift
        current_data = create_sample_dataset(300)
        # Introduce drift by shifting age distribution
        current_data['age'] = current_data['age'] + 10
        # Introduce concept drift by changing diagnosis correlation
        current_data['diagnosis'] = (current_data['mmse'] < 20).astype(int)
        current_data = pd.get_dummies(current_data, drop_first=True)
        
        # Align columns
        for col in reference_data.columns:
            if col not in current_data.columns:
                current_data[col] = 0
        current_data = current_data[reference_data.columns]
        
        print("Setting up drift monitoring...")
        
        # Create monitoring configuration
        alert_config = create_default_alert_config()
        response_config = create_default_response_config()
        
        # Baseline metrics (simulated)
        baseline_metrics = {
            'accuracy': 0.85,
            'roc_auc': 0.88,
            'f1_score': 0.82
        }
        
        # Current metrics (simulated degradation)
        current_metrics = {
            'accuracy': 0.78,
            'roc_auc': 0.82,
            'f1_score': 0.75
        }
        
        # Create drift monitor
        drift_monitor = create_enhanced_drift_monitor(
            reference_data=reference_data,
            baseline_metrics=baseline_metrics,
            alert_config=alert_config,
            response_config=response_config,
            drift_threshold=0.1
        )
        
        print("Running comprehensive drift detection...")
        
        # Detect drift
        drift_results = drift_monitor.detect_comprehensive_drift(
            current_data=current_data,
            current_metrics=current_metrics
        )
        
        print("\n📈 Drift Detection Results:")
        print(f"Data drift detected: {drift_results['data_drift'].get('overall_drift_detected', False)}")
        print(f"Model drift detected: {drift_results['model_drift'].get('performance_drift_detected', False)}")
        print(f"Concept drift detected: {drift_results['concept_drift'].get('concept_drift_detected', False)}")
        
        if drift_results['alerts']:
            print(f"\n🚨 Alerts Generated: {len(drift_results['alerts'])}")
            for i, alert in enumerate(drift_results['alerts'], 1):
                print(f"  {i}. {alert.severity.value.upper()} - {alert.description}")
                print(f"     Drift Score: {alert.drift_score:.4f}")
        
        if drift_results['recommended_actions']:
            print(f"\n💡 Recommended Actions: {len(drift_results['recommended_actions'])}")
            for i, action in enumerate(drift_results['recommended_actions'], 1):
                print(f"  {i}. {action.value.replace('_', ' ').title()}")
        
        # Get monitoring summary
        summary = drift_monitor.get_drift_summary(days_back=1)
        print(f"\n📊 Monitoring Summary:")
        print(f"Total alerts: {summary['total_alerts']}")
        print(f"Alerts by severity: {summary['alerts_by_severity']}")
        print(f"Alerts by type: {summary['alerts_by_type']}")
        
        print("✅ Enhanced Drift Monitoring demonstration completed!")
        
    except Exception as e:
        print(f"❌ Drift monitoring demonstration failed: {e}")
        logger.exception("Drift monitoring demo failed")

def demonstrate_integrated_system():
    """Demonstrate the complete integrated automation system."""
    print("\n" + "="*80)
    print("🌟 INTEGRATED AUTOMATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    try:
        from src.duetmind_adaptive.training.automation_system import (
            create_automation_system, setup_complete_system
        )
        
        print("Setting up complete automation system...")
        
        # Create sample data
        reference_data = create_sample_dataset(400)
        reference_data_encoded = pd.get_dummies(reference_data, drop_first=True)
        
        baseline_metrics = {
            'accuracy': 0.85,
            'roc_auc': 0.88,
            'f1_score': 0.82
        }
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Working directory: {temp_dir}")
            
            # Setup system
            system = setup_complete_system(
                reference_data=reference_data_encoded,
                baseline_metrics=baseline_metrics,
                working_dir=temp_dir
            )
            
            # Save sample data for workflow
            data_path = Path(temp_dir) / "sample_data.csv"
            reference_data.to_csv(data_path, index=False)
            
            print("✓ System initialized")
            
            # Get system status
            status = system.get_system_status()
            print(f"Components initialized: {status['components']}")
            
            print("\nCreating automated training workflow...")
            
            # Create workflow
            workflow_name = system.create_automated_training_workflow(
                data_path=str(data_path),
                target_column='diagnosis',
                pipeline_name='basic_classification',
                enable_automl=False  # Disable for demo speed
            )
            
            print(f"✓ Created workflow: {workflow_name}")
            
            # Get updated status
            updated_status = system.get_system_status()
            print(f"Active workflows: {updated_status['active_workflows']}")
            
            print("\nExecuting workflow...")
            
            # Run workflow (with timeout for demo)
            try:
                workflow_results = system.run_workflow(workflow_name, timeout=120)
                
                print(f"✓ Workflow completed")
                print(f"Task results: {len(workflow_results)} tasks")
                
                successful_tasks = sum(1 for r in workflow_results.values() if r.status.value == 'completed')
                print(f"Successful tasks: {successful_tasks}/{len(workflow_results)}")
                
            except Exception as e:
                print(f"⚠️ Workflow execution had issues: {e}")
            
            # Final system status
            final_status = system.get_system_status()
            print(f"\nFinal System Status:")
            print(f"  Workflows: {final_status['active_workflows']}")
            print(f"  Orchestrator: {final_status['orchestrator_status']['progress_percent']:.1f}% complete")
            
            # Save system state
            state_file = system.save_system_state()
            print(f"✓ System state saved to: {Path(state_file).name}")
            
            # Shutdown
            system.shutdown()
            
        print("✅ Integrated System demonstration completed!")
        
    except Exception as e:
        print(f"❌ Integrated system demonstration failed: {e}")
        logger.exception("Integrated system demo failed")

def main():
    """Run all demonstrations."""
    print("🎯 DUETMIND ADAPTIVE - AUTOMATION & SCALABILITY DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases all four automation & scalability components:")
    print("1. AutoML Integration - Automated hyperparameter optimization")  
    print("2. Pipeline Customization - Flexible, configurable pipelines")
    print("3. Scalable Orchestration - Workflow management and scheduling")
    print("4. Enhanced Drift Monitoring - Automated drift detection and response")
    print("5. Integrated System - All components working together")
    
    # Run demonstrations
    demonstrate_automl_integration()
    demonstrate_pipeline_customization() 
    demonstrate_scalable_orchestration()
    demonstrate_enhanced_drift_monitoring()
    demonstrate_integrated_system()
    
    print("\n" + "="*80)
    print("🎉 ALL DEMONSTRATIONS COMPLETED!")
    print("="*80)
    print("The Automation & Scalability features are now fully implemented and demonstrated.")
    print("\nNext steps:")
    print("- Integrate with existing DuetMind training pipelines")
    print("- Add production monitoring dashboards")
    print("- Implement Ray-based distributed computing")
    print("- Add more sophisticated drift detection algorithms")

if __name__ == "__main__":
    main()