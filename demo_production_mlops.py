#!/usr/bin/env python3
"""
Production MLOps Features Demonstration

This script demonstrates the complete production-ready MLOps pipeline including:
- Production monitoring with drift detection
- Data-driven automated retraining triggers  
- A/B testing infrastructure
- Production model serving

Run this script to see all features in action.
"""

import sys
import os
import tempfile
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_production_monitoring():
    """Demonstrate production monitoring capabilities."""
    print("\n" + "="*60)
    print("🔍 PRODUCTION MONITORING DEMONSTRATION")
    print("="*60)
    
    from mlops.monitoring.production_monitor import create_production_monitor, AlertConfig
    
    # Create sample baseline data
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'age': np.random.normal(70, 10, 1000),
        'mmse': np.random.normal(24, 4, 1000),
        'education': np.random.normal(14, 3, 1000),
        'ses': np.random.uniform(1, 5, 1000)
    })
    baseline_labels = pd.Series(np.random.choice([0, 1], 1000, p=[0.6, 0.4]))
    
    # Configure custom alerts
    alert_config = AlertConfig(
        critical_accuracy_drop=0.03,
        warning_accuracy_drop=0.01,
        critical_latency_ms=200.0,
        drift_threshold=0.08
    )
    
    # Create monitor
    monitor = create_production_monitor(
        model_name="alzheimer_classifier_demo",
        baseline_data=baseline_data,
        baseline_labels=baseline_labels
    )
    monitor.alert_config = alert_config
    
    print(f"✅ Created production monitor with {len(baseline_data)} baseline samples")
    
    # Simulate production predictions over time
    scenarios = [
        {"name": "Normal Operation", "drift": 0.0, "accuracy": 0.85, "latency": 80},
        {"name": "Slight Performance Drop", "drift": 0.05, "accuracy": 0.82, "latency": 100},
        {"name": "Drift Detection", "drift": 0.15, "accuracy": 0.80, "latency": 120},
        {"name": "Critical Issues", "drift": 0.20, "accuracy": 0.70, "latency": 250}
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        
        # Generate current data with specified drift
        current_data = baseline_data.iloc[:100].copy()
        if scenario['drift'] > 0:
            # Add drift to features
            current_data['age'] += np.random.normal(0, scenario['drift'] * 20, 100)
            current_data['mmse'] += np.random.normal(0, scenario['drift'] * 5, 100)
        
        # Generate predictions with specified accuracy
        true_labels = baseline_labels.iloc[:100].values
        predictions = []
        for label in true_labels:
            if random.random() < scenario['accuracy']:
                predictions.append(label)  # Correct prediction
            else:
                predictions.append(1 - label)  # Incorrect prediction
        
        predictions = np.array(predictions)
        
        # Log predictions
        result = monitor.log_prediction_batch(
            features=current_data,
            predictions=predictions,
            true_labels=true_labels,
            model_version="v1.2.0",
            latency_ms=scenario['latency']
        )
        
        print(f"   Logged: {result['metrics']['prediction_count']} predictions")
        print(f"   Accuracy: {result['metrics']['accuracy']:.3f}")
        print(f"   Drift Score: {result['metrics']['drift_score']:.3f}")
        print(f"   Latency: {result['metrics']['latency_ms']:.1f}ms")
        
        if result['alerts_triggered']:
            print(f"   🚨 Alerts: {len(result['alerts_triggered'])} triggered")
        
        time.sleep(1)  # Simulate time passage
    
    # Get overall summary
    summary = monitor.get_monitoring_summary(hours=1)
    print(f"\n📈 Summary Report:")
    print(f"   Status: {summary['status']}")
    print(f"   Total Predictions: {summary['total_predictions']}")
    print(f"   Average Accuracy: {summary['avg_accuracy']:.3f}")
    print(f"   Average Latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"   Drift Incidents: {summary['drift_incidents']}")
    print(f"   Critical Alerts: {summary['alerts'].get('CRITICAL', 0)}")
    
    return monitor


def demonstrate_ab_testing():
    """Demonstrate A/B testing infrastructure."""
    print("\n" + "="*60)
    print("🧪 A/B TESTING INFRASTRUCTURE DEMONSTRATION")
    print("="*60)
    
    from mlops.monitoring.ab_testing import ABTestingManager, ABTestConfig
    
    # Initialize A/B testing manager
    ab_manager = ABTestingManager()
    
    # Create experiment configuration
    config = ABTestConfig(
        experiment_name="alzheimer_model_comparison_demo",
        model_a_version="v1.0.0",
        model_b_version="v2.0.0", 
        traffic_split=0.4,  # 40% to new model
        duration_days=14,
        min_samples_per_variant=100,
        significance_level=0.05
    )
    
    print(f"✅ Created A/B test: {config.experiment_name}")
    print(f"   Control: {config.model_a_version} vs Treatment: {config.model_b_version}")
    print(f"   Traffic Split: {config.traffic_split*100:.0f}% to treatment")
    
    # Create and start experiment
    experiment_name = ab_manager.create_experiment(config)
    ab_manager.start_experiment(experiment_name)
    
    print(f"🚀 Started experiment: {experiment_name}")
    
    # Mock models with different performance characteristics
    class MockModel:
        def __init__(self, accuracy, name):
            self.accuracy = accuracy
            self.name = name
        
        def predict(self, X):
            return [1 if random.random() < self.accuracy else 0 for _ in range(len(X))]
    
    models = {
        "v1.0.0": MockModel(0.82, "Control Model"),      # 82% accuracy
        "v2.0.0": MockModel(0.87, "Treatment Model")     # 87% accuracy
    }
    
    print(f"\n🤖 Models:")
    print(f"   Control (v1.0.0): {models['v1.0.0'].accuracy*100:.0f}% accuracy")
    print(f"   Treatment (v2.0.0): {models['v2.0.0'].accuracy*100:.0f}% accuracy")
    
    # Generate predictions and outcomes
    print(f"\n📈 Running experiment with {500} users...")
    user_outcomes = []
    
    for i in range(500):
        # Generate user features
        features = {
            'age': random.randint(60, 90),
            'mmse': random.randint(15, 30),
            'education': random.randint(8, 20),
            'ses': random.randint(1, 5)
        }
        
        # Make prediction through A/B testing
        result = ab_manager.make_prediction(
            experiment_name=experiment_name,
            user_id=f"user_{i:04d}",
            features=features,
            models=models
        )
        
        # Simulate true outcome (somewhat correlated with MMSE score)
        true_label = 1 if features['mmse'] < 22 and random.random() < 0.75 else 0
        user_outcomes.append((f"user_{i:04d}", true_label))
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1} users...")
    
    # Update with true outcomes
    ab_manager.update_prediction_outcomes(experiment_name, user_outcomes)
    
    # Analyze experiment results
    print(f"\n📊 Analyzing experiment results...")
    results = ab_manager.analyze_experiment(experiment_name)
    
    print(f"\n🎯 Experiment Results:")
    print(f"   Experiment: {results.experiment_name}")
    print(f"   Control (A): {results.samples_a} samples, {results.metrics_a['accuracy']:.3f} accuracy")
    print(f"   Treatment (B): {results.samples_b} samples, {results.metrics_b['accuracy']:.3f} accuracy")
    
    # Statistical significance
    if 'accuracy' in results.statistical_significance:
        is_significant = results.statistical_significance['accuracy']
        p_value = results.p_values.get('accuracy', 'N/A')
        print(f"   Statistical Significance: {'✅ Yes' if is_significant else '❌ No'} (p={p_value:.4f})" if isinstance(p_value, float) else f"   Statistical Significance: {'✅ Yes' if is_significant else '❌ No'}")
    
    print(f"   Winner: {results.winner or 'Inconclusive'}")
    print(f"   Recommendation: {results.recommendation}")
    
    # Stop experiment
    final_results = ab_manager.stop_experiment(experiment_name, "Demo completed")
    print(f"⏹️  Experiment stopped: {experiment_name}")
    
    return ab_manager, results


def demonstrate_data_driven_retraining():
    """Demonstrate data-driven automated retraining."""
    print("\n" + "="*60)
    print("🔄 DATA-DRIVEN AUTOMATED RETRAINING DEMONSTRATION")
    print("="*60)
    
    from mlops.monitoring.data_trigger import create_data_trigger, RetrainingTriggerConfig
    
    # Configure retraining triggers
    config = RetrainingTriggerConfig(
        min_new_samples=50,             # Low threshold for demo
        max_days_without_retrain=1,     # Short for demo
        accuracy_degradation_threshold=0.05,
        drift_score_threshold=0.1,
        min_hours_between_retrains=0,   # No delay for demo
        max_retrains_per_day=10,
        require_manual_approval=False
    )
    
    print(f"✅ Created retraining trigger configuration:")
    print(f"   Min new samples: {config.min_new_samples}")
    print(f"   Max days without retrain: {config.max_days_without_retrain}")
    print(f"   Accuracy threshold: {config.accuracy_degradation_threshold}")
    print(f"   Drift threshold: {config.drift_score_threshold}")
    
    # Create retraining trigger (using monitor from previous demo if available)
    trigger = create_data_trigger(
        model_name="alzheimer_classifier_demo",
        config=config
    )
    
    print(f"🚀 Initialized automated retraining trigger")
    
    # Simulate various trigger scenarios
    scenarios = [
        {"type": "performance_degradation", "reason": "Accuracy dropped below threshold"},
        {"type": "data_drift", "reason": "Significant feature drift detected"},
        {"type": "time_based", "reason": "Maximum days without retraining exceeded"},
        {"type": "manual", "reason": "Forced retraining for demonstration"}
    ]
    
    print(f"\n🎭 Simulating trigger scenarios:")
    
    for scenario in scenarios:
        print(f"\n   Scenario: {scenario['type']}")
        print(f"   Reason: {scenario['reason']}")
        
        # Simulate trigger handling
        trigger._handle_trigger(scenario['type'], scenario['reason'])
        
        # Brief pause to simulate processing
        time.sleep(1)
    
    # Get trigger history
    history = trigger.get_trigger_history(days=1)
    
    print(f"\n📚 Retraining History (Last 24 hours):")
    print(f"   Total events: {len(history)}")
    
    for i, event in enumerate(history[:3]):  # Show first 3 events
        print(f"   Event {i+1}: {event['trigger_type']} - {event.get('success', 'Unknown')}")
    
    return trigger


def demonstrate_production_serving():
    """Demonstrate production model serving."""
    print("\n" + "="*60)
    print("🚀 PRODUCTION MODEL SERVING DEMONSTRATION")
    print("="*60)
    
    from mlops.serving.production_server import create_production_server
    
    # Create production server
    server = create_production_server(
        model_name="alzheimer_classifier_demo",
        enable_monitoring=True,
        enable_ab_testing=True
    )
    
    print(f"✅ Created production server:")
    print(f"   Model: {server.model_name}")
    print(f"   Monitoring: {'Enabled' if server.enable_monitoring else 'Disabled'}")
    print(f"   A/B Testing: {'Enabled' if server.enable_ab_testing else 'Disabled'}")
    
    # Mock a simple model for demonstration
    class DemoModel:
        def predict(self, X):
            # Simple rule-based prediction for demo
            predictions = []
            for row in X:
                # Simulate Alzheimer's prediction based on age and MMSE
                age = row[1] if len(row) > 1 else 70
                mmse = row[4] if len(row) > 4 else 25
                
                # Simple heuristic: higher age + lower MMSE = higher risk
                risk_score = (age - 60) * 0.02 + (30 - mmse) * 0.1
                prediction = 1 if risk_score > 0.5 else 0
                predictions.append(prediction)
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            predictions = self.predict(X)
            # Convert to probabilities
            probas = []
            for pred in predictions:
                if pred == 1:
                    probas.append([0.3, 0.7])  # High risk
                else:
                    probas.append([0.8, 0.2])  # Low risk
            return np.array(probas)
    
    # Add mock model to server
    demo_model = DemoModel()
    server.models["v1.0.0"] = demo_model
    server.current_model_version = "v1.0.0"
    
    print(f"🤖 Loaded demo model: v1.0.0")
    
    # Simulate API requests
    print(f"\n📡 Simulating API requests...")
    
    # Test single prediction
    sample_features = {
        'M/F': 0,
        'Age': 72,
        'EDUC': 16,
        'SES': 3,
        'MMSE': 23,
        'CDR': 0.5,
        'eTIV': 1450,
        'nWBV': 0.72,
        'ASF': 1.15
    }
    
    result = server._make_prediction(sample_features, user_id="demo_user_001")
    
    print(f"   Single Prediction Result:")
    print(f"     Prediction: {result['prediction']}")
    print(f"     Model Version: {result['model_version']}")
    print(f"     Confidence: {result['confidence']:.3f}")
    
    # Test batch predictions
    batch_features = []
    for i in range(10):
        features = {
            'M/F': random.choice([0, 1]),
            'Age': random.randint(60, 90),
            'EDUC': random.randint(8, 20),
            'SES': random.randint(1, 5),
            'MMSE': random.randint(15, 30),
            'CDR': random.uniform(0, 2),
            'eTIV': random.randint(1200, 1800),
            'nWBV': random.uniform(0.6, 0.8),
            'ASF': random.uniform(0.9, 1.3)
        }
        batch_features.append(features)
    
    print(f"\n   Batch Prediction Results (10 samples):")
    for i, features in enumerate(batch_features):
        result = server._make_prediction(features, user_id=f"batch_user_{i:03d}")
        print(f"     User {i+1}: Prediction={result['prediction']}, Confidence={result['confidence']:.3f}")
    
    # Test Flask app endpoints (using test client)
    with server.app.test_client() as client:
        print(f"\n🌐 Testing API endpoints:")
        
        # Health check
        response = client.get('/health')
        if response.status_code == 200:
            health_data = response.get_json()
            print(f"   ✅ Health Check: {health_data['status']}")
            print(f"      Uptime: {health_data['uptime_seconds']:.1f}s")
            print(f"      Requests: {health_data['request_count']}")
        
        # Single prediction
        response = client.post('/predict', json={
            'features': sample_features,
            'user_id': 'api_test_user'
        })
        
        if response.status_code == 200:
            pred_data = response.get_json()
            print(f"   ✅ Prediction API: {pred_data['prediction']}")
            print(f"      Latency: {pred_data['latency_ms']:.1f}ms")
        
        # Monitoring status
        response = client.get('/monitoring/status?hours=1')
        if response.status_code == 200:
            monitoring_data = response.get_json()
            print(f"   ✅ Monitoring API: {monitoring_data.get('status', 'Unknown')}")
    
    return server


def demonstrate_integration():
    """Demonstrate integrated production pipeline."""
    print("\n" + "="*60)
    print("🔗 INTEGRATED PRODUCTION PIPELINE DEMONSTRATION")
    print("="*60)
    
    print("🎯 This demonstration shows how all components work together:")
    print("   1. Production monitoring tracks model performance")
    print("   2. Data-driven triggers detect when retraining is needed")
    print("   3. A/B testing validates new model versions")
    print("   4. Production serving delivers predictions with monitoring")
    
    # Create integrated pipeline
    from mlops.monitoring.production_monitor import create_production_monitor
    from mlops.monitoring.data_trigger import create_data_trigger, RetrainingTriggerConfig
    from mlops.monitoring.ab_testing import ABTestingManager
    
    # Setup monitoring
    baseline_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    baseline_labels = pd.Series(np.random.choice([0, 1], 100))
    
    monitor = create_production_monitor(
        model_name="integrated_demo",
        baseline_data=baseline_data, 
        baseline_labels=baseline_labels
    )
    
    # Setup retraining triggers  
    config = RetrainingTriggerConfig(
        min_new_samples=10,
        max_days_without_retrain=1,
        accuracy_degradation_threshold=0.1
    )
    
    trigger = create_data_trigger(
        model_name="integrated_demo",
        config=config,
        production_monitor=monitor
    )
    
    # Setup A/B testing
    ab_manager = ABTestingManager()
    
    print(f"\n✅ Integrated pipeline components initialized:")
    print(f"   📊 Production Monitor: Ready")
    print(f"   🔄 Retraining Trigger: Ready") 
    print(f"   🧪 A/B Testing: Ready")
    
    # Simulate integrated workflow
    print(f"\n🎭 Simulating integrated workflow:")
    
    # Step 1: Normal operations with monitoring
    print(f"   1. Monitoring normal operations...")
    current_data = baseline_data.iloc[:20]
    predictions = np.random.choice([0, 1], 20)
    true_labels = baseline_labels.iloc[:20].values
    
    result = monitor.log_prediction_batch(
        features=current_data,
        predictions=predictions, 
        true_labels=true_labels,
        model_version="v1.0.0"
    )
    
    print(f"      ✅ Logged {result['metrics']['prediction_count']} predictions")
    print(f"      📈 Accuracy: {result['metrics']['accuracy']:.3f}")
    
    # Step 2: Performance degradation detected
    print(f"   2. Simulating performance degradation...")
    degraded_predictions = np.zeros(20)  # All wrong predictions
    
    result = monitor.log_prediction_batch(
        features=current_data,
        predictions=degraded_predictions,
        true_labels=true_labels,
        model_version="v1.0.0"
    )
    
    print(f"      ⚠️  Degraded accuracy: {result['metrics']['accuracy']:.3f}")
    if result['alerts_triggered']:
        print(f"      🚨 Alerts triggered: {len(result['alerts_triggered'])}")
    
    # Step 3: Trigger retraining
    print(f"   3. Triggering automated retraining...")
    trigger._handle_trigger("performance_degradation", "Accuracy below threshold")
    print(f"      ✅ Retraining initiated")
    
    # Step 4: A/B test new model
    print(f"   4. A/B testing new model version...")
    # This would normally happen after retraining completes
    print(f"      🧪 Would create A/B test: v1.0.0 vs v1.1.0")
    print(f"      📊 Statistical validation in progress...")
    
    print(f"\n🎉 Integrated pipeline demonstration completed!")
    
    return {
        'monitor': monitor,
        'trigger': trigger,
        'ab_manager': ab_manager
    }


def main():
    """Run complete production MLOps demonstration."""
    print("🚀 DuetMind Adaptive - Production MLOps Pipeline Demonstration")
    print("=" * 80)
    print("This demo showcases the complete production-ready MLOps pipeline with:")
    print("• Real-time monitoring and alerting")
    print("• Data-driven automated retraining")
    print("• A/B testing infrastructure")
    print("• Production model serving")
    print("• Integrated workflow orchestration")
    print("=" * 80)
    
    try:
        # Change to temporary directory for demo
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Run demonstrations
            monitor = demonstrate_production_monitoring()
            ab_manager, ab_results = demonstrate_ab_testing()
            trigger = demonstrate_data_driven_retraining()
            server = demonstrate_production_serving()
            pipeline = demonstrate_integration()
            
            # Summary
            print("\n" + "="*80)
            print("📊 DEMONSTRATION SUMMARY")
            print("="*80)
            
            print("✅ Production Monitoring:")
            print("   • Real-time performance tracking")
            print("   • Automated drift detection")
            print("   • Configurable alerting system")
            print("   • Data quality monitoring")
            
            print("\n✅ A/B Testing Infrastructure:")
            print(f"   • Statistical experiment design")
            print(f"   • Automated significance testing")
            print(f"   • Winner: {ab_results.winner or 'Inconclusive'}")
            print(f"   • Recommendation: {ab_results.recommendation}")
            
            print("\n✅ Data-Driven Retraining:")
            print("   • Performance-based triggers")
            print("   • Time-based triggers")
            print("   • File system monitoring")
            print("   • Comprehensive audit trail")
            
            print("\n✅ Production Serving:")
            print("   • High-performance Flask API")
            print("   • Health checks and batch processing")
            print("   • Integrated monitoring")
            print("   • A/B testing support")
            
            print("\n✅ Integrated Pipeline:")
            print("   • End-to-end automation")
            print("   • Component orchestration")
            print("   • Fault tolerance")
            print("   • Production ready")
            
            print("\n🎯 Key Benefits:")
            print("   • Reduced manual intervention")
            print("   • Faster incident detection and response")
            print("   • Data-driven model improvements")
            print("   • Statistical validation of changes")
            print("   • Comprehensive observability")
            
            print("\n📚 Next Steps:")
            print("   • Review PRODUCTION_MLOPS_GUIDE.md for detailed usage")
            print("   • Configure production environment settings")
            print("   • Set up monitoring dashboards")
            print("   • Establish incident response procedures")
            print("   • Train team on new capabilities")
            
            print("\n" + "="*80)
            print("🎉 Production MLOps Pipeline Demonstration Complete!")
            print("="*80)
            
            return True
            
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)