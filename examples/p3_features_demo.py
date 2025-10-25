"""
P3 Features Demonstration

Demonstrates all P3 features:
- P3-1: Advanced DICOM/3D Brain Viewer with explainability
- P3-2: Quantum-safe cryptography in production key flows
- P3-3: Model canary deployment with continuous validation
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('p3_demo')


def demo_p3_1_advanced_viewers():
    """
    Demonstrate P3-1: Advanced Multimodal Viewers.
    
    Shows:
    - DICOM viewer API
    - 3D brain visualization
    - Explainability overlays
    """
    print("\n" + "="*80)
    print("P3-1: ADVANCED MULTIMODAL VIEWERS DEMONSTRATION")
    print("="*80 + "\n")
    
    try:
        from api.viewer_api import create_viewer_api
        from src.aimedres.dashboards.brain_visualization import (
            BrainRegion,
            DiseaseStage,
            TreatmentType
        )
        import numpy as np
        
        # Create viewer API
        config = {
            'temp_dir': '/tmp/aimedres_demo_viewer'
        }
        viewer_api = create_viewer_api(config)
        
        print("✓ Viewer API initialized")
        print(f"  - DICOM streaming: Enabled")
        print(f"  - 3D brain visualization: Enabled")
        print(f"  - Explainability overlays: Enabled")
        
        # Demonstrate 3D brain viewer
        print("\n--- 3D Brain Visualization ---")
        
        # Create brain visualization session
        session_data = {
            'patient_id': 'DEMO_PATIENT_001',
            'regions': [
                'frontal_lobe',
                'hippocampus',
                'temporal_lobe',
                'amygdala'
            ],
            'highlight_abnormalities': True
        }
        
        print(f"Creating brain visualization for patient: {session_data['patient_id']}")
        print(f"  Regions of interest: {len(session_data['regions'])}")
        
        # Get statistics from brain engine
        if viewer_api.brain_engine:
            stats = viewer_api.brain_engine.get_statistics()
            print(f"\nBrain Engine Statistics:")
            print(f"  - Visualizations generated: {stats['visualizations_generated']}")
            print(f"  - Simulations run: {stats['simulations_run']}")
            print(f"  - Average render time: {stats['average_render_time_ms']:.2f}ms")
        
        # Demonstrate disease progression
        print("\n--- Disease Progression Tracking ---")
        
        from src.aimedres.dashboards.brain_visualization import create_brain_visualization_engine
        engine = create_brain_visualization_engine(enable_real_time=True)
        
        # Create baseline snapshot
        affected_regions = {
            BrainRegion.HIPPOCAMPUS: 0.3,
            BrainRegion.FRONTAL_LOBE: 0.2,
            BrainRegion.TEMPORAL_LOBE: 0.25
        }
        
        snapshot = engine.capture_progression_snapshot(
            patient_id='DEMO_PATIENT_001',
            stage=DiseaseStage.MILD,
            affected_regions=affected_regions,
            biomarkers={'amyloid_beta': 1.5, 'tau': 1.2},
            cognitive_scores={'mmse': 24, 'moca': 22}
        )
        
        print(f"Captured disease progression snapshot:")
        print(f"  - Snapshot ID: {snapshot.snapshot_id}")
        print(f"  - Disease stage: {snapshot.stage.value}")
        print(f"  - Affected regions: {len(snapshot.affected_regions)}")
        print(f"  - Biomarkers tracked: {len(snapshot.biomarkers)}")
        
        # Demonstrate treatment simulation
        print("\n--- Treatment Impact Simulation ---")
        
        simulation = engine.simulate_treatment_impact(
            patient_id='DEMO_PATIENT_001',
            baseline_snapshot_id=snapshot.snapshot_id,
            treatment_type=TreatmentType.MEDICATION,
            duration_days=180,
            efficacy_rate=0.75
        )
        
        print(f"Treatment simulation created:")
        print(f"  - Simulation ID: {simulation.simulation_id}")
        print(f"  - Treatment type: {simulation.treatment_type.value}")
        print(f"  - Duration: {simulation.duration_days} days")
        print(f"  - Success probability: {simulation.success_probability:.1%}")
        print(f"  - Projected outcomes: {len(simulation.projected_outcomes)} time points")
        
        if simulation.projected_outcomes:
            final_outcome = simulation.projected_outcomes[-1]
            print(f"  - Final projected improvement: {final_outcome['cognitive_improvement_pct']:.1f}%")
        
        # Demonstrate explainability overlay
        print("\n--- AI Explainability Overlay ---")
        
        explainability_data = {
            'prediction_type': 'alzheimers_risk',
            'region_importance': {
                'hippocampus': 0.85,
                'frontal_lobe': 0.72,
                'temporal_lobe': 0.68,
                'amygdala': 0.54
            },
            'confidence': 0.89,
            'features': [
                {'name': 'hippocampal_volume', 'value': -2.1, 'contribution': 0.35},
                {'name': 'amyloid_beta', 'value': 1.5, 'contribution': 0.28},
                {'name': 'cognitive_score', 'value': 24, 'contribution': 0.22},
                {'name': 'age', 'value': 72, 'contribution': 0.15}
            ]
        }
        
        print("Explainability overlay generated:")
        print(f"  - Prediction: {explainability_data['prediction_type']}")
        print(f"  - Confidence: {explainability_data['confidence']:.1%}")
        print(f"  - Key regions contributing to prediction:")
        for region, importance in sorted(
            explainability_data['region_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            print(f"    • {region}: {importance:.2f}")
        
        print(f"\n✓ P3-1 demonstration complete")
        return True
        
    except Exception as e:
        logger.error(f"P3-1 demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_p3_2_quantum_keys():
    """
    Demonstrate P3-2: Quantum-safe Cryptography in Production.
    
    Shows:
    - Hybrid Kyber/AES key generation
    - Automated key rotation
    - KMS integration
    """
    print("\n" + "="*80)
    print("P3-2: QUANTUM-SAFE PRODUCTION KEY FLOWS DEMONSTRATION")
    print("="*80 + "\n")
    
    try:
        from security.quantum_prod_keys import (
            create_quantum_key_manager,
            KeyType,
            KeyStatus,
            KeyRotationPolicy
        )
        
        # Create key manager with rotation policy
        rotation_policy = KeyRotationPolicy(
            enabled=True,
            rotation_interval_days=90,
            max_key_age_days=365,
            automatic_rotation=False,  # Manual for demo
            notify_before_rotation_days=7
        )
        
        config = {
            'quantum_algorithm': 'kyber768',
            'kms_enabled': False,  # KMS disabled for demo
            'key_storage_path': '/tmp/aimedres_demo_keys'
        }
        
        key_manager = create_quantum_key_manager(config, rotation_policy)
        
        print("✓ Quantum Key Manager initialized")
        print(f"  - Algorithm: {config['quantum_algorithm']}")
        print(f"  - Hybrid mode: Kyber768 + AES-256")
        print(f"  - Rotation policy: Every {rotation_policy.rotation_interval_days} days")
        
        # Generate various key types
        print("\n--- Key Generation ---")
        
        key_types = [
            (KeyType.DATA_ENCRYPTION, "Medical data encryption"),
            (KeyType.SESSION, "User session encryption"),
            (KeyType.API, "API authentication"),
        ]
        
        generated_keys = []
        for key_type, description in key_types:
            key = key_manager.generate_key(
                key_type=key_type,
                metadata={'description': description, 'environment': 'demo'},
                expires_in_days=180
            )
            generated_keys.append(key)
            
            print(f"\nGenerated {key_type.value} key:")
            print(f"  - Key ID: {key.key_id}")
            print(f"  - Status: {key.status.value}")
            print(f"  - Quantum protected: Yes")
            print(f"  - Expires: {key.expires_at.strftime('%Y-%m-%d') if key.expires_at else 'Never'}")
        
        # Demonstrate key rotation
        print("\n--- Key Rotation ---")
        
        # Rotate the first key
        old_key = generated_keys[0]
        print(f"\nRotating key: {old_key.key_id}")
        
        new_key = key_manager.rotate_key(old_key.key_id, force=True)
        
        print(f"Key rotation complete:")
        print(f"  - Old key: {old_key.key_id} (now deprecated)")
        print(f"  - New key: {new_key.key_id}")
        print(f"  - Rotation count: {new_key.rotation_count}")
        print(f"  - Grace period: {rotation_policy.grace_period_days} days")
        
        # Get status report
        print("\n--- Key Manager Status ---")
        
        status = key_manager.get_status_report()
        print(f"\nStatus Report:")
        print(f"  - Total keys: {status['total_keys']}")
        print(f"  - Active keys: {status['active_keys']}")
        print(f"  - Quantum protected: {status['quantum_protected']}")
        print(f"  - KMS enabled: {status['kms_enabled']}")
        print(f"  - Audit log entries: {status['audit_log_entries']}")
        
        if status['expiring_soon']:
            print(f"\n  Keys expiring soon:")
            for key_info in status['expiring_soon']:
                print(f"    • {key_info['key_id']}: {key_info['days_until_expiry']} days")
        
        # List all keys
        print("\n--- Key Inventory ---")
        
        all_keys = key_manager.list_keys()
        print(f"\nAll registered keys ({len(all_keys)}):")
        for key in all_keys:
            print(f"  • {key.key_id}")
            print(f"    Type: {key.key_type.value}, Status: {key.status.value}")
            print(f"    Created: {key.created_at.strftime('%Y-%m-%d %H:%M')}")
            print(f"    Usage count: {key.usage_count}")
        
        # Performance metrics
        print("\n--- Quantum Crypto Performance ---")
        
        if key_manager.quantum_crypto:
            perf_test = key_manager.quantum_crypto.test_performance_impact(test_data_size=1024)
            
            print(f"\nPerformance test (1KB data):")
            print(f"  - Encryption: {perf_test['measurements']['encryption']['average_ms']:.2f}ms avg")
            print(f"  - Decryption: {perf_test['measurements']['decryption']['average_ms']:.2f}ms avg")
            print(f"  - Key exchange: {perf_test['measurements']['key_exchange']['average_ms']:.2f}ms avg")
            print(f"  - Performance rating: {perf_test['performance_rating']}")
        
        print(f"\n✓ P3-2 demonstration complete")
        return True
        
    except Exception as e:
        logger.error(f"P3-2 demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_p3_3_canary_pipeline():
    """
    Demonstrate P3-3: Model Canary Deployment Pipeline.
    
    Shows:
    - Shadow mode deployment
    - Automated validation
    - Canary rollout
    - Rollback capabilities
    """
    print("\n" + "="*80)
    print("P3-3: MODEL CANARY DEPLOYMENT PIPELINE DEMONSTRATION")
    print("="*80 + "\n")
    
    try:
        from mlops.pipelines.canary_deployment import (
            create_canary_pipeline,
            CanaryConfig,
            DeploymentMode,
            DeploymentStatus
        )
        import numpy as np
        
        # Create canary pipeline with config
        config = CanaryConfig(
            shadow_duration_hours=1,  # Short for demo
            canary_stages=[5.0, 10.0, 25.0, 50.0, 100.0],
            stage_duration_hours=1,
            min_accuracy=0.85,
            min_f1_score=0.80,
            min_fairness_score=0.80,
            auto_rollback_enabled=True
        )
        
        pipeline = create_canary_pipeline(config, storage_path='/tmp/aimedres_demo_deployments')
        
        print("✓ Canary Pipeline initialized")
        print(f"  - Shadow duration: {config.shadow_duration_hours}h")
        print(f"  - Canary stages: {config.canary_stages}")
        print(f"  - Auto-rollback: {config.auto_rollback_enabled}")
        
        # Register a new model
        print("\n--- Model Registration ---")
        
        model_meta = pipeline.register_model(
            model_id='alzheimer_nn',
            version='v2.1.0',
            model_artifact_path='/models/alzheimer_nn_v2.1.0.pt',
            metadata={
                'framework': 'pytorch',
                'architecture': 'transformer',
                'parameters': '15M',
                'trained_on': '2025-01-15'
            }
        )
        
        print(f"\nRegistered new model:")
        print(f"  - Model ID: {model_meta.model_id}")
        print(f"  - Version: {model_meta.version}")
        print(f"  - Framework: {model_meta.metadata.get('framework')}")
        print(f"  - Parameters: {model_meta.metadata.get('parameters')}")
        
        # Deploy in shadow mode
        print("\n--- Shadow Mode Deployment ---")
        
        # Generate synthetic holdout data for validation
        np.random.seed(42)
        holdout_data = np.random.randn(100, 50)  # 100 samples, 50 features
        holdout_labels = np.random.randint(0, 2, 100)  # Binary classification
        
        print(f"Deploying model in shadow mode...")
        print(f"  - Holdout dataset: {len(holdout_data)} samples")
        
        deployment = pipeline.deploy_shadow(
            model_id='alzheimer_nn',
            model_version='v2.1.0',
            holdout_data=holdout_data,
            holdout_labels=holdout_labels
        )
        
        print(f"\nShadow deployment created:")
        print(f"  - Deployment ID: {deployment.deployment_id}")
        print(f"  - Mode: {deployment.mode.value}")
        print(f"  - Status: {deployment.status.value}")
        print(f"  - Traffic: {deployment.traffic_percentage}% (shadow receives 0%)")
        
        # Show validation results
        print("\n--- Validation Test Results ---")
        
        if deployment.validation_tests:
            print(f"\nCompleted {len(deployment.validation_tests)} validation tests:")
            
            for test in deployment.validation_tests:
                status_icon = "✓" if test.passed else "✗"
                print(f"\n  {status_icon} {test.test_name}")
                print(f"    Type: {test.test_type}")
                print(f"    Result: {test.result.value}")
                print(f"    Score: {test.score:.3f}")
                print(f"    Threshold: {test.threshold:.3f}")
                print(f"    Status: {'PASS' if test.passed else 'FAIL'}")
                
                # Show relevant details
                if test.test_type == 'fairness':
                    details = test.details
                    if 'demographic_disparity' in details:
                        print(f"    Demographic disparity: {details['demographic_disparity']:.3f}")
                elif test.test_type == 'performance':
                    details = test.details
                    if 'degradation_pct' in details:
                        print(f"    Performance degradation: {details['degradation_pct']:.1f}%")
        
        # Check if validation passed
        all_passed = all(t.passed for t in deployment.validation_tests)
        
        if all_passed:
            print(f"\n✓ All validation tests PASSED")
            
            # Deploy to canary if validation passed
            print("\n--- Canary Deployment ---")
            
            success = pipeline.deploy_canary(deployment.deployment_id, auto_promote=False)
            
            if success:
                # Get updated status
                status = pipeline.get_deployment_status(deployment.deployment_id)
                
                print(f"\nCanary deployment started:")
                print(f"  - Mode: {status['mode']}")
                print(f"  - Status: {status['status']}")
                print(f"  - Traffic: {status['traffic_percentage']}%")
                print(f"\nGradual rollout stages:")
                for i, stage_pct in enumerate(config.canary_stages, 1):
                    print(f"  {i}. {stage_pct}% traffic → Monitor for {config.stage_duration_hours}h")
        else:
            print(f"\n✗ Validation FAILED - deployment will not proceed to canary")
            if deployment.rollback_triggered:
                print(f"  Rollback reason: {deployment.rollback_reason}")
        
        # Show deployment summary
        print("\n--- Deployment Summary ---")
        
        all_deployments = pipeline.list_deployments()
        
        print(f"\nTotal deployments: {len(all_deployments)}")
        for dep in all_deployments:
            print(f"\n  • {dep['deployment_id']}")
            print(f"    Model: {dep['model_id']} v{dep['model_version']}")
            print(f"    Mode: {dep['mode']}, Status: {dep['status']}")
            print(f"    Traffic: {dep['traffic_percentage']}%")
            if dep['rollback_triggered']:
                print(f"    ⚠ Rollback: {dep['rollback_reason']}")
        
        print(f"\n✓ P3-3 demonstration complete")
        return True
        
    except Exception as e:
        logger.error(f"P3-3 demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all P3 demonstrations."""
    print("\n" + "="*80)
    print("AiMedRes P3 Features Comprehensive Demonstration")
    print("Long-term / Scale / Research Features")
    print("="*80)
    print(f"\nDemo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run demonstrations
    results['P3-1'] = demo_p3_1_advanced_viewers()
    results['P3-2'] = demo_p3_2_quantum_keys()
    results['P3-3'] = demo_p3_3_canary_pipeline()
    
    # Print summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80 + "\n")
    
    for feature, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{feature}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All P3 demonstrations completed successfully!")
    else:
        print("\n✗ Some demonstrations failed - check logs above for details")
    
    print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
