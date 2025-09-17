#!/usr/bin/env python3
"""
Demonstration of Imaging and Agent Integration for DuetMind Adaptive

This script demonstrates the complete integration of:
1. 3D CNN training with MLflow logging
2. Multimodal late fusion
3. Imaging insight generation and memory storage
4. Agent reasoning with imaging context
5. Continual learning triggers

Run with: python demo_imaging_agent_integration.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_3d_cnn_training():
    """Demonstrate 3D CNN training with MLflow logging"""
    logger.info("🧠 DEMO 1: 3D CNN Training with MLflow")
    logger.info("-" * 50)
    
    try:
        import torch
        from train_brain_mri import BrainMRI3DCNN
        import mlflow
        
        # Create 3D CNN model
        model = BrainMRI3DCNN(num_classes=2)
        logger.info(f"✅ Created 3D CNN with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 64, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")
        
        # Demonstrate MLflow integration (without full training)
        mlflow.set_experiment("demo_3d_cnn")
        with mlflow.start_run():
            mlflow.log_param("model_type", "3D_CNN")
            mlflow.log_param("input_shape", "64x64x64")
            mlflow.log_param("num_classes", 2)
            mlflow.log_metric("demo_accuracy", 0.85)
            logger.info("✅ MLflow logging successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 3D CNN demo failed: {e}")
        return False

def demo_multimodal_fusion():
    """Demonstrate multimodal late fusion with evaluation"""
    logger.info("\n🔗 DEMO 2: Multimodal Late Fusion")
    logger.info("-" * 50)
    
    try:
        from multimodal_data_integration import DataFusionProcessor
        
        # Create synthetic multimodal data
        np.random.seed(42)
        n_samples = 200
        
        # Simulate tabular clinical data
        tabular_data = pd.DataFrame({
            'age': np.random.normal(65, 12, n_samples),
            'mmse_score': np.random.normal(24, 5, n_samples),
            'education_years': np.random.normal(14, 3, n_samples),
            'apoe_status': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        })
        
        # Simulate imaging features
        imaging_data = pd.DataFrame({
            'total_brain_volume': np.random.normal(1450000, 200000, n_samples),
            'gray_matter_volume': np.random.normal(580000, 80000, n_samples),
            'white_matter_volume': np.random.normal(520000, 70000, n_samples),
            'csf_volume': np.random.normal(350000, 50000, n_samples),
            'cortical_thickness': np.random.normal(2.4, 0.3, n_samples),
            'snr_score': np.random.normal(15, 4, n_samples),
            'diagnosis': tabular_data['diagnosis']  # Same labels
        })
        
        logger.info(f"✅ Created synthetic data - Tabular: {tabular_data.shape}, Imaging: {imaging_data.shape}")
        
        # Perform late fusion
        processor = DataFusionProcessor()
        data_dict = {'clinical': tabular_data, 'imaging': imaging_data}
        
        results = processor.late_fusion(data_dict, 'diagnosis', use_mlflow=False)
        
        logger.info(f"✅ Late fusion completed with {len(results['modality_models'])} modalities")
        
        # Display results
        if 'modality_metrics' in results:
            for modality, metrics in results['modality_metrics'].items():
                logger.info(f"  {modality}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        if 'ensemble_metrics' in results:
            ensemble = results['ensemble_metrics']
            logger.info(f"  Ensemble: Accuracy={ensemble['accuracy']:.3f}, F1={ensemble['f1_score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Multimodal fusion demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_imaging_insights():
    """Demonstrate imaging insight generation and memory storage"""
    logger.info("\n💡 DEMO 3: Imaging Insight Generation")
    logger.info("-" * 50)
    
    try:
        from agent_memory.imaging_insights import (
            ImagingInsightSummarizer, 
            create_imaging_insight_from_features,
            create_imaging_insight_from_predictions
        )
        
        # Create sample imaging features
        imaging_features = {
            'total_brain_volume_mm3': 1420000,  # Slightly reduced
            'gray_matter_volume_mm3': 550000,   # Reduced (possible atrophy)
            'white_matter_volume_mm3': 510000,
            'qc_snr_basic': 18.5,              # Good quality
            'qc_motion_score': 0.2,            # Low motion
            'acquisition_date': '2024-01-15',
            'modality': 'T1w_MRI'
        }
        
        # Generate insight from features
        insight = create_imaging_insight_from_features(
            imaging_features, 
            patient_id='demo_patient_001'
        )
        
        logger.info(f"✅ Generated imaging insight: {insight.insight_id}")
        logger.info(f"  Clinical significance: {insight.clinical_significance}")
        logger.info(f"  Key findings: {len(insight.key_findings)} items")
        logger.info(f"  Confidence: {insight.confidence_score:.3f}")
        
        # Create sample ML predictions
        ml_predictions = {
            'prediction': 'mild_cognitive_impairment',
            'confidence': 0.78,
            'class_probabilities': {
                'normal': 0.22,
                'mild_cognitive_impairment': 0.78
            }
        }
        
        # Generate insight from predictions
        prediction_insight = create_imaging_insight_from_predictions(
            ml_predictions,
            imaging_features,
            patient_id='demo_patient_001'
        )
        
        logger.info(f"✅ Generated prediction insight: {prediction_insight.insight_id}")
        logger.info(f"  Memory content preview: {prediction_insight.summarize_for_memory()[:100]}...")
        
        return [insight, prediction_insight]
        
    except Exception as e:
        logger.error(f"❌ Imaging insights demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_agent_reasoning():
    """Demonstrate agent reasoning with imaging context"""
    logger.info("\n🤖 DEMO 4: Agent Reasoning with Imaging Context")
    logger.info("-" * 50)
    
    try:
        # Create temporary database for demo
        import sqlite3
        from agent_memory.embed_memory import AgentMemoryStore
        from agent_memory.live_reasoning import LiveReasoningAgent
        
        # Setup in-memory database for demo
        temp_db = ":memory:"
        memory_store = AgentMemoryStore(f"sqlite:///{temp_db}")
        
        # Create reasoning agent
        agent = LiveReasoningAgent(
            agent_id="demo_imaging_agent",
            memory_store=memory_store,
            similarity_threshold=0.6
        )
        
        logger.info("✅ Created reasoning agent with memory store")
        
        # Store some imaging insights in memory
        sample_insights = [
            {
                "content": "MRI T1w analysis shows reduced gray matter volume (550,000 mm³) suggesting possible mild atrophy. SNR quality good (18.5).",
                "type": "imaging_insight",
                "importance": 0.8,
                "metadata": {"modality": "T1w_MRI", "date": "2024-01-15"}
            },
            {
                "content": "Previous MRI comparison shows 5% volume reduction over 12 months, consistent with age-related changes.",
                "type": "imaging_insight", 
                "importance": 0.75,
                "metadata": {"modality": "T1w_MRI", "date": "2023-01-10"}
            }
        ]
        
        for insight in sample_insights:
            agent.store_imaging_insight(insight)
        
        logger.info(f"✅ Stored {len(sample_insights)} imaging insights in memory")
        
        # Test imaging-specific reasoning
        query = "What can you tell me about the brain volume changes in this patient?"
        
        result = agent.reason_with_context(query, reasoning_type="imaging_analysis")
        
        logger.info(f"✅ Reasoning completed with confidence: {result.confidence:.3f}")
        logger.info("📋 Reasoning Response:")
        logger.info("-" * 30)
        # Show first 300 characters of response
        response_preview = result.response[:300] + "..." if len(result.response) > 300 else result.response
        logger.info(response_preview)
        
        # Test retrieval of imaging insights
        imaging_insights = agent.retrieve_imaging_insights("brain volume atrophy", limit=3)
        logger.info(f"✅ Retrieved {len(imaging_insights)} relevant imaging insights")
        
        agent.end_session()
        return result
        
    except Exception as e:
        logger.error(f"❌ Agent reasoning demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_continual_learning_triggers():
    """Demonstrate continual learning trigger logic"""
    logger.info("\n🔄 DEMO 5: Continual Learning Triggers")
    logger.info("-" * 50)
    
    try:
        from mlops.monitoring.data_trigger import (
            DataDrivenRetrainingTrigger,
            RetrainingTriggerConfig
        )
        
        # Create temporary directories for demo
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "imaging_data"
            data_dir.mkdir()
            
            # Create some mock imaging files
            for i in range(150):  # More than new_study_threshold (100)
                mock_file = data_dir / f"study_{i:03d}.nii.gz"
                mock_file.touch()
            
            logger.info(f"✅ Created {len(list(data_dir.glob('*.nii.gz')))} mock imaging studies")
            
            # Configure triggers with imaging-specific settings
            config = RetrainingTriggerConfig(
                min_new_samples=50,
                new_study_threshold=100,  # 100 new imaging studies trigger
                imaging_drift_threshold=0.15,
                image_quality_threshold=0.8,
                data_directories=[str(data_dir)]
            )
            
            # Create trigger system
            trigger = DataDrivenRetrainingTrigger(
                model_name="demo_brain_mri_model",
                config=config
            )
            
            logger.info("✅ Created retraining trigger system")
            logger.info(f"  New study threshold: {config.new_study_threshold}")
            logger.info(f"  Imaging drift threshold: {config.imaging_drift_threshold}")
            logger.info(f"  Image quality threshold: {config.image_quality_threshold}")
            
            # Test trigger conditions
            new_studies_detected = trigger._check_new_imaging_studies()
            logger.info(f"✅ New imaging studies check: {new_studies_detected}")
            
            if new_studies_detected:
                logger.info("🚨 Would trigger retraining due to new imaging studies!")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Continual learning demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete imaging and agent integration demonstration"""
    
    print("=" * 80)
    print("🏥 DUETMIND ADAPTIVE - IMAGING & AGENT INTEGRATION DEMO")
    print("=" * 80)
    print()
    
    demo_results = {}
    
    # Run all demonstrations
    demo_results['3d_cnn'] = demo_3d_cnn_training()
    demo_results['multimodal_fusion'] = demo_multimodal_fusion()
    demo_results['imaging_insights'] = demo_imaging_insights()
    demo_results['agent_reasoning'] = demo_agent_reasoning()
    demo_results['continual_learning'] = demo_continual_learning_triggers()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DEMO SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for result in demo_results.values() if result)
    total_demos = len(demo_results)
    
    for demo_name, result in demo_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{demo_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Success Rate: {success_count}/{total_demos} ({success_count/total_demos*100:.1f}%)")
    
    if success_count == total_demos:
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("The imaging and agent integration system is working correctly.")
    else:
        print(f"\n⚠️  {total_demos - success_count} demo(s) failed. Check logs for details.")
    
    print("\n📚 Integration Features Demonstrated:")
    print("  • 3D CNN architecture for volumetric brain MRI analysis")
    print("  • MLflow experiment tracking and artifact logging")
    print("  • Multimodal late fusion with evaluation metrics")
    print("  • Imaging insight generation and structured memory storage")
    print("  • Agent reasoning with imaging-specific context retrieval")
    print("  • Continual learning triggers for new imaging studies")
    print()
    
    return success_count == total_demos

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)