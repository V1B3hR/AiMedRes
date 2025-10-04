#!/usr/bin/env python3
"""
Simple wrapper script for running Alzheimer's training as per problem statement.
This provides an easy entry point for the training functionality.
"""

import sys
import os
from pathlib import Path

def main():
    """Main entry point that runs the Alzheimer's training pipeline"""
    print("🧠 Alzheimer's Disease Training Pipeline")
    print("=" * 50)
    print("Starting training with datasets from Kaggle...")
    print("This may take several minutes to complete.")
    print()
    
    # Get the path to the actual training script
    repo_root = Path(__file__).parent
    training_script = repo_root / "files" / "training" / "train_alzheimers.py"
    
    if not training_script.exists():
        print(f"❌ Training script not found at: {training_script}")
        print("Please ensure you're running this from the repository root.")
        return False
    
    # Import and run the training pipeline
    try:
        # Add the repo root to Python path
        sys.path.insert(0, str(repo_root))
        
        from aimedres.training.train_alzheimers import AlzheimerTrainingPipeline
        
        # Create and run pipeline with reasonable defaults
        pipeline = AlzheimerTrainingPipeline(output_dir="outputs")
        
        print("📥 Downloading and loading Alzheimer's dataset...")
        print("🔄 Training multiple ML models...")
        print("📊 This includes: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network")
        print()
        
        # Run with parameters optimized for the problem statement datasets
        report = pipeline.run_full_pipeline(
            epochs=30,        # Good balance of training time and performance
            n_folds=5        # Standard cross-validation
        )
        
        print()
        print("🎉 Training completed successfully!")
        print()
        print("📈 Results Summary:")
        print("-" * 20)
        
        # Display key results
        if pipeline.classical_results:
            print("🤖 Classical Models (Cross-Validation):")
            for model_name, results in pipeline.classical_results.items():
                if 'error' not in results and 'accuracy' in results:
                    acc = results['accuracy']['mean']
                    f1 = results['macro_f1']['mean']
                    print(f"   {model_name:15}: Accuracy={acc:.1%}, F1={f1:.1%}")
        
        if pipeline.neural_results and 'error' not in pipeline.neural_results:
            nn = pipeline.neural_results
            print(f"🧠 Neural Network     : Accuracy={nn['accuracy']:.1%}, F1={nn['macro_f1']:.1%}")
        
        print()
        print("💾 All results saved to: outputs/")
        print("   📁 models/         - Trained models (.pkl, .pth files)")
        print("   📁 preprocessors/  - Data preprocessing components") 
        print("   📁 metrics/        - Training reports and metrics")
        print()
        print("ℹ️  Check outputs/metrics/training_summary.txt for detailed results")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print()
        print("Please install required packages:")
        print("pip install numpy pandas scikit-learn torch kagglehub xgboost lightgbm")
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print()
        print("Common solutions:")
        print("- Ensure internet connection for Kaggle dataset download")
        print("- Check that kagglehub is properly installed")
        print("- Verify sufficient disk space for models and data")
        return False

if __name__ == "__main__":
    print("Problem Statement: Run training @V1B3hR/AiMedRes/files/training/train_alzheimers.py")
    print("Datasets from: Kaggle Alzheimer's classification datasets")
    print()
    
    success = main()
    
    if success:
        print("✅ Training pipeline executed successfully!")
        print("The models are ready for Alzheimer's disease classification.")
    else:
        print("❌ Training pipeline failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)