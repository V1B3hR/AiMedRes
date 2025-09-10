#!/usr/bin/env python3
"""
Full Training Pipeline for DuetMind Adaptive
Comprehensive training system with multiple modes and data sources
"""

import argparse
import sys
import logging
from pathlib import Path
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("FullTraining")

def run_basic_training():
    """Run basic training with test data"""
    print("🔬 Running Basic Training...")
    try:
        from training import AlzheimerTrainer
        
        # Initialize trainer with test data
        trainer = AlzheimerTrainer()
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        model, results = trainer.train_model(X, y)
        trainer.save_model("basic_alzheimer_model.pkl")
        
        print(f"✅ Basic Training Completed!")
        print(f"📊 Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"📊 Test Accuracy: {results['test_accuracy']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Basic training failed: {e}")
        return False

def run_kaggle_training():
    """Run training with Kaggle dataset"""
    print("🌐 Running Kaggle Dataset Training...")
    try:
        # Run the complete training with Kaggle data
        result = subprocess.run([
            sys.executable, 
            "files/training/run_training_complete.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Kaggle Dataset Training Completed!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Kaggle training failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Kaggle training error: {e}")
        return False

def run_simulation_training():
    """Run agent simulation training"""
    print("🤖 Running Agent Simulation Training...")
    try:
        from training import run_training_simulation
        results, agents = run_training_simulation()
        
        print("✅ Simulation Training Completed!")
        print(f"👥 Trained {len(agents)} AI agents with ML capabilities")
        for agent in agents:
            state = agent.get_state()
            print(f"   {agent.name}: Knowledge={state['knowledge_graph_size']}, "
                  f"Status={state['status']}")
        return True
    except Exception as e:
        print(f"❌ Simulation training failed: {e}")
        return False

def run_comprehensive_training():
    """Run comprehensive training with all components"""
    print("🚀 Running Comprehensive Full Training...")
    
    # Track success of each component
    results = {}
    
    # 1. Basic training with test data
    print("\n" + "="*50)
    results['basic'] = run_basic_training()
    
    # 2. Extended training with enhanced ML
    print("\n" + "="*50)
    results['extended'] = run_extended_training()
    
    # 3. Advanced training with multiple models
    print("\n" + "="*50)
    results['advanced'] = run_advanced_training()
    
    # 4. Kaggle dataset training
    print("\n" + "="*50)
    results['kaggle'] = run_kaggle_training()
    
    # 5. Agent simulation training
    print("\n" + "="*50)
    results['simulation'] = run_simulation_training()
    
    # 6. Summary
    print("\n" + "="*50)
    print("🏁 COMPREHENSIVE TRAINING SUMMARY")
    print("="*50)
    
    total_success = 0
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {component.upper()}: {status}")
        if success:
            total_success += 1
    
    success_rate = total_success / len(results)
    if success_rate == 1.0:
        print(f"\n🎉 COMPREHENSIVE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    elif success_rate >= 0.6:
        print(f"\n⚠️  PARTIAL SUCCESS - Some components failed")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    else:
        print(f"\n💥 TRAINING FAILED - Most components failed")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    
    return success_rate >= 0.6

def run_extended_training():
    """Run extended training with enhanced ML capabilities"""
    print("🔬 Running Extended Training with Enhanced ML...")
    try:
        from training import AlzheimerTrainer, ExtendedTrainingRunner
        
        # Initialize extended trainer
        trainer = AlzheimerTrainer()
        extended_runner = ExtendedTrainingRunner(trainer)
        
        # Run extended training with cross-validation and hyperparameter tuning
        results = extended_runner.run_extended_training()
        
        print("✅ Extended Training Completed!")
        print(f"📊 Best Cross-Val Score: {results['best_cv_score']:.3f}")
        print(f"📊 Best Parameters: {results['best_params']}")
        print(f"📊 Model Performance: {results['final_accuracy']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Extended training failed: {e}")
        return False

def run_advanced_training():
    """Run advanced training with multiple models and ensemble methods"""
    print("🚀 Running Advanced Training with Multiple Models...")
    try:
        from training import AlzheimerTrainer, AdvancedTrainingRunner
        
        # Initialize advanced trainer
        trainer = AlzheimerTrainer()
        advanced_runner = AdvancedTrainingRunner(trainer)
        
        # Run advanced training with multiple models
        results = advanced_runner.run_advanced_training()
        
        print("✅ Advanced Training Completed!")
        print(f"📊 Models Trained: {len(results['model_results'])}")
        print(f"📊 Best Model: {results['best_model']}")
        print(f"📊 Best Accuracy: {results['best_accuracy']:.3f}")
        print(f"📊 Ensemble Accuracy: {results['ensemble_accuracy']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Advanced training failed: {e}")
        return False

def run_medical_ai_training():
    """Run medical AI focused training"""
    print("🏥 Running Medical AI Training...")
    try:
        # Use the comprehensive medical training
        result = subprocess.run([
            sys.executable, 
            "files/training/comprehensive_medical_ai_training.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Medical AI Training Completed!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Medical AI training failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Medical AI training error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Full Training Pipeline for DuetMind Adaptive")
    parser.add_argument(
        "--mode", 
        choices=["basic", "kaggle", "simulation", "comprehensive", "medical", "extended", "advanced"],
        default="comprehensive",
        help="Training mode: basic (test data), kaggle (real data), simulation (agents), comprehensive (all), medical (medical AI), extended (enhanced ML), advanced (multiple models)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🧠 DuetMind Adaptive - Full Training System")
    print(f"🎯 Mode: {args.mode.upper()}")
    print("="*60)
    
    try:
        if args.mode == "basic":
            success = run_basic_training()
        elif args.mode == "kaggle":
            success = run_kaggle_training()
        elif args.mode == "simulation":
            success = run_simulation_training()
        elif args.mode == "comprehensive":
            success = run_comprehensive_training()
        elif args.mode == "medical":
            success = run_medical_ai_training()
        elif args.mode == "extended":
            success = run_extended_training()
        elif args.mode == "advanced":
            success = run_advanced_training()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return False
        
        if success:
            print(f"\n🎊 SUCCESS: {args.mode.capitalize()} training completed!")
        else:
            print(f"\n💔 FAILURE: {args.mode.capitalize()} training failed!")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)