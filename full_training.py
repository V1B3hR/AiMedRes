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
        results = trainer.train_model(X, y)
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
    
    # 2. Kaggle dataset training
    print("\n" + "="*50)
    results['kaggle'] = run_kaggle_training()
    
    # 3. Agent simulation training
    print("\n" + "="*50)
    results['simulation'] = run_simulation_training()
    
    # 4. Summary
    print("\n" + "="*50)
    print("🏁 FULL TRAINING SUMMARY")
    print("="*50)
    
    total_success = 0
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {component.upper()}: {status}")
        if success:
            total_success += 1
    
    success_rate = total_success / len(results)
    if success_rate == 1.0:
        print(f"\n🎉 FULL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    elif success_rate >= 0.5:
        print(f"\n⚠️  PARTIAL SUCCESS - Some components failed")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    else:
        print(f"\n💥 TRAINING FAILED - Most components failed")
        print(f"📊 Success Rate: {success_rate*100:.0f}% ({total_success}/{len(results)} components)")
    
    return success_rate >= 0.5

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
        choices=["basic", "kaggle", "simulation", "comprehensive", "medical"],
        default="comprehensive",
        help="Training mode: basic (test data), kaggle (real data), simulation (agents), comprehensive (all), medical (medical AI)"
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