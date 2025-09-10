#!/usr/bin/env python3
"""
Command-line interface for running training in duetmind_adaptive
"""

import argparse
import sys
import logging
from pathlib import Path
from training import AlzheimerTrainer, run_training_simulation

def main():
    parser = argparse.ArgumentParser(description="Run training for duetmind_adaptive")
    parser.add_argument(
        "--data-path", 
        type=str, 
        help="Path to Alzheimer dataset CSV file (optional, will use test data if not provided)"
    )
    parser.add_argument(
        "--model-output", 
        type=str, 
        default="alzheimer_model.pkl",
        help="Path to save the trained model (default: alzheimer_model.pkl)"
    )
    parser.add_argument(
        "--mode", 
        choices=["train", "simulate", "both"],
        default="both",
        help="Mode: train (only train model), simulate (full simulation), both (default)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    if args.mode in ["train", "both"]:
        print("🧠 Starting DuetMind Adaptive Training...")
        
        # Initialize trainer
        trainer = AlzheimerTrainer(data_path=args.data_path)
        
        # Load and preprocess data
        df = trainer.load_data()
        X, y = trainer.preprocess_data(df)
        
        # Train model
        model, results = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model(args.model_output)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Training Accuracy: {results['train_accuracy']:.3f}")
        print(f"📊 Test Accuracy: {results['test_accuracy']:.3f}")
        print(f"💾 Model saved to: {args.model_output}")
        
        # Show top features
        print("\n🔍 Top Features by Importance:")
        top_features = sorted(results['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        for feature, importance in top_features:
            print(f"   {feature}: {importance:.3f}")
    
    if args.mode in ["simulate", "both"]:
        print("\n🤖 Running Full Training Simulation...")
        results, agents = run_training_simulation()
        print("✅ Simulation completed!")
        
        # Show agent states
        print(f"\n👥 Trained {len(agents)} AI agents with ML capabilities")
        for agent in agents:
            state = agent.get_state()
            print(f"   {agent.name}: Knowledge={state['knowledge_graph_size']}, "
                  f"Status={state['status']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)