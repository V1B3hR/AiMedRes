#!/usr/bin/env python3
"""
Main entry point for DuetMind Adaptive System
Supports both comprehensive training and simulation modes
"""

import argparse
import sys
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DuetMindMain")

def run_comprehensive_training() -> bool:
    """
    Run comprehensive training of the adaptive neural network system
    """
    logger.info("=== Starting Comprehensive Training ===")
    
    try:
        # Import training modules
        from comprehensive_training import ComprehensiveTrainer
        
        # Initialize trainer with configuration
        config = {
            'network_size': 50,
            'training_epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'adaptive_learning': True,
            'biological_cycles': True,
            'multi_agent_training': True
        }
        
        trainer = ComprehensiveTrainer(config)
        
        # Run training phases
        logger.info("Phase 1: Neural Network Foundation Training")
        neural_metrics = trainer.train_neural_foundation()
        
        logger.info("Phase 2: Adaptive Behavior Training")
        adaptive_metrics = trainer.train_adaptive_behaviors()
        
        logger.info("Phase 3: Multi-Agent Coordination Training")
        coordination_metrics = trainer.train_multi_agent_coordination()
        
        logger.info("Phase 4: Biological Cycle Integration Training")
        biological_metrics = trainer.train_biological_integration()
        
        # Consolidate and save results
        training_results = {
            'neural_foundation': neural_metrics,
            'adaptive_behaviors': adaptive_metrics,
            'coordination': coordination_metrics,
            'biological_integration': biological_metrics,
            'overall_success': True
        }
        
        trainer.save_trained_models()
        trainer.generate_training_report(training_results)
        
        logger.info("=== Comprehensive Training Complete ===")
        logger.info(f"Final training accuracy: {training_results.get('final_accuracy', 'N/A')}")
        logger.info(f"Models saved to: {trainer.model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_simulation() -> bool:
    """
    Run the adaptive labyrinth simulation
    """
    logger.info("=== Starting Adaptive Labyrinth Simulation ===")
    
    try:
        # Import simulation from existing neuralnet module
        from neuralnet import run_labyrinth_simulation
        
        # Run the simulation
        run_labyrinth_simulation()
        
        logger.info("=== Simulation Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return False

def run_both_modes() -> bool:
    """
    Run both training and simulation in sequence
    """
    logger.info("=== Running Both Training and Simulation ===")
    
    # Run training first
    training_success = run_comprehensive_training()
    if not training_success:
        logger.error("Training failed, skipping simulation")
        return False
    
    # Wait a moment between phases
    time.sleep(2)
    
    # Run simulation
    simulation_success = run_simulation()
    
    return training_success and simulation_success

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="DuetMind Adaptive System - Training and Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode training     # Run comprehensive training only
  python main.py --mode simulation   # Run simulation only  
  python main.py --mode both         # Run both training and simulation
  python main.py                     # Interactive mode (default)
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['training', 'simulation', 'both', 'interactive'],
        default='interactive',
        help='Execution mode (default: interactive)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON/YAML)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("DuetMind Adaptive System Starting...")
    logger.info(f"Mode: {args.mode}")
    
    success = False
    
    if args.mode == 'training':
        success = run_comprehensive_training()
    elif args.mode == 'simulation':
        success = run_simulation()
    elif args.mode == 'both':
        success = run_both_modes()
    elif args.mode == 'interactive':
        # Interactive mode - ask user what to do
        print("\n=== DuetMind Adaptive System ===")
        print("Please select an option:")
        print("1. Run comprehensive training")
        print("2. Run simulation")
        print("3. Run both training and simulation")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                if choice == '1':
                    success = run_comprehensive_training()
                    break
                elif choice == '2':
                    success = run_simulation()
                    break
                elif choice == '3':
                    success = run_both_modes()
                    break
                elif choice == '4':
                    print("Exiting...")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    
    # Return appropriate exit code
    if success:
        logger.info("Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()