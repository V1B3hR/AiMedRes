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
        # Use our existing full training system
        import subprocess
        import sys
        
        logger.info("Running comprehensive training via full_training.py")
        
        # Run the comprehensive training
        result = subprocess.run([
            sys.executable, 
            "full_training.py", 
            "--mode", "comprehensive"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("=== Comprehensive Training Complete ===")
            logger.info("All training components completed successfully")
            return True
        else:
            logger.error(f"Training failed: {result.stderr}")
            return False
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_simulation() -> bool:
    """
    Run the adaptive labyrinth simulation
    """
    logger.info("=== Starting Adaptive Labyrinth Simulation ===")
    
    try:
        # Import simulation from new authoritative module
        from labyrinth_simulation import run_labyrinth_simulation
        
        # Run the simulation
        result = run_labyrinth_simulation()
        
        logger.info("=== Simulation Complete ===")
        logger.info(f"Simulation completed {result['total_steps']} steps with {result['maze_master_interventions']} interventions")
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

def run_comprehensive_medical_ai() -> bool:
    """
    Run comprehensive training + simulation (medical AI context) with labyrinth components
    This implements the exact problem statement requirements
    """
    logger.info("=== Running Comprehensive Medical AI Training + Simulation ===")
    
    try:
        from files.files.training.comprehensive_training_simulation import MedicalAIComprehensiveSystem
        
        # Initialize and run the comprehensive system
        system = MedicalAIComprehensiveSystem()
        result = system.run_comprehensive_system()
        
        if result["status"] == "success":
            logger.info("=== Comprehensive Medical AI System Complete ===")
            logger.info(f"Training accuracy: {result['training_phase']['accuracy']:.3f}")
            logger.info(f"Simulation health score: {result['simulation_phase']['health_score']:.3f}")
            logger.info(f"Labyrinth components imported and functional: {result['system_integration']['labyrinth_components_imported']}")
            return True
        else:
            logger.error(f"Comprehensive system failed in {result.get('phase', 'unknown')} phase")
            return False
        
    except Exception as e:
        logger.error(f"Comprehensive medical AI system failed: {e}")
        return False

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="DuetMind Adaptive System - Training and Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode training       # Run comprehensive training only
  python main.py --mode simulation     # Run simulation only  
  python main.py --mode both           # Run both training and simulation
  python main.py --mode comprehensive  # Run comprehensive medical AI training + simulation
  python main.py                       # Interactive mode (default)
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['training', 'simulation', 'both', 'comprehensive', 'interactive'],
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
    elif args.mode == 'comprehensive':
        success = run_comprehensive_medical_ai()
    elif args.mode == 'interactive':
        # Interactive mode - ask user what to do
        print("\n=== DuetMind Adaptive System ===")
        print("Please select an option:")
        print("1. Run comprehensive training")
        print("2. Run simulation")
        print("3. Run both training and simulation")
        print("4. Run comprehensive medical AI training + simulation")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
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
                    success = run_comprehensive_medical_ai()
                    break
                elif choice == '5':
                    print("Exiting...")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
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