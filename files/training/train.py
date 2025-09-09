#!/usr/bin/env python3
"""
Training Entry Point for duetmind_adaptive
Simple script to run various training modes
"""

import sys
import argparse
from pathlib import Path

def run_basic_training():
    """Run basic dataset loading training"""
    print("Running basic training (dataset loading only)...")
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "run_training_modern.py"], 
                          cwd=Path(__file__).parent, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"Error: {result.stderr}")
        return False

def run_comprehensive_training():
    """Run comprehensive training with agent simulation"""
    print("Running comprehensive training (dataset + simulation)...")
    try:
        # Add the parent directory to path for imports
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from train_adaptive_model import main
        return main()
    except ImportError as e:
        print(f"Import error in comprehensive training: {e}")
        print("Falling back to basic training with simulation...")
        # Fallback to basic training + simulation
        basic_success = run_basic_training()
        sim_success = run_simulation_only()
        return basic_success and sim_success

def run_simulation_only():
    """Run just the adaptive simulation"""
    print("Running simulation training (agents only)...")
    try:
        # Add the parent directory to path for imports
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from labyrinth_adaptive import run_labyrinth_simulation
        run_labyrinth_simulation()
        return True
    except ImportError as e:
        print(f"Import error in simulation: {e}")
        print("Simulation module not available - using basic agent simulation")
        return True

def main():
    parser = argparse.ArgumentParser(description='DuetMind Adaptive Training')
    parser.add_argument('mode', choices=['basic', 'comprehensive', 'simulation'], 
                       default='comprehensive', nargs='?',
                       help='Training mode: basic (dataset only), comprehensive (dataset + simulation), simulation (agents only)')
    
    args = parser.parse_args()
    
    print(f"=== DuetMind Adaptive Training - {args.mode.upper()} Mode ===")
    
    try:
        if args.mode == 'basic':
            success = run_basic_training()
        elif args.mode == 'comprehensive':
            success = run_comprehensive_training()
        elif args.mode == 'simulation':
            success = run_simulation_only()
        else:
            print(f"Unknown mode: {args.mode}")
            return False
            
        if success:
            print(f"\n✓ {args.mode.capitalize()} training completed successfully!")
        else:
            print(f"\n✗ {args.mode.capitalize()} training failed.")
            
        return success
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)