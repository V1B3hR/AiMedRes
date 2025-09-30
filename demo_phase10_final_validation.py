#!/usr/bin/env python3
"""
Demo script for Phase 10: Final Model & System Validation

This script demonstrates how to use the Phase 10 debugging functionality
to perform final validation on trained models.

Usage:
    python demo_phase10_final_validation.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from debug.phase10_final_validation import Phase10FinalValidation


def main():
    """Run Phase 10 final validation demo"""
    print("=" * 70)
    print("Phase 10: Final Model & System Validation - Demo")
    print("=" * 70)
    print()
    
    print("This demo will:")
    print("  1. Validate models on held-out test data")
    print("  2. Perform end-to-end pipeline tests")
    print("  3. Document findings and next steps")
    print()
    
    # Create validator instance
    print("Initializing Phase 10 validator...")
    validator = Phase10FinalValidation(
        verbose=False,
        data_source='synthetic'
    )
    
    print("Running Phase 10 validation...")
    print()
    
    # Run Phase 10
    try:
        results = validator.run_phase_10()
        
        print()
        print("=" * 70)
        print("Demo Results Summary")
        print("=" * 70)
        print()
        
        # Print key results
        summary = results.get('summary', {})
        
        print(f"✅ Phase 10 completed successfully!")
        print(f"   • Models validated: {summary.get('total_models_validated', 0)}")
        print(f"   • Average test accuracy: {summary.get('average_test_accuracy', 0):.3f}")
        print(f"   • Best model: {summary.get('best_model', 'N/A')}")
        print(f"   • Pipeline tests passed: {summary.get('pipeline_tests_passed', 0)}/{summary.get('pipeline_tests_passed', 0) + summary.get('pipeline_tests_failed', 0)}")
        print(f"   • Execution time: {summary.get('execution_time_seconds', 0):.2f}s")
        print()
        
        # Show where results are saved
        results_file = Path(__file__).parent / "debug" / "phase10_results.json"
        print(f"📁 Results saved to: {results_file}")
        print()
        
        # Show next steps
        print("📝 Next Steps:")
        next_steps = results.get('subphase_10_3', {}).get('next_steps', [])
        for i, step in enumerate(next_steps[:5], 1):  # Show first 5
            print(f"   {i}. {step}")
        
        if len(next_steps) > 5:
            print(f"   ... and {len(next_steps) - 5} more (see results file)")
        
        print()
        print("=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Demo failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
