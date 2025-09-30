#!/usr/bin/env python3
"""
Demo script for Phase 7: Model Training & Evaluation

This script demonstrates the Phase 7 debugging capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from debug.phase7_model_training_evaluation import Phase7ModelTrainingEvaluator


def main():
    """Run Phase 7 demo"""
    print("=" * 70)
    print("DEMO: Phase 7 - Model Training & Evaluation")
    print("=" * 70)
    print()
    
    # Create evaluator
    evaluator = Phase7ModelTrainingEvaluator(verbose=True, data_source="synthetic")
    
    # Run Phase 7
    print("Running Phase 7 debugging process...")
    print()
    
    try:
        results = evaluator.run_phase_7()
        
        print()
        print("=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Key Results:")
        print("-" * 70)
        
        # Show comparison summary
        comparison = results['comparison_results'].get('comparison_summary', {})
        if comparison:
            print(f"✓ Best by Accuracy: {comparison['best_by_accuracy']['model']} "
                  f"({comparison['best_by_accuracy']['test_accuracy']:.3f})")
            print(f"✓ Best by F1 Score: {comparison['best_by_f1']['model']} "
                  f"({comparison['best_by_f1']['test_f1']:.3f})")
            print(f"✓ Least Overfitting: {comparison['least_overfit']['model']} "
                  f"(gap: {comparison['least_overfit']['overfitting_gap']:.3f})")
        
        print()
        print(f"✓ Total models trained: {len(evaluator.trained_models)}")
        print(f"✓ Baseline models: {len(evaluator.baseline_models)}")
        print()
        print(f"📊 Results saved to: {evaluator.output_dir / 'phase7_results.json'}")
        print(f"📈 Visualizations in: {evaluator.visualization_dir}")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("DEMO FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
