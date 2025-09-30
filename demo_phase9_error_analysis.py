#!/usr/bin/env python3
"""
Demo script for Phase 9: Error Analysis & Edge Cases

This script demonstrates the Phase 9 debugging capabilities with a simple interface.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from debug.phase9_error_analysis_edge_cases import Phase9ErrorAnalysis


def main():
    """Run Phase 9 demonstration"""
    print("=" * 80)
    print("PHASE 9 DEMO: ERROR ANALYSIS & EDGE CASES")
    print("=" * 80)
    print()
    print("This demo will:")
    print("  1. Generate synthetic medical data")
    print("  2. Train classification models (DecisionTree, RandomForest, GradientBoosting)")
    print("  3. Analyze misclassified samples and error patterns")
    print("  4. Investigate model bias across classes")
    print("  5. Test robustness on edge cases and adversarial examples")
    print("  6. Generate comprehensive visualizations")
    print()
    print("=" * 80)
    print()
    
    # Create analyzer
    analyzer = Phase9ErrorAnalysis(verbose=True, data_source='synthetic')
    
    try:
        # Run Phase 9
        print("Starting Phase 9 analysis...\n")
        results = analyzer.run_phase_9()
        
        print("\n" + "=" * 80)
        print("✅ PHASE 9 DEMO COMPLETE!")
        print("=" * 80)
        print()
        print("Results saved to:")
        print(f"  • JSON: {analyzer.output_dir / 'phase9_results.json'}")
        print(f"  • Visualizations: {analyzer.visualization_dir}")
        print()
        print("Generated visualizations:")
        print("  • 3 error distribution plots (confusion matrices, error rates, residuals)")
        print("  • 3 bias analysis plots (class performance, distributions, disparities)")
        print("  • 3 adversarial testing plots (robustness, consistency, stability)")
        print()
        
        # Print key findings
        summary = results['summary']
        print("Key Findings:")
        print(f"  • Average error rate: {summary.get('avg_error_rate', 0):.1%}")
        print(f"  • Average balanced accuracy: {summary.get('avg_balanced_accuracy', 0):.3f}")
        print(f"  • Bias detected: {'Yes' if summary.get('bias_detected', False) else 'No'}")
        print(f"  • Average robustness score: {summary.get('avg_robustness_score', 0):.3f}")
        print()
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Phase 9 demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
