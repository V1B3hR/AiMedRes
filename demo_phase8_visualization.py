#!/usr/bin/env python3
"""
Demo script for Phase 8: Model Visualization & Interpretability

This demo showcases the Phase 8 debugging capabilities for model interpretability.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from debug.phase8_model_visualization import Phase8ModelVisualization


def main():
    """Run Phase 8 demo with synthetic medical data"""
    
    print("=" * 80)
    print("Phase 8 Demo: Model Visualization & Interpretability")
    print("=" * 80)
    print()
    print("This demo will:")
    print("  1. Generate synthetic medical data (1000 samples)")
    print("  2. Train 3 tree-based models (DecisionTree, RandomForest, GradientBoosting)")
    print("  3. Generate feature importance plots")
    print("  4. Create partial dependence plots for top features")
    print("  5. Display enhanced confusion matrices with metrics")
    print()
    print("Starting Phase 8 analysis...")
    print()
    
    # Create visualizer
    visualizer = Phase8ModelVisualization(verbose=True, data_source='synthetic')
    
    # Run Phase 8
    try:
        results = visualizer.run_phase_8()
        
        print()
        print("=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print()
        print("📊 Summary:")
        print(f"  • Models analyzed: {len(results['models_analyzed'])}")
        print(f"  • Feature importance plots: {len(results['feature_importance']['feature_importance_plots'])}")
        print(f"  • Partial dependence plots: {len(results['partial_dependence']['pdp_plots_generated'])}")
        print(f"  • Confusion matrices: {len(results['confusion_matrices']['models_analyzed']) * 2}")
        print()
        print("📁 Output locations:")
        print(f"  • Results JSON: {visualizer.output_dir / 'phase8_results.json'}")
        print(f"  • Visualizations: {visualizer.visualization_dir}")
        print()
        print("🔍 Key insights:")
        
        # Show feature importance across models
        fi_results = results['feature_importance']
        if fi_results.get('top_features_by_model'):
            all_features = {}
            for model, features_list in fi_results['top_features_by_model'].items():
                for feat_info in features_list[:3]:  # Top 3 from each model
                    feat_name = feat_info['feature']
                    if feat_name not in all_features:
                        all_features[feat_name] = {'models': [], 'avg_importance': 0, 'count': 0}
                    all_features[feat_name]['models'].append(model)
                    all_features[feat_name]['avg_importance'] += feat_info['importance']
                    all_features[feat_name]['count'] += 1
            
            # Calculate average importance
            for feat in all_features:
                all_features[feat]['avg_importance'] /= all_features[feat]['count']
            
            # Sort by average importance
            sorted_features = sorted(all_features.items(), key=lambda x: -x[1]['avg_importance'])
            
            print()
            print("  Top 3 most important features (averaged across models):")
            for i, (feat, info) in enumerate(sorted_features[:3], 1):
                models_str = ', '.join(info['models'])
                print(f"    {i}. {feat}: {info['avg_importance']:.4f} ({models_str})")
        
        print()
        print("✅ Phase 8 demo completed successfully!")
        print()
        print("💡 Tip: View the generated plots in the debug/visualizations/ directory")
        print("    to gain insights into your model's behavior and predictions.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
