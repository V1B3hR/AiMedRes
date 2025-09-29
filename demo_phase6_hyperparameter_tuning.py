#!/usr/bin/env python3
"""
Demo Phase 6 Hyperparameter Tuning Implementation

This script demonstrates the Phase 6 hyperparameter tuning functionality
as specified in debuglist.md with comprehensive examples.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.phase6_hyperparameter_tuning import Phase6HyperparameterTuning, HyperparameterIdentifier, HyperparameterSearcher
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase6Demo")


def demo_hyperparameter_identification():
    """Demonstrate hyperparameter identification (Subphase 6.1)"""
    
    print("\n🔍 Phase 6.1 Demo: Hyperparameter Identification")
    print("-" * 50)
    
    identifier = HyperparameterIdentifier()
    
    # Demonstrate identification for all supported model types
    model_types = ['random_forest', 'logistic_regression', 'svm', 'mlp', 'decision_tree']
    
    for model_type in model_types:
        print(f"\n📋 {model_type.replace('_', ' ').title()} Hyperparameters:")
        
        # Get hyperparameter space
        params = identifier.get_hyperparameter_space(model_type)
        
        if params:
            for param_name, param_values in params.items():
                print(f"  • {param_name}: {param_values}")
            
            # Get model instance
            model = identifier.get_model_instance(model_type)
            print(f"  ✅ Model instance: {type(model).__name__}")
        else:
            print("  ⚠️  No hyperparameters defined")


def demo_search_methods():
    """Demonstrate different search methods (Subphase 6.2)"""
    
    print("\n🔍 Phase 6.2 Demo: Search Methods Comparison")
    print("-" * 50)
    
    # Create synthetic data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        n_informative=8, random_state=42
    )
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Initialize components
    searcher = HyperparameterSearcher(cv_folds=3, scoring='accuracy')  # Reduced CV for demo speed
    identifier = HyperparameterIdentifier()
    
    # Demonstrate on logistic regression (fast)
    model = identifier.get_model_instance('logistic_regression')
    param_space = identifier.get_hyperparameter_space('logistic_regression')
    
    print("📊 Comparing search methods on Logistic Regression:")
    
    # Grid Search
    grid_result = searcher.grid_search(model, param_space, X_scaled, y_encoded)
    print(f"  Grid Search: {grid_result['best_score']:.4f} in {grid_result['search_time']:.2f}s")
    
    # Random Search  
    random_result = searcher.random_search(model, param_space, X_scaled, y_encoded, n_iter=20)
    print(f"  Random Search: {random_result['best_score']:.4f} in {random_result['search_time']:.2f}s")
    
    # Bayesian Optimization
    try:
        bayesian_result = searcher.bayesian_optimization('logistic_regression', X_scaled, y_encoded, n_trials=20)
        if bayesian_result:
            print(f"  Bayesian Opt: {bayesian_result['best_score']:.4f} in {bayesian_result['search_time']:.2f}s")
    except Exception as e:
        print(f"  Bayesian Opt: Failed ({e})")


def demo_comprehensive_hyperparameter_tuning():
    """Demonstrate comprehensive hyperparameter tuning across multiple models and methods"""
    
    print("\n🚀 Phase 6 Demo: Comprehensive Hyperparameter Tuning")
    print("=" * 60)
    
    # Initialize Phase 6 tuning system
    phase6 = Phase6HyperparameterTuning(output_dir="debug")
    
    # Demo 1: Compare all search methods on Random Forest
    print("\n📋 Demo 1: Random Forest with All Search Methods")
    print("-" * 40)
    
    results_demo1 = phase6.run_hyperparameter_tuning(
        model_types=['random_forest'],
        methods=['grid_search', 'random_search', 'bayesian_optimization']
    )
    
    if results_demo1.get('summary'):
        print(f"🏆 Best Method for Random Forest: {results_demo1['summary']['best_method']}")
        print(f"📈 Best Score: {results_demo1['summary']['best_score']:.4f}")
    
    # Demo 2: Compare multiple models with Grid Search
    print("\n📋 Demo 2: Multiple Models with Grid Search")
    print("-" * 40)
    
    results_demo2 = phase6.run_hyperparameter_tuning(
        model_types=['random_forest', 'logistic_regression', 'decision_tree'],
        methods=['grid_search']
    )
    
    if results_demo2.get('summary'):
        print(f"🏆 Best Model with Grid Search: {results_demo2['summary']['best_model']}")
        print(f"📈 Best Score: {results_demo2['summary']['best_overall_score']:.4f}")
    
    # Demo 3: Fast comparison with Random Search
    print("\n📋 Demo 3: Fast Comparison with Random Search")
    print("-" * 40)
    
    results_demo3 = phase6.run_hyperparameter_tuning(
        model_types=['logistic_regression', 'decision_tree'],
        methods=['random_search']
    )
    
    if results_demo3.get('summary'):
        print(f"🏆 Best Model with Random Search: {results_demo3['summary']['best_model']}")
        print(f"📈 Best Score: {results_demo3['summary']['best_overall_score']:.4f}")


def main():
    """Main demo function"""
    
    print("🎯 AiMedRes Phase 6: Hyperparameter Tuning & Search Demo")
    print("=" * 70)
    
    # Demo hyperparameter identification
    demo_hyperparameter_identification()
    
    # Demo search methods comparison
    demo_search_methods()
    
    # Demo comprehensive tuning
    demo_comprehensive_hyperparameter_tuning()
    
    print("\n" + "=" * 70)
    print("✅ All Phase 6 demos completed successfully!")
    print("🔍 Key files generated:")
    print("  • debug/phase6_results.json - Detailed results")
    print("  • debug/visualizations/ - Comparison plots")
    print("  • debug/phase6_hyperparameter_tuning.py - Main implementation")
    print("=" * 70)


if __name__ == "__main__":
    main()