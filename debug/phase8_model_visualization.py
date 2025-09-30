#!/usr/bin/env python3
"""
Phase 8 Debugging Script: Model Visualization & Interpretability

This script implements Phase 8 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 8.1: Generate feature importance plots for tree-based models
- Subphase 8.2: Plot partial dependence for key features
- Subphase 8.3: Display confusion matrices for classifiers (precision, recall, F1)

Usage:
    python debug/phase8_model_visualization.py [--data-source] [--verbose]
"""

import sys
import os
import json
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))


class Phase8ModelVisualization:
    """Phase 8 debugging implementation for AiMedRes model visualization & interpretability"""
    
    def __init__(self, verbose: bool = False, data_source: str = "synthetic"):
        self.verbose = verbose
        self.data_source = data_source
        self.results = {}
        self.repo_root = Path(__file__).parent.parent
        self.output_dir = self.repo_root / "debug"
        self.visualization_dir = self.output_dir / "visualizations"
        self.visualization_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='[%(asctime)s] 🎨 %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("Phase8Visualization")
        
        # Load Phase 7 results if available
        self.phase7_results = None
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🎨" if level == "INFO" else "⚠️ " if level == "WARN" else "❌"
        print(f"[{timestamp}] {prefix} {message}")
        if self.verbose and level != "INFO":
            self.logger.info(message)

    def load_phase7_results(self) -> bool:
        """Load Phase 7 results and trained models"""
        self.log("Loading Phase 7 results...")
        
        phase7_path = self.output_dir / "phase7_results.json"
        if not phase7_path.exists():
            self.log("Phase 7 results not found. Running Phase 7 first...", "WARN")
            return False
        
        try:
            with open(phase7_path, 'r') as f:
                self.phase7_results = json.load(f)
            
            self.log(f"✓ Loaded Phase 7 results from {phase7_path}")
            self.log(f"  Data source: {self.phase7_results.get('data_source', 'unknown')}")
            self.log(f"  Models trained: {len(self.phase7_results.get('cv_results', {}).get('cv_results', {}))}")
            return True
            
        except Exception as e:
            self.log(f"Failed to load Phase 7 results: {e}", "ERROR")
            return False

    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
        """Generate synthetic medical data for testing"""
        np.random.seed(42)
        
        # Create features similar to medical datasets
        age = np.random.normal(65, 15, n_samples).clip(18, 95)
        bmi = np.random.normal(26, 5, n_samples).clip(15, 50)
        blood_pressure = np.random.normal(130, 20, n_samples).clip(90, 200)
        cholesterol = np.random.normal(200, 40, n_samples).clip(100, 400)
        glucose = np.random.normal(100, 30, n_samples).clip(70, 300)
        heart_rate = np.random.normal(75, 15, n_samples).clip(50, 120)
        
        # Create target based on features with some noise
        risk_score = (
            (age - 65) / 15 * 0.3 +
            (bmi - 26) / 5 * 0.2 +
            (blood_pressure - 130) / 20 * 0.2 +
            (cholesterol - 200) / 40 * 0.15 +
            (glucose - 100) / 30 * 0.15 +
            np.random.normal(0, 0.3, n_samples)
        )
        target = (risk_score > 0).astype(int)
        
        data = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'heart_rate': heart_rate,
            'target': target
        })
        
        return data, 'target'

    def load_data(self) -> Tuple[pd.DataFrame, str]:
        """Load data based on data_source"""
        self.log(f"Loading data from source: {self.data_source}")
        
        if self.data_source == "synthetic":
            return self.generate_synthetic_data()
        else:
            self.log(f"Loading from file: {self.data_source}")
            data = pd.read_csv(self.data_source)
            # Assume last column is target
            target_col = data.columns[-1]
            return data, target_col

    def prepare_data_and_models(self) -> bool:
        """Prepare data and train models for visualization"""
        self.log("Preparing data and models...")
        
        # Load data
        data, target_col = self.load_data()
        
        # Prepare features and target
        if target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            self.log(f"Encoding categorical features: {categorical_columns}")
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        self.log(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Convert back to DataFrame for easier handling
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        # Train tree-based models if not already available
        self.log("Training tree-based models for visualization...")
        
        tree_models = {
            'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for name, model in tree_models.items():
            self.log(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, y_pred)
            self.trained_models[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': test_acc
            }
            self.log(f"  ✓ {name} trained (Test Accuracy: {test_acc:.3f})")
        
        return True

    def subphase_8_1_feature_importance(self) -> Dict[str, Any]:
        """
        Subphase 8.1: Generate feature importance plots for tree-based models
        """
        self.log("\n" + "=" * 70)
        self.log("SUBPHASE 8.1: FEATURE IMPORTANCE FOR TREE-BASED MODELS")
        self.log("=" * 70)
        
        results = {
            'models_analyzed': [],
            'feature_importance_plots': [],
            'top_features_by_model': {}
        }
        
        for name, model_info in self.trained_models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                self.log(f"\nAnalyzing feature importance for {name}...")
                
                try:
                    # Get feature importances
                    importances = model.feature_importances_
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Save to CSV
                    csv_path = self.visualization_dir / f"feature_importance_{name}.csv"
                    importance_df.to_csv(csv_path, index=False)
                    self.log(f"  ✓ Saved feature importance data to {csv_path}")
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    top_n = min(15, len(importance_df))
                    top_features = importance_df.head(top_n)
                    
                    colors = sns.color_palette("viridis", top_n)
                    plt.barh(range(top_n), top_features['importance'].values, color=colors)
                    plt.yticks(range(top_n), top_features['feature'].values)
                    plt.xlabel('Importance', fontsize=12)
                    plt.ylabel('Feature', fontsize=12)
                    plt.title(f'Top {top_n} Feature Importances - {name}', fontsize=14, fontweight='bold')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.visualization_dir / f"feature_importance_{name}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.log(f"  ✓ Saved feature importance plot to {plot_path}")
                    
                    # Store results
                    results['models_analyzed'].append(name)
                    results['feature_importance_plots'].append(str(plot_path))
                    results['top_features_by_model'][name] = importance_df.head(5).to_dict('records')
                    
                    # Log top features
                    self.log(f"  Top 5 features for {name}:")
                    for idx, row in importance_df.head(5).iterrows():
                        self.log(f"    {row['feature']}: {row['importance']:.4f}")
                    
                except Exception as e:
                    self.log(f"  ✗ Failed to generate feature importance for {name}: {e}", "WARN")
        
        self.log(f"\n✅ Subphase 8.1 Complete: Analyzed {len(results['models_analyzed'])} models")
        return results

    def subphase_8_2_partial_dependence(self) -> Dict[str, Any]:
        """
        Subphase 8.2: Plot partial dependence for key features
        """
        self.log("\n" + "=" * 70)
        self.log("SUBPHASE 8.2: PARTIAL DEPENDENCE PLOTS")
        self.log("=" * 70)
        
        results = {
            'models_analyzed': [],
            'pdp_plots_generated': [],
            'features_analyzed': []
        }
        
        for name, model_info in self.trained_models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                self.log(f"\nGenerating partial dependence plots for {name}...")
                
                try:
                    # Get top 4 most important features
                    importances = model.feature_importances_
                    top_indices = np.argsort(importances)[-4:][::-1]
                    top_features = [self.feature_names[i] for i in top_indices]
                    
                    self.log(f"  Analyzing features: {', '.join(top_features)}")
                    
                    # Create 1D partial dependence plots
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes = axes.ravel()
                    
                    for idx, feature_idx in enumerate(top_indices):
                        feature_name = self.feature_names[feature_idx]
                        
                        # Calculate partial dependence
                        pd_result = partial_dependence(
                            model, self.X_train, [feature_idx], 
                            kind='average', grid_resolution=50
                        )
                        
                        # Plot
                        axes[idx].plot(pd_result['grid_values'][0], pd_result['average'][0], 
                                      linewidth=2, color='steelblue')
                        axes[idx].set_xlabel(feature_name, fontsize=10)
                        axes[idx].set_ylabel('Partial Dependence', fontsize=10)
                        axes[idx].set_title(f'PDP: {feature_name}', fontsize=11, fontweight='bold')
                        axes[idx].grid(alpha=0.3)
                    
                    plt.suptitle(f'Partial Dependence Plots - {name}', 
                               fontsize=14, fontweight='bold', y=1.00)
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = self.visualization_dir / f"partial_dependence_{name}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.log(f"  ✓ Saved 1D partial dependence plots to {plot_path}")
                    results['pdp_plots_generated'].append(str(plot_path))
                    
                    # Create 2D partial dependence plot for top 2 features
                    if len(top_indices) >= 2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        display = PartialDependenceDisplay.from_estimator(
                            model, self.X_train, 
                            [tuple(top_indices[:2])],
                            ax=ax,
                            grid_resolution=30
                        )
                        
                        plt.suptitle(f'2D Partial Dependence - {name}\n{top_features[0]} vs {top_features[1]}', 
                                   fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        
                        # Save plot
                        plot_path_2d = self.visualization_dir / f"partial_dependence_2d_{name}.png"
                        plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        self.log(f"  ✓ Saved 2D partial dependence plot to {plot_path_2d}")
                        results['pdp_plots_generated'].append(str(plot_path_2d))
                    
                    results['models_analyzed'].append(name)
                    results['features_analyzed'].extend(top_features)
                    
                except Exception as e:
                    self.log(f"  ✗ Failed to generate partial dependence plots for {name}: {e}", "WARN")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
        
        self.log(f"\n✅ Subphase 8.2 Complete: Generated {len(results['pdp_plots_generated'])} plots")
        return results

    def subphase_8_3_confusion_matrices(self) -> Dict[str, Any]:
        """
        Subphase 8.3: Display confusion matrices for classifiers (precision, recall, F1)
        """
        self.log("\n" + "=" * 70)
        self.log("SUBPHASE 8.3: ENHANCED CONFUSION MATRICES")
        self.log("=" * 70)
        
        results = {
            'models_analyzed': [],
            'confusion_matrices': {},
            'classification_reports': {}
        }
        
        for name, model_info in self.trained_models.items():
            self.log(f"\nGenerating confusion matrix for {name}...")
            
            try:
                y_pred = model_info['predictions']
                
                # Calculate confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Calculate per-class metrics
                report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
                
                # Create enhanced confusion matrix visualization
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Confusion Matrix with counts
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                           cbar_kws={'label': 'Count'})
                axes[0].set_xlabel('Predicted Label', fontsize=12)
                axes[0].set_ylabel('True Label', fontsize=12)
                axes[0].set_title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
                
                # Plot 2: Normalized Confusion Matrix (percentages)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                           cbar_kws={'label': 'Percentage'})
                axes[1].set_xlabel('Predicted Label', fontsize=12)
                axes[1].set_ylabel('True Label', fontsize=12)
                axes[1].set_title(f'Normalized Confusion Matrix - {name}', fontsize=14, fontweight='bold')
                
                plt.suptitle(f'Confusion Matrix Analysis - {name}\nAccuracy: {model_info["accuracy"]:.3f}',
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                
                # Save plot
                plot_path = self.visualization_dir / f"confusion_matrix_{name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.log(f"  ✓ Saved confusion matrix to {plot_path}")
                
                # Create detailed metrics visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract class-wise metrics
                classes = sorted([k for k in report.keys() if k.isdigit()])
                metrics = ['precision', 'recall', 'f1-score']
                
                x = np.arange(len(classes))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    values = [report[cls][metric] for cls in classes]
                    ax.bar(x + i * width, values, width, label=metric.capitalize())
                
                ax.set_xlabel('Class', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.set_title(f'Per-Class Metrics - {name}', fontsize=14, fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels(classes)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, 1.1)
                
                plt.tight_layout()
                
                # Save metrics plot
                metrics_path = self.visualization_dir / f"classification_metrics_{name}.png"
                plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.log(f"  ✓ Saved classification metrics to {metrics_path}")
                
                # Log metrics
                self.log(f"  Overall metrics for {name}:")
                self.log(f"    Accuracy:  {report['accuracy']:.3f}")
                self.log(f"    Macro Avg - Precision: {report['macro avg']['precision']:.3f}, "
                        f"Recall: {report['macro avg']['recall']:.3f}, "
                        f"F1: {report['macro avg']['f1-score']:.3f}")
                
                # Store results
                results['models_analyzed'].append(name)
                results['confusion_matrices'][name] = {
                    'matrix': cm.tolist(),
                    'normalized': cm_normalized.tolist(),
                    'plot_path': str(plot_path)
                }
                results['classification_reports'][name] = {
                    'accuracy': report['accuracy'],
                    'macro_avg': report['macro avg'],
                    'weighted_avg': report['weighted avg'],
                    'per_class': {cls: report[cls] for cls in classes}
                }
                
            except Exception as e:
                self.log(f"  ✗ Failed to generate confusion matrix for {name}: {e}", "WARN")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        self.log(f"\n✅ Subphase 8.3 Complete: Analyzed {len(results['models_analyzed'])} models")
        return results

    def run_phase_8(self) -> Dict[str, Any]:
        """Run complete Phase 8 debugging process"""
        self.log("🚀 Starting Phase 8: Model Visualization & Interpretability")
        
        # Prepare data and models
        if not self.prepare_data_and_models():
            raise RuntimeError("Failed to prepare data and models")
        
        # Run all subphases
        feature_importance_results = self.subphase_8_1_feature_importance()
        partial_dependence_results = self.subphase_8_2_partial_dependence()
        confusion_matrix_results = self.subphase_8_3_confusion_matrices()
        
        # Compile all results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'data_source': self.data_source,
            'data_shape': {
                'n_samples': len(self.X_train) + len(self.X_test),
                'n_features': len(self.feature_names),
                'n_classes': len(np.unique(self.y_test))
            },
            'models_analyzed': list(self.trained_models.keys()),
            'feature_importance': feature_importance_results,
            'partial_dependence': partial_dependence_results,
            'confusion_matrices': confusion_matrix_results
        }
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results

    def save_results(self, results: Dict[str, Any]):
        """Save Phase 8 results to JSON file"""
        results_path = self.output_dir / "phase8_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"💾 Results saved to {results_path}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of Phase 8 results"""
        self.log("=" * 70)
        self.log("PHASE 8 SUMMARY: MODEL VISUALIZATION & INTERPRETABILITY")
        self.log("=" * 70)
        
        self.log(f"\n📊 Models Analyzed: {len(results['models_analyzed'])}")
        for model_name in results['models_analyzed']:
            acc = self.trained_models[model_name]['accuracy']
            self.log(f"  • {model_name}: Accuracy {acc:.3f}")
        
        self.log(f"\n🎨 Visualizations Generated:")
        
        # Feature importance
        fi_results = results['feature_importance']
        self.log(f"  • Feature Importance Plots: {len(fi_results['feature_importance_plots'])}")
        
        # Partial dependence
        pd_results = results['partial_dependence']
        self.log(f"  • Partial Dependence Plots: {len(pd_results['pdp_plots_generated'])}")
        
        # Confusion matrices
        cm_results = results['confusion_matrices']
        self.log(f"  • Confusion Matrix Visualizations: {len(cm_results['models_analyzed']) * 2}")
        
        self.log(f"\n📁 All visualizations saved to: {self.visualization_dir}")
        
        # Highlight top features across models
        self.log(f"\n🔍 Key Insights:")
        top_features = fi_results.get('top_features_by_model', {})
        if top_features:
            all_features = {}
            for model, features_list in top_features.items():
                for feat_info in features_list:
                    feat_name = feat_info['feature']
                    if feat_name not in all_features:
                        all_features[feat_name] = []
                    all_features[feat_name].append(model)
            
            # Find features important across multiple models
            common_features = {f: models for f, models in all_features.items() if len(models) > 1}
            if common_features:
                self.log("  • Features important across multiple models:")
                for feat, models in sorted(common_features.items(), key=lambda x: -len(x[1]))[:5]:
                    self.log(f"    - {feat}: {', '.join(models)}")
        
        self.log("\n✅ Phase 8 Model Visualization & Interpretability COMPLETE")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 8: Model Visualization & Interpretability')
    parser.add_argument('--data-source', default='synthetic', 
                       help='Data source: synthetic, alzheimer, or path to CSV file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run Phase 8 debugging
    visualizer = Phase8ModelVisualization(
        verbose=args.verbose,
        data_source=args.data_source
    )
    
    try:
        results = visualizer.run_phase_8()
        print(f"\n✅ Phase 8 completed successfully!")
        print(f"Results saved to: {visualizer.output_dir / 'phase8_results.json'}")
        print(f"Visualizations saved to: {visualizer.visualization_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Phase 8 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
