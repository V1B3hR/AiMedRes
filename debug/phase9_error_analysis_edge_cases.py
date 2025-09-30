#!/usr/bin/env python3
"""
Phase 9 Debugging Script: Error Analysis & Edge Cases

This script implements Phase 9 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 9.1: Analyze misclassified samples and residuals
- Subphase 9.2: Investigate model bias (e.g., toward certain classes)
- Subphase 9.3: Test on edge cases and adversarial examples

Usage:
    python debug/phase9_error_analysis_edge_cases.py [--data-source] [--verbose]
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
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))


class Phase9ErrorAnalysis:
    """Phase 9 debugging implementation for AiMedRes error analysis & edge cases"""
    
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
            format='[%(asctime)s] 🔍 %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("Phase9ErrorAnalysis")
        
        # Load Phase 8 results if available
        self.phase8_results = None
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🔍" if level == "INFO" else "⚠️ " if level == "WARN" else "❌"
        print(f"[{timestamp}] {prefix} {message}")
        if self.verbose and level != "INFO":
            self.logger.info(message)

    def load_phase8_results(self) -> bool:
        """Load Phase 8 results and trained models"""
        self.log("Loading Phase 8 results...")
        
        phase8_path = self.output_dir / "phase8_results.json"
        if not phase8_path.exists():
            self.log("Phase 8 results not found. Will generate new models...", "WARN")
            return False
        
        try:
            with open(phase8_path, 'r') as f:
                self.phase8_results = json.load(f)
            
            self.log(f"✓ Loaded Phase 8 results from {phase8_path}")
            self.log(f"  Data source: {self.phase8_results.get('data_source', 'unknown')}")
            self.log(f"  Models available: {len(self.phase8_results.get('models_trained', []))}")
            return True
            
        except Exception as e:
            self.log(f"Error loading Phase 8 results: {e}", "ERROR")
            return False

    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, str]:
        """Generate synthetic medical data for testing"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'age': np.random.normal(55, 15, n_samples).clip(18, 90),
            'blood_pressure': np.random.normal(120, 20, n_samples).clip(80, 200),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(120, 300),
            'heart_rate': np.random.normal(75, 12, n_samples).clip(50, 120),
            'bmi': np.random.normal(27, 5, n_samples).clip(15, 45),
            'glucose': np.random.normal(100, 20, n_samples).clip(60, 200),
        }
        
        df = pd.DataFrame(data)
        
        # Generate target with some complexity
        risk_score = (
            0.3 * (df['age'] - 18) / 72 +
            0.2 * (df['blood_pressure'] - 80) / 120 +
            0.2 * (df['cholesterol'] - 120) / 180 +
            0.15 * (df['bmi'] - 15) / 30 +
            0.15 * (df['glucose'] - 60) / 140
        )
        
        # Add some noise
        risk_score = risk_score + np.random.normal(0, 0.1, n_samples)
        
        # Create multi-class target (0: Low, 1: Medium, 2: High risk)
        df['target'] = pd.cut(risk_score, bins=3, labels=[0, 1, 2]).astype(int)
        
        return df, 'target'

    def prepare_data_and_models(self) -> bool:
        """Prepare data and train models if needed"""
        self.log("Preparing data and models...")
        
        # Load or generate data
        data, target_col = self.generate_synthetic_data(n_samples=1000)
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Store feature and class names
        self.feature_names = list(X.columns)
        self.class_names = sorted(y.unique())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        # Train models
        self.log("Training models for error analysis...")
        models = {
            'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            self.trained_models[name] = {
                'model': model,
                'predictions_train': y_pred_train,
                'predictions_test': y_pred_test,
                'accuracy_train': accuracy_score(self.y_train, y_pred_train),
                'accuracy_test': accuracy_score(self.y_test, y_pred_test)
            }
            
            self.log(f"  ✓ {name}: Train Acc={self.trained_models[name]['accuracy_train']:.3f}, "
                    f"Test Acc={self.trained_models[name]['accuracy_test']:.3f}")
        
        self.log(f"✓ Prepared {len(self.trained_models)} models")
        return True

    def subphase_9_1_misclassified_analysis(self) -> Dict[str, Any]:
        """
        Subphase 9.1: Analyze misclassified samples and residuals
        
        Identifies misclassified samples, analyzes error patterns,
        and computes residuals for regression-style analysis.
        """
        self.log("\n" + "="*60)
        self.log("SUBPHASE 9.1: Analyzing Misclassified Samples & Residuals")
        self.log("="*60)
        
        results = {
            'models_analyzed': [],
            'misclassification_analysis': {},
            'error_patterns': {},
            'visualizations': []
        }
        
        for model_name, model_info in self.trained_models.items():
            self.log(f"\nAnalyzing errors for {model_name}...")
            
            # Get predictions and true labels
            y_pred = model_info['predictions_test']
            y_true = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            
            # Find misclassified samples
            misclassified_mask = y_pred != y_true
            misclassified_indices = np.where(misclassified_mask)[0]
            
            # Calculate error statistics
            n_misclassified = len(misclassified_indices)
            error_rate = n_misclassified / len(y_true)
            
            # Analyze error patterns by class
            error_by_class = {}
            for class_label in self.class_names:
                class_mask = y_true == class_label
                class_errors = np.sum(misclassified_mask & class_mask)
                class_total = np.sum(class_mask)
                error_by_class[str(class_label)] = {
                    'errors': int(class_errors),
                    'total': int(class_total),
                    'error_rate': float(class_errors / class_total if class_total > 0 else 0)
                }
            
            # Analyze confusion patterns (which classes are confused with which)
            confusion_patterns = {}
            for true_class in self.class_names:
                true_mask = y_true == true_class
                confused_predictions = y_pred[true_mask & misclassified_mask]
                if len(confused_predictions) > 0:
                    unique, counts = np.unique(confused_predictions, return_counts=True)
                    confusion_patterns[str(true_class)] = {
                        str(pred_class): int(count) 
                        for pred_class, count in zip(unique, counts)
                    }
            
            # Compute residuals (difference between predicted and true class)
            residuals = y_pred.astype(float) - y_true.astype(float)
            
            results['misclassification_analysis'][model_name] = {
                'total_misclassified': int(n_misclassified),
                'error_rate': float(error_rate),
                'error_by_class': error_by_class,
                'confusion_patterns': confusion_patterns,
                'residual_statistics': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'min': float(np.min(residuals)),
                    'max': float(np.max(residuals))
                }
            }
            
            results['models_analyzed'].append(model_name)
            
            self.log(f"  • Total misclassified: {n_misclassified}/{len(y_true)} ({error_rate:.1%})")
            self.log(f"  • Error rate by class:")
            for class_label, stats in error_by_class.items():
                self.log(f"    - Class {class_label}: {stats['errors']}/{stats['total']} ({stats['error_rate']:.1%})")
            
            # Visualize error distribution
            self._visualize_error_distribution(
                model_name, y_true, y_pred, misclassified_mask, residuals
            )
            results['visualizations'].append(f'error_distribution_{model_name}.png')
        
        self.log(f"\n✓ Subphase 9.1 complete - Analyzed {len(results['models_analyzed'])} models")
        return results

    def subphase_9_2_bias_investigation(self) -> Dict[str, Any]:
        """
        Subphase 9.2: Investigate model bias (e.g., toward certain classes)
        
        Analyzes class-wise performance, statistical bias metrics,
        and prediction distribution patterns.
        """
        self.log("\n" + "="*60)
        self.log("SUBPHASE 9.2: Investigating Model Bias")
        self.log("="*60)
        
        results = {
            'models_analyzed': [],
            'bias_metrics': {},
            'class_performance': {},
            'prediction_distribution': {},
            'statistical_tests': {},
            'visualizations': []
        }
        
        for model_name, model_info in self.trained_models.items():
            self.log(f"\nAnalyzing bias for {model_name}...")
            
            y_pred = model_info['predictions_test']
            y_true = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            
            # Class-wise performance metrics
            class_metrics = {}
            for class_label in self.class_names:
                # Binary classification metrics for each class
                y_true_binary = (y_true == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                class_metrics[str(class_label)] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(np.sum(y_true == class_label))
                }
            
            results['class_performance'][model_name] = class_metrics
            
            # Prediction distribution analysis
            pred_dist = {}
            true_dist = {}
            for class_label in self.class_names:
                pred_dist[str(class_label)] = int(np.sum(y_pred == class_label))
                true_dist[str(class_label)] = int(np.sum(y_true == class_label))
            
            results['prediction_distribution'][model_name] = {
                'predicted': pred_dist,
                'true': true_dist
            }
            
            # Calculate demographic parity (balanced prediction rates)
            # For multi-class: variance in prediction rates across classes
            pred_rates = [pred_dist[str(c)] / len(y_pred) for c in self.class_names]
            true_rates = [true_dist[str(c)] / len(y_true) for c in self.class_names]
            
            demographic_parity_diff = np.max(pred_rates) - np.min(pred_rates)
            
            # Calculate equalized odds (TPR and FPR should be equal across groups)
            # We'll use balanced accuracy as a proxy
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            
            # Statistical significance tests
            # Chi-square test for prediction distribution
            chi2_stat, chi2_pvalue = stats.chisquare(
                f_obs=list(pred_dist.values()),
                f_exp=list(true_dist.values())
            )
            
            results['bias_metrics'][model_name] = {
                'demographic_parity_difference': float(demographic_parity_diff),
                'balanced_accuracy': float(balanced_acc),
                'class_imbalance_ratio': float(max(true_rates) / min(true_rates) if min(true_rates) > 0 else 0)
            }
            
            results['statistical_tests'][model_name] = {
                'chi_square_stat': float(chi2_stat),
                'chi_square_pvalue': float(chi2_pvalue),
                'prediction_bias_significant': bool(chi2_pvalue < 0.05)
            }
            
            results['models_analyzed'].append(model_name)
            
            # Log findings
            self.log(f"  • Balanced Accuracy: {balanced_acc:.3f}")
            self.log(f"  • Demographic Parity Difference: {demographic_parity_diff:.3f}")
            self.log(f"  • Chi-square p-value: {chi2_pvalue:.4f}")
            
            self.log(f"  • Class-wise Performance:")
            for class_label, metrics in class_metrics.items():
                self.log(f"    - Class {class_label}: P={metrics['precision']:.3f}, "
                        f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            
            # Detect potential bias
            bias_warnings = []
            if demographic_parity_diff > 0.2:
                bias_warnings.append("High demographic parity difference detected")
            
            # Check for class imbalance in predictions vs truth
            max_diff = max(abs(pred_rates[i] - true_rates[i]) for i in range(len(self.class_names)))
            if max_diff > 0.15:
                bias_warnings.append("Significant prediction distribution bias")
            
            # Check for performance disparities
            f1_scores = [m['f1_score'] for m in class_metrics.values()]
            if max(f1_scores) - min(f1_scores) > 0.3:
                bias_warnings.append("Large performance disparity across classes")
            
            if bias_warnings:
                self.log(f"  ⚠️  Bias warnings:")
                for warning in bias_warnings:
                    self.log(f"    - {warning}")
            else:
                self.log(f"  ✓ No significant bias detected")
            
            # Visualize bias metrics
            self._visualize_bias_metrics(model_name, class_metrics, pred_dist, true_dist)
            results['visualizations'].append(f'bias_analysis_{model_name}.png')
        
        self.log(f"\n✓ Subphase 9.2 complete - Analyzed bias in {len(results['models_analyzed'])} models")
        return results

    def subphase_9_3_edge_cases_adversarial(self) -> Dict[str, Any]:
        """
        Subphase 9.3: Test on edge cases and adversarial examples
        
        Tests model robustness on boundary values, outliers,
        and adversarial perturbations.
        """
        self.log("\n" + "="*60)
        self.log("SUBPHASE 9.3: Testing Edge Cases & Adversarial Examples")
        self.log("="*60)
        
        results = {
            'models_analyzed': [],
            'edge_case_tests': {},
            'adversarial_tests': {},
            'robustness_scores': {},
            'visualizations': []
        }
        
        for model_name, model_info in self.trained_models.items():
            self.log(f"\nTesting {model_name} on edge cases...")
            
            model = model_info['model']
            
            # 1. Boundary value testing
            edge_cases = self._generate_edge_cases()
            edge_predictions = model.predict(edge_cases)
            
            # Count unique predictions for edge cases
            unique_preds, pred_counts = np.unique(edge_predictions, return_counts=True)
            edge_pred_dist = {str(pred): int(count) for pred, count in zip(unique_preds, pred_counts)}
            
            # 2. Adversarial perturbations (small random noise)
            adversarial_results = self._test_adversarial_robustness(model, model_info)
            
            # 3. Outlier testing
            outlier_results = self._test_outlier_handling(model)
            
            results['edge_case_tests'][model_name] = {
                'n_edge_cases': len(edge_cases),
                'edge_prediction_distribution': edge_pred_dist,
                'unique_predictions': len(unique_preds)
            }
            
            results['adversarial_tests'][model_name] = adversarial_results
            
            # Calculate overall robustness score
            robustness_score = (
                adversarial_results['robustness_rate'] * 0.5 +
                (1 - adversarial_results['avg_accuracy_drop']) * 0.5
            )
            
            results['robustness_scores'][model_name] = {
                'overall_score': float(robustness_score),
                'adversarial_robustness': float(adversarial_results['robustness_rate']),
                'stability_score': float(1 - adversarial_results['avg_accuracy_drop'])
            }
            
            results['models_analyzed'].append(model_name)
            
            self.log(f"  • Edge cases tested: {len(edge_cases)}")
            self.log(f"  • Adversarial robustness: {adversarial_results['robustness_rate']:.1%}")
            self.log(f"  • Average accuracy drop: {adversarial_results['avg_accuracy_drop']:.1%}")
            self.log(f"  • Overall robustness score: {robustness_score:.3f}")
            
            # Visualize adversarial testing results
            self._visualize_adversarial_tests(model_name, adversarial_results)
            results['visualizations'].append(f'adversarial_tests_{model_name}.png')
        
        self.log(f"\n✓ Subphase 9.3 complete - Tested {len(results['models_analyzed'])} models")
        return results

    def _generate_edge_cases(self) -> pd.DataFrame:
        """Generate edge case test samples (boundary values, extreme cases)"""
        edge_cases = []
        
        # Min/max values for each feature
        for feature in self.feature_names:
            # Minimum boundary
            min_case = {feat: self.X_test[feat].mean() for feat in self.feature_names}
            min_case[feature] = self.X_test[feature].min()
            edge_cases.append(min_case)
            
            # Maximum boundary
            max_case = {feat: self.X_test[feat].mean() for feat in self.feature_names}
            max_case[feature] = self.X_test[feature].max()
            edge_cases.append(max_case)
        
        # All minimum
        edge_cases.append({feat: self.X_test[feat].min() for feat in self.feature_names})
        
        # All maximum
        edge_cases.append({feat: self.X_test[feat].max() for feat in self.feature_names})
        
        return pd.DataFrame(edge_cases)

    def _test_adversarial_robustness(self, model, model_info) -> Dict[str, Any]:
        """Test model robustness to small adversarial perturbations"""
        y_pred_original = model_info['predictions_test']
        y_true = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        
        epsilon_values = [0.01, 0.05, 0.1, 0.2]
        perturbation_results = []
        
        for epsilon in epsilon_values:
            # Add random noise
            X_perturbed = self.X_test + np.random.normal(0, epsilon, self.X_test.shape)
            y_pred_perturbed = model.predict(X_perturbed)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred_perturbed)
            consistency = np.mean(y_pred_perturbed == y_pred_original)
            
            perturbation_results.append({
                'epsilon': float(epsilon),
                'accuracy': float(accuracy),
                'consistency_rate': float(consistency),
                'accuracy_drop': float(model_info['accuracy_test'] - accuracy)
            })
        
        # Overall robustness metrics
        avg_consistency = np.mean([r['consistency_rate'] for r in perturbation_results])
        avg_accuracy_drop = np.mean([r['accuracy_drop'] for r in perturbation_results])
        
        return {
            'perturbation_tests': perturbation_results,
            'robustness_rate': float(avg_consistency),
            'avg_accuracy_drop': float(avg_accuracy_drop),
            'stable_predictions': float(avg_consistency > 0.8)
        }

    def _test_outlier_handling(self, model) -> Dict[str, Any]:
        """Test model behavior on outlier samples"""
        # Generate outlier samples (3 std deviations away)
        outliers = []
        for _ in range(20):
            outlier = {}
            for feature in self.feature_names:
                mean = self.X_test[feature].mean()
                std = self.X_test[feature].std()
                # Random outlier direction
                outlier[feature] = mean + np.random.choice([-3, 3]) * std
            outliers.append(outlier)
        
        X_outliers = pd.DataFrame(outliers)
        predictions = model.predict(X_outliers)
        
        unique_preds, counts = np.unique(predictions, return_counts=True)
        
        return {
            'n_outliers_tested': len(outliers),
            'prediction_distribution': {str(p): int(c) for p, c in zip(unique_preds, counts)}
        }

    def _visualize_error_distribution(self, model_name: str, y_true, y_pred, 
                                     misclassified_mask, residuals):
        """Create visualization of error distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Error Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # 2. Error rate by class
        error_by_class = []
        for class_label in self.class_names:
            class_mask = y_true == class_label
            class_errors = np.sum(misclassified_mask & class_mask)
            class_total = np.sum(class_mask)
            error_by_class.append(class_errors / class_total if class_total > 0 else 0)
        
        axes[0, 1].bar(range(len(self.class_names)), error_by_class, color='coral')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Error Rate')
        axes[0, 1].set_title('Error Rate by Class')
        axes[0, 1].set_xticks(range(len(self.class_names)))
        axes[0, 1].set_xticklabels([f'Class {c}' for c in self.class_names])
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Residual distribution
        axes[1, 0].hist(residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
        axes[1, 0].set_xlabel('Residual (Predicted - True)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].legend()
        
        # 4. Misclassification pattern
        misclass_patterns = np.zeros((len(self.class_names), len(self.class_names)))
        for true_class in self.class_names:
            for pred_class in self.class_names:
                count = np.sum((y_true == true_class) & (y_pred == pred_class) & misclassified_mask)
                misclass_patterns[true_class, pred_class] = count
        
        sns.heatmap(misclass_patterns, annot=True, fmt='.0f', cmap='Reds', ax=axes[1, 1])
        axes[1, 1].set_title('Misclassification Patterns')
        axes[1, 1].set_xlabel('Predicted Class')
        axes[1, 1].set_ylabel('True Class')
        
        plt.tight_layout()
        output_path = self.visualization_dir / f'error_distribution_{model_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  ✓ Saved error distribution plot: {output_path}")

    def _visualize_bias_metrics(self, model_name: str, class_metrics: Dict,
                               pred_dist: Dict, true_dist: Dict):
        """Create visualization of bias metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Bias Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        classes = sorted([int(k) for k in class_metrics.keys()])
        
        # 1. Class-wise performance metrics
        precisions = [class_metrics[str(c)]['precision'] for c in classes]
        recalls = [class_metrics[str(c)]['recall'] for c in classes]
        f1_scores = [class_metrics[str(c)]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 0].bar(x - width, precisions, width, label='Precision', color='skyblue')
        axes[0, 0].bar(x, recalls, width, label='Recall', color='lightcoral')
        axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Class-wise Performance Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'Class {c}' for c in classes])
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Prediction distribution comparison
        true_counts = [true_dist[str(c)] for c in classes]
        pred_counts = [pred_dist[str(c)] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, pred_counts, width, label='Predicted', color='orange', alpha=0.8)
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Prediction Distribution vs True Distribution')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'Class {c}' for c in classes])
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Performance disparity heatmap
        metrics_matrix = np.array([precisions, recalls, f1_scores])
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=[f'Class {c}' for c in classes],
                   yticklabels=['Precision', 'Recall', 'F1-Score'],
                   ax=axes[1, 0], vmin=0, vmax=1)
        axes[1, 0].set_title('Performance Heatmap')
        
        # 4. Bias indicators
        bias_indicators = {
            'Precision\nVariance': np.var(precisions),
            'Recall\nVariance': np.var(recalls),
            'F1\nVariance': np.var(f1_scores),
            'Pred/True\nDivergence': np.mean([abs(pred_counts[i] - true_counts[i]) / max(true_counts[i], 1) 
                                               for i in range(len(classes))])
        }
        
        axes[1, 1].bar(range(len(bias_indicators)), list(bias_indicators.values()), 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_xticks(range(len(bias_indicators)))
        axes[1, 1].set_xticklabels(list(bias_indicators.keys()), rotation=0)
        axes[1, 1].set_ylabel('Variance / Divergence')
        axes[1, 1].set_title('Bias Indicators (Lower is Better)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.visualization_dir / f'bias_analysis_{model_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  ✓ Saved bias analysis plot: {output_path}")

    def _visualize_adversarial_tests(self, model_name: str, adversarial_results: Dict):
        """Create visualization of adversarial testing results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Adversarial Robustness: {model_name}', fontsize=16, fontweight='bold')
        
        perturbation_tests = adversarial_results['perturbation_tests']
        epsilons = [t['epsilon'] for t in perturbation_tests]
        accuracies = [t['accuracy'] for t in perturbation_tests]
        consistencies = [t['consistency_rate'] for t in perturbation_tests]
        accuracy_drops = [t['accuracy_drop'] for t in perturbation_tests]
        
        # 1. Accuracy under perturbation
        axes[0, 0].plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].axhline(adversarial_results['perturbation_tests'][0]['accuracy'], 
                          color='red', linestyle='--', label='Original Accuracy')
        axes[0, 0].set_xlabel('Perturbation Strength (ε)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy under Adversarial Perturbation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # 2. Prediction consistency
        axes[0, 1].plot(epsilons, consistencies, marker='s', linewidth=2, markersize=8, color='coral')
        axes[0, 1].axhline(0.8, color='green', linestyle='--', label='Robustness Threshold')
        axes[0, 1].set_xlabel('Perturbation Strength (ε)')
        axes[0, 1].set_ylabel('Consistency Rate')
        axes[0, 1].set_title('Prediction Consistency Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Accuracy drop
        axes[1, 0].bar(range(len(epsilons)), accuracy_drops, color='orange', alpha=0.7)
        axes[1, 0].set_xticks(range(len(epsilons)))
        axes[1, 0].set_xticklabels([f'{e:.2f}' for e in epsilons])
        axes[1, 0].set_xlabel('Perturbation Strength (ε)')
        axes[1, 0].set_ylabel('Accuracy Drop')
        axes[1, 0].set_title('Accuracy Degradation')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Overall robustness summary
        robustness_metrics = {
            'Robustness\nRate': adversarial_results['robustness_rate'],
            'Avg Accuracy\nDrop': adversarial_results['avg_accuracy_drop'],
            'Stability\nScore': 1 - adversarial_results['avg_accuracy_drop']
        }
        
        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' 
                 for v in robustness_metrics.values()]
        
        axes[1, 1].bar(range(len(robustness_metrics)), list(robustness_metrics.values()), 
                      color=colors, alpha=0.7)
        axes[1, 1].set_xticks(range(len(robustness_metrics)))
        axes[1, 1].set_xticklabels(list(robustness_metrics.keys()))
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Robustness Metrics Summary')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Good')
        axes[1, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Fair')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.visualization_dir / f'adversarial_tests_{model_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"  ✓ Saved adversarial tests plot: {output_path}")

    def run_phase_9(self) -> Dict[str, Any]:
        """Run complete Phase 9 error analysis & edge cases"""
        self.log("\n" + "="*70)
        self.log("🔍 PHASE 9: ERROR ANALYSIS & EDGE CASES")
        self.log("="*70)
        
        start_time = time.time()
        
        # Prepare data and models
        if not self.prepare_data_and_models():
            self.log("Failed to prepare data and models", "ERROR")
            return {}
        
        # Run all subphases
        results = {
            'phase': 9,
            'timestamp': datetime.now().isoformat(),
            'data_source': self.data_source,
            'n_models': len(self.trained_models),
            'models_trained': list(self.trained_models.keys()),
            'subphases': {}
        }
        
        # Subphase 9.1: Misclassified samples analysis
        results['subphases']['9.1_misclassified_analysis'] = self.subphase_9_1_misclassified_analysis()
        
        # Subphase 9.2: Bias investigation
        results['subphases']['9.2_bias_investigation'] = self.subphase_9_2_bias_investigation()
        
        # Subphase 9.3: Edge cases and adversarial testing
        results['subphases']['9.3_edge_cases_adversarial'] = self.subphase_9_3_edge_cases_adversarial()
        
        # Summary statistics
        execution_time = time.time() - start_time
        results['execution_time_seconds'] = execution_time
        results['summary'] = self._generate_summary(results)
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of Phase 9 results"""
        summary = {
            'total_models_analyzed': len(results['models_trained']),
            'subphases_completed': len(results['subphases']),
            'total_visualizations': sum(
                len(sp.get('visualizations', [])) 
                for sp in results['subphases'].values()
            )
        }
        
        # Aggregate key findings
        if '9.1_misclassified_analysis' in results['subphases']:
            sp9_1 = results['subphases']['9.1_misclassified_analysis']
            summary['avg_error_rate'] = np.mean([
                v['error_rate'] for v in sp9_1['misclassification_analysis'].values()
            ])
        
        if '9.2_bias_investigation' in results['subphases']:
            sp9_2 = results['subphases']['9.2_bias_investigation']
            summary['avg_balanced_accuracy'] = np.mean([
                v['balanced_accuracy'] for v in sp9_2['bias_metrics'].values()
            ])
            summary['bias_detected'] = any(
                v['prediction_bias_significant'] 
                for v in sp9_2['statistical_tests'].values()
            )
        
        if '9.3_edge_cases_adversarial' in results['subphases']:
            sp9_3 = results['subphases']['9.3_edge_cases_adversarial']
            summary['avg_robustness_score'] = np.mean([
                v['overall_score'] for v in sp9_3['robustness_scores'].values()
            ])
        
        return summary

    def save_results(self, results: Dict[str, Any]):
        """Save Phase 9 results to JSON file"""
        output_path = self.output_dir / "phase9_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"\n✓ Results saved to: {output_path}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print Phase 9 summary"""
        self.log("\n" + "="*70)
        self.log("📊 PHASE 9 SUMMARY")
        self.log("="*70)
        
        summary = results['summary']
        
        self.log(f"\n✅ Phase 9 Complete!")
        self.log(f"  • Models analyzed: {summary['total_models_analyzed']}")
        self.log(f"  • Subphases completed: {summary['subphases_completed']}/3")
        self.log(f"  • Visualizations created: {summary['total_visualizations']}")
        self.log(f"  • Execution time: {results['execution_time_seconds']:.1f}s")
        
        if 'avg_error_rate' in summary:
            self.log(f"\n📈 Error Analysis:")
            self.log(f"  • Average error rate: {summary['avg_error_rate']:.1%}")
        
        if 'avg_balanced_accuracy' in summary:
            self.log(f"\n⚖️  Bias Analysis:")
            self.log(f"  • Average balanced accuracy: {summary['avg_balanced_accuracy']:.3f}")
            self.log(f"  • Significant bias detected: {'Yes' if summary['bias_detected'] else 'No'}")
        
        if 'avg_robustness_score' in summary:
            self.log(f"\n🛡️  Robustness Testing:")
            self.log(f"  • Average robustness score: {summary['avg_robustness_score']:.3f}")
        
        self.log("\n✅ Phase 9 Error Analysis & Edge Cases COMPLETE")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 9: Error Analysis & Edge Cases')
    parser.add_argument('--data-source', default='synthetic', 
                       help='Data source: synthetic, alzheimer, or path to CSV file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run Phase 9 debugging
    analyzer = Phase9ErrorAnalysis(
        verbose=args.verbose,
        data_source=args.data_source
    )
    
    try:
        results = analyzer.run_phase_9()
        print(f"\n✅ Phase 9 completed successfully!")
        print(f"Results saved to: {analyzer.output_dir / 'phase9_results.json'}")
        print(f"Visualizations saved to: {analyzer.visualization_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Phase 9 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
