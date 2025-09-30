#!/usr/bin/env python3
"""
Phase 7 Debugging Script: Model Training & Evaluation

This script implements Phase 7 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 7.1: Train models with cross-validation
- Subphase 7.2: Record training, validation, and test metrics (accuracy, loss, etc.)
- Subphase 7.3: Compare results with baseline models

Usage:
    python debug/phase7_model_training_evaluation.py [--data-source] [--verbose]
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from training.training import create_test_alzheimer_data
    from scripts.data_loaders import DataLoader, CSVDataLoader, MockDataLoader
except ImportError as e:
    print(f"⚠️  Warning: Could not import some project modules: {e}")
    print("Will use synthetic data generation instead")


class Phase7ModelTrainingEvaluator:
    """Phase 7 debugging implementation for AiMedRes model training & evaluation"""
    
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
            format='[%(asctime)s] 🎯 %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("Phase7ModelTraining")
        
        # Initialize model storage
        self.trained_models = {}
        self.baseline_models = {}
        self.evaluation_metrics = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🎯" if level == "INFO" else "⚠️ " if level == "WARN" else "❌"
        print(f"[{timestamp}] {prefix} {message}")
        if self.verbose and level != "INFO":
            self.logger.info(message)

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
        
        # Create interactions and non-linear relationships
        risk_score = (
            0.3 * (age - 40) / 30 +
            0.2 * (bmi - 20) / 15 +
            0.2 * (blood_pressure - 100) / 50 +
            0.15 * (cholesterol - 150) / 100 +
            0.15 * (glucose - 80) / 80 +
            np.random.normal(0, 0.3, n_samples)
        )
        
        # Create target with 3 classes
        target = np.digitize(risk_score, bins=[0.33, 0.67])
        
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
        """Load data based on specified source"""
        self.log(f"Loading data from source: {self.data_source}")
        
        if self.data_source == "synthetic":
            return self.generate_synthetic_data()
        elif self.data_source == "alzheimer":
            try:
                data = create_test_alzheimer_data(n_samples=1000)
                return data, 'Diagnosis'
            except Exception as e:
                self.log(f"Failed to load Alzheimer data: {e}", "WARN")
                self.log("Falling back to synthetic data", "WARN")
                return self.generate_synthetic_data()
        else:
            # Try to load from CSV file
            try:
                data = pd.read_csv(self.data_source)
                # Assume last column is target
                target_col = data.columns[-1]
                return data, target_col
            except Exception as e:
                self.log(f"Failed to load data from {self.data_source}: {e}", "WARN")
                self.log("Falling back to synthetic data", "WARN")
                return self.generate_synthetic_data()

    def subphase_7_1_train_with_cross_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Subphase 7.1: Train models with cross-validation
        
        This subphase trains multiple models using stratified k-fold cross-validation
        to ensure robust generalization estimates.
        """
        self.log("=== SUBPHASE 7.1: TRAIN MODELS WITH CROSS-VALIDATION ===")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models to train
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm_rbf': SVC(kernel='rbf', random_state=42, probability=True),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
            'naive_bayes': GaussianNB()
        }
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for name, model in models.items():
            self.log(f"Training {name} with {n_folds}-fold cross-validation...")
            start_time = time.time()
            
            # Use scaled data for all models
            X_data = X_scaled
            
            # Perform cross-validation with multiple metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1': 'f1_macro'
            }
            
            try:
                cv_scores = cross_validate(
                    model, X_data, y, cv=skf, 
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Calculate mean and std for each metric
                results = {
                    'train_accuracy': {
                        'mean': cv_scores['train_accuracy'].mean(),
                        'std': cv_scores['train_accuracy'].std(),
                        'scores': cv_scores['train_accuracy'].tolist()
                    },
                    'test_accuracy': {
                        'mean': cv_scores['test_accuracy'].mean(),
                        'std': cv_scores['test_accuracy'].std(),
                        'scores': cv_scores['test_accuracy'].tolist()
                    },
                    'test_precision': {
                        'mean': cv_scores['test_precision'].mean(),
                        'std': cv_scores['test_precision'].std(),
                        'scores': cv_scores['test_precision'].tolist()
                    },
                    'test_recall': {
                        'mean': cv_scores['test_recall'].mean(),
                        'std': cv_scores['test_recall'].std(),
                        'scores': cv_scores['test_recall'].tolist()
                    },
                    'test_f1': {
                        'mean': cv_scores['test_f1'].mean(),
                        'std': cv_scores['test_f1'].std(),
                        'scores': cv_scores['test_f1'].tolist()
                    },
                    'overfitting_gap': cv_scores['train_accuracy'].mean() - cv_scores['test_accuracy'].mean(),
                    'training_time': time.time() - start_time
                }
                
                # Train final model on full data for later use
                model.fit(X_data, y)
                self.trained_models[name] = model
                
                cv_results[name] = results
                
                self.log(f"  ✓ {name}: Accuracy={results['test_accuracy']['mean']:.3f}±{results['test_accuracy']['std']:.3f}, "
                        f"F1={results['test_f1']['mean']:.3f}±{results['test_f1']['std']:.3f}, "
                        f"Time={results['training_time']:.2f}s")
                
                if results['overfitting_gap'] > 0.1:
                    self.log(f"    ⚠️  Overfitting detected (gap: {results['overfitting_gap']:.3f})", "WARN")
                    
            except Exception as e:
                self.log(f"  ✗ Failed to train {name}: {e}", "WARN")
                cv_results[name] = {'error': str(e)}
        
        return {
            'cv_results': cv_results,
            'n_folds': n_folds
            # Removed 'scaler' from return to avoid JSON serialization issues
        }

    def subphase_7_2_record_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Subphase 7.2: Record training, validation, and test metrics
        
        This subphase performs a train/test split and records comprehensive metrics
        for each trained model, including accuracy, precision, recall, F1, and loss.
        """
        self.log("=== SUBPHASE 7.2: RECORD TRAINING, VALIDATION, AND TEST METRICS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        metrics_record = {}
        
        for name, model in self.trained_models.items():
            self.log(f"Recording metrics for {name}...")
            
            try:
                # Retrain on training set
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Get probability predictions if available
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train_scaled)
                    y_test_proba = model.predict_proba(X_test_scaled)
                    
                    # Calculate log loss
                    train_loss = log_loss(y_train, y_train_proba)
                    test_loss = log_loss(y_test, y_test_proba)
                else:
                    train_loss = None
                    test_loss = None
                
                # Calculate comprehensive metrics
                metrics = {
                    'training_metrics': {
                        'accuracy': accuracy_score(y_train, y_train_pred),
                        'precision_macro': precision_score(y_train, y_train_pred, average='macro', zero_division=0),
                        'precision_weighted': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        'recall_macro': recall_score(y_train, y_train_pred, average='macro', zero_division=0),
                        'recall_weighted': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        'f1_macro': f1_score(y_train, y_train_pred, average='macro', zero_division=0),
                        'f1_weighted': f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                        'loss': train_loss
                    },
                    'test_metrics': {
                        'accuracy': accuracy_score(y_test, y_test_pred),
                        'precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
                        'precision_weighted': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                        'recall_macro': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
                        'recall_weighted': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                        'f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
                        'f1_weighted': f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
                        'loss': test_loss
                    },
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
                }
                
                metrics_record[name] = metrics
                
                self.log(f"  ✓ {name}:")
                self.log(f"    Train: Acc={metrics['training_metrics']['accuracy']:.3f}, "
                        f"F1={metrics['training_metrics']['f1_macro']:.3f}")
                self.log(f"    Test:  Acc={metrics['test_metrics']['accuracy']:.3f}, "
                        f"F1={metrics['test_metrics']['f1_macro']:.3f}")
                
                # Store for comparison
                self.evaluation_metrics.append({
                    'model': name,
                    'train_accuracy': metrics['training_metrics']['accuracy'],
                    'test_accuracy': metrics['test_metrics']['accuracy'],
                    'train_f1': metrics['training_metrics']['f1_macro'],
                    'test_f1': metrics['test_metrics']['f1_macro'],
                    'overfitting_gap': metrics['training_metrics']['accuracy'] - metrics['test_metrics']['accuracy']
                })
                
            except Exception as e:
                self.log(f"  ✗ Failed to record metrics for {name}: {e}", "WARN")
                metrics_record[name] = {'error': str(e)}
        
        return {
            'metrics_record': metrics_record,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

    def subphase_7_3_compare_with_baseline(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Subphase 7.3: Compare results with baseline models
        
        This subphase trains simple baseline models and compares their performance
        with the more complex models trained in previous subphases.
        """
        self.log("=== SUBPHASE 7.3: COMPARE RESULTS WITH BASELINE MODELS ===")
        
        # Define simple baseline models
        baseline_models = {
            'baseline_logistic': LogisticRegression(random_state=42, max_iter=1000),
            'baseline_decision_tree': DecisionTreeClassifier(random_state=42, max_depth=3)
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            self.log(f"Training baseline model: {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                self.baseline_models[name] = model
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Get predictions
                y_pred = model.predict(X_test_scaled)
                
                baseline_results[name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'overfitting_gap': train_score - test_score
                }
                
                self.log(f"  ✓ {name}: Train={train_score:.3f}, Test={test_score:.3f}, F1={baseline_results[name]['f1']:.3f}")
                
            except Exception as e:
                self.log(f"  ✗ Failed to train {name}: {e}", "WARN")
                baseline_results[name] = {'error': str(e)}
        
        # Perform comparison
        self.log("\n📊 COMPARISON SUMMARY:")
        self.log("-" * 70)
        self.log(f"{'Model':<30} {'Test Acc':<12} {'Test F1':<12} {'Overfit Gap':<12}")
        self.log("-" * 70)
        
        # Show baseline models first
        for name, results in baseline_results.items():
            if 'error' not in results:
                self.log(f"{name:<30} {results['test_accuracy']:>10.3f}  {results['f1']:>10.3f}  {results['overfitting_gap']:>10.3f}")
        
        self.log("-" * 70)
        
        # Show trained models
        for metric in self.evaluation_metrics:
            self.log(f"{metric['model']:<30} {metric['test_accuracy']:>10.3f}  {metric['test_f1']:>10.3f}  {metric['overfitting_gap']:>10.3f}")
        
        self.log("-" * 70)
        
        # Identify best models
        if self.evaluation_metrics:
            best_by_accuracy = max(self.evaluation_metrics, key=lambda x: x['test_accuracy'])
            best_by_f1 = max(self.evaluation_metrics, key=lambda x: x['test_f1'])
            least_overfit = min(self.evaluation_metrics, key=lambda x: abs(x['overfitting_gap']))
            
            self.log(f"\n🏆 Best by Test Accuracy: {best_by_accuracy['model']} ({best_by_accuracy['test_accuracy']:.3f})")
            self.log(f"🏆 Best by Test F1: {best_by_f1['model']} ({best_by_f1['test_f1']:.3f})")
            self.log(f"🏆 Least Overfitting: {least_overfit['model']} (gap: {least_overfit['overfitting_gap']:.3f})")
            
            comparison_summary = {
                'best_by_accuracy': best_by_accuracy,
                'best_by_f1': best_by_f1,
                'least_overfit': least_overfit
            }
        else:
            comparison_summary = {}
        
        return {
            'baseline_results': baseline_results,
            'comparison_summary': comparison_summary
        }

    def _create_visualizations(self, cv_results: Dict, metrics_record: Dict, comparison_results: Dict):
        """Create comprehensive visualizations for Phase 7"""
        self.log("Creating visualizations...")
        
        try:
            # 1. Cross-validation scores comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Extract CV results
            models = []
            cv_accuracies = []
            cv_f1_scores = []
            cv_stds_acc = []
            cv_stds_f1 = []
            
            for name, results in cv_results['cv_results'].items():
                if 'error' not in results:
                    models.append(name)
                    cv_accuracies.append(results['test_accuracy']['mean'])
                    cv_f1_scores.append(results['test_f1']['mean'])
                    cv_stds_acc.append(results['test_accuracy']['std'])
                    cv_stds_f1.append(results['test_f1']['std'])
            
            # Plot 1: CV Accuracy with error bars
            axes[0, 0].barh(models, cv_accuracies, xerr=cv_stds_acc, capsize=5, color='skyblue')
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_title('Cross-Validation Accuracy (with std)')
            axes[0, 0].grid(axis='x', alpha=0.3)
            
            # Plot 2: CV F1 Score with error bars
            axes[0, 1].barh(models, cv_f1_scores, xerr=cv_stds_f1, capsize=5, color='lightcoral')
            axes[0, 1].set_xlabel('F1 Score')
            axes[0, 1].set_title('Cross-Validation F1 Score (with std)')
            axes[0, 1].grid(axis='x', alpha=0.3)
            
            # Plot 3: Train vs Test Accuracy
            train_accs = []
            test_accs = []
            model_names = []
            for metric in self.evaluation_metrics:
                model_names.append(metric['model'])
                train_accs.append(metric['train_accuracy'])
                test_accs.append(metric['test_accuracy'])
            
            x = np.arange(len(model_names))
            width = 0.35
            axes[1, 0].bar(x - width/2, train_accs, width, label='Train', color='green', alpha=0.7)
            axes[1, 0].bar(x + width/2, test_accs, width, label='Test', color='orange', alpha=0.7)
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Train vs Test Accuracy')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Plot 4: Overfitting Gap
            overfit_gaps = [metric['overfitting_gap'] for metric in self.evaluation_metrics]
            colors = ['red' if gap > 0.1 else 'green' for gap in overfit_gaps]
            axes[1, 1].barh(model_names, overfit_gaps, color=colors, alpha=0.7)
            axes[1, 1].set_xlabel('Overfitting Gap (Train - Test)')
            axes[1, 1].set_title('Overfitting Analysis')
            axes[1, 1].axvline(x=0.1, color='orange', linestyle='--', label='Warning threshold (0.1)')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            viz_path = self.visualization_dir / 'phase7_model_training_evaluation.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"  ✓ Saved visualization to {viz_path}")
            
            # 2. Baseline comparison visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            all_models = []
            all_f1_scores = []
            all_colors = []
            
            # Add baseline models
            for name, results in comparison_results['baseline_results'].items():
                if 'error' not in results:
                    all_models.append(name)
                    all_f1_scores.append(results['f1'])
                    all_colors.append('lightblue')
            
            # Add trained models
            for metric in self.evaluation_metrics:
                all_models.append(metric['model'])
                all_f1_scores.append(metric['test_f1'])
                all_colors.append('lightgreen')
            
            bars = ax.barh(all_models, all_f1_scores, color=all_colors, alpha=0.7)
            ax.set_xlabel('F1 Score')
            ax.set_title('Baseline vs Advanced Models Comparison')
            ax.grid(axis='x', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='Baseline Models'),
                Patch(facecolor='lightgreen', label='Advanced Models')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            viz_path2 = self.visualization_dir / 'phase7_baseline_comparison.png'
            plt.savefig(viz_path2, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"  ✓ Saved baseline comparison to {viz_path2}")
            
        except Exception as e:
            self.log(f"  ✗ Failed to create visualizations: {e}", "WARN")

    def save_results(self, results: Dict[str, Any]):
        """Save Phase 7 results to JSON file"""
        results_path = self.output_dir / "phase7_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        self.log(f"Results saved to: {results_path}")

    def run_phase_7(self) -> Dict[str, Any]:
        """Run complete Phase 7 debugging process"""
        self.log("🚀 Starting Phase 7: Model Training & Evaluation")
        
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
        
        # Run all subphases
        cv_results = self.subphase_7_1_train_with_cross_validation(X, y)
        metrics_results = self.subphase_7_2_record_metrics(X, y)
        comparison_results = self.subphase_7_3_compare_with_baseline(X, y)
        
        # Create visualizations
        self._create_visualizations(cv_results, metrics_results, comparison_results)
        
        # Compile all results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'data_source': self.data_source,
            'data_shape': {'n_samples': X.shape[0], 'n_features': X.shape[1]},
            'cv_results': cv_results,
            'metrics_results': metrics_results,
            'comparison_results': comparison_results,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results

    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of Phase 7 results"""
        self.log("=" * 70)
        self.log("PHASE 7 SUMMARY: MODEL TRAINING & EVALUATION")
        self.log("=" * 70)
        
        comparison_summary = results['comparison_results'].get('comparison_summary', {})
        
        if comparison_summary:
            self.log(f"\n🏆 Top Performers:")
            if 'best_by_accuracy' in comparison_summary:
                best_acc = comparison_summary['best_by_accuracy']
                self.log(f"  • Best Accuracy: {best_acc['model']} ({best_acc['test_accuracy']:.3f})")
            
            if 'best_by_f1' in comparison_summary:
                best_f1 = comparison_summary['best_by_f1']
                self.log(f"  • Best F1 Score: {best_f1['model']} ({best_f1['test_f1']:.3f})")
            
            if 'least_overfit' in comparison_summary:
                least_over = comparison_summary['least_overfit']
                self.log(f"  • Least Overfitting: {least_over['model']} (gap: {least_over['overfitting_gap']:.3f})")
        
        # Count models with overfitting issues
        overfit_count = sum(1 for m in self.evaluation_metrics if m['overfitting_gap'] > 0.1)
        if overfit_count > 0:
            self.log(f"\n⚠️  {overfit_count} model(s) show signs of overfitting (gap > 0.1)")
        else:
            self.log(f"\n✅ No significant overfitting detected in any model")
        
        self.log(f"\n📈 Visualizations saved to: {self.visualization_dir}")
        self.log(f"📊 Total models trained: {len(self.trained_models)}")
        self.log(f"📊 Baseline models: {len(self.baseline_models)}")
        self.log("✅ Phase 7 Model Training & Evaluation COMPLETE")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 7: Model Training & Evaluation')
    parser.add_argument('--data-source', default='synthetic', 
                       help='Data source: synthetic, alzheimer, or path to CSV file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run Phase 7 debugging
    evaluator = Phase7ModelTrainingEvaluator(
        verbose=args.verbose,
        data_source=args.data_source
    )
    
    try:
        results = evaluator.run_phase_7()
        print(f"\n✅ Phase 7 completed successfully!")
        print(f"Results saved to: {evaluator.output_dir / 'phase7_results.json'}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Phase 7 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
