#!/usr/bin/env python3
"""
Phase 4 Debugging Script: Model Architecture Verification

This script implements Phase 4 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 4.1: Ensure model architecture matches problem needs (avoid under/overfitting)
- Subphase 4.2: Start with simple models for baseline (e.g., linear regression, decision tree)
- Subphase 4.3: Gradually increase complexity, logging performance changes

Usage:
    python debug/phase4_model_architecture_debug.py [--data-source] [--verbose]
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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


class Phase4ModelArchitectureDebugger:
    """Phase 4 debugging implementation for AiMedRes model architecture verification"""
    
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
            format='[%(asctime)s] 📊 %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger("Phase4ModelArchitecture")
        
        # Initialize model complexity progression
        self.baseline_models = {}
        self.complex_models = {}
        self.performance_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "📊" if level == "INFO" else "⚠️ " if level == "WARN" else "❌"
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
            0.2 * (bmi - 25) / 10 +
            0.2 * (blood_pressure - 120) / 50 +
            0.15 * (cholesterol - 200) / 100 +
            0.15 * (glucose - 100) / 50 +
            np.random.normal(0, 0.3, n_samples)
        )
        
        # Convert to binary classification
        target = (risk_score > 0.5).astype(int)
        
        df = pd.DataFrame({
            'age': age,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'heart_rate': heart_rate,
            'target': target
        })
        
        return df, 'target'

    def load_data(self) -> Tuple[pd.DataFrame, str]:
        """Load data based on specified source"""
        self.log(f"Loading data from source: {self.data_source}")
        
        if self.data_source == "synthetic":
            return self.generate_synthetic_data()
        elif self.data_source == "alzheimer":
            try:
                df = create_test_alzheimer_data(n_samples=1000)
                return df, 'diagnosis'
            except Exception as e:
                self.log(f"Failed to load Alzheimer data: {e}, falling back to synthetic", "WARN")
                return self.generate_synthetic_data()
        else:
            # Try to load from file
            try:
                data_path = Path(self.data_source)
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    # Assume last column is target for now
                    target_col = df.columns[-1]
                    return df, target_col
                else:
                    raise FileNotFoundError(f"Data file not found: {data_path}")
            except Exception as e:
                self.log(f"Failed to load data from {self.data_source}: {e}, using synthetic", "WARN")
                return self.generate_synthetic_data()

    def subphase_4_1_architecture_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Subphase 4.1: Ensure model architecture matches problem needs"""
        self.log("=== SUBPHASE 4.1: ARCHITECTURE ANALYSIS ===")
        
        analysis_results = {
            'data_characteristics': {},
            'architecture_recommendations': {},
            'overfitting_risk_factors': []
        }
        
        # Data characteristics analysis
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        feature_variance = X.var().mean()
        
        analysis_results['data_characteristics'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'samples_per_feature_ratio': n_samples / n_features,
            'feature_variance': feature_variance,
            'class_balance': pd.Series(y).value_counts().to_dict()
        }
        
        self.log(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
        self.log(f"Samples-to-features ratio: {n_samples/n_features:.2f}")
        
        # Architecture recommendations
        if n_samples / n_features < 10:
            analysis_results['overfitting_risk_factors'].append("Low samples-to-features ratio")
            analysis_results['architecture_recommendations']['complexity'] = "low"
            analysis_results['architecture_recommendations']['regularization'] = "strong"
        elif n_samples / n_features < 50:
            analysis_results['architecture_recommendations']['complexity'] = "moderate"
            analysis_results['architecture_recommendations']['regularization'] = "moderate"
        else:
            analysis_results['architecture_recommendations']['complexity'] = "high"
            analysis_results['architecture_recommendations']['regularization'] = "light"
        
        # Check class imbalance
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() / class_counts.max() < 0.5:
            analysis_results['overfitting_risk_factors'].append("Class imbalance detected")
            analysis_results['architecture_recommendations']['class_weight'] = "balanced"
        
        # Visualize data characteristics
        self._visualize_data_characteristics(X, y)
        
        self.log(f"Overfitting risk factors: {len(analysis_results['overfitting_risk_factors'])}")
        for factor in analysis_results['overfitting_risk_factors']:
            self.log(f"  - {factor}", "WARN")
            
        return analysis_results

    def subphase_4_2_baseline_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Subphase 4.2: Start with simple models for baseline"""
        self.log("=== SUBPHASE 4.2: BASELINE MODELS ===")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define baseline models (simple)
        baseline_models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        }
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            self.log(f"Training baseline model: {name}")
            
            # Use scaled data for LogisticRegression
            if name == 'logistic_regression':
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # Train model
            model.fit(X_tr, y_train)
            
            # Evaluate
            train_score = model.score(X_tr, y_train)
            test_score = model.score(X_te, y_test)
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
            
            # Predictions for detailed metrics
            y_pred = model.predict(X_te)
            
            model_results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_score': train_score - test_score,
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            baseline_results[name] = model_results
            self.baseline_models[name] = model
            
            # Log performance
            self.performance_log.append({
                'model': name,
                'complexity': 'low',
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_accuracy': cv_scores.mean(),
                'overfitting_score': train_score - test_score
            })
            
            self.log(f"  Train Acc: {train_score:.3f}, Test Acc: {test_score:.3f}, "
                    f"CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            if train_score - test_score > 0.1:
                self.log(f"  ⚠️  Potential overfitting detected (gap: {train_score - test_score:.3f})", "WARN")
        
        return baseline_results

    def subphase_4_3_progressive_complexity(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Subphase 4.3: Gradually increase complexity, logging performance changes"""
        self.log("=== SUBPHASE 4.3: PROGRESSIVE COMPLEXITY ===")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with increasing complexity
        complex_models = {
            'random_forest_simple': RandomForestClassifier(
                n_estimators=10, max_depth=5, random_state=42
            ),
            'random_forest_moderate': RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42
            ),
            'random_forest_complex': RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=42
            ),
            'svm_linear': SVC(kernel='linear', random_state=42, probability=True),
            'svm_rbf': SVC(kernel='rbf', random_state=42, probability=True),
            'mlp_simple': MLPClassifier(
                hidden_layer_sizes=(50,), max_iter=1000, random_state=42
            ),
            'mlp_complex': MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
            )
        }
        
        complexity_results = {}
        
        for name, model in complex_models.items():
            self.log(f"Training complex model: {name}")
            
            try:
                # Use scaled data for SVM and MLP
                if 'svm' in name or 'mlp' in name:
                    X_tr, X_te = X_train_scaled, X_test_scaled
                else:
                    X_tr, X_te = X_train, X_test
                
                # Train model
                model.fit(X_tr, y_train)
                
                # Evaluate
                train_score = model.score(X_tr, y_train)
                test_score = model.score(X_te, y_test)
                cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
                
                # Predictions for detailed metrics
                y_pred = model.predict(X_te)
                
                model_results = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'overfitting_score': train_score - test_score,
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                complexity_results[name] = model_results
                self.complex_models[name] = model
                
                # Determine complexity level
                if 'simple' in name or 'linear' in name:
                    complexity = 'moderate'
                else:
                    complexity = 'high'
                
                # Log performance
                self.performance_log.append({
                    'model': name,
                    'complexity': complexity,
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_accuracy': cv_scores.mean(),
                    'overfitting_score': train_score - test_score
                })
                
                self.log(f"  Train Acc: {train_score:.3f}, Test Acc: {test_score:.3f}, "
                        f"CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
                if train_score - test_score > 0.15:
                    self.log(f"  ⚠️  Significant overfitting detected (gap: {train_score - test_score:.3f})", "WARN")
                    
            except Exception as e:
                self.log(f"  ❌ Failed to train {name}: {e}", "ERROR")
                complexity_results[name] = {'error': str(e)}
        
        return complexity_results

    def _visualize_data_characteristics(self, X: pd.DataFrame, y: pd.Series):
        """Create visualizations for data characteristics"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Phase 4.1: Data Characteristics Analysis', fontsize=16)
            
            # 1. Feature distributions
            X.hist(bins=20, ax=axes[0, 0])
            axes[0, 0].set_title('Feature Distributions')
            
            # 2. Class balance
            class_counts = pd.Series(y).value_counts()
            axes[0, 1].bar(range(len(class_counts)), class_counts.values)
            axes[0, 1].set_title('Class Distribution')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Count')
            
            # 3. Feature correlation heatmap
            if X.shape[1] <= 10:  # Only for reasonable number of features
                corr = X.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('Feature Correlations')
            else:
                axes[1, 0].text(0.5, 0.5, f'Too many features ({X.shape[1]}) for correlation plot', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Correlations (Skipped)')
            
            # 4. Samples vs Features ratio visualization
            axes[1, 1].bar(['Samples', 'Features'], [X.shape[0], X.shape[1]])
            axes[1, 1].set_title('Dataset Dimensions')
            axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(self.visualization_dir / 'phase4_data_characteristics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Data characteristics visualization saved to {self.visualization_dir / 'phase4_data_characteristics.png'}")
            
        except Exception as e:
            self.log(f"Failed to create data characteristics visualization: {e}", "WARN")

    def _visualize_performance_progression(self):
        """Create visualization showing performance progression with model complexity"""
        try:
            if not self.performance_log:
                self.log("No performance data to visualize", "WARN")
                return
            
            df_perf = pd.DataFrame(self.performance_log)
            
            # Create performance progression plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Phase 4.3: Model Performance Progression', fontsize=16)
            
            # 1. Accuracy progression
            models = df_perf['model'].values
            x_pos = range(len(models))
            
            axes[0, 0].plot(x_pos, df_perf['train_accuracy'], 'o-', label='Train Accuracy', color='blue')
            axes[0, 0].plot(x_pos, df_perf['test_accuracy'], 's-', label='Test Accuracy', color='red')
            axes[0, 0].plot(x_pos, df_perf['cv_accuracy'], '^-', label='CV Accuracy', color='green')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Progression')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Overfitting progression
            axes[0, 1].bar(x_pos, df_perf['overfitting_score'], alpha=0.7, color='orange')
            axes[0, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Train - Test Accuracy')
            axes[0, 1].set_title('Overfitting Score (Train - Test)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Complexity vs Performance
            complexity_order = {'low': 1, 'moderate': 2, 'high': 3}
            df_perf['complexity_num'] = df_perf['complexity'].map(complexity_order)
            
            scatter = axes[1, 0].scatter(df_perf['complexity_num'], df_perf['test_accuracy'], 
                                       c=df_perf['overfitting_score'], cmap='RdYlBu_r', 
                                       s=100, alpha=0.7)
            axes[1, 0].set_xlabel('Model Complexity')
            axes[1, 0].set_ylabel('Test Accuracy')
            axes[1, 0].set_title('Complexity vs Performance')
            axes[1, 0].set_xticks([1, 2, 3])
            axes[1, 0].set_xticklabels(['Low', 'Moderate', 'High'])
            plt.colorbar(scatter, ax=axes[1, 0], label='Overfitting Score')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Best model recommendations
            best_test_model = df_perf.loc[df_perf['test_accuracy'].idxmax()]
            best_cv_model = df_perf.loc[df_perf['cv_accuracy'].idxmax()]
            least_overfitting = df_perf.loc[df_perf['overfitting_score'].idxmin()]
            
            recommendations = f"""Best Test Accuracy: {best_test_model['model']} ({best_test_model['test_accuracy']:.3f})
Best CV Accuracy: {best_cv_model['model']} ({best_cv_model['cv_accuracy']:.3f})
Least Overfitting: {least_overfitting['model']} ({least_overfitting['overfitting_score']:.3f})"""
            
            axes[1, 1].text(0.05, 0.95, recommendations, transform=axes[1, 1].transAxes, 
                            verticalalignment='top', fontsize=12, fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('Model Recommendations')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.visualization_dir / 'phase4_performance_progression.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"Performance progression visualization saved to {self.visualization_dir / 'phase4_performance_progression.png'}")
            
        except Exception as e:
            self.log(f"Failed to create performance progression visualization: {e}", "WARN")

    def generate_recommendations(self, architecture_analysis: Dict, baseline_results: Dict, 
                               complexity_results: Dict) -> Dict[str, Any]:
        """Generate architecture recommendations based on all analyses"""
        recommendations = {
            'best_baseline_model': None,
            'best_complex_model': None,
            'architecture_advice': [],
            'overfitting_warnings': [],
            'performance_summary': {}
        }
        
        # Find best performing models
        all_results = {**baseline_results, **complexity_results}
        
        # Filter out error results
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        if valid_results:
            # Best test accuracy
            best_test = max(valid_results.items(), key=lambda x: x[1]['test_accuracy'])
            recommendations['best_overall_model'] = {
                'name': best_test[0],
                'test_accuracy': best_test[1]['test_accuracy'],
                'overfitting_score': best_test[1]['overfitting_score']
            }
            
            # Best baseline
            baseline_valid = {k: v for k, v in baseline_results.items() if 'error' not in v}
            if baseline_valid:
                best_baseline = max(baseline_valid.items(), key=lambda x: x[1]['test_accuracy'])
                recommendations['best_baseline_model'] = {
                    'name': best_baseline[0],
                    'test_accuracy': best_baseline[1]['test_accuracy']
                }
            
            # Best complex model
            complex_valid = {k: v for k, v in complexity_results.items() if 'error' not in v}
            if complex_valid:
                best_complex = max(complex_valid.items(), key=lambda x: x[1]['test_accuracy'])
                recommendations['best_complex_model'] = {
                    'name': best_complex[0],
                    'test_accuracy': best_complex[1]['test_accuracy']
                }
        
        # Architecture advice based on analysis
        if architecture_analysis['overfitting_risk_factors']:
            recommendations['architecture_advice'].append(
                "Use regularization techniques (L1/L2, early stopping)"
            )
            recommendations['architecture_advice'].append(
                "Consider data augmentation or feature selection"
            )
        
        if architecture_analysis['data_characteristics']['samples_per_feature_ratio'] < 10:
            recommendations['architecture_advice'].append(
                "Prioritize simple models to avoid overfitting"
            )
        
        # Check for overfitting across all models
        for name, results in valid_results.items():
            if results['overfitting_score'] > 0.15:
                recommendations['overfitting_warnings'].append(
                    f"{name}: High overfitting (gap: {results['overfitting_score']:.3f})"
                )
        
        return recommendations

    def save_results(self, all_results: Dict[str, Any]):
        """Save all results to JSON file"""
        results_file = self.output_dir / "phase4_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        serializable_results = convert_numpy(all_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.log(f"Results saved to {results_file}")

    def run_phase_4(self) -> Dict[str, Any]:
        """Run complete Phase 4 debugging process"""
        self.log("🚀 Starting Phase 4: Model Architecture Verification")
        
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
        architecture_analysis = self.subphase_4_1_architecture_analysis(X, y)
        baseline_results = self.subphase_4_2_baseline_models(X, y)
        complexity_results = self.subphase_4_3_progressive_complexity(X, y)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            architecture_analysis, baseline_results, complexity_results
        )
        
        # Create visualizations
        self._visualize_performance_progression()
        
        # Compile all results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'data_source': self.data_source,
            'architecture_analysis': architecture_analysis,
            'baseline_results': baseline_results,
            'complexity_results': complexity_results,
            'recommendations': recommendations,
            'performance_log': self.performance_log
        }
        
        # Save results
        self.save_results(all_results)
        
        # Print summary
        self._print_summary(recommendations)
        
        return all_results

    def _print_summary(self, recommendations: Dict[str, Any]):
        """Print summary of Phase 4 results"""
        self.log("=" * 60)
        self.log("PHASE 4 SUMMARY: MODEL ARCHITECTURE VERIFICATION")
        self.log("=" * 60)
        
        if recommendations.get('best_overall_model'):
            best = recommendations['best_overall_model']
            self.log(f"🏆 Best Overall Model: {best['name']}")
            self.log(f"   Test Accuracy: {best['test_accuracy']:.3f}")
            self.log(f"   Overfitting Score: {best['overfitting_score']:.3f}")
        
        if recommendations.get('best_baseline_model'):
            baseline = recommendations['best_baseline_model']
            self.log(f"📊 Best Baseline Model: {baseline['name']} (Acc: {baseline['test_accuracy']:.3f})")
        
        if recommendations.get('best_complex_model'):
            complex_model = recommendations['best_complex_model']
            self.log(f"🧠 Best Complex Model: {complex_model['name']} (Acc: {complex_model['test_accuracy']:.3f})")
        
        if recommendations['overfitting_warnings']:
            self.log("⚠️  Overfitting Warnings:")
            for warning in recommendations['overfitting_warnings']:
                self.log(f"   - {warning}")
        
        if recommendations['architecture_advice']:
            self.log("💡 Architecture Recommendations:")
            for advice in recommendations['architecture_advice']:
                self.log(f"   - {advice}")
        
        self.log(f"📈 Visualizations saved to: {self.visualization_dir}")
        self.log("✅ Phase 4 Model Architecture Verification COMPLETE")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 4: Model Architecture Verification')
    parser.add_argument('--data-source', default='synthetic', 
                       help='Data source: synthetic, alzheimer, or path to CSV file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run Phase 4 debugging
    debugger = Phase4ModelArchitectureDebugger(
        verbose=args.verbose,
        data_source=args.data_source
    )
    
    try:
        results = debugger.run_phase_4()
        print(f"\n✅ Phase 4 completed successfully!")
        print(f"Results saved to: {debugger.output_dir / 'phase4_results.json'}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Phase 4 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())