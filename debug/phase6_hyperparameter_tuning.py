#!/usr/bin/env python3
"""
Phase 6 Debugging Script: Hyperparameter Tuning & Search

This script implements Phase 6 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 6.1: Identify key hyperparameters for tuning (learning rate, batch size, etc.)
- Subphase 6.2: Use grid search, random search, or Bayesian optimization for tuning
- Subphase 6.3: Track and visualize tuning results to identify optimal settings

Usage:
    python debug/phase6_hyperparameter_tuning.py [--data-source] [--verbose] [--method]
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available. Bayesian optimization will be skipped.")

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase6Debug")


class HyperparameterIdentifier:
    """Subphase 6.1: Identify key hyperparameters for tuning"""
    
    @staticmethod
    def get_hyperparameter_space(model_type: str) -> Dict[str, Any]:
        """Get hyperparameter space for different model types"""
        spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000, 3000]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'max_iter': [500, 1000, 2000]
            },
            'decision_tree': {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        }
        return spaces.get(model_type, {})
    
    @staticmethod
    def get_model_instance(model_type: str) -> Any:
        """Get model instance by type"""
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42),
            'mlp': MLPClassifier(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        return models.get(model_type)


class HyperparameterSearcher:
    """Subphase 6.2: Use grid search, random search, or Bayesian optimization for tuning"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'accuracy', random_state: int = 42):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.results = {}
    
    def grid_search(self, model, param_grid: Dict, X_train, y_train) -> Dict[str, Any]:
        """Perform grid search hyperparameter tuning"""
        logger.info(f"🔍 Starting Grid Search with {len(param_grid)} parameter combinations...")
        
        start_time = time.time()
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds, 
            scoring=self.scoring, n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        duration = time.time() - start_time
        
        results = {
            'method': 'Grid Search',
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'search_time': duration,
            'n_combinations_tested': len(grid_search.cv_results_['params']),
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }
        
        logger.info(f"✅ Grid Search completed in {duration:.2f}s")
        logger.info(f"📊 Best CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"🎯 Best Parameters: {grid_search.best_params_}")
        
        return results
    
    def random_search(self, model, param_grid: Dict, X_train, y_train, n_iter: int = 50) -> Dict[str, Any]:
        """Perform random search hyperparameter tuning"""
        logger.info(f"🎲 Starting Random Search with {n_iter} iterations...")
        
        start_time = time.time()
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=self.cv_folds,
            scoring=self.scoring, n_jobs=-1, random_state=self.random_state, verbose=0
        )
        random_search.fit(X_train, y_train)
        duration = time.time() - start_time
        
        results = {
            'method': 'Random Search',
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'search_time': duration,
            'n_combinations_tested': n_iter,
            'cv_results': random_search.cv_results_,
            'best_estimator': random_search.best_estimator_
        }
        
        logger.info(f"✅ Random Search completed in {duration:.2f}s")
        logger.info(f"📊 Best CV Score: {random_search.best_score_:.4f}")
        logger.info(f"🎯 Best Parameters: {random_search.best_params_}")
        
        return results
    
    def bayesian_optimization(self, model_type: str, X_train, y_train, n_trials: int = 50) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("⚠️  Optuna not available. Skipping Bayesian optimization.")
            return {}
        
        logger.info(f"🧠 Starting Bayesian Optimization with {n_trials} trials...")
        
        def objective(trial):
            # Define hyperparameter suggestions based on model type
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(random_state=self.random_state, **params)
            
            elif model_type == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.01, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': trial.suggest_int('max_iter', 1000, 3000)
                }
                model = LogisticRegression(random_state=self.random_state, **params)
            
            elif model_type == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
                }
                model = SVC(random_state=self.random_state, **params)
            
            else:
                raise ValueError(f"Bayesian optimization not implemented for {model_type}")
            
            # Evaluate using cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=self.scoring)
            return scores.mean()
        
        start_time = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        duration = time.time() - start_time
        
        results = {
            'method': 'Bayesian Optimization (Optuna)',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'search_time': duration,
            'n_trials': n_trials,
            'study': study
        }
        
        logger.info(f"✅ Bayesian Optimization completed in {duration:.2f}s")
        logger.info(f"📊 Best CV Score: {study.best_value:.4f}")
        logger.info(f"🎯 Best Parameters: {study.best_params}")
        
        return results


class ResultsVisualizer:
    """Subphase 6.3: Track and visualize tuning results to identify optimal settings"""
    
    def __init__(self, output_dir: str = "debug/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_search_comparison(self, results_list: List[Dict], model_name: str):
        """Compare different search methods"""
        if not results_list:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Hyperparameter Tuning Comparison - {model_name}', fontsize=16)
        
        # Extract data for comparison
        methods = [r['method'] for r in results_list if r]
        scores = [r['best_score'] for r in results_list if r]
        times = [r['search_time'] for r in results_list if r]
        n_evaluated = [r.get('n_combinations_tested', r.get('n_trials', 0)) for r in results_list if r]
        
        # Plot 1: Best scores comparison
        axes[0, 0].bar(methods, scores, color=['skyblue', 'lightgreen', 'lightcoral'][:len(methods)])
        axes[0, 0].set_title('Best CV Scores')
        axes[0, 0].set_ylabel('Cross-Validation Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Search time comparison
        axes[0, 1].bar(methods, times, color=['skyblue', 'lightgreen', 'lightcoral'][:len(methods)])
        axes[0, 1].set_title('Search Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Number of evaluations
        axes[1, 0].bar(methods, n_evaluated, color=['skyblue', 'lightgreen', 'lightcoral'][:len(methods)])
        axes[1, 0].set_title('Number of Evaluations')
        axes[1, 0].set_ylabel('Combinations/Trials Tested')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Efficiency (score per second)
        efficiency = [s/t if t > 0 else 0 for s, t in zip(scores, times)]
        axes[1, 1].bar(methods, efficiency, color=['skyblue', 'lightgreen', 'lightcoral'][:len(methods)])
        axes[1, 1].set_title('Efficiency (Score/Second)')
        axes[1, 1].set_ylabel('Score per Second')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'hyperparameter_comparison_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved comparison plot: {self.output_dir}/hyperparameter_comparison_{model_name.lower()}.png")
    
    def plot_bayesian_optimization_history(self, study, model_name: str):
        """Plot Bayesian optimization history if available"""
        if not OPTUNA_AVAILABLE or not study:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot optimization history
        trials = study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]
        
        axes[0].plot(trial_numbers, values, 'b-', alpha=0.7)
        axes[0].scatter(trial_numbers, values, c='blue', alpha=0.7)
        axes[0].set_title('Optimization History')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Objective Value')
        axes[0].grid(True, alpha=0.3)
        
        # Plot parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            axes[1].barh(params, importances)
            axes[1].set_title('Parameter Importance')
            axes[1].set_xlabel('Importance')
        except Exception:
            axes[1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'bayesian_optimization_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved Bayesian optimization plot: {self.output_dir}/bayesian_optimization_{model_name.lower()}.png")


class Phase6HyperparameterTuning:
    """Main class orchestrating Phase 6 hyperparameter tuning"""
    
    def __init__(self, output_dir: str = "debug"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "visualizations"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.identifier = HyperparameterIdentifier()
        self.searcher = HyperparameterSearcher()
        self.visualizer = ResultsVisualizer(str(self.results_dir))
        
        self.all_results = {}
    
    def load_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load sample data for hyperparameter tuning"""
        try:
            # Try to use existing data generation
            df = create_test_alzheimer_data(n_samples=300)
            
            # Preprocess data
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            # Encode categorical variables
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            
            # Scale numerical features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Encode target
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            
            logger.info(f"✅ Loaded data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            return X_scaled, y_encoded
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load project data: {e}")
            logger.info("🔧 Generating synthetic data...")
            
            # Generate synthetic data as fallback
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=300, n_features=10, n_classes=2, 
                                     n_informative=7, random_state=42)
            return X, y
    
    def run_hyperparameter_tuning(self, model_types: List[str] = None, 
                                 methods: List[str] = None) -> Dict[str, Any]:
        """Run complete hyperparameter tuning for specified models and methods"""
        
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'svm']
        
        if methods is None:
            methods = ['grid_search', 'random_search']
            if OPTUNA_AVAILABLE:
                methods.append('bayesian_optimization')
        
        logger.info("🚀 Starting Phase 6: Hyperparameter Tuning & Search")
        logger.info(f"📋 Models to tune: {model_types}")
        logger.info(f"📋 Methods to use: {methods}")
        
        # Load data
        X, y = self.load_sample_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
        
        phase6_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'models_tuned': {},
            'summary': {}
        }
        
        for model_type in model_types:
            logger.info(f"\n🔧 Tuning {model_type}...")
            
            # Subphase 6.1: Identify key hyperparameters
            param_space = self.identifier.get_hyperparameter_space(model_type)
            if not param_space:
                logger.warning(f"⚠️  No hyperparameter space defined for {model_type}")
                continue
            
            model = self.identifier.get_model_instance(model_type)
            if model is None:
                logger.warning(f"⚠️  Could not create model instance for {model_type}")
                continue
            
            logger.info(f"📝 Identified {len(param_space)} hyperparameters for {model_type}")
            
            model_results = []
            
            # Subphase 6.2: Apply different search methods
            if 'grid_search' in methods:
                try:
                    grid_result = self.searcher.grid_search(model, param_space, X_train, y_train)
                    model_results.append(grid_result)
                except Exception as e:
                    logger.error(f"❌ Grid search failed for {model_type}: {e}")
            
            if 'random_search' in methods:
                try:
                    random_result = self.searcher.random_search(model, param_space, X_train, y_train)
                    model_results.append(random_result)
                except Exception as e:
                    logger.error(f"❌ Random search failed for {model_type}: {e}")
            
            if 'bayesian_optimization' in methods and OPTUNA_AVAILABLE:
                try:
                    bayesian_result = self.searcher.bayesian_optimization(model_type, X_train, y_train)
                    if bayesian_result:
                        model_results.append(bayesian_result)
                except Exception as e:
                    logger.error(f"❌ Bayesian optimization failed for {model_type}: {e}")
            
            if model_results:
                # Store results
                phase6_results['models_tuned'][model_type] = {
                    'hyperparameter_space': param_space,
                    'search_results': model_results,
                    'best_method': max(model_results, key=lambda x: x['best_score'])['method'],
                    'best_score': max(model_results, key=lambda x: x['best_score'])['best_score']
                }
                
                # Subphase 6.3: Visualize results
                self.visualizer.plot_search_comparison(model_results, model_type)
                
                # If Bayesian optimization was used, create additional plots
                for result in model_results:
                    if result.get('method') == 'Bayesian Optimization (Optuna)':
                        self.visualizer.plot_bayesian_optimization_history(
                            result.get('study'), model_type)
        
        # Generate summary
        if phase6_results['models_tuned']:
            best_overall = max(
                phase6_results['models_tuned'].items(),
                key=lambda x: x[1]['best_score']
            )
            phase6_results['summary'] = {
                'best_model_type': best_overall[0],
                'best_score': best_overall[1]['best_score'],
                'best_method': best_overall[1]['best_method'],
                'total_models_tuned': len(phase6_results['models_tuned'])
            }
        
        # Save results
        results_file = self.output_dir / "phase6_results.json"
        with open(results_file, 'w') as f:
            # Convert non-serializable objects for JSON
            serializable_results = self._make_serializable(phase6_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info("✅ Phase 6: Hyperparameter Tuning & Search completed!")
        
        self.all_results = phase6_results
        return phase6_results
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            return obj


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Phase 6: Hyperparameter Tuning & Search')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'logistic_regression', 'svm', 'mlp', 'decision_tree'],
                       default=['random_forest', 'logistic_regression', 'svm'],
                       help='Models to tune')
    parser.add_argument('--methods', nargs='+',
                       choices=['grid_search', 'random_search', 'bayesian_optimization'],
                       default=['grid_search', 'random_search', 'bayesian_optimization'],
                       help='Search methods to use')
    parser.add_argument('--output-dir', default='debug',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize Phase 6 tuning
    phase6 = Phase6HyperparameterTuning(output_dir=args.output_dir)
    
    # Run hyperparameter tuning
    results = phase6.run_hyperparameter_tuning(
        model_types=args.models,
        methods=args.methods
    )
    
    # Print summary
    if results.get('summary'):
        print("\n" + "="*50)
        print("📊 PHASE 6 SUMMARY")
        print("="*50)
        print(f"🏆 Best Model: {results['summary']['best_model_type']}")
        print(f"📈 Best Score: {results['summary']['best_score']:.4f}")
        print(f"🔍 Best Method: {results['summary']['best_method']}")
        print(f"🎯 Models Tuned: {results['summary']['total_models_tuned']}")
        print("="*50)


if __name__ == "__main__":
    main()