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
from sklearn.metrics import accuracy_score, classification_report, make_scorer, confusion_matrix
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
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Warning: Optuna not available. Bayesian optimization will be disabled.")

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
    
    def identify_key_hyperparameters(self, model_types: List[str] = None) -> Dict[str, Dict]:
        """Identify key hyperparameters for specified model types"""
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression', 'svm', 'mlp', 'decision_tree']
        
        identified_params = {}
        for model_type in model_types:
            params = self.get_hyperparameter_space(model_type)
            if params:
                identified_params[model_type] = params
                logger.info(f"✅ Identified {len(params)} hyperparameters for {model_type}")
            else:
                logger.warning(f"⚠️  No hyperparameters defined for {model_type}")
        
        return identified_params


class HyperparameterSearcher:
    """Subphase 6.2: Use grid search, random search, or Bayesian optimization for tuning"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'accuracy', random_state: int = 42):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.results = {}
    
    def grid_search(self, model, param_grid: Dict, X_train, y_train) -> Dict[str, Any]:
        """Perform grid search hyperparameter tuning"""
        logger.info(f"🔍 Starting Grid Search with {len(param_grid)} parameter groups...")
        
        start_time = time.time()
        
        # Calculate total combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        logger.info(f"📊 Testing {total_combinations} parameter combinations")
        
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
            'n_combinations_tested': total_combinations,
            'cv_results': grid_search.cv_results_
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
            scoring=self.scoring, random_state=self.random_state, n_jobs=-1, verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        duration = time.time() - start_time
        
        results = {
            'method': 'Random Search',
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_,
            'search_time': duration,
            'n_trials': n_iter,
            'cv_results': random_search.cv_results_
        }
        
        logger.info(f"✅ Random Search completed in {duration:.2f}s")
        logger.info(f"📊 Best CV Score: {random_search.best_score_:.4f}")
        logger.info(f"🎯 Best Parameters: {random_search.best_params_}")
        
        return results
    
    def bayesian_optimization(self, model_type: str, X_train, y_train, n_trials: int = 50) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            logger.warning("⚠️  Optuna not available. Skipping Bayesian optimization.")
            return None
        
        logger.info(f"🧠 Starting Bayesian Optimization with {n_trials} trials...")
        
        start_time = time.time()
        
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
                if params['kernel'] in ['rbf', 'poly']:
                    params['gamma'] = trial.suggest_float('gamma', 0.001, 1, log=True)
                model = SVC(random_state=self.random_state, **params)
            
            elif model_type == 'mlp':
                hidden_size = trial.suggest_int('hidden_size', 50, 200)
                n_layers = trial.suggest_int('n_layers', 1, 2)
                if n_layers == 1:
                    hidden_layer_sizes = (hidden_size,)
                else:
                    hidden_layer_sizes = (hidden_size, hidden_size // 2)
                
                params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                    'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
                    'max_iter': trial.suggest_int('max_iter', 500, 2000)
                }
                model = MLPClassifier(random_state=self.random_state, **params)
            
            elif model_type == 'decision_tree':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                }
                model = DecisionTreeClassifier(random_state=self.random_state, **params)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=self.scoring)
            return scores.mean()
        
        # Create and run study
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        duration = time.time() - start_time
        
        results = {
            'method': 'Bayesian Optimization',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'search_time': duration,
            'n_trials': n_trials,
            'study': study  # Keep study object for visualization
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
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Time comparison
        axes[0, 1].bar(methods, times, color=['orange', 'purple', 'brown'][:len(methods)])
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
        axes[1, 1].bar(methods, efficiency, color=['gold', 'silver', 'bronze'][:len(methods)])
        axes[1, 1].set_title('Efficiency (Score/Second)')
        axes[1, 1].set_ylabel('Score per Second')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.output_dir / f'search_comparison_{model_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Comparison plot saved to: {save_path}")
        return save_path
    
    def plot_bayesian_optimization_history(self, study, model_name: str):
        """Plot Bayesian optimization history if available"""
        if not OPTUNA_AVAILABLE or study is None:
            return None
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Bayesian Optimization History - {model_name}', fontsize=16)
            
            # Plot 1: Optimization history
            trials = study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            trial_numbers = list(range(1, len(values) + 1))
            
            axes[0].plot(trial_numbers, values, 'b-', alpha=0.6, label='Trial values')
            best_values = []
            best_so_far = float('-inf')
            for value in values:
                if value > best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)
            
            axes[0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best value so far')
            axes[0].set_xlabel('Trial Number')
            axes[0].set_ylabel('CV Score')
            axes[0].set_title('Optimization Progress')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Parameter importance (if available)
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    params = list(importance.keys())
                    importances = list(importance.values())
                    
                    axes[1].barh(params, importances)
                    axes[1].set_xlabel('Importance')
                    axes[1].set_title('Parameter Importance')
                else:
                    axes[1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                               ha='center', va='center', transform=axes[1].transAxes)
            except Exception as e:
                axes[1].text(0.5, 0.5, f'Error calculating\nparameter importance:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1].transAxes)
            
            plt.tight_layout()
            save_path = self.output_dir / f'bayesian_optimization_{model_name}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Bayesian optimization plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.warning(f"⚠️  Could not create Bayesian optimization plot: {e}")
            return None
    
    def save_results_json(self, results: Dict, filename: str = "phase6_results.json"):
        """Save results to JSON file"""
        save_path = self.output_dir.parent / filename
        
        # Clean results for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if k == 'study':  # Skip study objects
                        continue
                    elif k == 'cv_results':  # Simplify cv_results
                        json_results[key][k] = {'mean_test_score': list(v.get('mean_test_score', []))}
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {save_path}")
        return save_path


class Phase6HyperparameterTuning:
    """Main class orchestrating Phase 6 hyperparameter tuning"""
    
    def __init__(self, output_dir: str = "debug", cv_folds: int = 5, random_state: int = 42):
        self.identifier = HyperparameterIdentifier()
        self.searcher = HyperparameterSearcher(cv_folds=cv_folds, random_state=random_state)
        self.visualizer = ResultsVisualizer(output_dir=f"{output_dir}/visualizations")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
    
    def run_hyperparameter_tuning(self, 
                                 model_types: List[str] = None,
                                 methods: List[str] = None,
                                 X_data=None, y_data=None) -> Dict[str, Any]:
        """Run complete hyperparameter tuning process"""
        logger.info("🚀 Starting Phase 6: Hyperparameter Tuning & Search")
        logger.info("=" * 60)
        
        if model_types is None:
            model_types = ['random_forest', 'logistic_regression']  # Start with fast models
        
        if methods is None:
            methods = ['grid_search', 'random_search']
            if OPTUNA_AVAILABLE:
                methods.append('bayesian_optimization')
        
        # Generate or use provided data
        if X_data is None or y_data is None:
            logger.info("📊 Generating synthetic dataset for tuning...")
            from sklearn.datasets import make_classification
            X_data, y_data = make_classification(
                n_samples=500, n_features=20, n_classes=3, 
                n_informative=15, n_redundant=3, random_state=self.random_state
            )
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Encode labels if necessary
        if y_data.dtype == 'object':
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y_data)
        else:
            y_encoded = y_data
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X_scaled.shape,
            'methods_used': methods,
            'models_tested': model_types,
            'summary': {}
        }
        
        # Phase 6.1: Identify hyperparameters
        logger.info("\n📋 Phase 6.1: Identifying key hyperparameters...")
        identified_params = self.identifier.identify_key_hyperparameters(model_types)
        results['identified_hyperparameters'] = identified_params
        
        # Phase 6.2 & 6.3: Search and visualize for each model
        best_overall_score = 0
        best_overall_method = None
        best_overall_model = None
        
        for model_type in model_types:
            logger.info(f"\n🔧 Testing {model_type.replace('_', ' ').title()}...")
            logger.info("-" * 40)
            
            model = self.identifier.get_model_instance(model_type)
            param_space = identified_params.get(model_type, {})
            
            if not param_space:
                logger.warning(f"⚠️  No hyperparameters defined for {model_type}")
                continue
            
            model_results = []
            
            # Run each search method
            for method in methods:
                logger.info(f"\n🔍 Running {method.replace('_', ' ').title()}...")
                
                try:
                    if method == 'grid_search':
                        result = self.searcher.grid_search(model, param_space, X_scaled, y_encoded)
                    elif method == 'random_search':
                        result = self.searcher.random_search(model, param_space, X_scaled, y_encoded, n_iter=20)
                    elif method == 'bayesian_optimization':
                        result = self.searcher.bayesian_optimization(model_type, X_scaled, y_encoded, n_trials=20)
                    else:
                        logger.warning(f"⚠️  Unknown method: {method}")
                        continue
                    
                    if result:
                        model_results.append(result)
                        
                        # Track best overall
                        if result['best_score'] > best_overall_score:
                            best_overall_score = result['best_score']
                            best_overall_method = method
                            best_overall_model = model_type
                
                except Exception as e:
                    logger.error(f"❌ Error running {method} for {model_type}: {e}")
            
            # Store results and create visualizations
            results[model_type] = model_results
            
            # Phase 6.3: Visualize results
            if model_results:
                self.visualizer.plot_search_comparison(model_results, model_type)
                
                # Create Bayesian optimization plot if available
                for result in model_results:
                    if result['method'] == 'Bayesian Optimization' and 'study' in result:
                        self.visualizer.plot_bayesian_optimization_history(result['study'], model_type)
        
        # Create summary
        results['summary'] = {
            'best_overall_score': best_overall_score,
            'best_method': best_overall_method,
            'best_model': best_overall_model,
            'total_models_tested': len(model_types),
            'total_methods_used': len(methods)
        }
        
        # Save results
        self.visualizer.save_results_json(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Phase 6 Hyperparameter Tuning Complete!")
        logger.info(f"🏆 Best Overall: {best_overall_method} on {best_overall_model} (Score: {best_overall_score:.4f})")
        logger.info("=" * 60)
        
        return results


def create_synthetic_data(data_type: str = "balanced") -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing"""
    from sklearn.datasets import make_classification
    
    if data_type == "balanced":
        return make_classification(
            n_samples=300, n_features=15, n_classes=3,
            n_informative=10, n_redundant=3, random_state=42
        )
    elif data_type == "imbalanced":
        return make_classification(
            n_samples=300, n_features=15, n_classes=2,
            n_informative=10, weights=[0.9, 0.1], random_state=42
        )
    else:
        return make_classification(
            n_samples=200, n_features=10, n_classes=2,
            n_informative=8, random_state=42
        )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Phase 6: Hyperparameter Tuning & Search")
    parser.add_argument("--data-source", choices=["synthetic", "alzheimer"], 
                       default="synthetic", help="Data source to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--method", choices=["all", "grid", "random", "bayesian"], 
                       default="all", help="Search methods to use")
    parser.add_argument("--models", nargs="+", 
                       choices=["random_forest", "logistic_regression", "svm", "mlp", "decision_tree"],
                       default=["random_forest", "logistic_regression"],
                       help="Models to tune")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine methods to use
    if args.method == "all":
        methods = ["grid_search", "random_search"]
        if OPTUNA_AVAILABLE:
            methods.append("bayesian_optimization")
    elif args.method == "grid":
        methods = ["grid_search"]
    elif args.method == "random":
        methods = ["random_search"]
    elif args.method == "bayesian":
        methods = ["bayesian_optimization"] if OPTUNA_AVAILABLE else ["random_search"]
    
    # Load or create data
    X_data, y_data = None, None
    if args.data_source == "synthetic":
        X_data, y_data = create_synthetic_data("balanced")
    elif args.data_source == "alzheimer":
        try:
            X_data, y_data = create_test_alzheimer_data()
        except Exception as e:
            logger.warning(f"⚠️  Could not load Alzheimer data: {e}")
            logger.info("📊 Using synthetic data instead")
            X_data, y_data = create_synthetic_data("balanced")
    
    # Run Phase 6
    phase6 = Phase6HyperparameterTuning()
    results = phase6.run_hyperparameter_tuning(
        model_types=args.models,
        methods=methods,
        X_data=X_data,
        y_data=y_data
    )
    
    # Print summary
    if results.get('summary'):
        print(f"\n🎯 Phase 6 Summary:")
        print(f"   📊 Models tested: {results['summary']['total_models_tested']}")
        print(f"   🔍 Methods used: {results['summary']['total_methods_used']}")
        print(f"   🏆 Best result: {results['summary']['best_method']} on {results['summary']['best_model']}")
        print(f"   📈 Best score: {results['summary']['best_overall_score']:.4f}")


if __name__ == "__main__":
    main()