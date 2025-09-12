#!/usr/bin/env python3
"""
Structured Alzheimer's Evaluation Pipeline
==========================================

Evaluation script for trained structured Alzheimer's disease prediction models.
Supports loading saved models, running inference on new data, and generating 
comprehensive evaluation reports.

Usage:
    python scripts/eval_alzheimers_structured.py --model-dir metrics/structured_alz/runs/seed_42 --test-data data/test_set.csv
    python scripts/eval_alzheimers_structured.py --aggregate-dir metrics/structured_alz --test-data data/test_set.csv
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path to import our modules  
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EvalAlzheimersStructured')


class ModelEvaluator:
    """
    Evaluator for trained Alzheimer's prediction models
    """
    
    def __init__(self, model_dir: Path):
        """
        Initialize evaluator with model directory
        
        Args:
            model_dir: Directory containing saved model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.run_metrics = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load saved model artifacts"""
        logger.info(f"Loading model artifacts from {self.model_dir}")
        
        # Load best model
        best_model_path = self.model_dir / 'best_model.pkl'
        if best_model_path.exists():
            self.model = joblib.load(best_model_path)
            logger.info("Loaded best model")
        else:
            # Fallback to final model
            final_model_path = self.model_dir / 'final_model.pkl'
            if final_model_path.exists():
                self.model = joblib.load(final_model_path)
                logger.info("Loaded final model (best model not found)")
            else:
                raise FileNotFoundError(f"No model found in {self.model_dir}")
        
        # Load preprocessor
        preprocessing_path = self.model_dir / 'preprocessing.pkl'
        if preprocessing_path.exists():
            self.preprocessor = joblib.load(preprocessing_path)
            logger.info("Loaded preprocessor")
        else:
            logger.warning("Preprocessor not found - raw features will be used")
        
        # Load feature names
        feature_names_path = self.model_dir / 'feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info("Loaded feature names")
        
        # Load run metrics
        run_metrics_path = self.model_dir / 'run_metrics.json'
        if run_metrics_path.exists():
            with open(run_metrics_path, 'r') as f:
                self.run_metrics = json.load(f)
            logger.info("Loaded run metrics")
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess evaluation data using saved preprocessor
        
        Args:
            df: Input DataFrame
            target_column: Name of target column (None for inference)
            
        Returns:
            Tuple of (X_transformed, y) where y is None for inference
        """
        if target_column and target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            y = None
            X = df
        
        if self.preprocessor is not None:
            X_transformed = self.preprocessor.transform(X)
        else:
            logger.warning("No preprocessor available - using raw features")
            X_transformed = X.values
        
        return X_transformed, y
    
    def evaluate(self, X, y) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'macro_f1': f1_score(y, y_pred, average='macro'),
            'weighted_f1': f1_score(y, y_pred, average='weighted'),
            'micro_f1': f1_score(y, y_pred, average='micro')
        }
        
        # Per-class metrics
        per_class_f1 = f1_score(y, y_pred, average=None)
        unique_classes = np.unique(y)
        
        for i, class_label in enumerate(unique_classes):
            metrics[f'f1_class_{class_label}'] = per_class_f1[i]
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC if applicable
        try:
            if len(unique_classes) == 2:
                # Binary classification
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_proba)
                elif hasattr(self.model, 'decision_function'):
                    y_scores = self.model.decision_function(X)
                    metrics['roc_auc'] = roc_auc_score(y, y_scores)
            elif len(unique_classes) > 2:
                # Multi-class
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X)
                    metrics['roc_auc_ovr'] = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
        except Exception as e:
            logger.debug(f"Could not compute ROC AUC: {e}")
        
        return metrics
    
    def predict(self, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on new data
        
        Args:
            X: Features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        y_pred = self.model.predict(X)
        
        try:
            y_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        except:
            y_proba = None
        
        return y_pred, y_proba


def load_test_data(data_path: Path, target_column: str = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load test data from file
    
    Args:
        data_path: Path to test data file
        target_column: Name of target column (auto-detect if None)
        
    Returns:
        Tuple of (DataFrame, target_column_name)
    """
    logger.info(f"Loading test data from {data_path}")
    
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        # Try CSV as default
        df = pd.read_csv(data_path)
    
    logger.info(f"Loaded test data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Auto-detect target column if not specified
    if target_column is None:
        candidates = [
            'diagnosis', 'Diagnosis', 'DIAGNOSIS',
            'target', 'Target', 'TARGET',
            'label', 'Label', 'LABEL',
            'class', 'Class', 'CLASS',
            'Group', 'group', 'GROUP'
        ]
        
        for candidate in candidates:
            if candidate in df.columns:
                target_column = candidate
                logger.info(f"Auto-detected target column: {target_column}")
                break
    
    return df, target_column


def save_evaluation_report(metrics: Dict[str, Any], output_path: Path, 
                          model_info: Dict[str, Any] = None):
    """
    Save comprehensive evaluation report
    
    Args:
        metrics: Evaluation metrics
        output_path: Path to save report
        model_info: Optional model information
    """
    report = {
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'model_info': model_info or {},
        'metrics': metrics
    }
    
    # Save JSON report
    json_path = output_path / 'evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown report
    markdown_content = f"""# Model Evaluation Report

## Overview

- **Evaluation Date**: {report['evaluation_timestamp']}
- **Model**: {model_info.get('best_model', 'Unknown') if model_info else 'Unknown'}
- **Test Samples**: {metrics.get('n_samples', 'Unknown')}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 0):.4f} |
| Macro F1 | {metrics.get('macro_f1', 0):.4f} |
| Weighted F1 | {metrics.get('weighted_f1', 0):.4f} |
| Micro F1 | {metrics.get('micro_f1', 0):.4f} |
"""
    
    if 'roc_auc' in metrics:
        markdown_content += f"| ROC AUC | {metrics['roc_auc']:.4f} |\n"
    
    if 'roc_auc_ovr' in metrics:
        markdown_content += f"| ROC AUC (OvR) | {metrics['roc_auc_ovr']:.4f} |\n"
    
    # Add per-class metrics
    markdown_content += "\n## Per-Class F1 Scores\n\n| Class | F1 Score |\n|-------|----------|\n"
    
    for key, value in metrics.items():
        if key.startswith('f1_class_'):
            class_name = key.replace('f1_class_', '')
            markdown_content += f"| {class_name} | {value:.4f} |\n"
    
    # Save markdown report
    markdown_path = output_path / 'evaluation_report.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Evaluation report saved to {output_path}")


def create_visualizations(metrics: Dict[str, Any], output_path: Path):
    """
    Create evaluation visualizations
    
    Args:
        metrics: Evaluation metrics containing confusion matrix
        output_path: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Confusion matrix visualization saved")
    
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available - skipping visualizations")
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")


def evaluate_single_model(model_dir: Path, test_data_path: Path, 
                         target_column: str = None, output_dir: Path = None) -> Dict[str, Any]:
    """
    Evaluate a single trained model
    
    Args:
        model_dir: Directory containing trained model artifacts
        test_data_path: Path to test data
        target_column: Name of target column
        output_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating model from {model_dir}")
    
    # Load model
    evaluator = ModelEvaluator(model_dir)
    
    # Load test data
    df, detected_target = load_test_data(test_data_path, target_column)
    target_column = target_column or detected_target
    
    if target_column is None:
        raise ValueError("Could not detect target column in test data")
    
    # Preprocess data
    X, y = evaluator.preprocess_data(df, target_column)
    
    # Evaluate
    metrics = evaluator.evaluate(X, y)
    metrics['n_samples'] = len(X)
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_evaluation_report(metrics, output_dir, evaluator.run_metrics)
        create_visualizations(metrics, output_dir)
    
    return metrics


def aggregate_model_evaluation(base_dir: Path, test_data_path: Path,
                              target_column: str = None) -> Dict[str, Any]:
    """
    Evaluate all models from multi-seed training
    
    Args:
        base_dir: Base directory containing runs/seed_* subdirectories
        test_data_path: Path to test data
        target_column: Name of target column
        
    Returns:
        Aggregated evaluation results
    """
    logger.info(f"Evaluating all models from {base_dir}")
    
    runs_dir = base_dir / 'runs'
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Find all seed directories
    seed_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found in {runs_dir}")
    
    logger.info(f"Found {len(seed_dirs)} model runs to evaluate")
    
    # Evaluate each model
    all_results = []
    
    for seed_dir in sorted(seed_dirs):
        try:
            logger.info(f"Evaluating {seed_dir.name}")
            
            metrics = evaluate_single_model(
                model_dir=seed_dir,
                test_data_path=test_data_path,
                target_column=target_column,
                output_dir=seed_dir / 'evaluation'
            )
            
            metrics['seed_dir'] = seed_dir.name
            all_results.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {seed_dir.name}: {e}")
            all_results.append({
                'seed_dir': seed_dir.name,
                'error': str(e)
            })
    
    # Aggregate results
    successful_results = [r for r in all_results if 'error' not in r]
    
    if not successful_results:
        raise RuntimeError("No models could be evaluated successfully")
    
    # Calculate aggregate statistics
    def collect_metric(metric_name: str) -> List[float]:
        return [r[metric_name] for r in successful_results if metric_name in r]
    
    key_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'micro_f1']
    if any('roc_auc' in r for r in successful_results):
        key_metrics.append('roc_auc')
    
    aggregate_stats = {
        'total_models': len(all_results),
        'successful_models': len(successful_results),
        'failed_models': len(all_results) - len(successful_results)
    }
    
    for metric in key_metrics:
        values = collect_metric(metric)
        if values:
            aggregate_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
    
    # Save aggregate results
    output_dir = base_dir / 'evaluation_aggregate'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'aggregate_evaluation.json', 'w') as f:
        json.dump({
            'aggregate_stats': aggregate_stats,
            'individual_results': all_results
        }, f, indent=2, default=str)
    
    # Create aggregate report
    markdown_content = f"""# Aggregate Model Evaluation

## Summary

- **Total Models**: {aggregate_stats['total_models']}
- **Successful**: {aggregate_stats['successful_models']}
- **Failed**: {aggregate_stats['failed_models']}

## Performance Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
"""
    
    for metric in key_metrics:
        if metric in aggregate_stats:
            stats = aggregate_stats[metric]
            clean_name = metric.replace('_', ' ').title()
            markdown_content += f"| {clean_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
    
    with open(output_dir / 'aggregate_evaluation.md', 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Aggregate evaluation saved to {output_dir}")
    
    return {
        'aggregate_stats': aggregate_stats,
        'individual_results': all_results
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained structured Alzheimer models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model-dir',
        type=Path,
        help='Directory containing single trained model artifacts'
    )
    model_group.add_argument(
        '--aggregate-dir',
        type=Path,
        help='Base directory containing multiple model runs (runs/seed_*)'
    )
    
    # Data specification
    parser.add_argument(
        '--test-data',
        type=Path,
        required=True,
        help='Path to test data (CSV or parquet)'
    )
    
    parser.add_argument(
        '--target-column',
        help='Name of target column (auto-detected if not specified)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save evaluation results (default: model-dir/evaluation)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.model_dir:
            # Single model evaluation
            output_dir = args.output_dir or (args.model_dir / 'evaluation')
            
            results = evaluate_single_model(
                model_dir=args.model_dir,
                test_data_path=args.test_data,
                target_column=args.target_column,
                output_dir=output_dir
            )
            
            print("\n" + "="*50)
            print("📊 MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"✅ Accuracy: {results.get('accuracy', 0):.4f}")
            print(f"🎯 Macro F1: {results.get('macro_f1', 0):.4f}")
            print(f"⚖️  Weighted F1: {results.get('weighted_f1', 0):.4f}")
            if 'roc_auc' in results:
                print(f"📈 ROC AUC: {results['roc_auc']:.4f}")
            print(f"📁 Results saved to: {output_dir}")
            print("="*50)
            
        else:
            # Aggregate evaluation
            results = aggregate_model_evaluation(
                base_dir=args.aggregate_dir,
                test_data_path=args.test_data,
                target_column=args.target_column
            )
            
            stats = results['aggregate_stats']
            
            print("\n" + "="*60)
            print("📊 AGGREGATE MODEL EVALUATION")
            print("="*60)
            print(f"🔢 Models: {stats['successful_models']}/{stats['total_models']} successful")
            
            if 'accuracy' in stats:
                acc_stats = stats['accuracy']
                print(f"✅ Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
            
            if 'macro_f1' in stats:
                f1_stats = stats['macro_f1']
                print(f"🎯 Macro F1: {f1_stats['mean']:.4f} ± {f1_stats['std']:.4f}")
            
            print(f"📁 Results saved to: {args.aggregate_dir / 'evaluation_aggregate'}")
            print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)