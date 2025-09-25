#!/usr/bin/env python3
"""
Enhanced Model Validation Module
Provides comprehensive model validation and performance assessment with cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from typing import Dict, Any, List, Tuple, Optional
import time
import logging

logger = logging.getLogger(__name__)

def validate_performance(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Enhanced model performance validation with comprehensive metrics
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary containing comprehensive validation metrics
    """
    start_time = time.time()
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate AUC if binary classification and probabilities available
        auc_score = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            except Exception:
                pass
        
        # Performance benchmarks from roadmap.md
        meets_sensitivity_target = report.get('macro avg', {}).get('recall', 0) >= 0.92
        meets_specificity_target = report.get('macro avg', {}).get('precision', 0) >= 0.87
        
        validation_time = time.time() - start_time
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_score': auc_score,
            'meets_sensitivity_target': meets_sensitivity_target,
            'meets_specificity_target': meets_specificity_target,
            'validation_time_ms': validation_time * 1000,
            'success': True,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'accuracy': 0.0,
            'timestamp': time.time()
        }

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Enhanced cross-validation with learning curves and performance analysis
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        cv_folds: Number of cross-validation folds
        
    Returns:
        Enhanced cross-validation results
    """
    start_time = time.time()
    
    try:
        # Use stratified k-fold for classification
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        # Learning curves for training analysis
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=skf, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        # Calculate comprehensive statistics
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_min': float(cv_scores.min()),
            'cv_max': float(cv_scores.max()),
            'learning_curve': {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            },
            'cross_validation_time_ms': (time.time() - start_time) * 1000,
            'success': True,
            'timestamp': time.time()
        }
        
        logger.info(f"Cross-validation completed: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
        return cv_results
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'timestamp': time.time()
        }

def generate_validation_report(model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """Generate comprehensive enhanced validation report"""
    
    logger.info("Generating comprehensive validation report...")
    
    # Performance on test set
    test_performance = validate_performance(model, X_test, y_test)
    
    # Cross-validation on training set  
    cv_results = cross_validate_model(model, X_train, y_train)
    
    # Model complexity metrics
    model_info = {
        'model_type': type(model).__name__,
        'n_features': X_train.shape[1],
        'n_training_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0]
    }
    
    # Performance summary against targets
    performance_summary = {
        'meets_accuracy_target': test_performance.get('accuracy', 0) >= 0.89,  # Current ADNI dataset target
        'meets_sensitivity_target': test_performance.get('meets_sensitivity_target', False),
        'meets_specificity_target': test_performance.get('meets_specificity_target', False),
        'cross_validation_stable': cv_results.get('cv_std', 1.0) < 0.05,  # Low variance indicator
    }
    
    report = {
        'test_performance': test_performance,
        'cross_validation': cv_results,
        'model_info': model_info,
        'performance_summary': performance_summary,
        'report_generated_at': time.time()
    }
    
    logger.info(f"Validation report completed - Accuracy: {test_performance.get('accuracy', 0):.3f}")
    return report

def alzheimer_specific_validation(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Alzheimer's disease specific validation metrics
    Focuses on early detection performance
    """
    try:
        y_pred = model.predict(X_test)
        
        # Alzheimer's specific metrics
        unique_classes = np.unique(y_test)
        
        # Early detection focus (if MCI class exists)
        mci_detection_accuracy = None
        if 'MCI' in unique_classes or 1 in unique_classes:
            mci_mask = (y_test == 'MCI') if 'MCI' in unique_classes else (y_test == 1)
            if mci_mask.sum() > 0:
                mci_detection_accuracy = accuracy_score(y_test[mci_mask], y_pred[mci_mask])
        
        # Clinical decision support metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'alzheimer_accuracy': accuracy_score(y_test, y_pred),
            'mci_detection_accuracy': mci_detection_accuracy,
            'early_detection_performance': mci_detection_accuracy or report.get('macro avg', {}).get('recall', 0),
            'clinical_utility_score': min(report.get('macro avg', {}).get('precision', 0), 
                                        report.get('macro avg', {}).get('recall', 0)),
            'success': True
        }
    except Exception as e:
        logger.error(f"Alzheimer's specific validation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

__all__ = [
    'validate_performance', 
    'cross_validate_model', 
    'generate_validation_report',
    'alzheimer_specific_validation'
]