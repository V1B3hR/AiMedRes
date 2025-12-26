#!/usr/bin/env python3
"""
Enhanced Model Validation Module
Provides comprehensive model validation and performance assessment with cross-validation
"""

import logging

# Essential imports only - ML libraries loaded lazily
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _lazy_import_validation():
    """Lazy import of validation libraries for better startup performance"""
    global np, pd, accuracy_score, classification_report, confusion_matrix
    global roc_auc_score, precision_recall_curve, cross_val_score
    global StratifiedKFold, learning_curve, f1_score, precision_score, recall_score

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve


def validate_performance(model, X_test, y_test) -> Dict[str, Any]:
    """
    Enhanced model performance validation with comprehensive metrics

    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary containing comprehensive validation metrics
    """
    _lazy_import_validation()  # Import sklearn when needed

    start_time = time.time()

    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        # Get prediction probabilities if available
        if hasattr(model, "predict_proba"):
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
                pass

        # Performance benchmarks from roadmap.md
        meets_sensitivity_target = report.get("macro avg", {}).get("recall", 0) >= 0.92
        meets_specificity_target = report.get("macro avg", {}).get("precision", 0) >= 0.87

        validation_time = time.time() - start_time

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "auc_score": auc_score,
            "meets_sensitivity_target": meets_sensitivity_target,
            "meets_specificity_target": meets_specificity_target,
            "validation_time_ms": validation_time * 1000,
            "success": True,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        return {"success": False, "error": str(e), "accuracy": 0.0, "timestamp": time.time()}


def cross_validate_model(model, X, y, cv_folds: int = 5) -> Dict[str, Any]:
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
    _lazy_import_validation()  # Import sklearn when needed

    start_time = time.time()

    try:
        # Use stratified k-fold for classification
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

        # Learning curves for training analysis
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=skf,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
        )

        # Calculate comprehensive statistics
        cv_results = {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "mean_score": float(cv_scores.mean()),  # For compatibility
            "cv_std": float(cv_scores.std()),
            "std_score": float(cv_scores.std()),  # For compatibility
            "cv_min": float(cv_scores.min()),
            "cv_max": float(cv_scores.max()),
            "execution_time": time.time() - start_time,
            "performance_target_met": (time.time() - start_time) < 0.1,  # 100ms target
            "learning_curve": {
                "train_sizes": train_sizes.tolist(),
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "val_scores_mean": val_scores.mean(axis=1).tolist(),
                "val_scores_std": val_scores.std(axis=1).tolist(),
            },
            "cross_validation_time_ms": (time.time() - start_time) * 1000,
            "success": True,
            "timestamp": time.time(),
        }

        logger.info(
            f"Cross-validation completed: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}"
        )
        return cv_results

    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "timestamp": time.time(),
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
        "model_type": type(model).__name__,
        "n_features": X_train.shape[1],
        "n_training_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
    }

    # Performance summary against targets
    performance_summary = {
        "meets_accuracy_target": test_performance.get("accuracy", 0)
        >= 0.89,  # Current ADNI dataset target
        "meets_sensitivity_target": test_performance.get("meets_sensitivity_target", False),
        "meets_specificity_target": test_performance.get("meets_specificity_target", False),
        "cross_validation_stable": cv_results.get("cv_std", 1.0) < 0.05,  # Low variance indicator
    }

    report = {
        "test_performance": test_performance,
        "cross_validation": cv_results,
        "model_info": model_info,
        "performance_summary": performance_summary,
        "report_generated_at": time.time(),
    }

    logger.info(
        f"Validation report completed - Accuracy: {test_performance.get('accuracy', 0):.3f}"
    )
    return report


def alzheimer_specific_validation(model, X_test, y_test) -> Dict[str, Any]:
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
        if "MCI" in unique_classes or 1 in unique_classes:
            mci_mask = (y_test == "MCI") if "MCI" in unique_classes else (y_test == 1)
            if mci_mask.sum() > 0:
                mci_detection_accuracy = accuracy_score(y_test[mci_mask], y_pred[mci_mask])

        # Clinical decision support metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "alzheimer_accuracy": accuracy_score(y_test, y_pred),
            "mci_detection_accuracy": mci_detection_accuracy,
            "early_detection_performance": mci_detection_accuracy
            or report.get("macro avg", {}).get("recall", 0),
            "clinical_utility_score": min(
                report.get("macro avg", {}).get("precision", 0),
                report.get("macro avg", {}).get("recall", 0),
            ),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Alzheimer's specific validation failed: {e}")
        return {"success": False, "error": str(e)}


def validate_alzheimer_pipeline(
    pipeline_func, data_path: Optional[str] = None, performance_target_ms: float = 100.0
) -> Dict[str, Any]:
    """
    Validate the complete Alzheimer's training pipeline with performance benchmarks

    Args:
        pipeline_func: Training pipeline function to validate
        data_path: Path to data file (optional)
        performance_target_ms: Performance target in milliseconds

    Returns:
        Comprehensive validation results
    """
    _lazy_import_validation()  # Import dependencies when needed

    start_time = time.time()
    logger.info("Starting Alzheimer's pipeline validation...")

    try:
        # Run the pipeline
        if data_path:
            results = pipeline_func(data_path=data_path)
        else:
            results = pipeline_func()

        total_time = time.time() - start_time
        performance_met = total_time * 1000 < performance_target_ms

        validation_summary = {
            "pipeline_validation": {
                "status": "success",
                "total_execution_time": total_time,
                "total_execution_time_ms": total_time * 1000,
                "performance_target_ms": performance_target_ms,
                "performance_target_met": performance_met,
                "pipeline_results": results,
            },
            "performance_analysis": {
                "speed_grade": "A" if total_time < 0.05 else "B" if total_time < 0.1 else "C",
                "meets_clinical_requirements": performance_met,
                "optimization_needed": not performance_met,
            },
        }

        logger.info(
            f"Pipeline validation {'✓' if performance_met else '✗'}: {total_time*1000:.1f}ms (target: {performance_target_ms}ms)"
        )
        return validation_summary

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return {
            "pipeline_validation": {
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "performance_target_met": False,
            }
        }


def run_comprehensive_validation_suite() -> Dict[str, Any]:
    """
    Run comprehensive validation suite for all training components
    """
    logger.info("🧪 Starting Comprehensive Validation Suite")

    suite_results = {
        "validation_suite": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_target_ms": 100.0,
            "tests_run": [],
            "tests_passed": 0,
            "tests_failed": 0,
        }
    }

    # Test 1: Import Performance
    logger.info("Testing import performance...")
    start_time = time.time()
    try:
        from .training import AlzheimerTrainer

        import_time = (time.time() - start_time) * 1000
        test_passed = import_time < 50  # 50ms import target

        suite_results["import_test"] = {
            "time_ms": import_time,
            "passed": test_passed,
            "target_ms": 50,
        }

        suite_results["validation_suite"]["tests_run"].append("import_performance")
        if test_passed:
            suite_results["validation_suite"]["tests_passed"] += 1
        else:
            suite_results["validation_suite"]["tests_failed"] += 1

    except Exception as e:
        suite_results["import_test"] = {"error": str(e), "passed": False}
        suite_results["validation_suite"]["tests_failed"] += 1

    # Test 2: Basic Training Performance
    logger.info("Testing basic training performance...")
    try:
        from .training import AlzheimerTrainer

        trainer = AlzheimerTrainer()

        start_time = time.time()
        data = trainer.load_data()
        load_time = (time.time() - start_time) * 1000

        start_time = time.time()
        X, y = trainer.preprocess_data(data)
        preprocess_time = (time.time() - start_time) * 1000

        total_time = load_time + preprocess_time
        test_passed = total_time < 100  # 100ms total target

        suite_results["training_performance_test"] = {
            "load_time_ms": load_time,
            "preprocess_time_ms": preprocess_time,
            "total_time_ms": total_time,
            "passed": test_passed,
            "target_ms": 100,
        }

        suite_results["validation_suite"]["tests_run"].append("training_performance")
        if test_passed:
            suite_results["validation_suite"]["tests_passed"] += 1
        else:
            suite_results["validation_suite"]["tests_failed"] += 1

    except Exception as e:
        suite_results["training_performance_test"] = {"error": str(e), "passed": False}
        suite_results["validation_suite"]["tests_failed"] += 1

    # Calculate overall results
    total_tests = (
        suite_results["validation_suite"]["tests_passed"]
        + suite_results["validation_suite"]["tests_failed"]
    )
    suite_results["validation_suite"]["success_rate"] = (
        suite_results["validation_suite"]["tests_passed"] / max(total_tests, 1) * 100
    )

    logger.info(
        f"🎯 Validation Suite Complete: {suite_results['validation_suite']['tests_passed']}/{total_tests} tests passed"
    )

    return suite_results


__all__ = [
    "validate_performance",
    "cross_validate_model",
    "generate_validation_report",
    "alzheimer_specific_validation",
    "validate_alzheimer_pipeline",
    "run_comprehensive_validation_suite",
]
