#!/usr/bin/env python3
"""
Cross-validation implementation for Phase 5 of AiMedRes debugging process.

This module implements:
- Phase 5.1: k-fold cross-validation for generalization check
- Phase 5.2: Stratified sampling for imbalanced datasets
- Phase 5.3: Leave-one-out cross-validation for small datasets
"""

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)

logger = logging.getLogger("CrossValidation")


class CrossValidationConfig:
    """Configuration for cross-validation strategies"""

    def __init__(
        self,
        k_folds: int = 5,
        small_dataset_threshold: int = 50,
        imbalance_threshold: float = 0.1,
        random_state: int = 42,
        scoring_metrics: Optional[List[str]] = None,
    ):
        """
        Initialize cross-validation configuration

        Args:
            k_folds: Number of folds for k-fold cross-validation
            small_dataset_threshold: Threshold below which to use LeaveOneOut
            imbalance_threshold: Minimum class ratio to consider dataset balanced
            random_state: Random state for reproducibility
            scoring_metrics: List of metrics to compute during cross-validation
        """
        self.k_folds = k_folds
        self.small_dataset_threshold = small_dataset_threshold
        self.imbalance_threshold = imbalance_threshold
        self.random_state = random_state
        self.scoring_metrics = scoring_metrics or [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]


class Phase5CrossValidator:
    """
    Implementation of Phase 5 cross-validation requirements for AiMedRes
    """

    def __init__(self, config: Optional[CrossValidationConfig] = None):
        """Initialize cross-validator with configuration"""
        self.config = config or CrossValidationConfig()

    def analyze_dataset_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze dataset to determine appropriate cross-validation strategy

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with dataset characteristics and recommendations
        """
        n_samples, n_features = X.shape
        class_counts = Counter(y)
        n_classes = len(class_counts)

        # Calculate class balance
        min_class_count = min(class_counts.values())
        max_class_count = max(class_counts.values())
        class_balance_ratio = min_class_count / max_class_count

        # Determine if dataset is imbalanced
        is_imbalanced = class_balance_ratio < self.config.imbalance_threshold

        # Determine if dataset is small
        is_small = n_samples < self.config.small_dataset_threshold

        # Recommended strategy
        if is_small:
            recommended_strategy = "leave_one_out"
        elif is_imbalanced:
            recommended_strategy = "stratified_k_fold"
        else:
            recommended_strategy = "k_fold"

        analysis = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "class_counts": dict(class_counts),
            "class_balance_ratio": class_balance_ratio,
            "is_imbalanced": is_imbalanced,
            "is_small": is_small,
            "recommended_strategy": recommended_strategy,
            "samples_per_fold": n_samples // self.config.k_folds if not is_small else 1,
        }

        logger.info(
            f"Dataset analysis: {n_samples} samples, {n_features} features, {n_classes} classes"
        )
        logger.info(f"Class balance ratio: {class_balance_ratio:.3f}")
        logger.info(f"Recommended CV strategy: {recommended_strategy}")

        return analysis

    def phase_5_1_k_fold_cv(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Phase 5.1: Use k-fold cross-validation for generalization check

        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with k-fold cross-validation results
        """
        logger.info(f"🔄 Phase 5.1: Running {self.config.k_folds}-fold cross-validation")

        # Standard k-fold cross-validation
        kfold = KFold(
            n_splits=self.config.k_folds, shuffle=True, random_state=self.config.random_state
        )

        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=kfold,
            scoring=self.config.scoring_metrics,
            return_train_score=True,
            n_jobs=-1,
        )

        # Calculate statistics
        results = {
            "strategy": "k_fold",
            "n_folds": self.config.k_folds,
            "cv_results": cv_results,
            "mean_scores": {},
            "std_scores": {},
            "generalization_gap": {},
        }

        for metric in self.config.scoring_metrics:
            test_scores = cv_results[f"test_{metric}"]
            train_scores = cv_results[f"train_{metric}"]

            results["mean_scores"][metric] = {
                "test": np.mean(test_scores),
                "train": np.mean(train_scores),
            }
            results["std_scores"][metric] = {
                "test": np.std(test_scores),
                "train": np.std(train_scores),
            }
            results["generalization_gap"][metric] = np.mean(train_scores) - np.mean(test_scores)

        # Check for overfitting
        accuracy_gap = results["generalization_gap"].get("accuracy", 0)
        if accuracy_gap > 0.1:  # 10% gap indicates potential overfitting
            logger.warning(
                f"⚠️  Potential overfitting detected. "
                f"Train-test accuracy gap: {accuracy_gap:.3f}"
            )

        logger.info(
            f"✅ K-fold CV completed. Mean test accuracy: "
            f"{results['mean_scores']['accuracy']['test']:.3f} ± "
            f"{results['std_scores']['accuracy']['test']:.3f}"
        )

        return results

    def phase_5_2_stratified_cv(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Phase 5.2: Apply stratified sampling for imbalanced datasets

        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with stratified cross-validation results
        """
        logger.info(f"🎯 Phase 5.2: Running stratified {self.config.k_folds}-fold cross-validation")

        # Stratified k-fold cross-validation
        stratified_kfold = StratifiedKFold(
            n_splits=self.config.k_folds, shuffle=True, random_state=self.config.random_state
        )

        # Perform stratified cross-validation
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=stratified_kfold,
            scoring=self.config.scoring_metrics,
            return_train_score=True,
            n_jobs=-1,
        )

        # Verify stratification by checking fold class distributions
        fold_class_distributions = []
        for fold_idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
            y_test_fold = y[test_idx]
            class_dist = Counter(y_test_fold)
            fold_class_distributions.append(class_dist)

        # Calculate statistics
        results = {
            "strategy": "stratified_k_fold",
            "n_folds": self.config.k_folds,
            "cv_results": cv_results,
            "fold_class_distributions": fold_class_distributions,
            "mean_scores": {},
            "std_scores": {},
            "generalization_gap": {},
        }

        for metric in self.config.scoring_metrics:
            test_scores = cv_results[f"test_{metric}"]
            train_scores = cv_results[f"train_{metric}"]

            results["mean_scores"][metric] = {
                "test": np.mean(test_scores),
                "train": np.mean(train_scores),
            }
            results["std_scores"][metric] = {
                "test": np.std(test_scores),
                "train": np.std(train_scores),
            }
            results["generalization_gap"][metric] = np.mean(train_scores) - np.mean(test_scores)

        # Verify class stratification worked correctly
        original_class_dist = Counter(y)
        total_samples = len(y)
        original_proportions = {
            cls: count / total_samples for cls, count in original_class_dist.items()
        }

        logger.info("📊 Class distribution verification:")
        logger.info(f"Original proportions: {original_proportions}")

        # Check each fold maintains similar proportions
        max_deviation = 0
        for fold_idx, fold_dist in enumerate(fold_class_distributions):
            fold_total = sum(fold_dist.values())
            fold_proportions = {cls: count / fold_total for cls, count in fold_dist.items()}

            # Calculate maximum deviation from original proportions
            for cls in original_proportions:
                deviation = abs(fold_proportions.get(cls, 0) - original_proportions[cls])
                max_deviation = max(max_deviation, deviation)

        results["stratification_quality"] = {
            "max_proportion_deviation": max_deviation,
            "is_well_stratified": max_deviation < 0.05,  # 5% tolerance
        }

        if max_deviation > 0.1:  # 10% deviation is concerning
            logger.warning(
                f"⚠️  Stratification may not be optimal. " f"Max deviation: {max_deviation:.3f}"
            )

        logger.info(
            f"✅ Stratified CV completed. Mean test accuracy: "
            f"{results['mean_scores']['accuracy']['test']:.3f} ± "
            f"{results['std_scores']['accuracy']['test']:.3f}"
        )

        return results

    def phase_5_3_leave_one_out_cv(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Phase 5.3: Optionally, use leave-one-out cross-validation for small datasets

        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with leave-one-out cross-validation results
        """
        n_samples = len(X)
        logger.info(f"🔍 Phase 5.3: Running Leave-One-Out cross-validation on {n_samples} samples")

        if n_samples > 100:
            logger.warning(f"⚠️  LOO-CV on {n_samples} samples may be computationally expensive")

        # Leave-one-out cross-validation
        loo = LeaveOneOut()

        # For LOO, we'll compute basic metrics manually due to computational constraints
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings about single-sample metrics

            # Use basic cross_val_score for key metrics
            accuracy_scores = cross_val_score(model, X, y, cv=loo, scoring="accuracy", n_jobs=-1)

            # For other metrics, we may need to be more careful with edge cases
            try:
                precision_scores = cross_val_score(
                    model, X, y, cv=loo, scoring="precision_macro", n_jobs=-1
                )
                recall_scores = cross_val_score(
                    model, X, y, cv=loo, scoring="recall_macro", n_jobs=-1
                )
                f1_scores = cross_val_score(model, X, y, cv=loo, scoring="f1_macro", n_jobs=-1)
            except Exception as e:
                logger.warning(f"Some metrics failed for LOO-CV: {e}")
                # Fallback to accuracy only
                precision_scores = accuracy_scores
                recall_scores = accuracy_scores
                f1_scores = accuracy_scores

        results = {
            "strategy": "leave_one_out",
            "n_folds": n_samples,
            "accuracy_scores": accuracy_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
            "mean_scores": {
                "accuracy": np.mean(accuracy_scores),
                "precision_macro": np.mean(precision_scores),
                "recall_macro": np.mean(recall_scores),
                "f1_macro": np.mean(f1_scores),
            },
            "std_scores": {
                "accuracy": np.std(accuracy_scores),
                "precision_macro": np.std(precision_scores),
                "recall_macro": np.std(recall_scores),
                "f1_macro": np.std(f1_scores),
            },
        }

        # Estimate confidence interval for accuracy (binomial)
        n_correct = np.sum(accuracy_scores)
        confidence_interval = self._calculate_binomial_ci(n_correct, n_samples)
        results["accuracy_confidence_interval"] = confidence_interval

        logger.info(
            f"✅ LOO-CV completed. Accuracy: {results['mean_scores']['accuracy']:.3f} "
            f"({confidence_interval[0]:.3f}-{confidence_interval[1]:.3f})"
        )

        return results

    def _calculate_binomial_ci(
        self, successes: int, trials: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate binomial confidence interval for accuracy"""
        from scipy import stats

        # Wilson score interval (more accurate for small samples)
        p = successes / trials
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def comprehensive_cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive cross-validation analysis following Phase 5 requirements

        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target labels

        Returns:
            Comprehensive cross-validation results
        """
        logger.info("🚀 Starting Phase 5: Comprehensive Cross-Validation Implementation")

        # Analyze dataset characteristics
        dataset_analysis = self.analyze_dataset_characteristics(X, y)

        # Store all results
        comprehensive_results = {
            "dataset_analysis": dataset_analysis,
            "phase_5_1_k_fold": None,
            "phase_5_2_stratified": None,
            "phase_5_3_leave_one_out": None,
            "recommended_strategy": dataset_analysis["recommended_strategy"],
        }

        # Phase 5.1: Always run k-fold CV for generalization check
        try:
            comprehensive_results["phase_5_1_k_fold"] = self.phase_5_1_k_fold_cv(model, X, y)
        except Exception as e:
            logger.error(f"Phase 5.1 k-fold CV failed: {e}")

        # Phase 5.2: Run stratified CV if dataset is imbalanced
        if (
            dataset_analysis["is_imbalanced"]
            or dataset_analysis["recommended_strategy"] == "stratified_k_fold"
        ):
            try:
                comprehensive_results["phase_5_2_stratified"] = self.phase_5_2_stratified_cv(
                    model, X, y
                )
            except Exception as e:
                logger.error(f"Phase 5.2 stratified CV failed: {e}")

        # Phase 5.3: Run LOO CV if dataset is small
        if (
            dataset_analysis["is_small"]
            or dataset_analysis["recommended_strategy"] == "leave_one_out"
        ):
            try:
                comprehensive_results["phase_5_3_leave_one_out"] = self.phase_5_3_leave_one_out_cv(
                    model, X, y
                )
            except Exception as e:
                logger.error(f"Phase 5.3 LOO CV failed: {e}")

        # Generate summary and recommendations
        summary = self._generate_cv_summary(comprehensive_results)
        comprehensive_results["summary"] = summary

        logger.info("✅ Phase 5: Comprehensive Cross-Validation completed successfully")
        return comprehensive_results

    def _generate_cv_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of cross-validation results"""
        dataset_analysis = results["dataset_analysis"]

        summary = {
            "recommended_strategy": dataset_analysis["recommended_strategy"],
            "dataset_characteristics": {
                "size": "small" if dataset_analysis["is_small"] else "adequate",
                "balance": "imbalanced" if dataset_analysis["is_imbalanced"] else "balanced",
                "n_samples": dataset_analysis["n_samples"],
                "n_classes": dataset_analysis["n_classes"],
            },
            "best_accuracy": None,
            "best_strategy": None,
            "recommendations": [],
        }

        # Find best performing strategy
        strategies = []
        if results["phase_5_1_k_fold"]:
            acc = results["phase_5_1_k_fold"]["mean_scores"]["accuracy"]["test"]
            strategies.append(("k_fold", acc))

        if results["phase_5_2_stratified"]:
            acc = results["phase_5_2_stratified"]["mean_scores"]["accuracy"]["test"]
            strategies.append(("stratified_k_fold", acc))

        if results["phase_5_3_leave_one_out"]:
            acc = results["phase_5_3_leave_one_out"]["mean_scores"]["accuracy"]
            strategies.append(("leave_one_out", acc))

        if strategies:
            best_strategy, best_accuracy = max(strategies, key=lambda x: x[1])
            summary["best_accuracy"] = best_accuracy
            summary["best_strategy"] = best_strategy

        # Generate recommendations
        if dataset_analysis["is_small"]:
            summary["recommendations"].append(
                "Dataset is small - Leave-One-Out CV provides most reliable estimate"
            )

        if dataset_analysis["is_imbalanced"]:
            summary["recommendations"].append(
                "Dataset is imbalanced - Stratified CV maintains class proportions"
            )

        # Check for overfitting
        for strategy_name, strategy_results in [
            ("k_fold", results["phase_5_1_k_fold"]),
            ("stratified_k_fold", results["phase_5_2_stratified"]),
        ]:
            if strategy_results and "generalization_gap" in strategy_results:
                gap = strategy_results["generalization_gap"].get("accuracy", 0)
                if gap > 0.1:
                    summary["recommendations"].append(
                        f"High generalization gap ({gap:.3f}) in {strategy_name} - consider regularization"
                    )

        return summary


def get_scipy_fallback():
    """Fallback for binomial CI calculation without scipy"""

    def normal_approx_ci(
        successes: int, trials: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Normal approximation for binomial confidence interval"""
        if trials == 0:
            return (0.0, 1.0)

        p = successes / trials
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z * np.sqrt(p * (1 - p) / trials)
        return (max(0, p - margin), min(1, p + margin))

    return normal_approx_ci


# Monkey patch if scipy is not available
try:
    from scipy import stats
except ImportError:
    logger.warning("SciPy not available, using normal approximation for confidence intervals")
    Phase5CrossValidator._calculate_binomial_ci = lambda self, s, t, c=0.95: get_scipy_fallback()(
        s, t, c
    )
