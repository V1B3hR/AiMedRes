"""
Enhanced Pipeline Customization for DuetMind Adaptive

Provides flexible, configurable training pipelines with dynamic configuration,
custom preprocessing steps, and modular architecture.
"""

import importlib
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps."""

    numerical_scaler: str = "standard"  # 'standard', 'minmax', 'robust', 'none'
    categorical_encoding: str = "onehot"  # 'onehot', 'label', 'target'
    missing_value_strategy: str = "median"  # 'mean', 'median', 'most_frequent', 'constant'
    feature_selection: Optional[str] = None  # 'selectkbest', 'rfe', 'none'
    outlier_detection: Optional[str] = None  # 'iqr', 'zscore', 'isolation_forest'
    custom_transformers: List[str] = None  # List of custom transformer class names


@dataclass
class ModelConfig:
    """Configuration for model training."""

    algorithm: str = "random_forest"
    hyperparameters: Dict[str, Any] = None
    cross_validation: Dict[str, Any] = None
    ensemble_method: Optional[str] = None  # 'voting', 'stacking', 'bagging'

    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.cross_validation is None:
            self.cross_validation = {"cv": 5, "scoring": "roc_auc"}


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    name: str = "default_pipeline"
    description: str = ""
    preprocessing: PreprocessingConfig = None
    model: ModelConfig = None
    evaluation: Dict[str, Any] = None
    output: Dict[str, Any] = None

    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = {"metrics": ["accuracy", "roc_auc", "f1"]}
        if self.output is None:
            self.output = {"save_model": True, "save_results": True}


class CustomTransformer(ABC, BaseEstimator, TransformerMixin):
    """Base class for custom transformers."""

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the transformer."""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform the data."""
        pass


class DynamicPipelineBuilder:
    """
    Dynamic pipeline builder that creates customized ML pipelines
    based on configuration specifications.
    """

    def __init__(self, config: Union[PipelineConfig, str, Dict[str, Any]]):
        """
        Initialize pipeline builder.

        Args:
            config: Pipeline configuration (object, file path, or dict)
        """
        self.config = self._load_config(config)
        self.pipeline = None
        self.custom_transformers = {}
        self._register_transformers()

    def _load_config(self, config: Union[PipelineConfig, str, Dict[str, Any]]) -> PipelineConfig:
        """Load configuration from various sources."""
        if isinstance(config, PipelineConfig):
            return config
        elif isinstance(config, str):
            # Load from file
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config}")

            with open(config_path, "r") as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            return self._dict_to_config(config_dict)
        elif isinstance(config, dict):
            return self._dict_to_config(config)
        else:
            raise ValueError("Config must be PipelineConfig, file path, or dictionary")

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig."""
        preprocessing_dict = config_dict.get("preprocessing", {})
        model_dict = config_dict.get("model", {})

        preprocessing = PreprocessingConfig(**preprocessing_dict)
        model = ModelConfig(**model_dict)

        return PipelineConfig(
            name=config_dict.get("name", "default_pipeline"),
            description=config_dict.get("description", ""),
            preprocessing=preprocessing,
            model=model,
            evaluation=config_dict.get("evaluation", {}),
            output=config_dict.get("output", {}),
        )

    def _register_transformers(self):
        """Register custom transformers."""
        if self.config.preprocessing.custom_transformers:
            for transformer_name in self.config.preprocessing.custom_transformers:
                try:
                    # Try to import the transformer dynamically
                    module_path, class_name = transformer_name.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    transformer_class = getattr(module, class_name)

                    if not issubclass(transformer_class, CustomTransformer):
                        logger.warning(
                            f"Transformer {transformer_name} does not inherit from CustomTransformer"
                        )

                    self.custom_transformers[class_name] = transformer_class
                    logger.info(f"Registered custom transformer: {class_name}")

                except Exception as e:
                    logger.error(f"Failed to register transformer {transformer_name}: {e}")

    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build preprocessing pipeline based on configuration."""
        steps = []

        # Identify column types
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Build preprocessing steps
        preprocessors = []

        # Numerical preprocessing
        if numerical_columns:
            numerical_steps = []

            # Missing value imputation
            if self.config.preprocessing.missing_value_strategy != "none":
                from sklearn.impute import SimpleImputer

                imputer = SimpleImputer(strategy=self.config.preprocessing.missing_value_strategy)
                numerical_steps.append(("imputer", imputer))

            # Scaling
            if self.config.preprocessing.numerical_scaler != "none":
                scaler_map = {
                    "standard": StandardScaler(),
                    "minmax": MinMaxScaler(),
                    "robust": RobustScaler(),
                }
                scaler = scaler_map.get(self.config.preprocessing.numerical_scaler)
                if scaler:
                    numerical_steps.append(("scaler", scaler))

            numerical_pipeline = Pipeline(numerical_steps)
            preprocessors.append(("num", numerical_pipeline, numerical_columns))

        # Categorical preprocessing
        if categorical_columns:
            categorical_steps = []

            # Missing value imputation
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="most_frequent")
            categorical_steps.append(("imputer", imputer))

            # Encoding
            if self.config.preprocessing.categorical_encoding == "onehot":
                from sklearn.preprocessing import OneHotEncoder

                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                categorical_steps.append(("encoder", encoder))
            elif self.config.preprocessing.categorical_encoding == "label":
                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()
                categorical_steps.append(("encoder", encoder))

            categorical_pipeline = Pipeline(categorical_steps)
            preprocessors.append(("cat", categorical_pipeline, categorical_columns))

        # Create column transformer
        if preprocessors:
            preprocessing_pipeline = ColumnTransformer(
                transformers=preprocessors, remainder="passthrough"
            )
            steps.append(("preprocessing", preprocessing_pipeline))

        # Feature selection
        if self.config.preprocessing.feature_selection:
            feature_selector = self._create_feature_selector()
            if feature_selector:
                steps.append(("feature_selection", feature_selector))

        # Custom transformers
        for name, transformer_class in self.custom_transformers.items():
            transformer = transformer_class()
            steps.append((f"custom_{name}", transformer))

        return Pipeline(steps) if steps else None

    def _create_feature_selector(self):
        """Create feature selection transformer."""
        method = self.config.preprocessing.feature_selection

        if method == "selectkbest":
            from sklearn.feature_selection import SelectKBest, f_classif

            return SelectKBest(score_func=f_classif, k=10)
        elif method == "rfe":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE

            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            return RFE(estimator, n_features_to_select=10)

        return None

    def build_model(self):
        """Build model based on configuration."""
        algorithm = self.config.model.algorithm
        params = self.config.model.hyperparameters

        # Model mapping
        model_map = {
            "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
            "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
            "svm": ("sklearn.svm", "SVC"),
            "xgboost": ("xgboost", "XGBClassifier"),
            "lightgbm": ("lightgbm", "LGBMClassifier"),
        }

        if algorithm not in model_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        module_name, class_name = model_map[algorithm]

        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            return model_class(**params)
        except ImportError:
            logger.error(f"Could not import {module_name}.{class_name}")
            raise
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    def build_complete_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build complete pipeline with preprocessing and model."""
        steps = []

        # Add preprocessing
        preprocessing_pipeline = self.build_preprocessing_pipeline(X)
        if preprocessing_pipeline:
            steps.extend(preprocessing_pipeline.steps)

        # Add model
        model = self.build_model()
        steps.append(("model", model))

        # Handle ensemble methods
        if self.config.model.ensemble_method:
            ensemble_pipeline = self._build_ensemble_pipeline(X)
            return ensemble_pipeline

        self.pipeline = Pipeline(steps)
        return self.pipeline

    def _build_ensemble_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build ensemble pipeline."""
        method = self.config.model.ensemble_method

        if method == "voting":
            from sklearn.ensemble import VotingClassifier

            # Create multiple models for voting
            estimators = []
            algorithms = ["random_forest", "logistic_regression", "svm"]

            for alg in algorithms:
                try:
                    temp_config = PipelineConfig(
                        model=ModelConfig(algorithm=alg, hyperparameters={})
                    )
                    temp_builder = DynamicPipelineBuilder(temp_config)
                    model = temp_builder.build_model()
                    estimators.append((alg, model))
                except Exception as e:
                    logger.warning(f"Could not create {alg} for ensemble: {e}")

            if len(estimators) >= 2:
                ensemble = VotingClassifier(estimators=estimators, voting="soft")

                # Build pipeline with preprocessing and ensemble
                steps = []
                preprocessing_pipeline = self.build_preprocessing_pipeline(X)
                if preprocessing_pipeline:
                    steps.extend(preprocessing_pipeline.steps)
                steps.append(("ensemble", ensemble))

                return Pipeline(steps)

        # Fallback to regular pipeline
        return self.build_complete_pipeline(X)

    def save_config(self, filepath: str):
        """Save pipeline configuration to file."""
        config_dict = asdict(self.config)

        filepath = Path(filepath)
        with open(filepath, "w") as f:
            if filepath.suffix in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False)
            elif filepath.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {filepath.suffix}")

        logger.info(f"Pipeline configuration saved to {filepath}")

    def get_config_template(self) -> Dict[str, Any]:
        """Get configuration template for reference."""
        return {
            "name": "example_pipeline",
            "description": "Example customizable pipeline",
            "preprocessing": {
                "numerical_scaler": "standard",
                "categorical_encoding": "onehot",
                "missing_value_strategy": "median",
                "feature_selection": "selectkbest",
                "outlier_detection": None,
                "custom_transformers": [],
            },
            "model": {
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                "cross_validation": {"cv": 5, "scoring": "roc_auc"},
                "ensemble_method": None,
            },
            "evaluation": {"metrics": ["accuracy", "roc_auc", "f1", "precision", "recall"]},
            "output": {"save_model": True, "save_results": True, "save_pipeline": True},
        }


class PipelineRegistry:
    """Registry for managing multiple pipeline configurations."""

    def __init__(self, registry_dir: str = "pipeline_configs"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.pipelines = {}
        self._load_existing_configs()

    def _load_existing_configs(self):
        """Load existing configurations from registry directory."""
        for config_file in self.registry_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    config_dict = yaml.safe_load(f)
                name = config_dict.get("name", config_file.stem)
                self.pipelines[name] = config_file
                logger.info(f"Loaded pipeline config: {name}")
            except Exception as e:
                logger.error(f"Error loading config {config_file}: {e}")

    def register_pipeline(self, name: str, config: Union[PipelineConfig, Dict[str, Any]]):
        """Register a new pipeline configuration."""
        config_path = self.registry_dir / f"{name}.yaml"

        if isinstance(config, PipelineConfig):
            config_dict = asdict(config)
        else:
            config_dict = config

        config_dict["name"] = name

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        self.pipelines[name] = config_path
        logger.info(f"Registered pipeline: {name}")

    def get_pipeline(self, name: str) -> DynamicPipelineBuilder:
        """Get pipeline builder by name."""
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not found in registry")

        return DynamicPipelineBuilder(str(self.pipelines[name]))

    def list_pipelines(self) -> List[str]:
        """List all registered pipeline names."""
        return list(self.pipelines.keys())

    def create_pipeline_variants(self, base_name: str, variants: Dict[str, Dict[str, Any]]):
        """Create multiple variants of a base pipeline."""
        base_builder = self.get_pipeline(base_name)
        base_config = asdict(base_builder.config)

        for variant_name, changes in variants.items():
            variant_config = base_config.copy()

            # Apply changes recursively
            def update_nested_dict(d, changes):
                for key, value in changes.items():
                    if isinstance(value, dict) and key in d:
                        update_nested_dict(d[key], value)
                    else:
                        d[key] = value

            update_nested_dict(variant_config, changes)
            variant_config["name"] = f"{base_name}_{variant_name}"

            self.register_pipeline(f"{base_name}_{variant_name}", variant_config)


# Factory functions
def create_pipeline_builder(
    config: Union[PipelineConfig, str, Dict[str, Any]],
) -> DynamicPipelineBuilder:
    """Factory function to create pipeline builder."""
    return DynamicPipelineBuilder(config)


def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig()


def load_config_template(filepath: str):
    """Save a configuration template to file."""
    builder = DynamicPipelineBuilder(PipelineConfig())
    template = builder.get_config_template()

    with open(filepath, "w") as f:
        yaml.dump(template, f, default_flow_style=False)

    logger.info(f"Configuration template saved to {filepath}")
