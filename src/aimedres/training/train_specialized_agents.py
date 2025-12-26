#!/usr/bin/env python3
"""
Specialized Medical Agents Training Pipeline

This script trains models for specialized medical agents (Radiologists, Neurologists, etc.)
that can work together for enhanced medical diagnosis and consensus-based decision making.

Features:
- Trains models for multiple specialized agent roles
- Supports cross-validation with configurable folds
- Neural network training with configurable epochs
- Comprehensive metrics reporting for multi-agent consensus
- Model persistence for deployment with specialized agents
"""

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Core ML libraries
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural network training will be skipped.")

    # Create dummy nn module to avoid NameError
    class DummyNN:
        class Module:
            pass

    nn = DummyNN()

# Kaggle dataset loading
try:
    import kagglehub

    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not available. Will attempt to use local data.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:

    class SpecializedAgentMLP(nn.Module):
        """
        Multi-Layer Perceptron for specialized medical agent models
        """

        def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [128, 64, 32]

            layers = []
            current_dim = input_dim

            # Build hidden layers
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(current_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                    ]
                )
                current_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(current_dim, num_classes))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

else:
    # Dummy class when PyTorch is not available
    class SpecializedAgentMLP:
        def __init__(self, *args, **kwargs):
            pass


class SpecializedAgentsTrainingPipeline:
    """
    Complete training pipeline for specialized medical agents
    """

    def __init__(self, output_dir: str = "outputs", n_folds: int = 5, epochs: int = 50):
        """
        Initialize the training pipeline

        Args:
            output_dir: Directory to save outputs
            n_folds: Number of cross-validation folds
            epochs: Number of training epochs for neural networks
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.n_folds = n_folds
        self.epochs = epochs

        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "preprocessors").mkdir(exist_ok=True)
        (self.output_dir / "agent_models").mkdir(exist_ok=True)

        # Initialize components
        self.data = None
        self.preprocessor = None
        self.label_encoder = None
        self.X_processed = None
        self.y_encoded = None
        self.feature_names = None

        # Results storage
        self.agent_results = {
            "radiologist": {},
            "neurologist": {},
            "pathologist": {},
            "consensus": {},
        }

        logger.info(f"Initialized pipeline with {n_folds} folds and {epochs} epochs")

    def download_dataset(self) -> str:
        """
        Download the Alzheimer's dataset from Kaggle for agent training

        Returns:
            Path to the downloaded dataset
        """
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("kagglehub is required but not installed. Run: pip install kagglehub")

        logger.info("Downloading Alzheimer's dataset for agent training...")

        try:
            # Use Alzheimer's dataset as it has good medical features
            path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
            logger.info(f"Dataset downloaded to: {path}")

            dataset_path = Path(path)
            csv_files = list(dataset_path.glob("*.csv"))

            if not csv_files:
                raise FileNotFoundError("No CSV files found in the downloaded dataset")

            dataset_file = csv_files[0]
            logger.info(f"Using dataset file: {dataset_file}")

            return str(dataset_file)

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the dataset for agent training

        Args:
            data_path: Optional path to dataset. If None, downloads from Kaggle

        Returns:
            Loaded dataset as pandas DataFrame
        """
        if data_path is None:
            data_path = self.download_dataset()

        logger.info(f"Loading dataset from: {data_path}")

        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")

            return self.data

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for training

        Returns:
            Processed features and labels
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Preprocessing data...")

        # Identify target column (diagnosis-related)
        possible_target_cols = ["Diagnosis", "diagnosis", "Group", "group", "Class", "class"]
        target_col = None

        for col in possible_target_cols:
            if col in self.data.columns:
                target_col = col
                break

        if target_col is None:
            raise ValueError(
                f"Could not identify target column. Available columns: {list(self.data.columns)}"
            )

        logger.info(f"Using target column: {target_col}")

        # Separate features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]

        # Remove ID columns if present
        id_cols = [
            col
            for col in X.columns
            if col.lower() in ["id", "patientid", "patient_id", "subject_id"]
        ]
        if id_cols:
            X = X.drop(columns=id_cols)
            logger.info(f"Removed ID columns: {id_cols}")

        # Encode target
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Target classes: {self.label_encoder.classes_}")
        logger.info(f"Class distribution: {np.bincount(self.y_encoded)}")

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        # Build preprocessing pipeline
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_transformer, categorical_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers)

        # Fit and transform
        self.X_processed = self.preprocessor.fit_transform(X)

        # Extract feature names
        self.feature_names = self._get_feature_names(numeric_cols, categorical_cols)

        logger.info(f"Preprocessing complete. Feature dimension: {self.X_processed.shape[1]}")

        # Save preprocessor
        preprocessor_path = self.output_dir / "preprocessors" / "agent_preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Saved preprocessor to: {preprocessor_path}")

        # Save label encoder
        label_encoder_path = self.output_dir / "preprocessors" / "label_encoder.pkl"
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Saved label encoder to: {label_encoder_path}")

        return self.X_processed, self.y_encoded

    def _get_feature_names(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
        """Extract feature names after preprocessing"""
        feature_names = []

        # Add numeric column names
        feature_names.extend(numeric_cols)

        # Add categorical column names (one-hot encoded)
        if categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for i, col in enumerate(categorical_cols):
                categories = cat_encoder.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])

        return feature_names

    def train_agent_model(self, agent_name: str, model_type: str = "rf") -> Dict[str, Any]:
        """
        Train a model for a specific agent type

        Args:
            agent_name: Name of the specialized agent (e.g., 'radiologist', 'neurologist')
            model_type: Type of model to train ('rf', 'gb', 'lr')

        Returns:
            Training results and metrics
        """
        if self.X_processed is None or self.y_encoded is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        logger.info(f"\n{'='*80}")
        logger.info(f"Training {agent_name.upper()} Model (Type: {model_type})")
        logger.info(f"{'='*80}")

        # Select model based on type
        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_type == "gb":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "lr":
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
        }

        # Add ROC AUC for binary classification
        if len(self.label_encoder.classes_) == 2:
            scoring["roc_auc"] = "roc_auc"

        logger.info(f"Running {self.n_folds}-fold cross-validation...")
        cv_results = cross_validate(
            model,
            self.X_processed,
            self.y_encoded,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        # Calculate mean scores
        results = {
            "agent_name": agent_name,
            "model_type": model_type,
            "n_folds": self.n_folds,
            "accuracy": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "balanced_accuracy": cv_results["test_balanced_accuracy"].mean(),
            "balanced_accuracy_std": cv_results["test_balanced_accuracy"].std(),
            "f1_weighted": cv_results["test_f1_weighted"].mean(),
            "f1_weighted_std": cv_results["test_f1_weighted"].std(),
        }

        if "test_roc_auc" in cv_results:
            results["roc_auc"] = cv_results["test_roc_auc"].mean()
            results["roc_auc_std"] = cv_results["test_roc_auc"].std()

        # Log results
        logger.info(f"\n{agent_name.upper()} Cross-Validation Results:")
        logger.info(
            f"  Accuracy:          {results['accuracy']:.4f} (+/- {results['accuracy_std']:.4f})"
        )
        logger.info(
            f"  Balanced Accuracy: {results['balanced_accuracy']:.4f} (+/- {results['balanced_accuracy_std']:.4f})"
        )
        logger.info(
            f"  F1 Weighted:       {results['f1_weighted']:.4f} (+/- {results['f1_weighted_std']:.4f})"
        )
        if "roc_auc" in results:
            logger.info(
                f"  ROC AUC:           {results['roc_auc']:.4f} (+/- {results['roc_auc_std']:.4f})"
            )

        # Train final model on full dataset
        logger.info(f"Training final {agent_name} model on full dataset...")
        model.fit(self.X_processed, self.y_encoded)

        # Save model
        model_path = self.output_dir / "agent_models" / f"{agent_name}_{model_type}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {agent_name} model to: {model_path}")

        # Store results
        self.agent_results[agent_name][model_type] = results

        return results

    def train_neural_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Train a neural network model for a specific agent

        Args:
            agent_name: Name of the specialized agent

        Returns:
            Training results and metrics
        """
        if not TORCH_AVAILABLE:
            logger.warning(
                f"PyTorch not available. Skipping neural network training for {agent_name}."
            )
            return {}

        if self.X_processed is None or self.y_encoded is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        logger.info(f"\n{'='*80}")
        logger.info(f"Training {agent_name.upper()} Neural Network")
        logger.info(f"{'='*80}")

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed,
            self.y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=self.y_encoded,
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize model
        input_dim = self.X_processed.shape[1]
        num_classes = len(self.label_encoder.classes_)
        model = SpecializedAgentMLP(input_dim, num_classes)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logger.info(f"Training on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        logger.info(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test_tensor.to(device)
            outputs = model(X_test_device)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted)
        balanced_acc = balanced_accuracy_score(y_test, predicted)
        f1 = f1_score(y_test, predicted, average="weighted")

        results = {
            "agent_name": agent_name,
            "model_type": "neural",
            "epochs": self.epochs,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_weighted": f1,
        }

        # Log results
        logger.info(f"\n{agent_name.upper()} Neural Network Results:")
        logger.info(f"  Accuracy:          {accuracy:.4f}")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"  F1 Weighted:       {f1:.4f}")

        # Save model
        model_path = self.output_dir / "agent_models" / f"{agent_name}_neural_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved {agent_name} neural model to: {model_path}")

        # Store results
        if "neural" not in self.agent_results[agent_name]:
            self.agent_results[agent_name]["neural"] = {}
        self.agent_results[agent_name]["neural"] = results

        return results

    def train_all_agents(self):
        """Train models for all specialized agents"""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING ALL SPECIALIZED AGENTS")
        logger.info("=" * 80)

        agents = [
            ("radiologist", "rf"),  # Random Forest for radiologist
            ("neurologist", "gb"),  # Gradient Boosting for neurologist
            ("pathologist", "lr"),  # Logistic Regression for pathologist
        ]

        # Train classical models for each agent
        for agent_name, model_type in agents:
            try:
                self.train_agent_model(agent_name, model_type)
            except Exception as e:
                logger.error(f"Error training {agent_name} model: {e}")

        # Train neural networks for each agent
        if TORCH_AVAILABLE:
            for agent_name, _ in agents:
                try:
                    self.train_neural_agent(agent_name)
                except Exception as e:
                    logger.error(f"Error training {agent_name} neural network: {e}")

        # Calculate consensus metrics
        self.calculate_consensus_metrics()

        # Save all results
        self.save_results()

    def calculate_consensus_metrics(self):
        """Calculate metrics for multi-agent consensus"""
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING CONSENSUS METRICS")
        logger.info("=" * 80)

        # Aggregate agent performances
        all_accuracies = []
        all_f1_scores = []

        for agent_name, agent_results in self.agent_results.items():
            if agent_name == "consensus":
                continue

            for model_type, metrics in agent_results.items():
                if "accuracy" in metrics:
                    all_accuracies.append(metrics["accuracy"])
                if "f1_weighted" in metrics:
                    all_f1_scores.append(metrics["f1_weighted"])

        if all_accuracies:
            consensus_accuracy = np.mean(all_accuracies)
            consensus_std = np.std(all_accuracies)

            self.agent_results["consensus"] = {
                "mean_accuracy": consensus_accuracy,
                "accuracy_std": consensus_std,
                "mean_f1": np.mean(all_f1_scores) if all_f1_scores else 0.0,
                "f1_std": np.std(all_f1_scores) if all_f1_scores else 0.0,
                "n_agents": len([a for a in self.agent_results.keys() if a != "consensus"]),
                "diversity_score": consensus_std,  # Higher diversity indicates more varied opinions
            }

            logger.info(f"Consensus Metrics:")
            logger.info(f"  Mean Accuracy:     {consensus_accuracy:.4f} (+/- {consensus_std:.4f})")
            logger.info(f"  Mean F1:           {self.agent_results['consensus']['mean_f1']:.4f}")
            logger.info(f"  Diversity Score:   {consensus_std:.4f}")
            logger.info(f"  Number of Agents:  {self.agent_results['consensus']['n_agents']}")

    def save_results(self):
        """Save all training results to JSON"""
        results_path = self.output_dir / "metrics" / "agent_training_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            return obj

        serializable_results = {}
        for agent, models in self.agent_results.items():
            serializable_results[agent] = {}
            if isinstance(models, dict):
                for model_type, metrics in models.items():
                    if isinstance(metrics, dict):
                        serializable_results[agent][model_type] = {
                            k: convert_to_json_serializable(v) for k, v in metrics.items()
                        }
                    else:
                        serializable_results[agent][model_type] = convert_to_json_serializable(
                            metrics
                        )
            else:
                serializable_results[agent] = convert_to_json_serializable(models)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {results_path}")

        # Create summary file
        summary_path = self.output_dir / "agent_training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SPECIALIZED MEDICAL AGENTS TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Folds: {self.n_folds}\n")
            f.write(f"Number of Epochs: {self.epochs}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            for agent, models in self.agent_results.items():
                f.write(f"\n{agent.upper()} AGENT:\n")
                f.write("-" * 40 + "\n")
                if isinstance(models, dict):
                    for model_type, metrics in models.items():
                        f.write(f"  Model Type: {model_type}\n")
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                if isinstance(value, float):
                                    f.write(f"    {metric}: {value:.4f}\n")
                                else:
                                    f.write(f"    {metric}: {value}\n")
                        else:
                            f.write(f"    Value: {metrics}\n")
                        f.write("\n")
                else:
                    f.write(f"  Value: {models}\n\n")

        logger.info(f"✅ Summary saved to: {summary_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train models for specialized medical agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="specialized_agents_results",
        help="Directory to save training outputs",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset CSV file (if None, downloads from Kaggle)",
    )

    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")

    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs for neural networks"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("SPECIALIZED MEDICAL AGENTS TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Cross-validation folds: {args.folds}")
    logger.info(f"Neural network epochs: {args.epochs}")
    logger.info("=" * 80 + "\n")

    try:
        # Initialize pipeline
        pipeline = SpecializedAgentsTrainingPipeline(
            output_dir=args.output_dir, n_folds=args.folds, epochs=args.epochs
        )

        # Load and preprocess data
        pipeline.load_data(args.data_path)
        pipeline.preprocess_data()

        # Train all agent models
        pipeline.train_all_agents()

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {args.output_dir}")
        logger.info(f"Models saved to: {args.output_dir}/agent_models/")
        logger.info(f"Metrics saved to: {args.output_dir}/metrics/")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
