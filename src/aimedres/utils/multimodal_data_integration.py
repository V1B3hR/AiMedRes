#!/usr/bin/env python3
"""
Multi-modal Data Integration System for Advanced Medical Data Processing
Supports imaging, genetic, longitudinal data and privacy-preserving federated learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
import kagglehub
from contextlib import contextmanager

# Machine learning imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import base components
from aimedres.utils.data_loaders import DataLoader

logger = logging.getLogger("MultimodalDataIntegration")


@dataclass
class DataModality:
    """Represents a specific data modality"""
    name: str
    data_type: str  # 'tabular', 'imaging', 'genetic', 'longitudinal'
    source_path: str
    preprocessing_config: Dict[str, Any]
    privacy_level: str  # 'public', 'restricted', 'private'


class MultiModalDataLoader(DataLoader):
    """Advanced data loader for multi-modal medical data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modalities = {}
        self.loaded_data = {}
        self.fusion_strategy = config.get('fusion_strategy', 'concatenation')
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data"""
        return len(data) > 0 and len(data.columns) > 0
        
    def register_modality(self, modality: DataModality):
        """Register a new data modality"""
        self.modalities[modality.name] = modality
        
    def load_lung_disease_dataset(self) -> pd.DataFrame:
        """Load lung disease dataset from Kaggle"""
        try:
            logger.info("Downloading lung disease dataset from Kaggle...")
            
            # Download dataset using kagglehub
            path = kagglehub.dataset_download("fatemehmehrparvar/lung-disease")
            
            # Find CSV files in the downloaded path
            csv_files = list(Path(path).glob("*.csv"))
            
            if not csv_files:
                logger.warning("No CSV files found in downloaded dataset")
                return self._create_mock_lung_disease_data()
            
            # Load the first CSV file found
            data_file = csv_files[0]
            logger.info(f"Loading data from: {data_file}")
            
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load Kaggle dataset: {e}")
            logger.info("Creating mock lung disease data for demonstration...")
            return self._create_mock_lung_disease_data()
    
    def _create_mock_lung_disease_data(self) -> pd.DataFrame:
        """Create mock lung disease data for testing purposes"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            # Patient demographics
            'age': np.random.randint(20, 90, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'smoking_history': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.4, 0.3, 0.3]),
            'pack_years': np.random.exponential(10, n_samples),
            
            # Clinical measurements
            'fev1': np.random.normal(2.5, 0.8, n_samples),  # Forced Expiratory Volume
            'fvc': np.random.normal(3.2, 1.0, n_samples),   # Forced Vital Capacity
            'fev1_fvc_ratio': np.random.normal(0.78, 0.1, n_samples),
            'oxygen_saturation': np.random.normal(96, 3, n_samples),
            
            # Imaging features (simulated CT scan measurements)
            'lung_opacity_score': np.random.uniform(0, 100, n_samples),
            'emphysema_score': np.random.uniform(0, 100, n_samples),
            'fibrosis_score': np.random.uniform(0, 100, n_samples),
            'nodule_count': np.random.poisson(2, n_samples),
            'largest_nodule_size': np.random.exponential(5, n_samples),
            
            # Genetic factors (simulated)
            'alpha1_antitrypsin_deficiency': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'cftr_mutation': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            'genetic_risk_score': np.random.normal(50, 15, n_samples),
            
            # Laboratory values
            'c_reactive_protein': np.random.exponential(3, n_samples),
            'white_blood_cells': np.random.normal(7000, 2000, n_samples),
            'eosinophil_count': np.random.exponential(200, n_samples),
        }
        
        # Create disease labels based on risk factors
        disease_prob = (
            0.1 +
            0.3 * (data['age'] > 65) +
            0.2 * (data['smoking_history'] == 'Current') +
            0.15 * (data['pack_years'] > 20) +
            0.2 * (data['fev1_fvc_ratio'] < 0.7) +
            0.1 * (data['emphysema_score'] > 50) +
            0.1 * np.random.random(n_samples)  # Random component
        )
        
        # Generate disease categories
        disease_categories = []
        for prob in disease_prob:
            if prob > 0.8:
                disease = 'COPD'
            elif prob > 0.6:
                disease = 'Asthma' 
            elif prob > 0.4:
                disease = 'Pneumonia'
            elif prob > 0.2:
                disease = 'Fibrosis'
            else:
                disease = 'Normal'
            disease_categories.append(disease)
        
        data['diagnosis'] = disease_categories
        
        df = pd.DataFrame(data)
        
        # Ensure realistic constraints
        df['fev1'] = np.clip(df['fev1'], 0.5, 5.0)
        df['fvc'] = np.clip(df['fvc'], 0.8, 6.0)
        df['fev1_fvc_ratio'] = np.clip(df['fev1_fvc_ratio'], 0.3, 1.0)
        df['oxygen_saturation'] = np.clip(df['oxygen_saturation'], 85, 100)
        df['pack_years'] = np.clip(df['pack_years'], 0, 80)
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """Load and integrate multi-modal data"""
        
        # Load primary lung disease dataset
        lung_data = self.load_lung_disease_dataset()
        self.loaded_data['lung_disease'] = lung_data
        
        # If other modalities are registered, load and integrate them
        integrated_data = lung_data.copy()
        
        for modality_name, modality in self.modalities.items():
            try:
                modality_data = self._load_modality_data(modality)
                self.loaded_data[modality_name] = modality_data
                integrated_data = self._fuse_data(integrated_data, modality_data, modality_name)
            except Exception as e:
                logger.warning(f"Failed to load modality {modality_name}: {e}")
        
        return integrated_data
    
    def _load_modality_data(self, modality: DataModality) -> pd.DataFrame:
        """Load data for a specific modality"""
        
        if modality.data_type == 'tabular':
            return pd.read_csv(modality.source_path)
        elif modality.data_type == 'longitudinal':
            return self._load_longitudinal_data(modality.source_path)
        elif modality.data_type == 'genetic':
            return self._load_genetic_data(modality.source_path)
        elif modality.data_type == 'imaging':
            return self._load_imaging_features(modality.source_path)
        else:
            raise ValueError(f"Unsupported data type: {modality.data_type}")
    
    def _load_longitudinal_data(self, source_path: str) -> pd.DataFrame:
        """Load and process longitudinal data"""
        # This would typically load time-series medical data
        # For now, create mock longitudinal data
        np.random.seed(42)
        n_patients = 200
        n_timepoints = 5
        
        longitudinal_data = []
        for patient_id in range(n_patients):
            for timepoint in range(n_timepoints):
                record = {
                    'patient_id': patient_id,
                    'timepoint': timepoint,
                    'days_from_baseline': timepoint * 90,  # Every 3 months
                    'fev1_longitudinal': 2.5 + np.random.normal(0, 0.1) - timepoint * 0.05,
                    'symptom_score': np.random.randint(0, 10),
                    'medication_compliance': np.random.uniform(0.7, 1.0),
                }
                longitudinal_data.append(record)
        
        return pd.DataFrame(longitudinal_data)
    
    def _load_genetic_data(self, source_path: str) -> pd.DataFrame:
        """Load and process genetic data"""
        # Mock genetic variant data
        np.random.seed(42)
        n_patients = 500
        
        genetic_variants = [
            'rs1800470_TGFB1', 'rs1982073_TGFB1', 'rs1800896_IL10',
            'rs1800795_IL6', 'rs361525_TNF', 'rs1800629_TNF',
            'rs2010963_VEGFA', 'rs699947_VEGFA', 'rs4880_SOD2'
        ]
        
        genetic_data = {'patient_id': range(n_patients)}
        
        for variant in genetic_variants:
            # Simulate genotype calls (0=homozygous reference, 1=heterozygous, 2=homozygous alternate)
            genetic_data[variant] = np.random.choice([0, 1, 2], n_patients, p=[0.7, 0.25, 0.05])
        
        return pd.DataFrame(genetic_data)
    
    def _load_imaging_features(self, source_path: str) -> pd.DataFrame:
        """Load and process medical imaging features"""
        # Mock imaging analysis results
        np.random.seed(42)
        n_patients = 300
        
        imaging_data = {
            'patient_id': range(n_patients),
            'lung_volume_ml': np.random.normal(5000, 1000, n_patients),
            'air_trapping_percent': np.random.uniform(0, 30, n_patients),
            'ground_glass_opacity_score': np.random.uniform(0, 100, n_patients),
            'reticulation_score': np.random.uniform(0, 50, n_patients),
            'honeycombing_present': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
            'pleural_effusion_volume': np.random.exponential(50, n_patients),
        }
        
        return pd.DataFrame(imaging_data)
    
    def _fuse_data(self, primary_data: pd.DataFrame, modality_data: pd.DataFrame, 
                   modality_name: str) -> pd.DataFrame:
        """Fuse multi-modal data using specified strategy"""
        
        if self.fusion_strategy == 'concatenation':
            # Simple concatenation (assumes same number of rows)
            if len(primary_data) == len(modality_data):
                # Add prefix to avoid column name conflicts
                modality_data_prefixed = modality_data.add_prefix(f"{modality_name}_")
                return pd.concat([primary_data, modality_data_prefixed], axis=1)
            else:
                logger.warning(f"Size mismatch for {modality_name}, skipping fusion")
                return primary_data
                
        elif self.fusion_strategy == 'patient_id_merge':
            # Merge on patient ID
            if 'patient_id' in modality_data.columns:
                # Create patient IDs in primary data if not present
                if 'patient_id' not in primary_data.columns:
                    primary_data['patient_id'] = range(len(primary_data))
                
                return primary_data.merge(modality_data, on='patient_id', how='left')
            else:
                logger.warning(f"No patient_id in {modality_name}, using concatenation")
                return self._fuse_data(primary_data, modality_data, modality_name)
        
        return primary_data


class DataFusionProcessor:
    """Advanced data fusion techniques for multi-modal integration"""
    
    def __init__(self):
        self.fusion_models = {}
        self.dimensionality_reducers = {}
        
    def early_fusion(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Early fusion: concatenate all features before training"""
        
        logger.info("Performing early fusion...")
        
        fused_data = None
        for modality_name, data in data_dict.items():
            if fused_data is None:
                fused_data = data.copy()
            else:
                # Add modality prefix and concatenate
                data_prefixed = data.add_prefix(f"{modality_name}_")
                fused_data = pd.concat([fused_data, data_prefixed], axis=1)
        
        return fused_data
    
    def late_fusion(self, data_dict: Dict[str, pd.DataFrame], 
                   target_column: str, use_mlflow: bool = True, 
                   mlflow_experiment: str = "multimodal_late_fusion") -> Dict[str, Any]:
        """Late fusion: train separate models and combine predictions with evaluation"""
        
        logger.info("Performing late fusion...")
        
        if use_mlflow:
            import mlflow
            import mlflow.sklearn
            mlflow.set_experiment(mlflow_experiment)  # noqa: F821
        
        # Use MLflow context if enabled
        if use_mlflow:
            with mlflow.start_run():  # noqa: F821
                return self._perform_late_fusion_with_mlflow(data_dict, target_column, True)
        else:
            return self._perform_late_fusion_with_mlflow(data_dict, target_column, False)
    
    def _perform_late_fusion_with_mlflow(self, data_dict: Dict[str, pd.DataFrame], 
                                        target_column: str, use_mlflow: bool) -> Dict[str, Any]:
        """Internal method to perform late fusion with optional MLflow logging."""
        """Internal method to perform late fusion with optional MLflow logging."""
        
        if use_mlflow:
            mlflow.log_param("fusion_type", "late_fusion")  # noqa: F821
            mlflow.log_param("num_modalities", len(data_dict))  # noqa: F821
            mlflow.log_param("modalities", list(data_dict.keys()))  # noqa: F821
        
        modality_models = {}
        modality_predictions = {}
        modality_metrics = {}
        
        for modality_name, data in data_dict.items():
            if target_column not in data.columns:
                logger.warning(f"Target column {target_column} not found in {modality_name}")
                continue
                
            logger.info(f"Training model for {modality_name}...")
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            if use_mlflow:
                mlflow.log_param(f"{modality_name}_features", X.shape[1])  # noqa: F821
                mlflow.log_param(f"{modality_name}_samples", X.shape[0])  # noqa: F821
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate individual modality
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            modality_metrics[modality_name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            if use_mlflow:
                mlflow.log_metric(f"{modality_name}_accuracy", accuracy)  # noqa: F821
                mlflow.log_metric(f"{modality_name}_precision", report['weighted avg']['precision'])  # noqa: F821
                mlflow.log_metric(f"{modality_name}_recall", report['weighted avg']['recall'])  # noqa: F821
                mlflow.log_metric(f"{modality_name}_f1", report['weighted avg']['f1-score'])  # noqa: F821
            
            modality_models[modality_name] = model
            modality_predictions[modality_name] = y_pred_proba
            
            logger.info(f"{modality_name} - Accuracy: {accuracy:.4f}")
        
        # Combine predictions (simple averaging)
        ensemble_predictions = {}
        if modality_predictions:
            combined_predictions = np.mean(list(modality_predictions.values()), axis=0)
            ensemble_pred_labels = np.argmax(combined_predictions, axis=1)
            
            # Evaluate ensemble if we have test data
            if len(modality_predictions) > 0:
                # Use the same test split as the last modality for ensemble evaluation
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred_labels)
                ensemble_report = classification_report(y_test, ensemble_pred_labels, output_dict=True)
                
                ensemble_predictions = {
                    'accuracy': ensemble_accuracy,
                    'precision': ensemble_report['weighted avg']['precision'],
                    'recall': ensemble_report['weighted avg']['recall'],
                    'f1_score': ensemble_report['weighted avg']['f1-score']
                }
                
                if use_mlflow:
                    mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)  # noqa: F821
                    mlflow.log_metric("ensemble_precision", ensemble_report['weighted avg']['precision'])  # noqa: F821
                    mlflow.log_metric("ensemble_recall", ensemble_report['weighted avg']['recall'])  # noqa: F821
                    mlflow.log_metric("ensemble_f1", ensemble_report['weighted avg']['f1-score'])  # noqa: F821
                    
                    # Log the ensemble as an artifact
                    import joblib
                    ensemble_path = "ensemble_models.pkl"
                    joblib.dump(modality_models, ensemble_path)
                    mlflow.log_artifact(ensemble_path)  # noqa: F821
                
                logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}")
            
            return {
                'modality_models': modality_models,
                'modality_predictions': modality_predictions,
                'combined_predictions': combined_predictions,
                'modality_metrics': modality_metrics,
                'ensemble_metrics': ensemble_predictions
            }
        
        return {}
    
    def hierarchical_fusion(self, data_dict: Dict[str, pd.DataFrame], 
                          hierarchy: List[List[str]]) -> pd.DataFrame:
        """Hierarchical fusion: fuse data in stages according to hierarchy"""
        
        logger.info("Performing hierarchical fusion...")
        
        fused_levels = {}
        
        for level_idx, level_modalities in enumerate(hierarchy):
            level_data = []
            
            for modality in level_modalities:
                if modality in data_dict:
                    level_data.append(data_dict[modality])
                elif modality in fused_levels:
                    level_data.append(fused_levels[modality])
            
            if level_data:
                # Concatenate data at this level
                level_fused = pd.concat(level_data, axis=1)
                
                # Apply dimensionality reduction if needed
                if level_fused.shape[1] > 100:
                    pca = PCA(n_components=min(50, level_fused.shape[1]))
                    level_reduced = pca.fit_transform(level_fused.select_dtypes(include=[np.number]))
                    level_fused = pd.DataFrame(level_reduced, 
                                             columns=[f'PC_{i+1}' for i in range(level_reduced.shape[1])])
                
                fused_levels[f'level_{level_idx}'] = level_fused
        
        # Return final fused data
        return fused_levels.get(f'level_{len(hierarchy)-1}', pd.DataFrame())


class PrivacyPreservingFederatedLearning:
    """Privacy-preserving federated learning for distributed medical data"""
    
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.global_model = None
        self.client_models = {}
        self.aggregation_history = []
        
    def add_differential_privacy_noise(self, gradients: np.ndarray, 
                                     sensitivity: float = 1.0, 
                                     epsilon: float = 0.1) -> np.ndarray:
        """Add differential privacy noise to gradients"""
        
        # Laplace noise for differential privacy
        noise_scale = sensitivity / epsilon
        noise = np.random.laplace(0, noise_scale, gradients.shape)
        
        return gradients + noise
    
    def secure_aggregation(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform secure aggregation of client model updates"""
        
        logger.info(f"Aggregating updates from {len(client_updates)} clients...")
        
        if not client_updates:
            return {}
        
        # Simple federated averaging (in practice, would use more sophisticated secure aggregation)
        aggregated_weights = {}
        
        # Get model structure from first client
        first_client = client_updates[0]
        model_weights = first_client['model_weights']
        
        # Initialize aggregated weights
        for key in model_weights.keys():
            aggregated_weights[key] = np.zeros_like(model_weights[key])
        
        # Sum all client weights
        total_samples = sum(client['num_samples'] for client in client_updates)
        
        for client_update in client_updates:
            client_weights = client_update['model_weights']
            client_samples = client_update['num_samples']
            weight_factor = client_samples / total_samples
            
            for key in client_weights.keys():
                aggregated_weights[key] += client_weights[key] * weight_factor
        
        # Add differential privacy noise if enabled
        if self.privacy_budget > 0:
            for key in aggregated_weights.keys():
                aggregated_weights[key] = self.add_differential_privacy_noise(
                    aggregated_weights[key], 
                    epsilon=self.privacy_budget / len(aggregated_weights)
                )
        
        return {
            'aggregated_weights': aggregated_weights,
            'num_clients': len(client_updates),
            'total_samples': total_samples
        }
    
    def simulate_federated_training(self, distributed_data: List[pd.DataFrame], 
                                  target_column: str, num_rounds: int = 10) -> Dict[str, Any]:
        """Simulate federated learning training across multiple data silos"""
        
        logger.info(f"Starting federated training with {len(distributed_data)} clients for {num_rounds} rounds...")
        
        federated_results = []
        
        for round_num in range(num_rounds):
            logger.info(f"Federated round {round_num + 1}/{num_rounds}")
            
            client_updates = []
            
            # Train local models on each client's data
            for client_id, client_data in enumerate(distributed_data):
                if target_column not in client_data.columns:
                    continue
                
                # Prepare client data
                X_client = client_data.drop(columns=[target_column])
                y_client = client_data[target_column]
                
                # Handle categorical variables
                categorical_columns = X_client.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    le = LabelEncoder()
                    X_client[col] = le.fit_transform(X_client[col].astype(str))
                
                # Train local model
                local_model = RandomForestClassifier(n_estimators=50, random_state=42 + round_num)
                local_model.fit(X_client, y_client)
                
                # Simulate model weights (for demonstration)
                model_weights = {
                    f'feature_weights_{i}': np.random.random(10) 
                    for i in range(min(5, X_client.shape[1]))
                }
                
                client_update = {
                    'client_id': client_id,
                    'model_weights': model_weights,
                    'num_samples': len(client_data),
                    'local_accuracy': local_model.score(X_client, y_client)
                }
                
                client_updates.append(client_update)
            
            # Aggregate client updates
            aggregated_result = self.secure_aggregation(client_updates)
            
            round_result = {
                'round': round_num + 1,
                'num_participating_clients': len(client_updates),
                'average_local_accuracy': np.mean([c['local_accuracy'] for c in client_updates]),
                'aggregation_result': aggregated_result
            }
            
            federated_results.append(round_result)
            
            logger.info(f"  Round {round_num + 1} completed - Avg accuracy: {round_result['average_local_accuracy']:.3f}")
        
        return {
            'federated_rounds': federated_results,
            'final_global_model': aggregated_result,
            'privacy_budget_used': self.privacy_budget
        }


class MultiModalMedicalAI:
    """Comprehensive multi-modal medical AI system"""
    
    def __init__(self):
        self.data_loader = None
        self.fusion_processor = DataFusionProcessor()
        self.federated_learner = PrivacyPreservingFederatedLearning()
        self.trained_models = {}
        
    def setup_data_integration(self, config: Dict[str, Any]):
        """Setup multi-modal data integration"""
        
        self.data_loader = MultiModalDataLoader(config)
        
        # Register additional modalities if specified
        if 'modalities' in config:
            for modality_config in config['modalities']:
                modality = DataModality(**modality_config)
                self.data_loader.register_modality(modality)
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive multi-modal analysis"""
        
        logger.info("🚀 Starting comprehensive multi-modal medical AI analysis...")
        
        # Load integrated data
        integrated_data = self.data_loader.load_data()
        logger.info(f"Loaded integrated dataset: {integrated_data.shape}")
        
        # Analyze data distribution
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(integrated_data),
                'total_features': len(integrated_data.columns),
                'feature_types': integrated_data.dtypes.value_counts().to_dict()
            }
        }
        
        # If diagnosis column exists, perform classification analysis
        if 'diagnosis' in integrated_data.columns:
            analysis_results.update(self._perform_classification_analysis(integrated_data))
        
        # Perform clustering analysis for unsupervised insights
        analysis_results.update(self._perform_clustering_analysis(integrated_data))
        
        # If multiple data modalities are loaded, perform fusion analysis
        if len(self.data_loader.loaded_data) > 1:
            analysis_results.update(self._perform_fusion_analysis())
        
        return analysis_results
    
    def _perform_classification_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform classification analysis on the integrated dataset"""
        
        logger.info("📊 Performing classification analysis...")
        
        # Prepare data for classification
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'classification_results': {
                'accuracy': accuracy,
                'num_classes': len(np.unique(y)),
                'class_distribution': y.value_counts().to_dict(),
                'top_features': top_features,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        }
    
    def _perform_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis for patient stratification"""
        
        logger.info("🔍 Performing clustering analysis...")
        
        # Select numerical features for clustering
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            return {'clustering_results': 'No numerical features available for clustering'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data.fillna(0))
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(10, scaled_data.shape[1]))
        reduced_data = pca.fit_transform(scaled_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_data)
        
        # Analyze clusters
        cluster_summary = {}
        for cluster_id in range(5):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = numerical_data[cluster_mask]
            
            cluster_summary[f'cluster_{cluster_id}'] = {
                'size': np.sum(cluster_mask),
                'mean_values': cluster_data.mean().to_dict(),
                'std_values': cluster_data.std().to_dict()
            }
        
        return {
            'clustering_results': {
                'num_clusters': 5,
                'cluster_summary': cluster_summary,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
            }
        }
    
    def _perform_fusion_analysis(self) -> Dict[str, Any]:
        """Perform data fusion analysis across modalities"""
        
        logger.info("🔗 Performing data fusion analysis...")
        
        fusion_results = {}
        
        # Early fusion
        early_fused = self.fusion_processor.early_fusion(self.data_loader.loaded_data)
        fusion_results['early_fusion_shape'] = early_fused.shape
        
        # Late fusion (if diagnosis column available)
        target_col = 'diagnosis'
        if target_col in early_fused.columns:
            late_fusion_result = self.fusion_processor.late_fusion(
                self.data_loader.loaded_data, target_col
            )
            if late_fusion_result:
                fusion_results['late_fusion_models'] = list(late_fusion_result['modality_models'].keys())
        
        return {'fusion_analysis': fusion_results}


def run_multimodal_demo() -> Dict[str, Any]:
    """Run a comprehensive multi-modal medical AI demonstration"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuration for multi-modal system
    config = {
        'fusion_strategy': 'concatenation',
        'modalities': [
            {
                'name': 'longitudinal',
                'data_type': 'longitudinal', 
                'source_path': 'mock_longitudinal.csv',
                'preprocessing_config': {},
                'privacy_level': 'restricted'
            },
            {
                'name': 'genetic',
                'data_type': 'genetic',
                'source_path': 'mock_genetic.csv', 
                'preprocessing_config': {},
                'privacy_level': 'private'
            },
            {
                'name': 'imaging',
                'data_type': 'imaging',
                'source_path': 'mock_imaging.csv',
                'preprocessing_config': {},
                'privacy_level': 'restricted'
            }
        ]
    }
    
    # Initialize multi-modal AI system
    multimodal_ai = MultiModalMedicalAI()
    multimodal_ai.setup_data_integration(config)
    
    # Run comprehensive analysis
    results = multimodal_ai.run_comprehensive_analysis()
    
    # Demonstrate federated learning
    logger.info("🔐 Demonstrating privacy-preserving federated learning...")
    
    # Create simulated distributed datasets
    main_data = multimodal_ai.data_loader.load_data()
    if 'diagnosis' in main_data.columns:
        # Split data to simulate different medical institutions
        n_clients = 3
        client_datasets = []
        for i in range(n_clients):
            start_idx = i * len(main_data) // n_clients
            end_idx = (i + 1) * len(main_data) // n_clients
            client_data = main_data.iloc[start_idx:end_idx].copy()
            client_datasets.append(client_data)
        
        # Run federated learning simulation
        federated_results = multimodal_ai.federated_learner.simulate_federated_training(
            client_datasets, 'diagnosis', num_rounds=5
        )
        
        results['federated_learning'] = federated_results
    
    # Save results
    results_path = Path("multimodal_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Multi-modal analysis completed! Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Run the multi-modal demonstration
    results = run_multimodal_demo()
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-MODAL MEDICAL AI ANALYSIS SUMMARY")
    print("="*60)
    
    if 'data_summary' in results:
        print(f"Total Samples: {results['data_summary']['total_samples']}")
        print(f"Total Features: {results['data_summary']['total_features']}")
    
    if 'classification_results' in results:
        print(f"Classification Accuracy: {results['classification_results']['accuracy']:.3f}")
        print(f"Number of Classes: {results['classification_results']['num_classes']}")
    
    if 'clustering_results' in results:
        print(f"Clustering: {results['clustering_results']['num_clusters']} clusters identified")
    
    if 'federated_learning' in results:
        fed_results = results['federated_learning']
        final_accuracy = fed_results['federated_rounds'][-1]['average_local_accuracy']
        print(f"Federated Learning: Final accuracy {final_accuracy:.3f}")
        print(f"Privacy Budget Used: {fed_results['privacy_budget_used']}")
    
    print("="*60)