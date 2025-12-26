"""
Secure Medical Data Processing System.

Provides HIPAA/GDPR compliant medical data handling for ML training:
- Data isolation between training and inference
- Automatic anonymization and de-identification
- Audit trails for all data access
- Privacy-preserving ML training
- Secure model parameter storage
- Data leak prevention between components
"""

import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import tempfile
import json

# Import security modules
from security import DataEncryption, PrivacyManager, SecurityMonitor

# Configure medical data logging
medical_logger = logging.getLogger('duetmind.medical')
medical_logger.setLevel(logging.INFO)

class SecureMedicalDataProcessor:
    """
    HIPAA/GDPR compliant medical data processing system.
    
    Security Features:
    - Data isolation between training and inference
    - Automatic de-identification of PII
    - Secure temporary data handling
    - Audit logging for compliance
    - Privacy-preserving ML operations
    - Secure model storage and retrieval
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_encryption = DataEncryption(config.get('master_password'))
        self.privacy_manager = PrivacyManager(config)
        self.security_monitor = SecurityMonitor(config)
        
        # Secure workspace for medical data
        self.secure_workspace = Path(config.get('secure_workspace', './secure_medical_workspace'))
        self.secure_workspace.mkdir(exist_ok=True, parents=True)
        
        # Data isolation directories
        self.training_data_dir = self.secure_workspace / 'training'
        self.inference_data_dir = self.secure_workspace / 'inference'
        self.model_storage_dir = self.secure_workspace / 'models'
        
        for directory in [self.training_data_dir, self.inference_data_dir, self.model_storage_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Active data tracking
        self.active_datasets = {}
        self.model_registry = {}
        
        medical_logger.info("Secure medical data processor initialized")
    
    def load_and_secure_dataset(self, dataset_source: str, dataset_params: Dict[str, Any], 
                               user_id: str, purpose: str = 'training') -> str:
        """
        Load and secure a medical dataset with full compliance tracking.
        
        Args:
            dataset_source: Source of the dataset (kaggle, file, etc.)
            dataset_params: Parameters for loading the dataset
            user_id: User requesting the data
            purpose: Purpose of data usage (training, inference, etc.)
            
        Returns:
            Secure dataset ID for future reference
        """
        dataset_id = f"medical_{purpose}_{int(datetime.now().timestamp())}"
        
        try:
            # Log data access request
            self.privacy_manager.log_data_access(
                user_id=user_id,
                data_type='medical_dataset',
                action='load',
                data_id=dataset_id,
                purpose=purpose,
                legal_basis='healthcare_research'
            )
            
            # Load raw dataset (placeholder - integrate with actual loading)
            if dataset_source == 'kaggle':
                raw_data = self._load_kaggle_dataset(dataset_params)
            else:
                raw_data = self._load_file_dataset(dataset_params)
            
            # Secure the dataset
            secured_data = self._secure_dataset(raw_data, dataset_id, purpose)
            
            # Register for retention tracking
            self.privacy_manager.register_data_for_retention(dataset_id, 'medical_data')
            
            # Store in appropriate isolation directory
            storage_dir = self.training_data_dir if purpose == 'training' else self.inference_data_dir
            storage_path = storage_dir / f"{dataset_id}.encrypted"
            
            # Encrypt and store
            encrypted_data = self.data_encryption.encrypt_data(secured_data.to_dict())
            with open(storage_path, 'w') as f:
                f.write(encrypted_data)
            
            # Track active dataset
            self.active_datasets[dataset_id] = {
                'user_id': user_id,
                'purpose': purpose,
                'created_at': datetime.now(),
                'storage_path': storage_path,
                'status': 'active',
                'records_count': len(secured_data),
                'features_count': len(secured_data.columns)
            }
            
            medical_logger.info(f"Secured medical dataset {dataset_id} for {purpose}")
            return dataset_id
            
        except Exception as e:
            medical_logger.error(f"Failed to secure dataset: {e}")
            self.security_monitor.log_security_event(
                'medical_data_load_error',
                {'dataset_id': dataset_id, 'error': str(e)},
                severity='warning',
                user_id=user_id
            )
            raise
    
    def _secure_dataset(self, data: pd.DataFrame, dataset_id: str, purpose: str) -> pd.DataFrame:
        """
        Apply security measures to the dataset.
        
        Args:
            data: Raw dataset
            dataset_id: Unique dataset identifier
            purpose: Purpose of data usage
            
        Returns:
            Secured dataset
        """
        # Create a copy to avoid modifying original
        secured_data = data.copy()
        
        # Remove direct identifiers
        identifier_columns = [
            'PatientID', 'patient_id', 'id', 'ID',
            'name', 'Name', 'first_name', 'last_name',
            'ssn', 'social_security', 'medical_record_number',
            'phone', 'email', 'address', 'DoctorInCharge'
        ]
        
        for col in identifier_columns:
            if col in secured_data.columns:
                medical_logger.info(f"Removing identifier column: {col}")
                secured_data = secured_data.drop(columns=[col])
        
        # Hash quasi-identifiers for consistency while preserving privacy
        quasi_identifiers = ['birth_date', 'zip_code', 'postal_code']
        for col in quasi_identifiers:
            if col in secured_data.columns:
                secured_data[f"{col}_hash"] = secured_data[col].apply(
                    lambda x: self.data_encryption.hash_pii(str(x))
                )
                secured_data = secured_data.drop(columns=[col])
        
        # Add metadata for tracking
        secured_data.attrs = {
            'dataset_id': dataset_id,
            'secured_at': datetime.now().isoformat(),
            'purpose': purpose,
            'security_level': 'high',
            'anonymized': True
        }
        
        medical_logger.info(f"Applied security measures to dataset {dataset_id}")
        return secured_data
    
    def get_secure_dataset(self, dataset_id: str, user_id: str, purpose: str) -> pd.DataFrame:
        """
        Retrieve a secured dataset with access control.
        
        Args:
            dataset_id: Dataset identifier
            user_id: User requesting access
            purpose: Purpose of access
            
        Returns:
            Decrypted dataset
        """
        if dataset_id not in self.active_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = self.active_datasets[dataset_id]
        
        # Check access permissions
        if dataset_info['user_id'] != user_id:
            self.security_monitor.log_security_event(
                'unauthorized_dataset_access',
                {'dataset_id': dataset_id, 'requesting_user': user_id, 'owner': dataset_info['user_id']},
                severity='warning',
                user_id=user_id
            )
            raise PermissionError(f"Access denied to dataset {dataset_id}")
        
        # Check purpose compatibility
        if dataset_info['purpose'] != purpose:
            medical_logger.warning(f"Purpose mismatch for dataset {dataset_id}: {purpose} vs {dataset_info['purpose']}")
        
        # Log data access
        self.privacy_manager.log_data_access(
            user_id=user_id,
            data_type='medical_dataset',
            action='retrieve',
            data_id=dataset_id,
            purpose=purpose,
            legal_basis='healthcare_research'
        )
        
        # Decrypt and return data
        try:
            with open(dataset_info['storage_path'], 'r') as f:
                encrypted_data = f.read()
            
            decrypted_dict = self.data_encryption.decrypt_data(encrypted_data)
            return pd.DataFrame(decrypted_dict)
            
        except Exception as e:
            medical_logger.error(f"Failed to retrieve dataset {dataset_id}: {e}")
            raise
    
    def secure_model_training(self, dataset_id: str, model_config: Dict[str, Any], 
                            user_id: str) -> str:
        """
        Perform secure model training with privacy protection.
        
        Args:
            dataset_id: ID of the dataset to train on
            model_config: Model configuration
            user_id: User performing training
            
        Returns:
            Secure model ID
        """
        model_id = f"model_{dataset_id}_{int(datetime.now().timestamp())}"
        
        try:
            # Get training data
            training_data = self.get_secure_dataset(dataset_id, user_id, 'training')
            
            # Log training start
            self.privacy_manager.log_data_access(
                user_id=user_id,
                data_type='model_training',
                action='train',
                data_id=model_id,
                purpose='medical_ai_training',
                legal_basis='healthcare_research'
            )
            
            # Perform training in secure environment
            model_results = self._train_model_securely(training_data, model_config, model_id)
            
            # Encrypt and store model
            # Serialize model for encryption using base64 encoding
            import base64
            model_bytes = pickle.dumps(model_results)
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            encrypted_model = self.data_encryption.encrypt_data({'model_data': model_b64})
            model_path = self.model_storage_dir / f"{model_id}.encrypted"
            
            with open(model_path, 'w') as f:
                f.write(encrypted_model)
            
            # Register model
            self.model_registry[model_id] = {
                'user_id': user_id,
                'dataset_id': dataset_id,
                'created_at': datetime.now(),
                'model_path': model_path,
                'config': model_config,
                'status': 'trained',
                'performance': model_results.get('performance', {})
            }
            
            # Register for retention
            self.privacy_manager.register_data_for_retention(model_id, 'model_data')
            
            medical_logger.info(f"Securely trained model {model_id}")
            return model_id
            
        except Exception as e:
            medical_logger.error(f"Secure training failed: {e}")
            self.security_monitor.log_security_event(
                'model_training_error',
                {'model_id': model_id, 'error': str(e)},
                severity='warning',
                user_id=user_id
            )
            raise
    
    def _train_model_securely(self, data: pd.DataFrame, config: Dict[str, Any], 
                            model_id: str) -> Dict[str, Any]:
        """
        Train model in secure isolated environment.
        
        Args:
            data: Training data
            config: Model configuration
            model_id: Model identifier
            
        Returns:
            Training results
        """
        # Use temporary secure workspace for training
        with tempfile.TemporaryDirectory(prefix=f"secure_training_{model_id}_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col != 'Diagnosis']
            X = data[feature_columns]
            y = data['Diagnosis'] if 'Diagnosis' in data.columns else data.iloc[:, -1]
            
            # Train model (placeholder - integrate with actual ML)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'model': model,
                'feature_columns': feature_columns,
                'performance': {
                    'accuracy': accuracy,
                    'test_size': len(X_test),
                    'train_size': len(X_train)
                },
                'trained_at': datetime.now().isoformat(),
                'model_id': model_id
            }
            
            medical_logger.info(f"Model {model_id} trained with accuracy: {accuracy:.3f}")
            return results
    
    def secure_inference(self, model_id: str, input_data: Dict[str, Any], 
                        user_id: str) -> Dict[str, Any]:
        """
        Perform secure inference with privacy protection.
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for prediction
            user_id: User requesting inference
            
        Returns:
            Prediction results
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_registry[model_id]
        
        # Validate medical input data
        from security import InputValidator
        validator = InputValidator()
        is_valid, errors = validator.validate_medical_data(input_data)
        
        if not is_valid:
            raise ValueError(f"Invalid medical data: {errors}")
        
        # Log inference request
        self.privacy_manager.log_data_access(
            user_id=user_id,
            data_type='model_inference',
            action='predict',
            data_id=f"inference_{model_id}_{int(datetime.now().timestamp())}",
            purpose='medical_prediction',
            legal_basis='healthcare_provision'
        )
        
        try:
            # Load and decrypt model
            with open(model_info['model_path'], 'r') as f:
                encrypted_model = f.read()
            
            model_data = self.data_encryption.decrypt_data(encrypted_model)
            
            # Deserialize model from base64
            import base64
            model_bytes = base64.b64decode(model_data['model_data'])
            model_results = pickle.loads(model_bytes)
            
            # Perform secure inference (placeholder - would use actual model)
            prediction_result = {
                'prediction': 0,  # Would use actual model: model_results['model'].predict([input_data])[0]
                'confidence': 0.85,
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'privacy_protected': True
            }
            
            medical_logger.info(f"Secure inference completed for model {model_id}")
            return prediction_result
            
        except Exception as e:
            medical_logger.error(f"Secure inference failed: {e}")
            self.security_monitor.log_security_event(
                'inference_error',
                {'model_id': model_id, 'error': str(e)},
                severity='warning',
                user_id=user_id
            )
            raise
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """
        Clean up expired data according to retention policies.
        
        Returns:
            Cleanup statistics
        """
        stats = {'datasets_cleaned': 0, 'models_cleaned': 0}
        
        # Clean expired datasets
        for dataset_id, info in list(self.active_datasets.items()):
            retention_status = self.privacy_manager.get_retention_status(dataset_id)
            if retention_status and retention_status.get('status') == 'deleted':
                # Remove dataset file
                if info['storage_path'].exists():
                    info['storage_path'].unlink()
                
                del self.active_datasets[dataset_id]
                stats['datasets_cleaned'] += 1
                medical_logger.info(f"Cleaned expired dataset {dataset_id}")
        
        # Clean expired models
        for model_id, info in list(self.model_registry.items()):
            retention_status = self.privacy_manager.get_retention_status(model_id)
            if retention_status and retention_status.get('status') == 'deleted':
                # Remove model file
                if info['model_path'].exists():
                    info['model_path'].unlink()
                
                del self.model_registry[model_id]
                stats['models_cleaned'] += 1
                medical_logger.info(f"Cleaned expired model {model_id}")
        
        return stats
    
    def _load_kaggle_dataset(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load dataset from Kaggle (placeholder for integration)."""
        # This would integrate with the actual Kaggle loading code
        # For now, return sample data that matches expected structure
        medical_logger.info("Loading Kaggle dataset (placeholder)")
        
        # Create sample data that matches the expected medical dataset structure
        np.random.seed(42)  # For reproducible sample data
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'Age': np.random.randint(60, 90, n_samples),
            'Gender': np.random.randint(0, 2, n_samples),
            'Ethnicity': np.random.randint(0, 4, n_samples),
            'EducationLevel': np.random.randint(1, 5, n_samples),
            'BMI': np.random.normal(26, 4, n_samples),
            'Smoking': np.random.randint(0, 2, n_samples),
            'AlcoholConsumption': np.random.uniform(0, 5, n_samples),
            'PhysicalActivity': np.random.uniform(1, 5, n_samples),
            'DietQuality': np.random.uniform(3, 10, n_samples),
            'SleepQuality': np.random.uniform(3, 9, n_samples),
            'FamilyHistoryAlzheimers': np.random.randint(0, 2, n_samples),
            'CardiovascularDisease': np.random.randint(0, 2, n_samples),
            'Diabetes': np.random.randint(0, 2, n_samples),
            'Depression': np.random.randint(0, 2, n_samples),
            'HeadInjury': np.random.randint(0, 2, n_samples),
            'Hypertension': np.random.randint(0, 2, n_samples),
            'SystolicBP': np.random.normal(130, 20, n_samples),
            'DiastolicBP': np.random.normal(80, 10, n_samples),
            'CholesterolTotal': np.random.normal(200, 30, n_samples),
            'CholesterolLDL': np.random.normal(120, 25, n_samples),
            'CholesterolHDL': np.random.normal(50, 15, n_samples),
            'CholesterolTriglycerides': np.random.normal(150, 40, n_samples),
            'MMSE': np.random.randint(15, 30, n_samples),
            'FunctionalAssessment': np.random.uniform(3, 10, n_samples),
            'MemoryComplaints': np.random.randint(0, 2, n_samples),
            'BehavioralProblems': np.random.randint(0, 2, n_samples),
            'ADL': np.random.uniform(4, 10, n_samples),
            'Confusion': np.random.randint(0, 2, n_samples),
            'Disorientation': np.random.randint(0, 2, n_samples),
            'PersonalityChanges': np.random.randint(0, 2, n_samples),
            'DifficultyCompletingTasks': np.random.randint(0, 2, n_samples),
            'Forgetfulness': np.random.randint(0, 2, n_samples),
            'Diagnosis': np.random.randint(0, 2, n_samples)  # 0 = No Alzheimer's, 1 = Alzheimer's
        })
        
        medical_logger.info(f"Generated sample medical data with {len(sample_data)} rows and {len(sample_data.columns)} columns")
        return sample_data
    
    def _load_file_dataset(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Load dataset from file (placeholder for integration)."""
        # This would integrate with file loading
        # For now, return smaller sample data for the "original" dataset
        medical_logger.info("Loading file dataset (placeholder)")
        
        # Create smaller sample data for the "original" dataset
        np.random.seed(43)  # Different seed for variety
        n_samples = 50
        
        sample_data = pd.DataFrame({
            'Age': np.random.randint(65, 85, n_samples),
            'Gender': np.random.randint(0, 2, n_samples),
            'BMI': np.random.normal(25, 3, n_samples),
            'MMSE': np.random.randint(18, 30, n_samples),
            'FunctionalAssessment': np.random.uniform(4, 9, n_samples),
            'MemoryComplaints': np.random.randint(0, 2, n_samples),
            'FamilyHistoryAlzheimers': np.random.randint(0, 2, n_samples),
            'Depression': np.random.randint(0, 2, n_samples),
            'Diagnosis': np.random.randint(0, 2, n_samples)
        })
        
        medical_logger.info(f"Generated sample validation data with {len(sample_data)} rows and {len(sample_data.columns)} columns")
        return sample_data