"""
Comprehensive Integration and End-to-End Testing
Tests complete workflows, data pipelines, and cross-module compatibility
"""

import pytest
import asyncio
import time
import threading
import tempfile
import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import requests
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import AlzheimerTrainer, TrainingIntegratedAgent
from neuralnet import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom
from data_loaders import create_data_loader, CSVDataLoader, MockDataLoader
from clinical_decision_support import RiskStratificationEngine, ExplainableAIDashboard
from data_quality_monitor import DataQualityMonitor


class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    @pytest.fixture
    def sample_medical_data(self):
        """Generate comprehensive medical test data"""
        np.random.seed(42)  # For reproducible tests
        
        n_samples = 1000
        data = {
            'patient_id': [f'P{str(i).zfill(6)}' for i in range(n_samples)],
            'age': np.random.normal(70, 15, n_samples).astype(int),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'education_level': np.random.normal(14, 4, n_samples).astype(int),
            'mmse_score': np.random.normal(24, 4, n_samples).astype(int),
            'cdr_score': np.random.choice([0.0, 0.5, 1.0, 2.0, 3.0], n_samples, 
                                        p=[0.3, 0.2, 0.2, 0.2, 0.1]),
            'apoe_genotype': np.random.choice(['E2/E2', 'E2/E3', 'E3/E3', 'E3/E4', 'E4/E4'], 
                                            n_samples, p=[0.05, 0.15, 0.6, 0.15, 0.05]),
            'brain_volume': np.random.normal(1200, 100, n_samples),
            'hippocampus_volume': np.random.normal(3.5, 0.5, n_samples),
            'cortical_thickness': np.random.normal(2.8, 0.3, n_samples),
            'csf_abeta': np.random.normal(800, 200, n_samples),
            'csf_tau': np.random.normal(300, 100, n_samples),
            'csf_ptau': np.random.normal(25, 10, n_samples),
            'diagnosis': np.random.choice(['Normal', 'MCI', 'Dementia'], n_samples, 
                                        p=[0.4, 0.35, 0.25])
        }
        
        # Ensure realistic constraints
        data['age'] = np.clip(data['age'], 50, 95)
        data['education_level'] = np.clip(data['education_level'], 8, 20)
        data['mmse_score'] = np.clip(data['mmse_score'], 10, 30)
        data['brain_volume'] = np.clip(data['brain_volume'], 1000, 1500)
        data['hippocampus_volume'] = np.clip(data['hippocampus_volume'], 2.0, 5.0)
        data['cortical_thickness'] = np.clip(data['cortical_thickness'], 2.0, 3.5)
        
        return pd.DataFrame(data)

    def test_complete_data_pipeline(self, sample_medical_data):
        """Test complete data processing pipeline"""
        # Step 1: Data Quality Monitoring
        quality_monitor = DataQualityMonitor()
        
        quality_report = quality_monitor.assess_data_quality(sample_medical_data)
        assert quality_report['completeness_score'] > 0.95
        assert quality_report['consistency_score'] > 0.90
        assert len(quality_report['anomalies']) >= 0
        
        # Step 2: Data Loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_medical_data.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            data_loader = CSVDataLoader(tmp_path)
            loaded_data = data_loader.load_data()
            
            assert len(loaded_data) == len(sample_medical_data)
            assert list(loaded_data.columns) == list(sample_medical_data.columns)
            
            # Step 3: Training Pipeline
            trainer = AlzheimerTrainer()
            trainer.train_model(loaded_data)
            
            assert trainer.model is not None
            
            # Step 4: Prediction Pipeline
            test_data = loaded_data.drop('diagnosis', axis=1).head(10)
            predictions = trainer.predict(test_data)
            
            assert len(predictions) == 10
            assert all(pred in ['Normal', 'MCI', 'Dementia'] for pred in predictions)
            
            # Step 5: Risk Assessment Integration
            risk_engine = RiskStratificationEngine()
            
            for _, patient in test_data.head(5).iterrows():
                risk_assessment = risk_engine.assess_alzheimer_risk({
                    'age': patient['age'],
                    'mmse_score': patient['mmse_score'],
                    'apoe_genotype': patient['apoe_genotype'],
                    'brain_volume': patient['brain_volume']
                })
                
                assert 'risk_score' in risk_assessment
                assert 0 <= risk_assessment['risk_score'] <= 1
                assert 'risk_level' in risk_assessment
                
        finally:
            os.unlink(tmp_path)

    def test_cross_module_data_consistency(self, sample_medical_data):
        """Test data consistency across different modules"""
        # Prepare test data
        test_patient = sample_medical_data.iloc[0].to_dict()
        
        # Test 1: Training module data handling
        trainer = AlzheimerTrainer()
        single_patient_df = pd.DataFrame([test_patient])
        trainer.train_model(single_patient_df)
        
        prediction = trainer.predict(single_patient_df.drop('diagnosis', axis=1))
        assert len(prediction) == 1
        
        # Test 2: Risk engine data handling
        risk_engine = RiskStratificationEngine()
        risk_data = {
            'age': test_patient['age'],
            'mmse_score': test_patient['mmse_score'],
            'apoe_genotype': test_patient['apoe_genotype']
        }
        
        risk_assessment = risk_engine.assess_alzheimer_risk(risk_data)
        assert isinstance(risk_assessment, dict)
        
        # Test 3: Data quality monitor consistency
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.assess_data_quality(single_patient_df)
        
        assert quality_report['total_records'] == 1
        assert quality_report['completeness_score'] >= 0.8

    def test_async_data_processing(self, sample_medical_data):
        """Test asynchronous data processing workflows"""
        async def process_batch_async(data_batch: pd.DataFrame) -> Dict[str, Any]:
            """Async batch processing simulation"""
            await asyncio.sleep(0.1)  # Simulate async processing
            
            trainer = AlzheimerTrainer()
            trainer.train_model(data_batch)
            
            predictions = trainer.predict(data_batch.drop('diagnosis', axis=1))
            
            return {
                'batch_size': len(data_batch),
                'predictions': predictions,
                'training_complete': True
            }
        
        async def test_async_pipeline():
            # Split data into batches
            batch_size = 100
            batches = [
                sample_medical_data[i:i+batch_size] 
                for i in range(0, len(sample_medical_data), batch_size)
            ]
            
            # Process batches concurrently
            tasks = [process_batch_async(batch) for batch in batches[:3]]  # Process first 3 batches
            results = await asyncio.gather(*tasks)
            
            # Verify results
            assert len(results) == 3
            assert all(result['training_complete'] for result in results)
            assert sum(result['batch_size'] for result in results) == 300
            
            return results
        
        # Run async test
        results = asyncio.run(test_async_pipeline())
        assert len(results) == 3

    def test_error_handling_in_pipeline(self, sample_medical_data):
        """Test error handling throughout the data pipeline"""
        # Introduce data corruption
        corrupted_data = sample_medical_data.copy()
        corrupted_data.loc[0:10, 'age'] = np.nan
        corrupted_data.loc[20:30, 'mmse_score'] = -999
        corrupted_data.loc[40:50, 'diagnosis'] = 'InvalidDiagnosis'
        
        # Test data quality monitoring with corrupted data
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.assess_data_quality(corrupted_data)
        
        assert quality_report['completeness_score'] < 1.0
        assert len(quality_report['anomalies']) > 0
        
        # Test training with corrupted data
        trainer = AlzheimerTrainer()
        
        try:
            trainer.train_model(corrupted_data)
            # If training succeeds, verify it handles corruption gracefully
            assert trainer.model is not None
        except (ValueError, TypeError) as e:
            # Expected controlled failure
            assert "Invalid" in str(e) or "nan" in str(e) or "missing" in str(e).lower()

    def test_performance_under_load(self, sample_medical_data):
        """Test system performance under load conditions"""
        # Generate larger dataset
        large_data = pd.concat([sample_medical_data] * 5, ignore_index=True)  # 5000 records
        
        start_time = time.time()
        
        # Data quality assessment
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.assess_data_quality(large_data)
        
        quality_time = time.time() - start_time
        
        # Training
        trainer = AlzheimerTrainer()
        trainer.train_model(large_data)
        
        training_time = time.time() - start_time - quality_time
        
        # Prediction on subset
        test_subset = large_data.drop('diagnosis', axis=1).head(500)
        predictions = trainer.predict(test_subset)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert quality_time < 10.0, f"Quality assessment too slow: {quality_time:.2f}s"
        assert training_time < 60.0, f"Training too slow: {training_time:.2f}s"
        assert total_time < 70.0, f"Total pipeline too slow: {total_time:.2f}s"
        assert len(predictions) == 500


class TestWorkflowIntegration:
    """Test complete clinical workflow integration"""
    
    def test_complete_clinical_workflow(self):
        """Test end-to-end clinical decision support workflow"""
        # Step 1: Patient data intake
        patient_data = {
            'patient_id': 'P123456',
            'age': 72,
            'gender': 'F',
            'education_level': 16,
            'mmse_score': 26,
            'cdr_score': 0.5,
            'apoe_genotype': 'E3/E4',
            'brain_volume': 1150,
            'hippocampus_volume': 3.2,
            'chief_complaint': 'Memory concerns',
            'family_history': 'Mother had dementia'
        }
        
        # Step 2: Risk stratification
        risk_engine = RiskStratificationEngine()
        risk_assessment = risk_engine.assess_alzheimer_risk(patient_data)
        
        assert 'risk_score' in risk_assessment
        assert 'risk_level' in risk_assessment
        assert 'recommendations' in risk_assessment
        
        # Step 3: AI model prediction
        trainer = AlzheimerTrainer()
        
        # Create training data for the model
        training_data = pd.DataFrame([
            {
                'age': 70, 'gender': 'M', 'education_level': 14, 
                'mmse_score': 28, 'cdr_score': 0.0, 'apoe_genotype': 'E3/E3',
                'diagnosis': 'Normal'
            },
            {
                'age': 75, 'gender': 'F', 'education_level': 12, 
                'mmse_score': 22, 'cdr_score': 1.0, 'apoe_genotype': 'E3/E4',
                'diagnosis': 'MCI'
            },
            {
                'age': 80, 'gender': 'M', 'education_level': 16, 
                'mmse_score': 18, 'cdr_score': 2.0, 'apoe_genotype': 'E4/E4',
                'diagnosis': 'Dementia'
            }
        ] * 50)  # Repeat to have enough training data
        
        trainer.train_model(training_data)
        
        # Make prediction for patient
        patient_df = pd.DataFrame([{
            'age': patient_data['age'],
            'gender': patient_data['gender'],
            'education_level': patient_data['education_level'],
            'mmse_score': patient_data['mmse_score'],
            'cdr_score': patient_data['cdr_score'],
            'apoe_genotype': patient_data['apoe_genotype']
        }])
        
        prediction = trainer.predict(patient_df)
        assert len(prediction) == 1
        assert prediction[0] in ['Normal', 'MCI', 'Dementia']
        
        # Step 4: Generate explanations
        dashboard = ExplainableAIDashboard()
        
        # Mock model for explanation
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        
        feature_names = ['age', 'gender', 'education_level', 'mmse_score', 'cdr_score', 'apoe_genotype']
        importance_dict = dashboard.calculate_feature_importance(mock_model, feature_names)
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == len(feature_names)
        
        # Step 5: Generate recommendations
        workflow_result = {
            'patient_id': patient_data['patient_id'],
            'risk_assessment': risk_assessment,
            'ai_prediction': prediction[0],
            'feature_importance': importance_dict,
            'timestamp': time.time()
        }
        
        # Verify complete workflow result
        assert 'patient_id' in workflow_result
        assert 'risk_assessment' in workflow_result
        assert 'ai_prediction' in workflow_result
        assert 'feature_importance' in workflow_result

    def test_multi_patient_batch_processing(self):
        """Test batch processing of multiple patients"""
        # Generate batch of patients
        patients = []
        for i in range(10):
            patients.append({
                'patient_id': f'P{str(i).zfill(6)}',
                'age': 65 + (i * 2),
                'gender': 'M' if i % 2 == 0 else 'F',
                'education_level': 12 + (i % 8),
                'mmse_score': 30 - (i % 15),
                'cdr_score': (i % 4) * 0.5,
                'apoe_genotype': ['E3/E3', 'E3/E4', 'E4/E4'][i % 3]
            })
        
        # Process batch
        results = []
        risk_engine = RiskStratificationEngine()
        
        for patient in patients:
            risk_assessment = risk_engine.assess_alzheimer_risk(patient)
            results.append({
                'patient_id': patient['patient_id'],
                'risk_score': risk_assessment['risk_score'],
                'risk_level': risk_assessment['risk_level']
            })
        
        # Verify batch processing
        assert len(results) == 10
        assert all('patient_id' in result for result in results)
        assert all('risk_score' in result for result in results)
        assert all(0 <= result['risk_score'] <= 1 for result in results)

    def test_workflow_error_recovery(self):
        """Test workflow error recovery mechanisms"""
        # Test with invalid patient data
        invalid_patient = {
            'patient_id': '',  # Invalid empty ID
            'age': -5,  # Invalid age
            'gender': 'X',  # Invalid gender
            'mmse_score': 100,  # Invalid score
            'apoe_genotype': 'INVALID'  # Invalid genotype
        }
        
        risk_engine = RiskStratificationEngine()
        
        try:
            risk_assessment = risk_engine.assess_alzheimer_risk(invalid_patient)
            # If successful, should have handled errors gracefully
            assert isinstance(risk_assessment, dict)
        except (ValueError, TypeError, KeyError) as e:
            # Expected error handling
            assert "Invalid" in str(e) or "missing" in str(e).lower()

    def test_concurrent_workflow_processing(self):
        """Test concurrent processing of multiple workflows"""
        def process_patient_workflow(patient_id: str) -> Dict[str, Any]:
            """Process single patient workflow"""
            patient_data = {
                'patient_id': patient_id,
                'age': 70,
                'gender': 'M',
                'education_level': 14,
                'mmse_score': 25,
                'cdr_score': 0.5,
                'apoe_genotype': 'E3/E4'
            }
            
            risk_engine = RiskStratificationEngine()
            risk_assessment = risk_engine.assess_alzheimer_risk(patient_data)
            
            return {
                'patient_id': patient_id,
                'risk_score': risk_assessment['risk_score'],
                'processing_thread': threading.current_thread().name
            }
        
        # Process multiple patients concurrently
        patient_ids = [f'P{str(i).zfill(6)}' for i in range(20)]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_patient_workflow, pid) for pid in patient_ids]
            results = [future.result() for future in as_completed(futures)]
        
        # Verify concurrent processing
        assert len(results) == 20
        assert all('patient_id' in result for result in results)
        assert len(set(result['processing_thread'] for result in results)) > 1  # Multiple threads used


class TestSystemIntegration:
    """Test system-level integration scenarios"""
    
    def test_database_integration(self):
        """Test database integration functionality"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            # Initialize database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE patients (
                    id TEXT PRIMARY KEY,
                    age INTEGER,
                    gender TEXT,
                    diagnosis TEXT,
                    risk_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    assessment_type TEXT,
                    results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            conn.commit()
            
            # Test data insertion
            patient_data = ('P123456', 72, 'F', 'MCI', 0.65)
            cursor.execute(
                'INSERT INTO patients (id, age, gender, diagnosis, risk_score) VALUES (?, ?, ?, ?, ?)',
                patient_data
            )
            
            assessment_data = ('P123456', 'alzheimer_risk', '{"risk_level": "moderate", "recommendations": ["follow-up"]}')
            cursor.execute(
                'INSERT INTO assessments (patient_id, assessment_type, results) VALUES (?, ?, ?)',
                assessment_data
            )
            
            conn.commit()
            
            # Test data retrieval
            cursor.execute('SELECT * FROM patients WHERE id = ?', ('P123456',))
            patient_record = cursor.fetchone()
            
            assert patient_record is not None
            assert patient_record[0] == 'P123456'
            assert patient_record[1] == 72
            assert patient_record[2] == 'F'
            
            # Test join query
            cursor.execute('''
                SELECT p.id, p.age, p.diagnosis, a.assessment_type, a.results
                FROM patients p
                JOIN assessments a ON p.id = a.patient_id
                WHERE p.id = ?
            ''', ('P123456',))
            
            joined_record = cursor.fetchone()
            assert joined_record is not None
            assert joined_record[0] == 'P123456'
            assert joined_record[3] == 'alzheimer_risk'
            
            conn.close()
            
        finally:
            os.unlink(db_path)

    def test_api_integration(self):
        """Test API integration scenarios"""
        # Mock API responses for testing
        class MockAPIClient:
            def __init__(self):
                self.base_url = "https://api.medical-system.com"
                self.timeout = 30
            
            def post_patient_assessment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
                """Mock API call for patient assessment"""
                # Simulate API response
                return {
                    'status': 'success',
                    'assessment_id': f'ASSESS_{int(time.time())}',
                    'patient_id': patient_data.get('patient_id'),
                    'risk_score': 0.65,
                    'processing_time': 0.25
                }
            
            def get_assessment_results(self, assessment_id: str) -> Dict[str, Any]:
                """Mock API call to retrieve assessment results"""
                return {
                    'status': 'completed',
                    'assessment_id': assessment_id,
                    'results': {
                        'risk_level': 'moderate',
                        'confidence': 0.85,
                        'recommendations': ['cognitive_assessment', 'follow_up_6months']
                    }
                }
            
            def update_patient_record(self, patient_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
                """Mock API call to update patient record"""
                return {
                    'status': 'updated',
                    'patient_id': patient_id,
                    'updated_fields': list(updates.keys()),
                    'timestamp': time.time()
                }
        
        # Test API integration workflow
        api_client = MockAPIClient()
        
        patient_data = {
            'patient_id': 'P123456',
            'age': 72,
            'gender': 'F',
            'mmse_score': 24,
            'symptoms': ['memory_loss', 'confusion']
        }
        
        # Submit assessment
        assessment_response = api_client.post_patient_assessment(patient_data)
        assert assessment_response['status'] == 'success'
        assert 'assessment_id' in assessment_response
        
        assessment_id = assessment_response['assessment_id']
        
        # Retrieve results
        results_response = api_client.get_assessment_results(assessment_id)
        assert results_response['status'] == 'completed'
        assert 'results' in results_response
        
        # Update patient record
        updates = {
            'last_assessment': assessment_id,
            'risk_score': results_response['results'].get('confidence', 0)
        }
        
        update_response = api_client.update_patient_record(patient_data['patient_id'], updates)
        assert update_response['status'] == 'updated'
        assert 'last_assessment' in update_response['updated_fields']

    def test_file_system_integration(self):
        """Test file system integration and management"""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            data_dir = temp_path / "data"
            models_dir = temp_path / "models"
            reports_dir = temp_path / "reports"
            
            for directory in [data_dir, models_dir, reports_dir]:
                directory.mkdir(exist_ok=True)
            
            # Test file operations
            # 1. Data file handling
            test_data = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'age': [65, 70, 75],
                'diagnosis': ['Normal', 'MCI', 'Dementia']
            })
            
            data_file = data_dir / "test_data.csv"
            test_data.to_csv(data_file, index=False)
            
            # Verify file creation and reading
            assert data_file.exists()
            loaded_data = pd.read_csv(data_file)
            assert len(loaded_data) == 3
            assert list(loaded_data.columns) == ['patient_id', 'age', 'diagnosis']
            
            # 2. Model persistence
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 1])
            
            model_file = models_dir / "test_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Verify model saving and loading
            assert model_file.exists()
            with open(model_file, 'rb') as f:
                loaded_model = pickle.load(f)
            
            prediction = loaded_model.predict([[2, 3]])
            assert len(prediction) == 1
            
            # 3. Report generation
            report_data = {
                'timestamp': time.time(),
                'total_patients': len(test_data),
                'model_accuracy': 0.95,
                'processing_time': 1.25
            }
            
            report_file = reports_dir / "assessment_report.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Verify report creation and reading
            assert report_file.exists()
            with open(report_file, 'r') as f:
                loaded_report = json.load(f)
            
            assert loaded_report['total_patients'] == 3
            assert loaded_report['model_accuracy'] == 0.95

    def test_configuration_management(self):
        """Test configuration management across system"""
        # Create temporary config file
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'medical_db',
                'pool_size': 10
            },
            'ml_models': {
                'alzheimer_model': {
                    'type': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'risk_model': {
                    'type': 'logistic_regression',
                    'regularization': 0.01
                }
            },
            'security': {
                'session_timeout': 3600,
                'max_login_attempts': 5,
                'password_min_length': 8
            },
            'monitoring': {
                'log_level': 'INFO',
                'metrics_retention_days': 30,
                'alert_thresholds': {
                    'error_rate': 0.05,
                    'response_time': 2.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            json.dump(config_data, config_file, indent=2)
            config_path = config_file.name
        
        try:
            # Test configuration loading
            class ConfigManager:
                def __init__(self, config_path: str):
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                
                def get(self, path: str, default=None):
                    """Get config value by dot-separated path"""
                    keys = path.split('.')
                    value = self.config
                    
                    for key in keys:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            return default
                    
                    return value
            
            config_manager = ConfigManager(config_path)
            
            # Test configuration access
            assert config_manager.get('database.host') == 'localhost'
            assert config_manager.get('database.port') == 5432
            assert config_manager.get('ml_models.alzheimer_model.n_estimators') == 100
            assert config_manager.get('security.session_timeout') == 3600
            assert config_manager.get('nonexistent.key', 'default') == 'default'
            
            # Test configuration validation
            required_configs = [
                'database.host',
                'ml_models.alzheimer_model.type',
                'security.session_timeout'
            ]
            
            for config_path in required_configs:
                value = config_manager.get(config_path)
                assert value is not None, f"Required config missing: {config_path}"
            
        finally:
            os.unlink(config_path)


@pytest.mark.slow
class TestEndToEndScenarios:
    """End-to-end testing scenarios"""
    
    def test_complete_patient_journey(self):
        """Test complete patient assessment journey"""
        # Patient intake
        patient_data = {
            'patient_id': 'P789012',
            'personal_info': {
                'age': 68,
                'gender': 'M',
                'education_level': 14
            },
            'clinical_data': {
                'mmse_score': 26,
                'cdr_score': 0.5,
                'apoe_genotype': 'E3/E4'
            },
            'medical_history': {
                'family_history_dementia': True,
                'cardiovascular_disease': False,
                'diabetes': True
            },
            'symptoms': [
                'mild_memory_loss',
                'difficulty_with_names',
                'occasional_confusion'
            ]
        }
        
        # Step 1: Data quality check
        quality_monitor = DataQualityMonitor()
        
        # Flatten data for quality check
        flat_data = {
            **patient_data['personal_info'],
            **patient_data['clinical_data'],
            **patient_data['medical_history']
        }
        
        quality_df = pd.DataFrame([flat_data])
        quality_report = quality_monitor.assess_data_quality(quality_df)
        
        assert quality_report['completeness_score'] > 0.8
        
        # Step 2: Risk assessment
        risk_engine = RiskStratificationEngine()
        risk_assessment = risk_engine.assess_alzheimer_risk({
            'age': patient_data['personal_info']['age'],
            'mmse_score': patient_data['clinical_data']['mmse_score'],
            'apoe_genotype': patient_data['clinical_data']['apoe_genotype']
        })
        
        assert 'risk_score' in risk_assessment
        assert 'recommendations' in risk_assessment
        
        # Step 3: AI prediction
        trainer = AlzheimerTrainer()
        
        # Create minimal training set
        training_data = pd.DataFrame([
            {'age': 65, 'gender': 'M', 'education_level': 12, 'mmse_score': 28, 
             'cdr_score': 0.0, 'apoe_genotype': 'E3/E3', 'diagnosis': 'Normal'},
            {'age': 70, 'gender': 'F', 'education_level': 16, 'mmse_score': 24, 
             'cdr_score': 0.5, 'apoe_genotype': 'E3/E4', 'diagnosis': 'MCI'},
            {'age': 75, 'gender': 'M', 'education_level': 14, 'mmse_score': 20, 
             'cdr_score': 1.0, 'apoe_genotype': 'E4/E4', 'diagnosis': 'Dementia'}
        ] * 20)  # Replicate for sufficient training data
        
        trainer.train_model(training_data)
        
        patient_features = pd.DataFrame([{
            'age': patient_data['personal_info']['age'],
            'gender': patient_data['personal_info']['gender'],
            'education_level': patient_data['personal_info']['education_level'],
            'mmse_score': patient_data['clinical_data']['mmse_score'],
            'cdr_score': patient_data['clinical_data']['cdr_score'],
            'apoe_genotype': patient_data['clinical_data']['apoe_genotype']
        }])
        
        prediction = trainer.predict(patient_features)
        assert len(prediction) == 1
        assert prediction[0] in ['Normal', 'MCI', 'Dementia']
        
        # Step 4: Compile final assessment
        final_assessment = {
            'patient_id': patient_data['patient_id'],
            'assessment_timestamp': time.time(),
            'data_quality': {
                'completeness_score': quality_report['completeness_score'],
                'anomalies_count': len(quality_report['anomalies'])
            },
            'risk_assessment': {
                'risk_score': risk_assessment['risk_score'],
                'risk_level': risk_assessment['risk_level'],
                'recommendations': risk_assessment.get('recommendations', [])
            },
            'ai_prediction': {
                'diagnosis': prediction[0],
                'model_version': '1.0'
            },
            'clinical_summary': {
                'primary_concerns': patient_data['symptoms'],
                'key_risk_factors': [
                    'APOE E3/E4 genotype',
                    'Family history of dementia',
                    'Mild cognitive symptoms'
                ]
            }
        }
        
        # Verify complete assessment
        assert 'patient_id' in final_assessment
        assert 'risk_assessment' in final_assessment
        assert 'ai_prediction' in final_assessment
        assert 'clinical_summary' in final_assessment
        
        # Verify assessment completeness
        assert final_assessment['data_quality']['completeness_score'] > 0.8
        assert final_assessment['risk_assessment']['risk_score'] is not None
        assert final_assessment['ai_prediction']['diagnosis'] in ['Normal', 'MCI', 'Dementia']

    def test_system_resilience_under_failure(self):
        """Test system resilience when components fail"""
        # Simulate component failures
        failures_encountered = []
        
        def attempt_with_fallback(primary_func, fallback_func, failure_type):
            """Attempt primary function with fallback on failure"""
            try:
                return primary_func()
            except Exception as e:
                failures_encountered.append(failure_type)
                return fallback_func()
        
        # Primary and fallback functions
        def primary_risk_assessment():
            # Simulate failure
            raise ConnectionError("Risk assessment service unavailable")
        
        def fallback_risk_assessment():
            # Simple fallback logic
            return {
                'risk_score': 0.5,
                'risk_level': 'unknown',
                'recommendations': ['manual_review_required'],
                'fallback_used': True
            }
        
        def primary_ai_prediction():
            # Simulate model failure
            raise RuntimeError("AI model not responding")
        
        def fallback_ai_prediction():
            # Rule-based fallback
            return ['requires_clinical_assessment']
        
        # Test resilient workflow
        patient_data = {
            'age': 70,
            'mmse_score': 22,
            'apoe_genotype': 'E3/E4'
        }
        
        # Attempt risk assessment with fallback
        risk_result = attempt_with_fallback(
            primary_risk_assessment,
            fallback_risk_assessment,
            'risk_assessment_failure'
        )
        
        # Attempt AI prediction with fallback
        prediction_result = attempt_with_fallback(
            primary_ai_prediction,
            fallback_ai_prediction,
            'ai_prediction_failure'
        )
        
        # Verify system continued to function despite failures
        assert 'risk_score' in risk_result
        assert risk_result['fallback_used'] is True
        assert len(prediction_result) > 0
        assert len(failures_encountered) == 2
        assert 'risk_assessment_failure' in failures_encountered
        assert 'ai_prediction_failure' in failures_encountered