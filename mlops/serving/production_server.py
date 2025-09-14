#!/usr/bin/env python3
"""
Production Model Serving Server with A/B Testing and Monitoring.
High-performance Flask server for model inference with built-in monitoring.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path
import joblib
import pickle
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import mlflow
import mlflow.sklearn
from threading import Lock
import hashlib
import time

from ..monitoring.production_monitor import ProductionMonitor, create_production_monitor
from ..monitoring.ab_testing import ABTestingManager, ABTestConfig
from .model_loader import ModelLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionServer:
    """Production model serving server with A/B testing and monitoring."""
    
    def __init__(self, 
                 model_name: str = "alzheimer_classifier",
                 mlflow_tracking_uri: str = "sqlite:///mlflow.db",
                 enable_monitoring: bool = True,
                 enable_ab_testing: bool = True):
        """Initialize production server."""
        self.model_name = model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.enable_monitoring = enable_monitoring
        self.enable_ab_testing = enable_ab_testing
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Model management
        self.model_loader = ModelLoader(mlflow_tracking_uri)
        self.models = {}
        self.current_model_version = None
        self.model_lock = Lock()
        
        # Monitoring
        self.monitor = None
        if enable_monitoring:
            self.monitor = create_production_monitor(model_name)
        
        # A/B Testing
        self.ab_manager = None
        if enable_ab_testing:
            self.ab_manager = ABTestingManager(mlflow_tracking_uri)
        
        # Server stats
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"Production server initialized for {model_name}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model_name': self.model_name,
                'current_model_version': self.current_model_version,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'request_count': self.request_count,
                'error_count': self.error_count,
                'models_loaded': list(self.models.keys())
            })
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available models."""
            return jsonify({
                'models': list(self.models.keys()),
                'current_version': self.current_model_version
            })
        
        @self.app.route('/models/<version>/load', methods=['POST'])
        def load_model(version):
            """Load a specific model version."""
            try:
                success = self.load_model_version(version)
                if success:
                    return jsonify({'status': 'success', 'version': version})
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500
            except Exception as e:
                logger.error(f"Error loading model {version}: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint."""
            start_time = time.time()
            self.request_count += 1
            
            try:
                # Parse request
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                # Extract features and user info
                features = data.get('features', {})
                user_id = data.get('user_id', 'anonymous')
                experiment_name = data.get('experiment_name')
                
                if not features:
                    return jsonify({'error': 'No features provided'}), 400
                
                # Make prediction
                result = self._make_prediction(features, user_id, experiment_name)
                
                # Add timing info
                result['latency_ms'] = (time.time() - start_time) * 1000
                result['timestamp'] = datetime.now().isoformat()
                
                return jsonify(result)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/predict/batch', methods=['POST'])
        def predict_batch():
            """Batch prediction endpoint."""
            start_time = time.time()
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400
                
                batch_features = data.get('batch_features', [])
                user_ids = data.get('user_ids', [])
                experiment_name = data.get('experiment_name')
                
                if not batch_features:
                    return jsonify({'error': 'No batch features provided'}), 400
                
                # Ensure user_ids has same length as batch_features
                if len(user_ids) != len(batch_features):
                    user_ids = [f'user_{i}' for i in range(len(batch_features))]
                
                # Make batch predictions
                results = []
                for features, user_id in zip(batch_features, user_ids):
                    try:
                        result = self._make_prediction(features, user_id, experiment_name)
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e), 'user_id': user_id})
                
                return jsonify({
                    'results': results,
                    'batch_size': len(results),
                    'total_latency_ms': (time.time() - start_time) * 1000,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Batch prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/feedback', methods=['POST'])
        def feedback():
            """Endpoint for prediction feedback (true labels)."""
            try:
                data = request.get_json()
                user_id = data.get('user_id')
                true_label = data.get('true_label')
                experiment_name = data.get('experiment_name')
                
                if not user_id or true_label is None:
                    return jsonify({'error': 'user_id and true_label required'}), 400
                
                # Update A/B testing results if applicable
                if self.ab_manager and experiment_name:
                    self.ab_manager.update_prediction_outcomes(
                        experiment_name, [(user_id, true_label)]
                    )
                
                return jsonify({'status': 'success', 'message': 'Feedback recorded'})
                
            except Exception as e:
                logger.error(f"Feedback error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/monitoring/status', methods=['GET'])
        def monitoring_status():
            """Get monitoring status."""
            if not self.monitor:
                return jsonify({'error': 'Monitoring not enabled'}), 404
            
            try:
                hours = request.args.get('hours', 24, type=int)
                summary = self.monitor.get_monitoring_summary(hours=hours)
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Monitoring status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/experiments', methods=['GET'])
        def list_experiments():
            """List A/B testing experiments."""
            if not self.ab_manager:
                return jsonify({'error': 'A/B testing not enabled'}), 404
            
            try:
                experiments = self.ab_manager.list_experiments()
                return jsonify({'experiments': experiments})
            except Exception as e:
                logger.error(f"List experiments error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/experiments/<experiment_name>/status', methods=['GET'])
        def experiment_status(experiment_name):
            """Get experiment status."""
            if not self.ab_manager:
                return jsonify({'error': 'A/B testing not enabled'}), 404
            
            try:
                status = self.ab_manager.get_experiment_status(experiment_name)
                return jsonify(status)
            except Exception as e:
                logger.error(f"Experiment status error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/experiments/<experiment_name>/results', methods=['GET'])
        def experiment_results(experiment_name):
            """Get experiment results."""
            if not self.ab_manager:
                return jsonify({'error': 'A/B testing not enabled'}), 404
            
            try:
                results = self.ab_manager.analyze_experiment(experiment_name)
                return jsonify(results.__dict__)
            except Exception as e:
                logger.error(f"Experiment results error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _make_prediction(self, features: Dict[str, Any], 
                        user_id: str = 'anonymous',
                        experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Make a prediction with monitoring and A/B testing."""
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # A/B testing prediction
        if self.ab_manager and experiment_name and experiment_name in self.ab_manager.active_experiments:
            result = self.ab_manager.make_prediction(
                experiment_name, user_id, features, self.models
            )
            prediction = result['prediction']
            model_version = result['model_version']
            variant = result['variant']
            latency_ms = result['latency_ms']
        else:
            # Regular prediction with current model
            with self.model_lock:
                if not self.models or not self.current_model_version:
                    raise ValueError("No model loaded")
                
                model = self.models[self.current_model_version]
                model_version = self.current_model_version
                variant = "production"
                
                # Make prediction
                start_time = time.time()
                feature_array = self._prepare_features(features_df)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_array)[0]
                    prediction = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                else:
                    prediction = int(model.predict(feature_array)[0])
                    confidence = 0.8  # Default confidence
                
                latency_ms = (time.time() - start_time) * 1000
        
        # Log to monitoring system
        if self.monitor:
            try:
                # For monitoring, we need prediction as array
                predictions_array = np.array([prediction])
                
                self.monitor.log_prediction_batch(
                    features=features_df,
                    predictions=predictions_array,
                    model_version=model_version,
                    latency_ms=latency_ms
                )
            except Exception as e:
                logger.warning(f"Monitoring log error: {e}")
        
        return {
            'prediction': prediction,
            'model_version': model_version,
            'variant': variant,
            'user_id': user_id,
            'confidence': locals().get('confidence', 0.8)
        }
    
    def _prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model prediction."""
        # This is a simplified feature preparation
        # In practice, you'd need proper preprocessing pipeline
        
        # Define expected feature order (should match training)
        expected_features = [
            'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
        ]
        
        # Create feature array with default values
        feature_array = []
        for feature in expected_features:
            if feature in features_df.columns:
                value = features_df[feature].iloc[0]
                # Handle different data types
                if pd.isna(value):
                    value = 0.0
                elif isinstance(value, str):
                    # Simple categorical encoding
                    value = hash(value) % 100 / 100.0
                else:
                    value = float(value)
                feature_array.append(value)
            else:
                # Default value for missing features
                feature_array.append(0.0)
        
        return np.array(feature_array).reshape(1, -1)
    
    def load_model_version(self, version: str) -> bool:
        """Load a specific model version."""
        try:
            with self.model_lock:
                logger.info(f"Loading model version: {version}")
                
                # Try to load from MLflow first
                model = self.model_loader.load_model_version(self.model_name, version)
                
                if model is not None:
                    self.models[version] = model
                    self.current_model_version = version
                    logger.info(f"Successfully loaded model version: {version}")
                    return True
                else:
                    logger.error(f"Failed to load model version: {version}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading model {version}: {e}")
            return False
    
    def load_latest_model(self) -> bool:
        """Load the latest model version."""
        try:
            latest_version = self.model_loader.get_latest_model_version(self.model_name)
            if latest_version:
                return self.load_model_version(latest_version)
            else:
                logger.warning("No model versions found")
                return False
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return False
    
    def setup_baseline_monitoring(self, baseline_data_path: str):
        """Setup baseline data for monitoring."""
        if not self.monitor:
            logger.warning("Monitoring not enabled")
            return
        
        try:
            # Load baseline data
            if baseline_data_path.endswith('.csv'):
                baseline_df = pd.read_csv(baseline_data_path)
            elif baseline_data_path.endswith('.parquet'):
                baseline_df = pd.read_parquet(baseline_data_path)
            else:
                logger.error("Unsupported baseline data format")
                return
            
            # Separate features and labels (assuming last column is label)
            baseline_features = baseline_df.iloc[:, :-1]
            baseline_labels = baseline_df.iloc[:, -1]
            
            self.monitor.set_baseline_metrics(baseline_features, baseline_labels)
            logger.info(f"Baseline monitoring setup with {len(baseline_df)} samples")
            
        except Exception as e:
            logger.error(f"Error setting up baseline monitoring: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitor:
            self.monitor.start_monitoring()
            logger.info("Started continuous monitoring")
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the production server."""
        logger.info(f"Starting production server on {host}:{port}")
        
        # Load initial model
        self.load_latest_model()
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self.start_monitoring()
        
        # Run Flask app
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def create_production_server(model_name: str = "alzheimer_classifier",
                           **kwargs) -> ProductionServer:
    """Factory function to create a production server."""
    return ProductionServer(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Model Server')
    parser.add_argument('--model-name', default='alzheimer_classifier', help='Model name')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    parser.add_argument('--no-ab-testing', action='store_true', help='Disable A/B testing')
    parser.add_argument('--baseline-data', help='Path to baseline data for monitoring')
    
    args = parser.parse_args()
    
    # Create server
    server = create_production_server(
        model_name=args.model_name,
        enable_monitoring=not args.no_monitoring,
        enable_ab_testing=not args.no_ab_testing
    )
    
    # Setup baseline monitoring if provided
    if args.baseline_data:
        server.setup_baseline_monitoring(args.baseline_data)
    
    # Run server
    server.run(host=args.host, port=args.port, debug=args.debug)