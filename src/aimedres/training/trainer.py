"""
Secure Trainer for DuetMind Adaptive

High-level training interface with security and safety features.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from .pipeline import TrainingPipeline, TrainingResult
from ..core.neural_network import AdaptiveNeuralNetwork
from ..core.agent import DuetMindAgent
from ..core.config import DuetMindConfig

logger = logging.getLogger(__name__)

class SecureTrainer:
    """
    High-level secure trainer interface
    """
    
    def __init__(self, config: Optional[DuetMindConfig] = None):
        """Initialize trainer with configuration"""
        self.config = config or DuetMindConfig()
        self.pipeline = TrainingPipeline(
            enable_safety_monitoring=True,
            max_training_time=3600.0,  # 1 hour
            checkpoint_interval=10
        )
        
        logger.info("Secure trainer initialized")
    
    def train_network_on_synthetic_data(self, 
                                      input_size: int = 32,
                                      output_size: int = 2,
                                      num_samples: int = 1000,
                                      epochs: int = 10) -> TrainingResult:
        """
        Train network on synthetic data for demonstration
        
        Args:
            input_size: Neural network input size
            output_size: Neural network output size  
            num_samples: Number of synthetic samples
            epochs: Training epochs
            
        Returns:
            Training result
        """
        logger.info(f"Training network on synthetic data: {num_samples} samples, {epochs} epochs")
        
        # Create network
        network = AdaptiveNeuralNetwork(
            input_size=input_size,
            hidden_layers=[64, 32, 16],
            output_size=output_size,
            enable_safety_monitoring=False  # Pipeline handles safety
        )
        
        # Generate synthetic data
        X_train = np.random.randn(num_samples, input_size)
        y_train = np.random.randint(0, output_size, num_samples)
        
        # Split validation data
        split_idx = int(0.8 * num_samples)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # Train network
        result = self.pipeline.train_network(
            network=network,
            training_data=X_train,
            training_labels=y_train,
            validation_data=X_val,
            validation_labels=y_val,
            epochs=epochs,
            batch_size=32
        )
        
        return result
    
    def train_agent_scenarios(self, 
                            agent_name: str = "TrainingAgent",
                            num_scenarios: int = 100,
                            epochs: int = 20) -> TrainingResult:
        """
        Train agent on cognitive scenarios
        
        Args:
            agent_name: Name for the training agent
            num_scenarios: Number of training scenarios
            epochs: Training epochs
            
        Returns:
            Training result
        """
        logger.info(f"Training agent on {num_scenarios} scenarios for {epochs} epochs")
        
        # Create agent
        agent = DuetMindAgent(
            name=agent_name,
            neural_config={
                'input_size': 32,
                'hidden_layers': [64, 32, 16],
                'output_size': 2
            },
            enable_safety_monitoring=False  # Pipeline handles safety
        )
        
        # Generate training scenarios
        scenarios = []
        for i in range(num_scenarios):
            scenario = {
                'inputs': np.random.randn(32).tolist(),
                'context': {'scenario_id': i, 'type': 'synthetic'},
                'expected_response': f"Response to scenario {i}"
            }
            scenarios.append(scenario)
        
        # Train agent
        result = self.pipeline.train_agent(
            agent=agent,
            training_scenarios=scenarios,
            epochs=epochs
        )
        
        # Cleanup
        agent.shutdown()
        
        return result
    
    def shutdown(self):
        """Shutdown trainer"""
        logger.info("Shutting down secure trainer")
        self.pipeline.shutdown()
        logger.info("Secure trainer shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()