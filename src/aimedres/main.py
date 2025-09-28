#!/usr/bin/env python3
"""
AiMedRes - Secure Main Entry Point

Production-ready main entry point with comprehensive security,
configuration management, and operational safety.
"""

import argparse
import sys
import logging
import time
import signal
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .core.config import DuetMindConfig
from .core.neural_network import AdaptiveNeuralNetwork
from .core.agent import DuetMindAgent
from .utils.safety import SafetyMonitor
from .security.validation import InputValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('aimedres.log')
    ]
)
logger = logging.getLogger("AiMedRes")

class AiMedResApplication:
    """
    Main AiMedRes application with security and safety features
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize application with configuration"""
        if isinstance(config_path, DuetMindConfig):
            # If config object passed directly
            self.config = config_path
        else:
            # If config path passed
            self.config = DuetMindConfig(config_path)
        self.safety_monitor = None
        self.agents: Dict[str, DuetMindAgent] = {}
        self.neural_networks: Dict[str, AdaptiveNeuralNetwork] = {}
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("AiMedRes application initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown()
    
    def start_safety_monitoring(self):
        """Start comprehensive safety monitoring"""
        self.safety_monitor = SafetyMonitor(
            max_error_rate=0.1,
            max_response_time=30.0,
            max_memory_mb=self.config.neural_network.batch_size * 10.0,  # Scale with batch size
            max_cpu_percent=80.0
        )
        self.safety_monitor.start()
        logger.info("Safety monitoring started")
    
    def create_neural_network(self, network_id: str, **kwargs) -> AdaptiveNeuralNetwork:
        """Create a new neural network with configuration"""
        config = {
            'input_size': kwargs.get('input_size', self.config.neural_network.input_size),
            'hidden_layers': kwargs.get('hidden_layers', self.config.neural_network.hidden_layers),
            'output_size': kwargs.get('output_size', self.config.neural_network.output_size),
            'learning_rate': kwargs.get('learning_rate', self.config.neural_network.learning_rate),
            'enable_safety_monitoring': True
        }
        
        network = AdaptiveNeuralNetwork(**config)
        self.neural_networks[network_id] = network
        
        logger.info(f"Created neural network: {network_id}")
        return network
    
    def create_agent(self, agent_id: Optional[str] = None, name: Optional[str] = None, **kwargs) -> DuetMindAgent:
        """Create a new AiMedRes agent"""
        neural_config = {
            'input_size': kwargs.get('input_size', self.config.neural_network.input_size),
            'hidden_layers': kwargs.get('hidden_layers', self.config.neural_network.hidden_layers),
            'output_size': kwargs.get('output_size', self.config.neural_network.output_size),
            'learning_rate': kwargs.get('learning_rate', self.config.neural_network.learning_rate)
        }
        
        agent = DuetMindAgent(
            agent_id=agent_id,
            name=name,
            neural_config=neural_config,
            enable_safety_monitoring=True,
            enable_input_validation=True
        )
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Created agent: {agent.name} ({agent.agent_id})")
        return agent
    
    def run_training_mode(self):
        """Run comprehensive training mode"""
        logger.info("=== Starting Training Mode ===")
        
        try:
            # Create training neural network
            training_network = self.create_neural_network("training_network")
            
            # Create synthetic training data
            import numpy as np
            X_train = np.random.randn(1000, self.config.neural_network.input_size)
            
            # Simulate training process
            training_results = []
            
            for epoch in range(min(10, self.config.neural_network.epochs)):  # Limited epochs for demo
                epoch_start = time.time()
                
                # Check safety before training step
                if self.safety_monitor and not self.safety_monitor.is_safe_to_operate():
                    logger.error("Training halted due to safety concerns")
                    break
                
                # Simulate training batch
                batch_results = []
                for i in range(0, len(X_train), self.config.neural_network.batch_size):
                    batch = X_train[i:i + self.config.neural_network.batch_size]
                    
                    for sample in batch:
                        try:
                            prediction = training_network.predict(sample)
                            batch_results.append(prediction)
                        except Exception as e:
                            logger.error(f"Training error on sample: {e}")
                
                epoch_time = time.time() - epoch_start
                
                result = {
                    'epoch': epoch + 1,
                    'samples_processed': len(batch_results),
                    'epoch_time': epoch_time,
                    'network_health': training_network.get_network_health()
                }
                
                training_results.append(result)
                logger.info(f"Epoch {epoch + 1} complete: {len(batch_results)} samples in {epoch_time:.2f}s")
            
            logger.info(f"Training complete: {len(training_results)} epochs")
            return {"status": "success", "epochs": training_results}
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_simulation_mode(self):
        """Run agent simulation mode"""
        logger.info("=== Starting Simulation Mode ===")
        
        try:
            # Create simulation agents
            agents = []
            for i in range(3):
                agent = self.create_agent(
                    name=f"SimAgent-{i+1}",
                    input_size=32,
                    hidden_layers=[64, 32, 16],
                    output_size=2
                )
                agents.append(agent)
            
            # Run simulation steps
            simulation_results = []
            
            for step in range(20):  # 20 simulation steps
                step_start = time.time()
                
                # Check safety
                if self.safety_monitor and not self.safety_monitor.is_safe_to_operate():
                    logger.error("Simulation halted due to safety concerns")
                    break
                
                step_results = []
                
                # Each agent thinks and potentially communicates
                for agent in agents:
                    try:
                        # Generate random input for thinking
                        import numpy as np
                        inputs = np.random.randn(32)
                        
                        # Agent thinks
                        thought = agent.think(inputs)
                        step_results.append({
                            'agent_id': agent.agent_id,
                            'agent_name': agent.name,
                            'thought': thought,
                            'status': agent.get_status()
                        })
                        
                        # Random communication
                        if step % 5 == 0 and len(agents) > 1:
                            other_agents = [a for a in agents if a.agent_id != agent.agent_id]
                            target = np.random.choice(other_agents)
                            
                            message = f"Hello from {agent.name} at step {step}"
                            comm_result = agent.communicate(message, target.agent_id)
                            
                            if comm_result.get('success'):
                                logger.info(f"{agent.name} communicated with {target.name}")
                        
                    except Exception as e:
                        logger.error(f"Agent {agent.name} simulation error: {e}")
                
                step_time = time.time() - step_start
                
                simulation_results.append({
                    'step': step + 1,
                    'step_time': step_time,
                    'agents': step_results
                })
                
                logger.info(f"Simulation step {step + 1} complete: {len(step_results)} agents in {step_time:.3f}s")
                
                # Occasional sleep cycles
                if step > 0 and step % 10 == 0:
                    logger.info("Initiating agent sleep cycles...")
                    for agent in agents:
                        agent.sleep(1.0)  # 1 hour sleep
            
            logger.info(f"Simulation complete: {len(simulation_results)} steps")
            return {"status": "success", "steps": simulation_results}
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_comprehensive_mode(self):
        """Run comprehensive training + simulation"""
        logger.info("=== Starting Comprehensive Mode ===")
        
        # Run training first
        training_results = self.run_training_mode()
        
        if training_results.get("status") == "success":
            # Run simulation
            simulation_results = self.run_simulation_mode()
            
            return {
                "status": "success",
                "training": training_results,
                "simulation": simulation_results
            }
        else:
            return training_results
    
    def run_interactive_mode(self):
        """Run interactive mode with user commands"""
        logger.info("=== Starting Interactive Mode ===")
        print("AiMedRes Interactive Mode")
        print("Commands: train, simulate, comprehensive, agents, networks, status, quit")
        
        while True:
            try:
                command = input("\naimedres> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "train":
                    result = self.run_training_mode()
                    print(f"Training result: {result['status']}")
                elif command == "simulate":
                    result = self.run_simulation_mode()
                    print(f"Simulation result: {result['status']}")
                elif command == "comprehensive":
                    result = self.run_comprehensive_mode()
                    print(f"Comprehensive result: {result['status']}")
                elif command == "agents":
                    print(f"Active agents: {len(self.agents)}")
                    for agent_id, agent in self.agents.items():
                        status = agent.get_status()
                        print(f"  {status['name']}: {status['state']} (energy: {status['biological_state']['energy']:.1f})")
                elif command == "networks":
                    print(f"Neural networks: {len(self.neural_networks)}")
                    for net_id, network in self.neural_networks.items():
                        health = network.get_network_health()
                        print(f"  {net_id}: {health['network_state']} ({health['active_nodes']} active nodes)")
                elif command == "status":
                    self.print_system_status()
                elif command == "help":
                    print("Available commands:")
                    print("  train       - Run training mode")
                    print("  simulate    - Run simulation mode")
                    print("  comprehensive - Run both training and simulation")
                    print("  agents      - List active agents")
                    print("  networks    - List neural networks")
                    print("  status      - Show system status")
                    print("  quit        - Exit application")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit.")
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                print(f"Error: {e}")
    
    def print_system_status(self):
        """Print comprehensive system status"""
        print("\n=== AiMedRes System Status ===")
        
        # Configuration
        print(f"Configuration loaded: {bool(self.config)}")
        
        # Safety monitoring
        if self.safety_monitor:
            status = self.safety_monitor.get_status()
            print(f"Safety monitoring: {status['safety_state']} (safe: {status['is_safe']})")
            print(f"Operations: {status['metrics']['total_operations']} (errors: {status['metrics']['error_count']})")
        else:
            print("Safety monitoring: Disabled")
        
        # Agents
        print(f"Active agents: {len(self.agents)}")
        
        # Neural networks
        print(f"Neural networks: {len(self.neural_networks)}")
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            print(f"Memory usage: {memory_mb:.1f} MB")
            print(f"CPU usage: {cpu_percent:.1f}%")
        except:
            print("Resource monitoring: Unavailable")
    
    def shutdown(self):
        """Graceful application shutdown"""
        logger.info("Starting graceful shutdown...")
        self.running = False
        
        # Shutdown agents
        for agent_id, agent in self.agents.items():
            try:
                agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_id}: {e}")
        
        # Shutdown neural networks
        for net_id, network in self.neural_networks.items():
            try:
                network.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down network {net_id}: {e}")
        
        # Stop safety monitoring
        if self.safety_monitor:
            self.safety_monitor.stop()
        
        logger.info("Graceful shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AiMedRes System - Advanced AI Medical Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m aimedres --mode training       # Run training only
  python -m aimedres --mode simulation     # Run simulation only  
  python -m aimedres --mode comprehensive  # Run both training and simulation
  python -m aimedres                       # Interactive mode (default)
        """)
    
    parser.add_argument('--mode', 
                       choices=['training', 'simulation', 'both', 'comprehensive', 'interactive'],
                       default='interactive',
                       help='Execution mode (default: interactive)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', type=Path,
                       help='Path to configuration file (JSON/YAML)')
    parser.add_argument('--no-safety-monitoring', action='store_true',
                       help='Disable safety monitoring (not recommended for production)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    try:
        # Create and configure application
        with AiMedResApplication(args.config) as app:
            # Start safety monitoring unless disabled
            if not args.no_safety_monitoring:
                app.start_safety_monitoring()
            
            # Run selected mode
            if args.mode == 'training':
                result = app.run_training_mode()
                print(f"Training completed with status: {result.get('status', 'unknown')}")
            
            elif args.mode == 'simulation':
                result = app.run_simulation_mode()
                print(f"Simulation completed with status: {result.get('status', 'unknown')}")
            
            elif args.mode in ['both', 'comprehensive']:
                result = app.run_comprehensive_mode()
                print(f"Comprehensive run completed with status: {result.get('status', 'unknown')}")
            
            elif args.mode == 'interactive':
                app.run_interactive_mode()
            
            else:
                logger.error(f"Unknown mode: {args.mode}")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())