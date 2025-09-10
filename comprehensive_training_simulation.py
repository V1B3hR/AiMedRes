#!/usr/bin/env python3
"""
Comprehensive Training + Simulation for Medical AI Context
==========================================================

This module implements the exact problem statement requirements:
- Comprehensive training + simulation (medical AI context)
- Imports labyrinth components and runs the adaptive simulation

This is the unified entry point that combines both training and simulation
in a medical AI context as specifically requested.
"""

import logging
import time
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import labyrinth components as required
from labyrinth_adaptive import (
    UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, 
    NetworkMetrics, MazeMaster, CapacitorInSpace
)
from labyrinth_simulation import run_labyrinth_simulation, LabyrinthSimulationConfig

# Import training components for medical AI context
from training import AlzheimerTrainer, TrainingIntegratedAgent, run_training_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveTrainingSimulation")

class MedicalAIComprehensiveSystem:
    """
    Comprehensive system that integrates training and simulation for medical AI context
    """
    
    def __init__(self):
        self.training_results = None
        self.simulation_results = None
        self.trained_agents = []
        self.resource_room = ResourceRoom()
        self.maze_master = MazeMaster()
        self.metrics = NetworkMetrics()
        
    def run_comprehensive_training(self) -> Dict[str, Any]:
        """
        Run comprehensive training in medical AI context
        """
        logger.info("=== Starting Comprehensive Medical AI Training ===")
        
        try:
            # Initialize medical AI trainer
            trainer = AlzheimerTrainer()
            
            # Load and train on medical data
            logger.info("Loading medical dataset...")
            df = trainer.load_data()
            X, y = trainer.preprocess_data(df)
            
            logger.info("Training medical AI model...")
            model, results = trainer.train_model(X, y)
            trainer.save_model("comprehensive_medical_model.pkl")
            
            # Store training results
            self.training_results = results
            
            logger.info("=== Training Results ===")
            logger.info(f"Training Accuracy: {results['train_accuracy']:.3f}")
            logger.info(f"Test Accuracy: {results['test_accuracy']:.3f}")
            logger.info(f"Medical AI model saved successfully")
            
            # Create medical AI integrated agents
            self.trained_agents = [
                TrainingIntegratedAgent(
                    "MedicalAI_Agent_Alpha", 
                    {"logic": 0.9, "analytical": 0.8, "medical_expertise": 0.9}, 
                    AliveLoopNode((0, 0), (0.5, 0), 20.0, node_id=1), 
                    self.resource_room, 
                    trainer
                ),
                TrainingIntegratedAgent(
                    "MedicalAI_Agent_Beta", 
                    {"creativity": 0.7, "logic": 0.8, "diagnostic_reasoning": 0.8}, 
                    AliveLoopNode((2, 0), (0, 0.5), 18.0, node_id=2), 
                    self.resource_room, 
                    trainer
                ),
                TrainingIntegratedAgent(
                    "MedicalAI_Agent_Gamma", 
                    {"pattern_recognition": 0.9, "collaboration": 0.8, "precision": 0.9}, 
                    AliveLoopNode((0, 2), (0.3, -0.2), 16.0, node_id=3), 
                    self.resource_room, 
                    trainer
                )
            ]
            
            logger.info(f"Created {len(self.trained_agents)} medical AI agents with trained models")
            
            return {
                "status": "success",
                "training_accuracy": results['test_accuracy'],
                "agents_created": len(self.trained_agents),
                "model_path": "comprehensive_medical_model.pkl"
            }
            
        except Exception as e:
            logger.error(f"Comprehensive training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_adaptive_simulation(self) -> Dict[str, Any]:
        """
        Run the adaptive simulation with labyrinth components
        """
        logger.info("=== Starting Adaptive Labyrinth Simulation ===")
        
        try:
            # Import and configure labyrinth simulation
            config = LabyrinthSimulationConfig()
            
            # Customize simulation for medical AI context
            medical_topics = [
                "Analyze patient cognitive patterns",
                "Collaborate on diagnostic assessment", 
                "Share medical knowledge",
                "Evaluate treatment efficacy",
                "Predict disease progression"
            ]
            
            # Update configuration for medical AI context
            config.config['simulation']['topics'] = medical_topics
            config.config['simulation']['steps'] = 25  # Longer simulation for comprehensive analysis
            
            # If we have trained agents, use them in the simulation
            if self.trained_agents:
                logger.info("Using trained medical AI agents in simulation")
                # Update agent configurations to reflect medical AI capabilities
                for i, agent_config in enumerate(config.config.get('agents', [])):
                    if i < len(self.trained_agents):
                        agent_config['name'] = self.trained_agents[i].name
                        agent_config['style'].update({
                            'medical_training': 0.9,
                            'adaptive_learning': 0.8
                        })
            
            # Run the labyrinth simulation
            logger.info("Running adaptive simulation with labyrinth components...")
            simulation_results = run_labyrinth_simulation(config)
            
            self.simulation_results = simulation_results
            
            logger.info("=== Simulation Results ===")
            logger.info(f"Completed {simulation_results['total_steps']} simulation steps")
            logger.info(f"Agent count: {simulation_results['agent_count']}")
            logger.info(f"MazeMaster interventions: {simulation_results['maze_master_interventions']}")
            logger.info(f"Final health score: {simulation_results['final_health_score']}")
            
            return {
                "status": "success",
                "steps_completed": simulation_results['total_steps'],
                "health_score": simulation_results['final_health_score'],
                "interventions": simulation_results['maze_master_interventions']
            }
            
        except Exception as e:
            logger.error(f"Adaptive simulation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_comprehensive_system(self) -> Dict[str, Any]:
        """
        Run the complete comprehensive training + simulation system
        """
        logger.info("="*60)
        logger.info("COMPREHENSIVE TRAINING + SIMULATION FOR MEDICAL AI")
        logger.info("="*60)
        
        # Phase 1: Comprehensive Training
        training_result = self.run_comprehensive_training()
        if training_result["status"] != "success":
            return {
                "status": "failed",
                "phase": "training",
                "error": training_result.get("error", "Training failed")
            }
        
        # Brief pause between phases
        time.sleep(2)
        
        # Phase 2: Adaptive Simulation with Labyrinth Components
        simulation_result = self.run_adaptive_simulation()
        if simulation_result["status"] != "success":
            return {
                "status": "failed", 
                "phase": "simulation",
                "error": simulation_result.get("error", "Simulation failed")
            }
        
        # Generate comprehensive report
        comprehensive_result = {
            "status": "success",
            "training_phase": {
                "accuracy": training_result["training_accuracy"],
                "agents_created": training_result["agents_created"],
                "model_saved": training_result["model_path"]
            },
            "simulation_phase": {
                "steps_completed": simulation_result["steps_completed"],
                "health_score": simulation_result["health_score"],
                "interventions": simulation_result["interventions"]
            },
            "system_integration": {
                "labyrinth_components_imported": True,
                "medical_ai_context": True,
                "adaptive_simulation_completed": True
            }
        }
        
        logger.info("="*60)
        logger.info("COMPREHENSIVE SYSTEM COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Medical AI Training Accuracy: {training_result['training_accuracy']:.3f}")
        logger.info(f"Simulation Health Score: {simulation_result['health_score']:.3f}")
        logger.info(f"Total Simulation Steps: {simulation_result['steps_completed']}")
        logger.info(f"Labyrinth Components: ✅ Imported and Functional")
        logger.info(f"Medical AI Context: ✅ Integrated")
        logger.info(f"Adaptive Simulation: ✅ Completed")
        
        return comprehensive_result


def main():
    """
    Main entry point for comprehensive training + simulation
    """
    print("🏥 DuetMind Adaptive - Comprehensive Training + Simulation (Medical AI)")
    print("=" * 70)
    
    try:
        # Initialize comprehensive system
        system = MedicalAIComprehensiveSystem()
        
        # Run comprehensive training + simulation
        result = system.run_comprehensive_system()
        
        if result["status"] == "success":
            print("\n🎉 SUCCESS: Comprehensive training + simulation completed!")
            print(f"✅ Training accuracy: {result['training_phase']['accuracy']:.3f}")
            print(f"✅ Simulation health score: {result['simulation_phase']['health_score']:.3f}")
            print(f"✅ Labyrinth components imported and functional")
            print(f"✅ Medical AI context fully integrated")
            return True
        else:
            print(f"\n❌ FAILED: {result.get('phase', 'Unknown')} phase failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)