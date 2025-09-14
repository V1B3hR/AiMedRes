#!/usr/bin/env python3
"""
Comprehensive Training and Simulation System for duetmind_adaptive
Integrates real data training with adaptive agent simulation
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import training system
from files.training.alzheimer_training_system import (
    load_alzheimer_data, preprocess_data, train_model, 
    evaluate_model, save_model, load_model, predict
)

# Import simulation system
from labyrinth_adaptive import (
    UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, 
    MazeMaster, NetworkMetrics, CapacitorInSpace
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ComprehensiveSystem")

class MedicalKnowledgeAgent(UnifiedAdaptiveAgent):
    """Enhanced agent with medical reasoning capabilities"""
    
    def __init__(self, name, cognitive_profile, alive_node, resource_room, medical_model=None):
        # Pass cognitive_profile as style to parent class
        super().__init__(name, cognitive_profile, alive_node, resource_room)
        self.medical_model = medical_model
        self.cognitive_profile = cognitive_profile  # Store separately for medical reasoning
        self.medical_knowledge = {}
        self.patient_assessments = []
        
    def medical_reasoning(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform medical reasoning using trained model"""
        if self.medical_model is None:
            return {"error": "No medical model available", "reasoning": "Cannot assess without trained model"}
            
        try:
            # Convert patient data to DataFrame format expected by model
            feature_cols = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
            patient_df = pd.DataFrame([patient_data], columns=feature_cols)
            
            # Make prediction
            results = predict(self.medical_model, patient_df)
            assessment = results[0] if results else {"error": "Prediction failed"}
            
            # Add reasoning based on cognitive profile
            reasoning = self._generate_medical_reasoning(patient_data, assessment)
            
            assessment['agent_reasoning'] = reasoning
            assessment['agent_name'] = self.name
            assessment['cognitive_profile'] = self.cognitive_profile
            
            self.patient_assessments.append(assessment)
            return assessment
            
        except Exception as e:
            logger.error(f"Medical reasoning error for {self.name}: {e}")
            return {"error": str(e), "reasoning": "Assessment failed due to technical error"}
    
    def _generate_medical_reasoning(self, patient_data: Dict[str, Any], assessment: Dict[str, Any]) -> str:
        """Generate reasoning explanation based on cognitive profile"""
        reasoning_parts = []
        
        # Base reasoning on cognitive strengths
        if self.cognitive_profile.get('analytical', 0) > 0.7:
            reasoning_parts.append(f"Analytical assessment shows {assessment.get('prediction', 'unknown')} with {assessment.get('confidence', 0):.1%} confidence")
            
        if self.cognitive_profile.get('logic', 0) > 0.7:
            key_indicators = []
            if patient_data.get('MMSE', 0) < 24:
                key_indicators.append("low MMSE score")
            if patient_data.get('CDR', 0) > 0.5:
                key_indicators.append("elevated CDR")
            if key_indicators:
                reasoning_parts.append(f"Key logical indicators: {', '.join(key_indicators)}")
                
        if self.cognitive_profile.get('creativity', 0) > 0.7:
            reasoning_parts.append("Creative analysis suggests considering holistic patient context")
            
        return "; ".join(reasoning_parts) if reasoning_parts else "Standard assessment completed"
    
    def collaborate_on_case(self, other_agents: List['MedicalKnowledgeAgent'], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents on a medical case"""
        collaboration_results = {
            'patient_data': patient_data,
            'individual_assessments': [],
            'consensus_prediction': None,
            'confidence_scores': [],
            'collaborative_reasoning': []
        }
        
        # Get individual assessments
        my_assessment = self.medical_reasoning(patient_data)
        collaboration_results['individual_assessments'].append({
            'agent': self.name,
            'assessment': my_assessment
        })
        
        for agent in other_agents:
            if agent != self and agent.medical_model is not None:
                agent_assessment = agent.medical_reasoning(patient_data)
                collaboration_results['individual_assessments'].append({
                    'agent': agent.name,
                    'assessment': agent_assessment
                })
        
        # Calculate consensus
        predictions = []
        confidences = []
        
        for result in collaboration_results['individual_assessments']:
            assessment = result['assessment']
            if 'prediction' in assessment and 'confidence' in assessment:
                predictions.append(1 if assessment['prediction'] == 'Demented' else 0)
                confidences.append(assessment['confidence'])
        
        if predictions:
            consensus_score = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            collaboration_results['consensus_prediction'] = 'Demented' if consensus_score >= 0.5 else 'Nondemented'
            collaboration_results['confidence_scores'] = confidences
            collaboration_results['consensus_confidence'] = avg_confidence
            collaboration_results['agreement_level'] = 1.0 - np.std(predictions)
        
        return collaboration_results

class ComprehensiveSystem:
    """Main system orchestrating training and simulation"""
    
    def __init__(self):
        self.medical_model = None
        self.training_data = None
        self.agents = []
        self.simulation_results = {}
        self.medical_cases = []
        
    def run_comprehensive_training(self, save_model_path="models/alzheimer_model.pkl"):
        """Run complete training pipeline"""
        logger.info("=== Starting Comprehensive Training ===")
        
        try:
            # Load and preprocess data
            logger.info("Loading Alzheimer's disease dataset...")
            df = load_alzheimer_data(file_path="alzheimer.csv")
            X, y = preprocess_data(df)
            self.training_data = {'X': X, 'y': y, 'df': df}
            
            # Train model
            logger.info("Training medical model...")
            clf, X_test, y_test = train_model(X, y)
            
            # Evaluate
            logger.info("Evaluating model performance...")
            y_pred = evaluate_model(clf, X_test, y_test)
            
            # Save model
            logger.info(f"Saving model to {save_model_path}...")
            save_model(clf, save_model_path)
            
            self.medical_model = clf
            
            logger.info("=== Training Complete ===")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def load_trained_model(self, model_path="models/alzheimer_model.pkl"):
        """Load previously trained model"""
        self.medical_model = load_model(model_path)
        return self.medical_model is not None
    
    def create_medical_agents(self):
        """Create agents with medical reasoning capabilities"""
        logger.info("Creating medical knowledge agents...")
        
        resource_room = ResourceRoom()
        
        agent_configs = [
            {
                'name': 'Dr_Analytical',
                'profile': {'analytical': 0.9, 'logic': 0.8, 'precision': 0.85},
                'position': (0, 0),
                'velocity': (0.3, 0),
                'energy': 18.0
            },
            {
                'name': 'Dr_Creative',
                'profile': {'creativity': 0.9, 'intuition': 0.8, 'holistic': 0.7},
                'position': (2, 0),
                'velocity': (0, 0.3),
                'energy': 16.0
            },
            {
                'name': 'Dr_Balanced',
                'profile': {'analytical': 0.7, 'creativity': 0.7, 'logic': 0.6, 'empathy': 0.8},
                'position': (1, 1),
                'velocity': (0.2, 0.2),
                'energy': 17.0
            }
        ]
        
        self.agents = []
        for i, config in enumerate(agent_configs):
            alive_node = AliveLoopNode(
                position=config['position'],
                velocity=config['velocity'],
                initial_energy=config['energy'],
                node_id=i+1
            )
            
            agent = MedicalKnowledgeAgent(
                name=config['name'],
                cognitive_profile=config['profile'],
                alive_node=alive_node,
                resource_room=resource_room,
                medical_model=self.medical_model
            )
            
            self.agents.append(agent)
        
        logger.info(f"Created {len(self.agents)} medical agents")
        return self.agents
    
    def generate_medical_cases(self, num_cases=5):
        """Generate realistic medical case scenarios"""
        logger.info(f"Generating {num_cases} medical case scenarios...")
        
        cases = []
        np.random.seed(42)  # For reproducible cases
        
        for i in range(num_cases):
            # Generate realistic patient data
            age = np.random.randint(65, 95)
            gender = np.random.choice([0, 1])  # 0=Female, 1=Male
            education = np.random.randint(8, 20)
            ses = np.random.uniform(1.0, 5.0)
            
            # MMSE and CDR are correlated with dementia status
            if np.random.random() < 0.4:  # 40% chance of dementia indicators
                mmse = np.random.uniform(10, 23)  # Lower cognitive scores
                cdr = np.random.uniform(0.5, 3.0)  # Higher impairment
            else:
                mmse = np.random.uniform(24, 30)  # Normal cognitive scores
                cdr = np.random.uniform(0.0, 0.5)  # Minimal impairment
                
            etiv = np.random.uniform(1200, 2000)
            nwbv = np.random.uniform(0.6, 0.8)
            asf = np.random.uniform(0.8, 1.3)
            
            case = {
                'case_id': f"CASE_{i+1:03d}",
                'patient_data': {
                    'M/F': gender,
                    'Age': age,
                    'EDUC': education,
                    'SES': ses,
                    'MMSE': mmse,
                    'CDR': cdr,
                    'eTIV': etiv,
                    'nWBV': nwbv,
                    'ASF': asf
                },
                'description': f"Patient {i+1}: {age}yo {'M' if gender==1 else 'F'}, MMSE={mmse:.1f}, CDR={cdr:.1f}"
            }
            cases.append(case)
        
        self.medical_cases = cases
        logger.info("Medical cases generated successfully")
        return cases
    
    def run_medical_simulation(self, steps=10):
        """Run simulation with medical reasoning scenarios"""
        logger.info("=== Starting Medical Simulation ===")
        
        if not self.agents:
            logger.error("No agents available. Create agents first.")
            return None
            
        if not self.medical_cases:
            logger.error("No medical cases available. Generate cases first.")
            return None
        
        maze_master = MazeMaster()
        metrics = NetworkMetrics()
        capacitors = [CapacitorInSpace((1,1), capacity=10.0, initial_energy=5.0)]
        
        simulation_results = {
            'steps': [],
            'collaborative_assessments': [],
            'agent_performance': {agent.name: [] for agent in self.agents},
            'network_health': []
        }
        
        for step in range(1, steps + 1):
            logger.info(f"\n--- Medical Simulation Step {step} ---")
            
            step_results = {
                'step': step,
                'agent_states': [],
                'medical_activities': []
            }
            
            # Regular agent reasoning and movement
            for agent in self.agents:
                agent.reason(f"Medical consultation step {step}")
                agent.alive_node.move()
                
                step_results['agent_states'].append(agent.get_state())
            
            # Every few steps, present a medical case for collaboration
            if step % 3 == 0 and self.medical_cases:
                case = self.medical_cases[(step // 3 - 1) % len(self.medical_cases)]
                logger.info(f"Presenting medical case: {case['description']}")
                
                # Main agent evaluates case and collaborates with others
                main_agent = self.agents[0]
                other_agents = self.agents[1:]
                
                collaboration_result = main_agent.collaborate_on_case(other_agents, case['patient_data'])
                collaboration_result['case_info'] = case
                collaboration_result['step'] = step
                
                simulation_results['collaborative_assessments'].append(collaboration_result)
                step_results['medical_activities'].append(collaboration_result)
                
                logger.info(f"Consensus: {collaboration_result.get('consensus_prediction', 'N/A')} "
                          f"(confidence: {collaboration_result.get('consensus_confidence', 0):.3f})")
            
            # System governance and monitoring
            maze_master.govern_agents(self.agents)
            metrics.update(self.agents)
            health_score = metrics.health_score()
            simulation_results['network_health'].append(health_score)
            
            logger.info(f"Network Health Score: {health_score:.3f}")
            
            simulation_results['steps'].append(step_results)
        
        self.simulation_results = simulation_results
        logger.info("=== Medical Simulation Complete ===")
        return simulation_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report of training and simulation results"""
        logger.info("Generating comprehensive system report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': {},
            'simulation_summary': {},
            'medical_assessments': {},
            'system_performance': {}
        }
        
        # Training summary
        if self.training_data:
            report['training_summary'] = {
                'dataset_size': len(self.training_data['X']),
                'features_used': list(self.training_data['X'].columns),
                'target_distribution': self.training_data['y'].value_counts().to_dict(),
                'model_available': self.medical_model is not None
            }
        
        # Simulation summary
        if self.simulation_results:
            report['simulation_summary'] = {
                'total_steps': len(self.simulation_results['steps']),
                'agents_count': len(self.agents),
                'medical_cases_evaluated': len(self.simulation_results['collaborative_assessments']),
                'average_network_health': np.mean(self.simulation_results['network_health']) if self.simulation_results['network_health'] else 0
            }
            
            # Medical assessment analysis
            assessments = self.simulation_results['collaborative_assessments']
            if assessments:
                consensus_predictions = [a.get('consensus_prediction') for a in assessments]
                avg_confidence = np.mean([a.get('consensus_confidence', 0) for a in assessments])
                
                report['medical_assessments'] = {
                    'total_cases': len(assessments),
                    'consensus_predictions': dict(pd.Series(consensus_predictions).value_counts()),
                    'average_confidence': avg_confidence,
                    'average_agreement': np.mean([a.get('agreement_level', 0) for a in assessments])
                }
        
        # System performance
        report['system_performance'] = {
            'training_completed': self.medical_model is not None,
            'agents_created': len(self.agents),
            'simulation_completed': bool(self.simulation_results),
            'integration_success': all([
                self.medical_model is not None,
                len(self.agents) > 0,
                bool(self.simulation_results)
            ])
        }
        
        return report

def main():
    """Main execution function"""
    print("=" * 60)
    print("COMPREHENSIVE TRAINING AND SIMULATION ON REAL DATA")
    print("duetmind_adaptive - Medical AI Agent System")
    print("=" * 60)
    
    system = ComprehensiveSystem()
    
    try:
        # Step 1: Train medical model on real data
        success = system.run_comprehensive_training()
        if not success:
            logger.error("Training failed. Attempting to load existing model...")
            if not system.load_trained_model():
                logger.error("No trained model available. Exiting.")
                return
        
        # Step 2: Create medical knowledge agents
        agents = system.create_medical_agents()
        
        # Step 3: Generate realistic medical cases
        cases = system.generate_medical_cases(num_cases=6)
        
        # Step 4: Run integrated simulation
        results = system.run_medical_simulation(steps=15)
        
        # Step 5: Generate comprehensive report
        report = system.generate_comprehensive_report()
        
        # Display final results
        print("\n" + "=" * 60)
        print("COMPREHENSIVE SYSTEM REPORT")
        print("=" * 60)
        
        print(f"\nTraining Summary:")
        print(f"  Dataset size: {report['training_summary'].get('dataset_size', 'N/A')}")
        print(f"  Features used: {len(report['training_summary'].get('features_used', []))}")
        print(f"  Model trained: {report['training_summary'].get('model_available', False)}")
        
        print(f"\nSimulation Summary:")
        print(f"  Total steps: {report['simulation_summary'].get('total_steps', 'N/A')}")
        print(f"  Medical agents: {report['simulation_summary'].get('agents_count', 'N/A')}")
        print(f"  Cases evaluated: {report['simulation_summary'].get('medical_cases_evaluated', 'N/A')}")
        print(f"  Avg network health: {report['simulation_summary'].get('average_network_health', 0):.3f}")
        
        print(f"\nMedical Assessment Results:")
        med_assess = report.get('medical_assessments', {})
        print(f"  Total cases: {med_assess.get('total_cases', 'N/A')}")
        print(f"  Average confidence: {med_assess.get('average_confidence', 0):.3f}")
        print(f"  Agent agreement: {med_assess.get('average_agreement', 0):.3f}")
        
        print(f"\nSystem Performance:")
        perf = report['system_performance']
        print(f"  Integration success: {perf.get('integration_success', False)}")
        print(f"  All components working: {all(perf.values())}")
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TRAINING AND SIMULATION COMPLETE")
        print("Real data successfully integrated with adaptive agent simulation")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        logger.error(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()