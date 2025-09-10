#!/usr/bin/env python3
"""
Usage Examples for Comprehensive Training and Simulation System
Demonstrates various ways to use the duetmind_adaptive system
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_1_basic_training():
    """Example 1: Basic training workflow"""
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC TRAINING WORKFLOW")
    print("="*60)
    
    from training.alzheimer_training_system import (
        load_alzheimer_data, preprocess_data, train_model, 
        evaluate_model, save_model
    )
    
    # Load data
    print("1. Loading Alzheimer's dataset...")
    df = load_alzheimer_data("alzheimer.csv")
    
    # Preprocess
    print("2. Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Train
    print("3. Training model...")
    clf, X_test, y_test = train_model(X, y)
    
    # Evaluate
    print("4. Evaluating model...")
    y_pred = evaluate_model(clf, X_test, y_test)
    
    # Save
    print("5. Saving model...")
    save_model(clf, "models/example_model.pkl")
    
    print("✓ Basic training complete!")
    return clf

def example_2_data_quality_check():
    """Example 2: Data quality validation"""
    print("\n" + "="*60)
    print("EXAMPLE 2: DATA QUALITY VALIDATION")
    print("="*60)
    
    from data_quality_monitor import DataQualityMonitor
    from training.alzheimer_training_system import load_alzheimer_data
    
    # Load data
    print("1. Loading dataset for quality check...")
    df = load_alzheimer_data("alzheimer.csv")
    
    # Validate quality
    print("2. Running comprehensive quality validation...")
    monitor = DataQualityMonitor()
    report = monitor.validate_alzheimer_dataset(df)
    
    # Display results
    print("3. Quality validation results:")
    print(f"   Overall Score: {report['overall_score']:.3f}/1.0")
    print(f"   Completeness: {report['completeness']['score']:.3f}")
    print(f"   Consistency: {report['consistency']['score']:.3f}")
    print(f"   Validity: {report['validity']['score']:.3f}")
    print(f"   Ready for Training: {'Yes' if report['overall_score'] >= 0.7 else 'No'}")
    
    print("✓ Data quality validation complete!")
    return report

def example_3_simple_simulation():
    """Example 3: Basic agent simulation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: BASIC AGENT SIMULATION")
    print("="*60)
    
    from labyrinth_simulation import run_labyrinth_simulation
    
    print("Running basic labyrinth simulation...")
    print("(This demonstrates the core adaptive agent system)")
    
    # Run simulation (output will be logged)
    run_labyrinth_simulation()
    
    print("✓ Basic simulation complete!")

def example_4_medical_agent_reasoning():
    """Example 4: Medical agent with real data"""
    print("\n" + "="*60)
    print("EXAMPLE 4: MEDICAL AGENT REASONING")
    print("="*60)
    
    from comprehensive_training_simulation import MedicalKnowledgeAgent, ComprehensiveSystem
    from training.alzheimer_training_system import load_model
    from labyrinth_adaptive import AliveLoopNode, ResourceRoom
    
    # Load or train model
    print("1. Loading trained model...")
    model = load_model("models/alzheimer_model.pkl")
    if model is None:
        print("   No model found, training new one...")
        model = example_1_basic_training()
    
    # Create agent
    print("2. Creating medical agent...")
    resource_room = ResourceRoom()
    alive_node = AliveLoopNode((0, 0), (0.1, 0.1), 15.0, node_id=1)
    
    agent = MedicalKnowledgeAgent(
        name="Dr_Example",
        cognitive_profile={'analytical': 0.8, 'logic': 0.7, 'precision': 0.9},
        alive_node=alive_node,
        resource_room=resource_room,
        medical_model=model
    )
    
    # Test medical reasoning
    print("3. Testing medical reasoning...")
    test_patient = {
        'M/F': 1,      # Male
        'Age': 75,
        'EDUC': 16,
        'SES': 2.0,
        'MMSE': 24,    # Borderline cognitive score
        'CDR': 0.5,    # Very mild dementia
        'eTIV': 1500,
        'nWBV': 0.70,
        'ASF': 1.0
    }
    
    assessment = agent.medical_reasoning(test_patient)
    
    print("4. Assessment results:")
    print(f"   Prediction: {assessment.get('prediction', 'N/A')}")
    print(f"   Confidence: {assessment.get('confidence', 0):.3f}")
    print(f"   Agent Reasoning: {assessment.get('agent_reasoning', 'N/A')}")
    
    print("✓ Medical reasoning example complete!")
    return agent, assessment

def example_5_comprehensive_system():
    """Example 5: Full comprehensive system"""
    print("\n" + "="*60)
    print("EXAMPLE 5: COMPREHENSIVE SYSTEM")
    print("="*60)
    
    from comprehensive_training_simulation import ComprehensiveSystem
    
    print("1. Initializing comprehensive system...")
    system = ComprehensiveSystem()
    
    # Try to load existing model first
    print("2. Loading or training model...")
    if not system.load_trained_model():
        print("   Training new model...")
        success = system.run_comprehensive_training()
        if not success:
            print("   ❌ Training failed!")
            return None
    
    # Create agents
    print("3. Creating medical agents...")
    agents = system.create_medical_agents()
    
    # Generate cases
    print("4. Generating medical cases...")
    cases = system.generate_medical_cases(num_cases=3)
    
    # Run simulation
    print("5. Running medical simulation...")
    results = system.run_medical_simulation(steps=6)
    
    # Generate report
    print("6. Generating comprehensive report...")
    report = system.generate_comprehensive_report()
    
    print("7. System performance summary:")
    print(f"   Training completed: {report['system_performance']['training_completed']}")
    print(f"   Agents created: {report['system_performance']['agents_created']}")
    print(f"   Simulation completed: {report['system_performance']['simulation_completed']}")
    print(f"   Integration success: {report['system_performance']['integration_success']}")
    
    if results and 'collaborative_assessments' in results:
        assessments = results['collaborative_assessments']
        if assessments:
            print(f"   Medical cases evaluated: {len(assessments)}")
            avg_confidence = sum(a.get('consensus_confidence', 0) for a in assessments) / len(assessments)
            print(f"   Average assessment confidence: {avg_confidence:.3f}")
    
    print("✓ Comprehensive system example complete!")
    return system, results, report

def example_6_custom_scenarios():
    """Example 6: Custom medical scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 6: CUSTOM MEDICAL SCENARIOS")
    print("="*60)
    
    from comprehensive_training_simulation import ComprehensiveSystem
    import numpy as np
    
    system = ComprehensiveSystem()
    
    # Load model
    if not system.load_trained_model():
        print("No model available. Please run training first.")
        return None
    
    # Create agents
    agents = system.create_medical_agents()
    
    # Create custom scenarios
    print("1. Creating custom medical scenarios...")
    
    scenarios = [
        {
            'name': 'Mild Cognitive Impairment',
            'data': {'M/F': 0, 'Age': 68, 'EDUC': 18, 'SES': 1.5, 'MMSE': 26, 'CDR': 0.5, 'eTIV': 1450, 'nWBV': 0.75, 'ASF': 0.95}
        },
        {
            'name': 'Advanced Alzheimer\'s',
            'data': {'M/F': 1, 'Age': 85, 'EDUC': 12, 'SES': 3.0, 'MMSE': 15, 'CDR': 2.0, 'eTIV': 1300, 'nWBV': 0.60, 'ASF': 1.15}
        },
        {
            'name': 'Healthy Aging',
            'data': {'M/F': 0, 'Age': 72, 'EDUC': 16, 'SES': 2.0, 'MMSE': 29, 'CDR': 0.0, 'eTIV': 1550, 'nWBV': 0.78, 'ASF': 0.90}
        }
    ]
    
    print("2. Running collaborative assessments...")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['name']}")
        
        # Get collaborative assessment
        main_agent = agents[0]
        other_agents = agents[1:]
        
        collaboration = main_agent.collaborate_on_case(other_agents, scenario['data'])
        
        print(f"   Consensus: {collaboration.get('consensus_prediction', 'N/A')}")
        print(f"   Confidence: {collaboration.get('consensus_confidence', 0):.3f}")
        print(f"   Agreement: {collaboration.get('agreement_level', 0):.3f}")
        
        # Show individual agent assessments
        for assessment in collaboration.get('individual_assessments', []):
            agent_name = assessment['agent']
            agent_result = assessment['assessment']
            print(f"   {agent_name}: {agent_result.get('prediction', 'N/A')} ({agent_result.get('confidence', 0):.3f})")
    
    print("\n✓ Custom scenarios example complete!")
    return scenarios, agents

def main():
    """Run all examples"""
    print("COMPREHENSIVE TRAINING AND SIMULATION EXAMPLES")
    print("duetmind_adaptive - Medical AI Agent System")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    examples = [
        ("Basic Training", example_1_basic_training),
        ("Data Quality Check", example_2_data_quality_check),
        ("Simple Simulation", example_3_simple_simulation),
        ("Medical Agent Reasoning", example_4_medical_agent_reasoning),
        ("Comprehensive System", example_5_comprehensive_system),
        ("Custom Scenarios", example_6_custom_scenarios)
    ]
    
    print(f"\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print(f"\nTo run specific example: python3 {__file__} <number>")
    print(f"To run all examples: python3 {__file__}")
    
    # Check if specific example requested
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            if 1 <= example_num <= len(examples):
                name, func = examples[example_num - 1]
                print(f"\nRunning Example {example_num}: {name}")
                result = func()
                print(f"\n✓ Example {example_num} completed successfully!")
                return result
            else:
                print(f"\nInvalid example number. Choose 1-{len(examples)}")
                return None
        except ValueError:
            print(f"\nInvalid input. Please provide a number 1-{len(examples)}")
            return None
    
    # Run all examples
    print(f"\nRunning all examples...")
    results = {}
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\nStarting Example {i}: {name}")
            results[name] = func()
            print(f"✓ Example {i} completed successfully!")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    # Summary
    print(f"\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    successful = sum(1 for result in results.values() if result is not None)
    total = len(results)
    
    print(f"Examples completed: {successful}/{total}")
    
    for name, result in results.items():
        status = "✓ Success" if result is not None else "❌ Failed"
        print(f"  {name}: {status}")
    
    print(f"\nExecution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("For detailed documentation, see: COMPREHENSIVE_TRAINING_DOCUMENTATION.md")
    
    return results

if __name__ == "__main__":
    main()