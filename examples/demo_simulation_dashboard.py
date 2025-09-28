#!/usr/bin/env python3
"""
Demo script for the Web-Based Simulation Dashboard

This script demonstrates the key features of the simulation dashboard:
1. Scenario creation and management
2. Simulation execution with real-time monitoring
3. Intervention capabilities
4. Metrics collection and monitoring
5. Real-time updates and safety monitoring
"""

import asyncio
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer demo-token", "Content-Type": "application/json"}

def demo_scenario_creation():
    """Demonstrate scenario creation"""
    print("🔧 Demo: Scenario Creation")
    print("=" * 40)
    
    # Create a complex Alzheimer's patient scenario
    scenario_data = {
        "name": "Advanced Alzheimer's Risk Assessment",
        "description": "Complex scenario with multiple risk factors and timeline events",
        "patient_profile": {
            "patient_id": "patient_alz_001",
            "age": 73,
            "gender": "Male",
            "conditions": ["Mild Cognitive Impairment", "Hypertension", "Type 2 Diabetes"],
            "medications": ["Metformin", "Lisinopril", "Aricept"],
            "vitals": {
                "blood_pressure": 145.0,
                "heart_rate": 68.0,
                "temperature": 98.6,
                "respiratory_rate": 16.0
            },
            "lab_values": {
                "MMSE": 26.0,
                "CDR": 0.5,
                "glucose": 145.0,
                "cholesterol": 220.0
            },
            "medical_history": ["Family history of Alzheimer's", "Previous stroke", "Depression"]
        },
        "timeline_events": [
            {
                "event_id": "event_baseline",
                "timestamp": "2024-01-01T09:00:00Z",
                "event_type": "initial_assessment",
                "parameters": {"assessment_type": "baseline"},
                "description": "Initial cognitive assessment"
            },
            {
                "event_id": "event_mmse_decline",
                "timestamp": "2024-01-15T10:30:00Z",
                "event_type": "lab_result",
                "parameters": {"MMSE": 23.0},
                "description": "MMSE score shows decline"
            },
            {
                "event_id": "event_medication_change",
                "timestamp": "2024-01-22T14:00:00Z",
                "event_type": "medication_change",
                "parameters": {
                    "add": ["Memantine"],
                    "remove": []
                },
                "description": "Added Memantine for cognitive support"
            },
            {
                "event_id": "event_vital_change",
                "timestamp": "2024-02-01T11:00:00Z",
                "event_type": "vital_change",
                "parameters": {"blood_pressure": 160.0},
                "description": "Blood pressure elevation"
            }
        ],
        "parameters": {
            "simulation_speed": "normal",
            "enable_interventions": True,
            "safety_monitoring": True
        }
    }
    
    # Create the scenario
    response = requests.post(f"{BASE_URL}/api/scenarios", headers=HEADERS, json=scenario_data)
    
    if response.status_code == 200:
        result = response.json()
        scenario_id = result["scenario_id"]
        print(f"✅ Created scenario: {scenario_id}")
        print(f"   Name: {scenario_data['name']}")
        print(f"   Patient: {scenario_data['patient_profile']['age']}yo {scenario_data['patient_profile']['gender']}")
        print(f"   Conditions: {', '.join(scenario_data['patient_profile']['conditions'])}")
        print(f"   Timeline events: {len(scenario_data['timeline_events'])}")
        return scenario_id
    else:
        print(f"❌ Failed to create scenario: {response.status_code}")
        print(response.text)
        return None

def demo_simulation_execution(scenario_id):
    """Demonstrate simulation execution"""
    print(f"\n🚀 Demo: Simulation Execution")
    print("=" * 40)
    
    # Start the simulation
    response = requests.post(
        f"{BASE_URL}/api/simulations/start",
        headers=HEADERS,
        json={"scenario_id": scenario_id}
    )
    
    if response.status_code == 200:
        result = response.json()
        simulation_id = result["simulation_id"]
        print(f"✅ Started simulation: {simulation_id}")
        
        # Monitor simulation progress
        for i in range(5):
            time.sleep(1)
            status_response = requests.get(
                f"{BASE_URL}/api/simulations/{simulation_id}/status",
                headers=HEADERS
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   Status: {status.get('status', 'unknown')}")
                print(f"   Current tick: {status.get('current_tick', 0)}")
                if status.get('status') == 'completed':
                    break
            else:
                print(f"   Failed to get status: {status_response.status_code}")
        
        return simulation_id
    else:
        print(f"❌ Failed to start simulation: {response.status_code}")
        print(response.text)
        return None

def demo_intervention_panel(simulation_id):
    """Demonstrate intervention capabilities"""
    print(f"\n⚡ Demo: Intervention Panel")
    print("=" * 40)
    
    interventions = [
        {"type": "pause", "description": "Pause simulation for analysis"},
        {"type": "force_fallback", "description": "Force fallback to deterministic model"},
        {"type": "inject_memory", "description": "Inject synthetic memory event"}
    ]
    
    for intervention in interventions:
        print(f"\n   Applying intervention: {intervention['type']}")
        print(f"   Description: {intervention['description']}")
        
        response = requests.post(
            f"{BASE_URL}/api/simulations/{simulation_id}/intervention",
            headers=HEADERS,
            json={
                "scenario_id": simulation_id,
                "intervention_type": intervention["type"],
                "parameters": {}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Intervention applied: {result.get('intervention_id')}")
            print(f"   Status: {result.get('status')}")
        else:
            print(f"   ❌ Intervention failed: {response.status_code}")
        
        time.sleep(0.5)

def demo_metrics_monitoring():
    """Demonstrate metrics collection and monitoring"""
    print(f"\n📊 Demo: Metrics Monitoring")
    print("=" * 40)
    
    response = requests.get(f"{BASE_URL}/api/metrics", headers=HEADERS)
    
    if response.status_code == 200:
        metrics = response.json()
        print("   Current Metrics:")
        print(f"   • Active Simulations: {metrics.get('active_simulations', 0)}")
        print(f"   • Completed Simulations: {metrics.get('completed_simulations', 0)}")
        print(f"   • Failed Simulations: {metrics.get('failed_simulations', 0)}")
        print(f"   • Scenario Throughput: {metrics.get('scenario_throughput', 0):.1f} scenarios/hour")
        print(f"   • Average Latency: {metrics.get('avg_latency_ms', 0):.1f} ms")
        print(f"   • Reproducibility Rate: {metrics.get('reproducibility_rate', 0):.2%}")
        print(f"   • Intervention Frequency: {metrics.get('intervention_frequency', 0)}")
        
        return metrics
    else:
        print(f"❌ Failed to get metrics: {response.status_code}")
        return None

def demo_scenario_management():
    """Demonstrate scenario management"""
    print(f"\n📋 Demo: Scenario Management")
    print("=" * 40)
    
    response = requests.get(f"{BASE_URL}/api/scenarios", headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        scenarios = data.get("scenarios", [])
        print(f"   Found {len(scenarios)} scenarios:")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"   {i}. {scenario['name']}")
            print(f"      ID: {scenario['scenario_id']}")
            print(f"      Created by: {scenario['created_by']}")
            print(f"      Created at: {scenario['created_at']}")
            print(f"      Description: {scenario.get('description', 'No description')}")
            print()
        
        return scenarios
    else:
        print(f"❌ Failed to list scenarios: {response.status_code}")
        return None

def demo_data_contract():
    """Demonstrate the simulation tick data contract"""
    print(f"\n📋 Demo: Simulation Data Contract")
    print("=" * 40)
    
    example_tick = {
        "scenario_id": "scn-123",
        "tick": 42,
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "risk_assessor": {
                "state": "computing",
                "latency_ms": 54,
                "risk_score": 0.73
            },
            "treatment_advisor": {
                "state": "completed",
                "latency_ms": 89,
                "confidence": 0.91
            }
        },
        "patient_context_hash": "abc123",
        "recommendations": [
            {
                "code": "RX_ADJUST",
                "confidence": 0.91,
                "type": "medication_adjustment",
                "description": "Consider adjusting medication dosage"
            }
        ],
        "safety_state": "warning",
        "memory_events": [
            {
                "type": "episodic_consolidated",
                "count": 5,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "metrics": {
            "avg_latency_ms": 71.5,
            "max_latency_ms": 89,
            "agent_count": 2,
            "success_rate": 1.0
        }
    }
    
    print("   Example Simulation Tick Data Contract:")
    print(json.dumps(example_tick, indent=2, default=str))

def main():
    """Run the complete simulation dashboard demo"""
    print("🎯 AiMedRes Simulation Dashboard Demo")
    print("=" * 50)
    print()
    
    try:
        # Test server connectivity
        response = requests.get(f"{BASE_URL}/api/metrics", headers=HEADERS)
        if response.status_code != 200:
            print("❌ Dashboard server is not running or not accessible")
            print("   Please start the server with: python simulation_dashboard.py")
            return
        
        print("✅ Dashboard server is running and accessible")
        print()
        
        # Demo 1: Scenario Creation
        scenario_id = demo_scenario_creation()
        if not scenario_id:
            return
        
        # Demo 2: Scenario Management
        demo_scenario_management()
        
        # Demo 3: Simulation Execution
        simulation_id = demo_simulation_execution(scenario_id)
        if not simulation_id:
            return
        
        # Demo 4: Intervention Panel
        demo_intervention_panel(simulation_id)
        
        # Demo 5: Metrics Monitoring
        demo_metrics_monitoring()
        
        # Demo 6: Data Contract
        demo_data_contract()
        
        print(f"\n🎉 Dashboard Demo Completed Successfully!")
        print("=" * 50)
        print(f"\n📱 Key Features Demonstrated:")
        print("   ✅ Scenario Builder - Compose patient profiles & timeline events")
        print("   ✅ Execution Orchestrator - Launch simulation workers with isolation")
        print("   ✅ Real-Time Metrics Layer - Monitor agent states & performance")
        print("   ✅ Intervention Panel - Manual override capabilities")
        print("   ✅ Metrics Collection - Track throughput, latency, intervention frequency")
        print("   ✅ Data Contract - Standardized simulation tick format")
        
        print(f"\n🌐 Dashboard Access:")
        print(f"   • Web Interface: {BASE_URL}/")
        print(f"   • API Documentation: {BASE_URL}/docs")
        print(f"   • Metrics Endpoint: {BASE_URL}/api/metrics")
        
        print(f"\n🏗️ Architecture Components:")
        print("   • Frontend SPA ↔ API Gateway (FastAPI) ↔ Orchestrator Service ↔ Worker Pool")
        print("   • Metrics & Events: Redis Streams (configurable)")
        print("   • Persistence: SQLite (demo) / Postgres (production)")
        print("   • Authentication & RBAC: HTTP Bearer (extensible)")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()