"""
Test suite for Quantum-Enhanced Computing (P20)

Tests hybrid quantum ML, molecular simulation, quantum optimization,
and benchmarking capabilities.
"""

import pytest
import time
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aimedres.core.quantum_computing import (
    QuantumComputingSystem,
    QuantumCircuitBuilder,
    HybridQuantumMLEngine,
    MolecularSimulationEngine,
    QuantumOptimizationEngine,
    QuantumBenchmarkingSystem,
    QuantumBackend,
    CircuitType,
    OptimizationProblem,
    MoleculeType,
    QuantumCircuit,
    HybridModel,
    Molecule,
    SimulationResult,
    OptimizationResult,
    BenchmarkResult,
    create_quantum_computing_system
)


class TestQuantumCircuitBuilder:
    """Tests for quantum circuit building."""
    
    def test_builder_initialization(self):
        """Test circuit builder initialization."""
        builder = QuantumCircuitBuilder(QuantumBackend.SIMULATOR)
        assert builder.backend == QuantumBackend.SIMULATOR
        assert len(builder.circuits) == 0
        assert len(builder.circuit_library) > 0
    
    def test_create_circuit(self):
        """Test creating quantum circuit."""
        builder = QuantumCircuitBuilder()
        
        circuit = builder.create_circuit(
            circuit_type=CircuitType.VARIATIONAL,
            num_qubits=4,
            depth=3
        )
        
        assert circuit.circuit_type == CircuitType.VARIATIONAL
        assert circuit.num_qubits == 4
        assert circuit.depth == 3
        assert circuit.gate_count > 0
        assert len(circuit.parameters) > 0
        assert circuit.circuit_id in builder.circuits
    
    def test_create_qaoa_circuit(self):
        """Test creating QAOA circuit."""
        builder = QuantumCircuitBuilder()
        
        circuit = builder.create_qaoa_circuit(num_qubits=6, num_layers=2)
        
        assert circuit.circuit_type == CircuitType.QAOA
        assert circuit.num_qubits == 6
        assert circuit.depth == 2
    
    def test_create_vqe_circuit(self):
        """Test creating VQE circuit."""
        builder = QuantumCircuitBuilder()
        
        circuit = builder.create_vqe_circuit(num_qubits=4)
        
        assert circuit.circuit_type == CircuitType.VQE
        assert circuit.num_qubits == 4
    
    def test_optimize_circuit(self):
        """Test circuit optimization."""
        builder = QuantumCircuitBuilder()
        
        original = builder.create_circuit(CircuitType.VARIATIONAL, 6, 5)
        optimized = builder.optimize_circuit(original)
        
        assert optimized.num_qubits == original.num_qubits
        assert optimized.depth < original.depth
        assert optimized.gate_count < original.gate_count
        assert 'optimized_from' in optimized.metadata


class TestHybridQuantumMLEngine:
    """Tests for hybrid quantum-classical ML."""
    
    def test_engine_initialization(self):
        """Test ML engine initialization."""
        engine = HybridQuantumMLEngine(QuantumBackend.SIMULATOR)
        assert engine.quantum_backend == QuantumBackend.SIMULATOR
        assert len(engine.models) == 0
        assert len(engine.predictions) == 0
    
    def test_train_hybrid_model(self):
        """Test training hybrid model."""
        engine = HybridQuantumMLEngine()
        
        model = engine.train_hybrid_model(
            quantum_layers=3,
            classical_layers=2,
            training_samples=1000,
            epochs=10
        )
        
        assert model.quantum_layers == 3
        assert model.classical_layers == 2
        assert 0.75 <= model.accuracy <= 0.95
        assert model.training_time_seconds >= 0
        assert model.inference_time_ms > 0
        assert model.model_id in engine.models
    
    def test_predict(self):
        """Test making predictions."""
        engine = HybridQuantumMLEngine()
        
        # Train model first
        model = engine.train_hybrid_model(2, 2, 500, 5)
        
        # Make prediction
        input_data = {"feature_1": 0.5, "feature_2": 0.7}
        prediction = engine.predict(model.model_id, input_data)
        
        assert 'prediction_id' in prediction
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 0 <= prediction['confidence'] <= 1
        assert prediction['quantum_enhanced'] is True
    
    def test_compare_quantum_vs_classical(self):
        """Test quantum vs classical comparison."""
        engine = HybridQuantumMLEngine()
        
        comparison = engine.compare_quantum_vs_classical(
            quantum_layers=3,
            classical_layers=2,
            samples=1000
        )
        
        assert 'quantum_model' in comparison
        assert 'classical_model' in comparison
        assert 'quantum_advantage' in comparison
        
        # Check quantum advantage metrics
        advantage = comparison['quantum_advantage']
        assert 'accuracy_gain' in advantage
        assert 'speedup_training' in advantage
        assert 'speedup_inference' in advantage


class TestMolecularSimulationEngine:
    """Tests for molecular simulation."""
    
    def test_engine_initialization(self):
        """Test simulation engine initialization."""
        engine = MolecularSimulationEngine()
        assert len(engine.molecules) == 0
        assert len(engine.simulations) == 0
        assert engine.circuit_builder is not None
    
    def test_register_molecule(self):
        """Test registering molecule."""
        engine = MolecularSimulationEngine()
        
        molecule = Molecule(
            molecule_id="mol_001",
            name="Aspirin",
            molecule_type=MoleculeType.DRUG_COMPOUND,
            formula="C9H8O4",
            num_atoms=21,
            num_bonds=21,
            molecular_weight=180.16,
            structure_data={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        )
        
        engine.register_molecule(molecule)
        
        assert "mol_001" in engine.molecules
        assert engine.molecules["mol_001"].name == "Aspirin"
    
    def test_simulate_ground_state(self):
        """Test ground state simulation."""
        engine = MolecularSimulationEngine()
        
        # Register molecule
        molecule = Molecule(
            molecule_id="mol_sim",
            name="Water",
            molecule_type=MoleculeType.DRUG_COMPOUND,
            formula="H2O",
            num_atoms=3,
            num_bonds=2,
            molecular_weight=18.015,
            structure_data={}
        )
        engine.register_molecule(molecule)
        
        # Simulate
        result = engine.simulate_ground_state("mol_sim", method="VQE")
        
        assert result.molecule_id == "mol_sim"
        assert result.convergence_achieved is True
        assert len(result.energy_levels) > 0
        assert result.ground_state_energy < 0  # Negative energy
        assert len(result.excited_states) > 0
        assert result.simulation_time_seconds >= 0
    
    def test_predict_binding_affinity(self):
        """Test binding affinity prediction."""
        engine = MolecularSimulationEngine()
        
        # Register two molecules
        for i, (name, weight) in enumerate([("DrugA", 200.0), ("TargetB", 50000.0)]):
            molecule = Molecule(
                molecule_id=f"mol_bind_{i}",
                name=name,
                molecule_type=MoleculeType.PROTEIN if i == 1 else MoleculeType.DRUG_COMPOUND,
                formula=f"C{10+i}H{20+i}",
                num_atoms=30 + i * 10,
                num_bonds=30 + i * 10,
                molecular_weight=weight,
                structure_data={}
            )
            engine.register_molecule(molecule)
        
        # Predict binding
        result = engine.predict_binding_affinity("mol_bind_0", "mol_bind_1")
        
        assert 'prediction_id' in result
        assert 'binding_affinity_nm' in result
        assert result['binding_strength'] in ['strong', 'moderate', 'weak']
        assert 0 <= result['confidence'] <= 1
    
    def test_optimize_molecule_structure(self):
        """Test molecule structure optimization."""
        engine = MolecularSimulationEngine()
        
        # Register molecule
        molecule = Molecule(
            molecule_id="mol_opt",
            name="TestMolecule",
            molecule_type=MoleculeType.DRUG_COMPOUND,
            formula="C10H15N",
            num_atoms=26,
            num_bonds=26,
            molecular_weight=149.23,
            structure_data={}
        )
        engine.register_molecule(molecule)
        
        # Optimize
        result = engine.optimize_molecule_structure("mol_opt")
        
        assert 'optimization_id' in result
        assert result['converged'] is True
        assert result['optimized_energy'] < result['initial_energy']
        assert result['energy_reduction'] > 0


class TestQuantumOptimizationEngine:
    """Tests for quantum optimization."""
    
    def test_engine_initialization(self):
        """Test optimization engine initialization."""
        engine = QuantumOptimizationEngine()
        assert len(engine.optimization_results) == 0
        assert len(engine.problem_registry) == 0
        assert engine.circuit_builder is not None
    
    def test_solve_optimization_problem(self):
        """Test solving optimization problem."""
        engine = QuantumOptimizationEngine()
        
        problem_data = {"constraint_1": 10, "constraint_2": 20}
        result = engine.solve_optimization_problem(
            problem_type=OptimizationProblem.DRUG_DISCOVERY,
            problem_data=problem_data,
            num_qubits=6,
            num_layers=2
        )
        
        assert result.problem_type == OptimizationProblem.DRUG_DISCOVERY
        assert result.convergence_reached is True
        assert 0.7 <= result.objective_value <= 0.95
        assert result.quantum_advantage > 1.0  # Should have speedup
        assert result.num_iterations > 0
    
    def test_optimize_drug_candidate(self):
        """Test drug candidate optimization."""
        engine = QuantumOptimizationEngine()
        
        candidate_properties = {
            "molecular_weight": 300,
            "lipophilicity": 2.5,
            "solubility": "high"
        }
        
        result = engine.optimize_drug_candidate(candidate_properties)
        
        assert result.problem_type == OptimizationProblem.DRUG_DISCOVERY
        assert result.convergence_reached is True
        assert result.quantum_advantage >= 2.0
    
    def test_optimize_treatment_plan(self):
        """Test treatment plan optimization."""
        engine = QuantumOptimizationEngine()
        
        patient_data = {"age": 65, "conditions": ["diabetes", "hypertension"]}
        treatment_options = ["medication_a", "medication_b", "combination"]
        
        result = engine.optimize_treatment_plan(patient_data, treatment_options)
        
        assert result.problem_type == OptimizationProblem.TREATMENT_PLANNING
        assert result.convergence_reached is True
        assert 'optimal_plan' in result.optimal_solution


class TestQuantumBenchmarkingSystem:
    """Tests for quantum benchmarking."""
    
    def test_system_initialization(self):
        """Test benchmarking system initialization."""
        system = QuantumBenchmarkingSystem()
        assert len(system.benchmarks) == 0
        assert len(system.roi_analyses) == 0
    
    def test_benchmark_quantum_vs_classical(self):
        """Test benchmarking quantum vs classical."""
        system = QuantumBenchmarkingSystem()
        
        benchmark = system.benchmark_quantum_vs_classical(
            problem_type="molecular_simulation",
            problem_size=100,
            quantum_backend=QuantumBackend.SIMULATOR
        )
        
        assert benchmark.problem_type == "molecular_simulation"
        assert benchmark.quantum_time_seconds > 0
        assert benchmark.classical_time_seconds > 0
        assert benchmark.speedup_factor > 0
        assert benchmark.quantum_accuracy > 0
        assert benchmark.classical_accuracy > 0
        assert 'roi' in vars(benchmark)
    
    def test_evaluate_roi(self):
        """Test ROI evaluation."""
        system = QuantumBenchmarkingSystem()
        
        roi_analysis = system.evaluate_roi(
            use_case="drug_discovery",
            annual_volume=1000,
            quantum_cost_per_run=1.0,
            classical_cost_per_run=2.0,
            speedup_factor=3.0
        )
        
        assert 'analysis_id' in roi_analysis
        assert roi_analysis['use_case'] == "drug_discovery"
        assert 'cost_savings_annual' in roi_analysis
        assert 'speedup_factor' in roi_analysis
        assert 'recommend_quantum' in roi_analysis
        assert 'payback_period_years' in roi_analysis
    
    def test_get_benchmark_summary(self):
        """Test getting benchmark summary."""
        system = QuantumBenchmarkingSystem()
        
        # Run some benchmarks
        for i in range(3):
            system.benchmark_quantum_vs_classical(
                f"problem_{i}",
                50 + i * 10,
                QuantumBackend.SIMULATOR
            )
        
        summary = system.get_benchmark_summary()
        
        assert summary['total_benchmarks'] == 3
        assert 'average_speedup' in summary
        assert 'average_quantum_accuracy' in summary
        assert 'average_classical_accuracy' in summary
        assert len(summary['problems_benchmarked']) == 3


class TestQuantumComputingSystem:
    """Tests for integrated quantum computing system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = create_quantum_computing_system(QuantumBackend.SIMULATOR)
        
        assert system.backend == QuantumBackend.SIMULATOR
        assert system.circuit_builder is not None
        assert system.hybrid_ml is not None
        assert system.molecular is not None
        assert system.optimization is not None
        assert system.benchmarking is not None
    
    def test_system_statistics(self):
        """Test comprehensive system statistics."""
        system = create_quantum_computing_system()
        
        stats = system.get_system_statistics()
        
        assert 'backend' in stats
        assert 'circuit_builder' in stats
        assert 'hybrid_ml' in stats
        assert 'molecular_simulation' in stats
        assert 'optimization' in stats
        assert 'benchmarking' in stats
        
        # Verify statistics structure
        assert 'total_circuits' in stats['circuit_builder']
        assert 'total_models' in stats['hybrid_ml']
        assert 'registered_molecules' in stats['molecular_simulation']
        assert 'total_optimizations' in stats['optimization']
        assert 'total_benchmarks' in stats['benchmarking']


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def test_drug_discovery_workflow(self):
        """Test complete drug discovery workflow."""
        system = create_quantum_computing_system()
        
        # 1. Register target molecule (protein)
        target = Molecule(
            molecule_id="target_protein",
            name="Target Protein",
            molecule_type=MoleculeType.PROTEIN,
            formula="Complex",
            num_atoms=1000,
            num_bonds=1050,
            molecular_weight=50000.0,
            structure_data={}
        )
        system.molecular.register_molecule(target)
        
        # 2. Register drug candidate
        drug = Molecule(
            molecule_id="drug_candidate",
            name="Drug Candidate X",
            molecule_type=MoleculeType.DRUG_COMPOUND,
            formula="C20H25N3O",
            num_atoms=49,
            num_bonds=51,
            molecular_weight=323.43,
            structure_data={}
        )
        system.molecular.register_molecule(drug)
        
        # 3. Simulate drug molecule
        simulation = system.molecular.simulate_ground_state("drug_candidate")
        assert simulation.convergence_achieved is True
        
        # 4. Predict binding affinity
        binding = system.molecular.predict_binding_affinity("drug_candidate", "target_protein")
        assert 'binding_affinity_nm' in binding
        
        # 5. Optimize drug structure
        optimization = system.molecular.optimize_molecule_structure("drug_candidate")
        assert optimization['converged'] is True
        
        # 6. Optimize for drug discovery problem
        drug_opt = system.optimization.optimize_drug_candidate({
            "molecular_weight": drug.molecular_weight,
            "binding_affinity": binding['binding_affinity_nm']
        })
        assert drug_opt.problem_type == OptimizationProblem.DRUG_DISCOVERY
    
    def test_quantum_ml_workflow(self):
        """Test quantum ML workflow."""
        system = create_quantum_computing_system()
        
        # 1. Create quantum circuit
        circuit = system.circuit_builder.create_circuit(
            CircuitType.VARIATIONAL,
            num_qubits=6,
            depth=4
        )
        assert circuit is not None
        
        # 2. Optimize circuit
        optimized_circuit = system.circuit_builder.optimize_circuit(circuit)
        assert optimized_circuit.depth < circuit.depth
        
        # 3. Train hybrid model
        model = system.hybrid_ml.train_hybrid_model(
            quantum_layers=3,
            classical_layers=2,
            training_samples=500,
            epochs=5
        )
        assert model.accuracy > 0.75
        
        # 4. Make predictions
        prediction = system.hybrid_ml.predict(model.model_id, {"data": "test"})
        assert 'confidence' in prediction
        
        # 5. Compare quantum vs classical
        comparison = system.hybrid_ml.compare_quantum_vs_classical(2, 2, 300)
        assert 'quantum_advantage' in comparison
    
    def test_optimization_workflow(self):
        """Test optimization workflow."""
        system = create_quantum_computing_system()
        
        # 1. Solve treatment planning problem
        treatment_result = system.optimization.optimize_treatment_plan(
            patient_data={"age": 70, "conditions": ["heart_disease"]},
            treatment_options=["med_a", "med_b", "surgery"]
        )
        assert treatment_result.convergence_reached is True
        
        # 2. Benchmark the optimization
        benchmark = system.benchmarking.benchmark_quantum_vs_classical(
            "treatment_optimization",
            problem_size=50,
            quantum_backend=system.backend
        )
        assert benchmark.speedup_factor > 0
        
        # 3. Evaluate ROI
        roi = system.benchmarking.evaluate_roi(
            use_case="treatment_planning",
            annual_volume=5000,
            quantum_cost_per_run=0.50,
            classical_cost_per_run=1.00,
            speedup_factor=benchmark.speedup_factor
        )
        assert 'recommend_quantum' in roi
    
    def test_benchmarking_workflow(self):
        """Test comprehensive benchmarking workflow."""
        system = create_quantum_computing_system()
        
        # Run multiple benchmarks
        problems = [
            ("molecular_simulation", 100),
            ("optimization", 80),
            ("ml_training", 120)
        ]
        
        for problem_type, size in problems:
            benchmark = system.benchmarking.benchmark_quantum_vs_classical(
                problem_type,
                size,
                QuantumBackend.SIMULATOR
            )
            assert benchmark.speedup_factor > 0
        
        # Get summary
        summary = system.benchmarking.get_benchmark_summary()
        assert summary['total_benchmarks'] == 3
        assert summary['average_speedup'] > 0
        
        # Evaluate ROI for best performing
        roi = system.benchmarking.evaluate_roi(
            "best_use_case",
            annual_volume=10000,
            quantum_cost_per_run=0.20,
            classical_cost_per_run=0.50,
            speedup_factor=summary['average_speedup']
        )
        assert 'cost_savings_annual' in roi


class TestPerformanceMetrics:
    """Tests for performance and accuracy metrics."""
    
    def test_circuit_depth_reduction(self):
        """Test circuit depth reduction through optimization."""
        builder = QuantumCircuitBuilder()
        
        original = builder.create_circuit(CircuitType.QAOA, 8, 6)
        optimized = builder.optimize_circuit(original)
        
        reduction = original.depth - optimized.depth
        assert reduction > 0
        assert reduction >= original.depth * 0.15  # At least 15% reduction
    
    def test_quantum_advantage_speedup(self):
        """Test quantum advantage provides speedup."""
        engine = HybridQuantumMLEngine()
        
        comparison = engine.compare_quantum_vs_classical(3, 2, 1000)
        
        # Quantum should have some advantage
        assert comparison['quantum_advantage']['speedup_training'] > 0
        assert comparison['quantum_advantage']['speedup_inference'] > 0
    
    def test_molecular_simulation_accuracy(self):
        """Test molecular simulation achieves convergence."""
        engine = MolecularSimulationEngine()
        
        molecule = Molecule(
            molecule_id="accuracy_test",
            name="Test Molecule",
            molecule_type=MoleculeType.DRUG_COMPOUND,
            formula="C6H6",
            num_atoms=12,
            num_bonds=12,
            molecular_weight=78.11,
            structure_data={}
        )
        engine.register_molecule(molecule)
        
        result = engine.simulate_ground_state("accuracy_test")
        
        assert result.convergence_achieved is True
        assert result.iterations > 0
        assert result.ground_state_energy < 0  # Physical constraint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
