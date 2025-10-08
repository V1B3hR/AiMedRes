#!/usr/bin/env python3
"""
Authoritative Labyrinth Simulation Module
=========================================

This module provides the consolidated and configurable labyrinth simulation functionality.
Replaces duplicate run_labyrinth_simulation functions from labyrinth_adaptive.py and neuralnet.py.

Features:
- Externalized configuration via JSON file and CLI arguments
- Consolidated single source of truth for simulation logic
- Configurable simulation parameters (steps, topics, thresholds)
"""

import json
import time
import logging
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import the core classes (keep using existing modules for classes)
from labyrinth_adaptive import (
    UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom, 
    NetworkMetrics, MazeMaster, CapacitorInSpace
)

logger = logging.getLogger("LabyrinthSimulation")

class LabyrinthSimulationConfig:
    """Configuration manager for labyrinth simulation parameters."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "labyrinth_config.json"
        self.config = self._load_default_config()
        if Path(self.config_file).exists():
            self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "simulation": {
                "steps": 20,
                "sleep_delay": 0.2,
                "topics": ["Find exit", "Share wisdom", "Collaborate"]
            },
            "maze_master": {
                "confusion_escape_thresh": 0.85,
                "entropy_escape_thresh": 1.5,
                "soft_advice_thresh": 0.65
            },
            "agents": [
                {
                    "name": "AgentA",
                    "style": {"logic": 0.8, "creativity": 0.5},
                    "position": [0, 0],
                    "velocity": [0.5, 0],
                    "energy": 15.0,
                    "node_id": 1
                },
                {
                    "name": "AgentB", 
                    "style": {"creativity": 0.9, "analytical": 0.7},
                    "position": [2, 0],
                    "velocity": [0, 0.5],
                    "energy": 12.0,
                    "node_id": 2
                },
                {
                    "name": "AgentC",
                    "style": {"logic": 0.6, "expressiveness": 0.8},
                    "position": [0, 2],
                    "velocity": [0.3, -0.2],
                    "energy": 10.0,
                    "node_id": 3
                }
            ],
            "capacitors": [
                {
                    "position": [1, 1],
                    "capacity": 8.0,
                    "initial_energy": 3.0
                }
            ]
        }
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                # Deep merge with defaults
                self._deep_merge(self.config, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_file}: {e}")
            logger.info("Using default configuration")
    
    def _deep_merge(self, default: Dict, override: Dict):
        """Deep merge override config into default config."""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._deep_merge(default[key], value)
            else:
                default[key] = value
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments."""
        if hasattr(args, 'steps') and args.steps:
            self.config['simulation']['steps'] = args.steps
        if hasattr(args, 'topics') and args.topics:
            self.config['simulation']['topics'] = args.topics
        if hasattr(args, 'confusion_thresh') and args.confusion_thresh:
            self.config['maze_master']['confusion_escape_thresh'] = args.confusion_thresh
        if hasattr(args, 'entropy_thresh') and args.entropy_thresh:
            self.config['maze_master']['entropy_escape_thresh'] = args.entropy_thresh
        if hasattr(args, 'advice_thresh') and args.advice_thresh:
            self.config['maze_master']['soft_advice_thresh'] = args.advice_thresh
    
    def get(self, path: str, default=None):
        """Get configuration value by dot-separated path."""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


def run_labyrinth_simulation(config: Optional[LabyrinthSimulationConfig] = None) -> Dict[str, Any]:
    """
    Run the unified adaptive labyrinth simulation with configurable parameters.
    
    Args:
        config: Configuration object. If None, loads from default config file.
    
    Returns:
        Dictionary containing simulation results
    """
    if config is None:
        config = LabyrinthSimulationConfig()
    
    logger.info("=== Unified Adaptive Labyrinth Simulation ===")
    
    # Initialize core components
    resource_room = ResourceRoom()
    
    # Create MazeMaster with configurable thresholds
    maze_master = MazeMaster(
        confusion_escape_thresh=config.get('maze_master.confusion_escape_thresh', 0.85),
        entropy_escape_thresh=config.get('maze_master.entropy_escape_thresh', 1.5),
        soft_advice_thresh=config.get('maze_master.soft_advice_thresh', 0.65)
    )
    metrics = NetworkMetrics()

    # Create agents from configuration
    agents = []
    for agent_config in config.get('agents', []):
        agent = UnifiedAdaptiveAgent(
            agent_config['name'], 
            agent_config['style'], 
            AliveLoopNode(
                tuple(agent_config['position']), 
                tuple(agent_config['velocity']), 
                agent_config['energy'], 
                node_id=agent_config['node_id']
            ), 
            resource_room
        )
        agents.append(agent)
    
    # Create capacitors from configuration
    capacitors = []
    for cap_config in config.get('capacitors', []):
        capacitor = CapacitorInSpace(
            tuple(cap_config['position']),
            capacity=cap_config['capacity'],
            initial_energy=cap_config['initial_energy']
        )
        capacitors.append(capacitor)

    # Get simulation parameters
    steps = config.get('simulation.steps', 20)
    topics = config.get('simulation.topics', ["Find exit", "Share wisdom", "Collaborate"])
    sleep_delay = config.get('simulation.sleep_delay', 0.2)
    
    logger.info(f"Starting simulation with {steps} steps, {len(agents)} agents, {len(capacitors)} capacitors")
    
    # Simulation results tracking
    results = {
        'total_steps': steps,
        'agent_count': len(agents),
        'capacitor_count': len(capacitors),
        'maze_master_interventions': 0,
        'final_health_score': 0.0,
        'config_used': config.config
    }
    
    # Main simulation loop
    for step in range(1, steps + 1):
        logger.info(f"\n--- Step {step} ---")
        for i, agent in enumerate(agents):
            topic = topics[step % len(topics)]
            agent.reason(f"{topic} at step {step}")
            agent.alive_node.move()
            if step % 5 == 0:
                agent.teleport_to_resource_room({"topic": topic, "step": step, "energy": agent.alive_node.energy})
                retrieved = agent.retrieve_from_resource_room()
        
        maze_master.govern_agents(agents)
        metrics.update(agents)
        health_score = metrics.health_score()
        
        logger.info(f"Network Health Score: {health_score}")
        for capacitor in capacitors:
            logger.info(capacitor.status())
        for agent in agents:
            logger.info(f"{agent.name} state: {agent.get_state()}")
        
        time.sleep(sleep_delay)
    
    # Record final results
    results['maze_master_interventions'] = maze_master.interventions
    results['final_health_score'] = metrics.health_score()

    logger.info("\n=== Simulation Complete ===")
    logger.info(f"Total MazeMaster interventions: {maze_master.interventions}")
    logger.info(f"Final health score: {results['final_health_score']}")
    
    return results


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for simulation configuration."""
    parser = argparse.ArgumentParser(
        description="Configurable Labyrinth Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        help='Number of simulation steps'
    )
    
    parser.add_argument(
        '--topics',
        nargs='+',
        help='List of topics for agent reasoning'
    )
    
    parser.add_argument(
        '--confusion-thresh',
        type=float,
        help='MazeMaster confusion escape threshold'
    )
    
    parser.add_argument(
        '--entropy-thresh',
        type=float,
        help='MazeMaster entropy escape threshold'
    )
    
    parser.add_argument(
        '--advice-thresh',
        type=float,
        help='MazeMaster soft advice threshold'
    )
    
    return parser


def main():
    """Main entry point for standalone simulation execution."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = LabyrinthSimulationConfig(args.config)
    config.update_from_args(args)
    
    # Run simulation
    results = run_labyrinth_simulation(config)
    
    print(f"\n=== Simulation Results ===")
    print(f"Steps completed: {results['total_steps']}")
    print(f"Agent count: {results['agent_count']}")
    print(f"MazeMaster interventions: {results['maze_master_interventions']}")
    print(f"Final health score: {results['final_health_score']}")


if __name__ == "__main__":
    main()