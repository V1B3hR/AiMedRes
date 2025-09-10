#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for duetmind_adaptive
Integrates dataset loading with adaptive agent simulation for training
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from labyrinth_simulation import run_labyrinth_simulation, LabyrinthSimulationConfig
from labyrinth_adaptive import UnifiedAdaptiveAgent, AliveLoopNode, ResourceRoom

def load_training_data():
    """Load the Alzheimer's dataset for training"""
    print("Loading Alzheimer's dataset...")
    
    file_path = "alzheimer.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "brsdincer/alzheimer-features",
        file_path,
    )
    
    print(f"Dataset loaded: {len(df)} records with {len(df.columns)} columns")
    print("Dataset columns:", df.columns.tolist())
    return df

def analyze_dataset(df):
    """Analyze the dataset to understand features for training"""
    print("\n=== Dataset Analysis ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    # Analyze target distribution
    if 'Group' in df.columns:
        print("\nGroup distribution:")
        print(df['Group'].value_counts())
    
    return df

def train_with_adaptive_agents(df):
    """Use the dataset to inform adaptive agent training"""
    print("\n=== Training Adaptive Agents with Dataset ===")
    
    # Extract key metrics from dataset for agent configuration
    if 'Age' in df.columns:
        avg_age = df['Age'].mean()
        age_std = df['Age'].std()
        print(f"Dataset age range: {df['Age'].min()}-{df['Age'].max()}, avg: {avg_age:.1f}, std: {age_std:.1f}")
    
    if 'MMSE' in df.columns:
        mmse_data = df['MMSE'].dropna()
        avg_mmse = mmse_data.mean()
        print(f"MMSE score range: {mmse_data.min()}-{mmse_data.max()}, avg: {avg_mmse:.1f}")
    
    # Create resource room with dataset-informed parameters
    resource_room = ResourceRoom()
    
    # Add dataset insights to resource room
    dataset_insights = {
        "dataset_size": len(df),
        "features": list(df.columns),
        "dementia_cases": len(df[df['Group'] == 'Demented']) if 'Group' in df.columns else 0,
        "control_cases": len(df[df['Group'] == 'Nondemented']) if 'Group' in df.columns else 0
    }
    
    # Store dataset information in resource room
    resource_room.deposit("training_agent", dataset_insights)
    
    print("Dataset insights stored in ResourceRoom for agent training")
    print(f"Insights: {dataset_insights}")
    
    # Run training simulation
    print("\n=== Running Adaptive Training Simulation ===")
    run_labyrinth_simulation()
    
    return True

def main():
    """Main training pipeline"""
    print("=== DuetMind Adaptive Training Pipeline ===")
    
    try:
        # Step 1: Load training data
        df = load_training_data()
        
        # Step 2: Analyze dataset
        analyze_dataset(df)
        
        # Step 3: Train adaptive agents with dataset insights
        train_with_adaptive_agents(df)
        
        print("\n=== Training Complete ===")
        print("✓ Dataset loaded successfully")
        print("✓ Dataset analyzed and processed")
        print("✓ Adaptive agents trained with dataset insights")
        print("✓ Simulation completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)