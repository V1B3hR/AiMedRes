#!/usr/bin/env python3
"""
Single-File Federated Learning Module for AiMedRes.

This module provides a complete, asynchronous framework for orchestrating
federated learning rounds. It is designed as a core component of the AiMedRes
platform to enable collaborative model training across multiple institutions
without sharing sensitive patient data.
"""

import asyncio
import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

# ==============================================================================
# 1. CONFIGURATION (Simulating configs/federated_learning_config.yaml)
# ==============================================================================

FEDERATED_LEARNING_CONFIG = {
    "aggregation_strategy": "federated_averaging",
    "min_clients_for_aggregation": 3,
    "round_timeout_seconds": 3600,  # 1 hour
    "model_id": "alzheimer_prediction_v3",
}

# ==============================================================================
# 2. DATA MODELS (Pydantic for robust, validated data exchange)
# ==============================================================================

class ModelWeights(BaseModel):
    """A container for model weights, designed for serialization."""
    tensors: Dict[str, List[float]] = Field(..., description="Model weights/tensors, flattened for JSON serialization.")
    tensor_shapes: Dict[str, List[int]] = Field(..., description="Original shapes of the tensors.")

    @classmethod
    def from_numpy(cls, numpy_weights: Dict[str, np.ndarray]) -> 'ModelWeights':
        """Creates a ModelWeights object from a dictionary of NumPy arrays."""
        return cls(
            tensors={name: arr.flatten().tolist() for name, arr in numpy_weights.items()},
            tensor_shapes={name: list(arr.shape) for name, arr in numpy_weights.items()}
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Converts the ModelWeights object back to a dictionary of NumPy arrays."""
        return {
            name: np.array(flat_tensor).reshape(self.tensor_shapes[name])
            for name, flat_tensor in self.tensors.items()
        }

class GlobalModelState(BaseModel):
    """Represents the state of the global model on the central server."""
    model_id: str
    model_version: int = Field(..., ge=0)
    weights: ModelWeights
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ClientUpdate(BaseModel):
    """Represents a model update submitted by a single FL client."""
    client_id: str
    model_id: str
    base_model_version: int
    weight_update: ModelWeights
    num_samples: int = Field(..., gt=0, description="Number of samples used for training.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Client-side training metrics (e.g., loss, accuracy).")

# ==============================================================================
# 3. DECOUPLED DEPENDENCIES (Protocols for storage and queuing)
# ==============================================================================

class ModelStore(ABC):
    """Abstract protocol for persisting and retrieving global model states."""
    @abstractmethod
    async def save_model(self, model_state: GlobalModelState):
        pass

    @abstractmethod
    async def get_latest_model(self, model_id: str) -> Optional[GlobalModelState]:
        pass

class UpdateQueue(ABC):
    """Abstract protocol for queuing client updates for aggregation."""
    @abstractmethod
    async def submit_update(self, round_id: str, update: ClientUpdate):
        pass

    @abstractmethod
    async def get_updates_for_round(self, round_id: str) -> List[ClientUpdate]:
        pass

# --- Concrete implementations for demonstration ---

class InMemoryModelStore(ModelStore):
    """Simple in-memory model store for demonstration."""
    def __init__(self):
        self._store: Dict[str, GlobalModelState] = {}

    async def save_model(self, model_state: GlobalModelState):
        key = f"{model_state.model_id}_v{model_state.model_version}"
        self._store[key] = model_state
        logging.info(f"Saved model {key} to in-memory store.")

    async def get_latest_model(self, model_id: str) -> Optional[GlobalModelState]:
        versions = [ms for ms in self._store.values() if ms.model_id == model_id]
        if not versions:
            return None
        return max(versions, key=lambda ms: ms.model_version)

class InMemoryUpdateQueue(UpdateQueue):
    """Simple in-memory update queue for demonstration."""
    def __init__(self):
        self._queue: Dict[str, List[ClientUpdate]] = defaultdict(list)

    async def submit_update(self, round_id: str, update: ClientUpdate):
        self._queue[round_id].append(update)
        logging.info(f"Client '{update.client_id}' submitted update for round '{round_id}'.")

    async def get_updates_for_round(self, round_id: str) -> List[ClientUpdate]:
        return self._queue.pop(round_id, [])

# ==============================================================================
# 4. AGGREGATION STRATEGY PATTERN (For algorithmic flexibility)
# ==============================================================================

class AggregationStrategy(ABC):
    """Abstract protocol for model aggregation algorithms."""
    @abstractmethod
    def aggregate(self, base_weights: Dict[str, np.ndarray], updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        pass

class FederatedAveraging(AggregationStrategy):
    """Implements the standard Federated Averaging (FedAvg) algorithm."""
    def aggregate(self, base_weights: Dict[str, np.ndarray], updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        if not updates:
            return base_weights

        total_samples = sum(update.num_samples for update in updates)
        if total_samples == 0:
            return base_weights

        # Initialize aggregated weights with zeros
        aggregated_weights = {name: np.zeros_like(arr) for name, arr in base_weights.items()}

        # Perform weighted average of weight updates
        for update in updates:
            client_weights = update.weight_update.to_numpy()
            weighting_factor = update.num_samples / total_samples
            for name in aggregated_weights.keys():
                aggregated_weights[name] += client_weights[name] * weighting_factor
        
        return aggregated_weights

# ==============================================================================
# 5. MAIN MODULE FACADE (The primary entry point for AiMedRes)
# ==============================================================================

class FederatedLearningModule:
    """
    Orchestrates the entire federated learning process, managing rounds,
    clients, and model aggregation.
    """
    def __init__(self, config: Dict[str, Any], model_store: ModelStore, update_queue: UpdateQueue):
        self.config = config
        self.model_store = model_store
        self.update_queue = update_queue
        self.model_id = config["model_id"]
        self.min_clients = config["min_clients_for_aggregation"]
        
        strategy_map = {"federated_averaging": FederatedAveraging}
        strategy_name = config["aggregation_strategy"]
        self.aggregator = strategy_map[strategy_name]()
        
        self.current_round_id: Optional[str] = None
        self.round_updates: Dict[str, List[ClientUpdate]] = defaultdict(list)
        logging.info(f"FederatedLearningModule initialized for model '{self.model_id}'.")

    async def get_latest_global_model(self) -> Optional[GlobalModelState]:
        """Provides the latest version of the global model for clients to download."""
        return await self.model_store.get_latest_model(self.model_id)

    async def submit_client_update(self, update: ClientUpdate):
        """Accepts and queues a trained model update from a client."""
        latest_model = await self.get_latest_global_model()
        if not latest_model or update.base_model_version != latest_model.model_version:
            raise ValueError("Client update is based on an outdated model version.")
        
        round_id = f"round_v{latest_model.model_version}"
        self.round_updates[round_id].append(update)
        logging.info(f"Update from client '{update.client_id}' received for round '{round_id}'. "
                     f"Total updates for round: {len(self.round_updates[round_id])}")

    async def run_aggregation_round(self) -> Optional[GlobalModelState]:
        """
        Triggers the aggregation process for the current round if enough
        client updates have been received.
        """
        latest_model = await self.get_latest_global_model()
        if not latest_model:
            logging.warning("Aggregation skipped: No base model found.")
            return None

        round_id = f"round_v{latest_model.model_version}"
        updates_for_round = self.round_updates.get(round_id, [])

        if len(updates_for_round) < self.min_clients:
            logging.info(f"Aggregation skipped for round '{round_id}': "
                         f"Have {len(updates_for_round)}/{self.min_clients} required updates.")
            return None

        logging.info(f"Starting aggregation for round '{round_id}' with {len(updates_for_round)} updates.")
        
        # Perform aggregation
        base_weights_np = latest_model.weights.to_numpy()
        aggregated_weights_np = self.aggregator.aggregate(base_weights_np, updates_for_round)
        
        # Create and save the new global model state
        new_model_state = GlobalModelState(
            model_id=self.model_id,
            model_version=latest_model.model_version + 1,
            weights=ModelWeights.from_numpy(aggregated_weights_np),
            metadata={
                "aggregated_from_clients": [u.client_id for u in updates_for_round],
                "num_updates": len(updates_for_round),
                "avg_client_loss": np.mean([u.metrics.get("loss", 0) for u in updates_for_round])
            }
        )
        
        await self.model_store.save_model(new_model_state)
        
        # Clean up processed updates
        del self.round_updates[round_id]
        
        logging.info(f"Aggregation complete. New global model version: {new_model_state.model_version}.")
        return new_model_state

# ==============================================================================
# 6. EXAMPLE USAGE (Simulating a full federated learning round)
# ==============================================================================

async def main():
    """Demonstrates the setup and execution of a federated learning round."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("--- 1. Initializing Dependencies and Module ---")
    model_store = InMemoryModelStore()
    update_queue = InMemoryUpdateQueue() # Note: The current facade uses an in-memory dict, not this queue.
    fl_module = FederatedLearningModule(FEDERATED_LEARNING_CONFIG, model_store, update_queue)

    print("\n--- 2. Creating Genesis (Initial) Global Model ---")
    # In a real scenario, this would be a pre-trained model.
    initial_weights_np = {
        "layer1.weights": np.random.randn(128, 64).astype(np.float32),
        "layer1.bias": np.zeros(64).astype(np.float32),
    }
    genesis_model = GlobalModelState(
        model_id=fl_module.model_id,
        model_version=0,
        weights=ModelWeights.from_numpy(initial_weights_np)
    )
    await model_store.save_model(genesis_model)
    print(f"Genesis model v{genesis_model.model_version} created.")

    print("\n--- 3. Simulating Client Actions ---")
    client_ids = ["hospital-A", "clinic-B", "research-C", "lab-D"]
    for client_id in client_ids:
        # a. Client downloads the latest global model
        global_model = await fl_module.get_latest_global_model()
        print(f"Client '{client_id}' downloaded global model v{global_model.model_version}.")
        
        # b. Client simulates local training (e.g., adding small random noise)
        local_weights_np = global_model.weights.to_numpy()
        for name in local_weights_np:
            local_weights_np[name] += np.random.randn(*local_weights_np[name].shape).astype(np.float32) * 0.01

        # c. Client creates and submits an update
        client_update = ClientUpdate(
            client_id=client_id,
            model_id=fl_module.model_id,
            base_model_version=global_model.model_version,
            weight_update=ModelWeights.from_numpy(local_weights_np),
            num_samples=np.random.randint(100, 500),
            metrics={"loss": np.random.rand() * 0.5 + 0.1}
        )
        await fl_module.submit_client_update(client_update)

    print("\n--- 4. Attempting Aggregation (Might be skipped if not enough clients) ---")
    # This first attempt should fail because we only have 4 clients and min is 3, but we submitted 4.
    # Let's adjust the config for the demo to succeed.
    fl_module.min_clients = 3
    new_global_model = await fl_module.run_aggregation_round()

    if new_global_model:
        print(f"✅ SUCCESS: Aggregation complete. New global model is v{new_global_model.model_version}.")
        print(f"Metadata: {new_global_model.metadata}")
        
        # Verification: Check if the new model weights are different from the genesis model
        genesis_w = genesis_model.weights.to_numpy()["layer1.bias"]
        new_w = new_global_model.weights.to_numpy()["layer1.bias"]
        assert not np.array_equal(genesis_w, new_w), "New weights should be different!"
        print("Verification successful: New model weights have been updated.")
    else:
        print("❌ FAILURE: Aggregation did not run.")

if __name__ == "__main__":
    asyncio.run(main())
