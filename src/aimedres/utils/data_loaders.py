#!/usr/bin/env python3
"""
Data Loaders for AiMedRes
Base classes and utilities for loading various types of medical data
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("DataLoaders")


class DataLoader(ABC):
    """Abstract base class for data loaders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loaded_data = {}
        
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source"""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data"""
        if data is None or data.empty:
            logger.warning("Data is empty or None")
            return False
        return True