"""
Data loader abstractions for duetmind_adaptive
Provides mockable interfaces for different data sources
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data and return as DataFrame"""
        pass
    
    @abstractmethod  
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that loaded data meets expected schema"""
        pass


class KaggleDataLoader(DataLoader):
    """Concrete implementation for loading Kaggle datasets"""
    
    def __init__(self, dataset_name: str, file_path: str):
        self.dataset_name = dataset_name
        self.file_path = file_path
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from Kaggle using kagglehub"""
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            logger.info(f"Loading Kaggle dataset: {self.dataset_name}/{self.file_path}")
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                self.dataset_name,
                self.file_path,
                **kwargs
            )
            
            if not self.validate_data(df):
                raise ValueError("Loaded data does not match expected schema")
                
            logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except ImportError:
            raise ImportError("kagglehub is not available. Please install with: pip install kagglehub")
        except Exception as e:
            logger.error(f"Failed to load Kaggle dataset: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate Kaggle dataset structure"""
        if df is None or df.empty:
            return False
        return len(df.columns) > 0 and len(df) > 0


class CSVDataLoader(DataLoader):
    """Concrete implementation for loading CSV files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            logger.info(f"Loading CSV file: {self.file_path}")
            df = pd.read_csv(self.file_path, **kwargs)
            
            if not self.validate_data(df):
                raise ValueError("Loaded data does not match expected schema")
                
            logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate CSV data structure"""
        if df is None or df.empty:
            return False
        return len(df.columns) > 0 and len(df) > 0


class MockDataLoader(DataLoader):
    """Mock implementation for testing"""
    
    def __init__(self, mock_data: Optional[pd.DataFrame] = None):
        self.mock_data = mock_data
        self._load_called = False
        self._validate_called = False
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Return mock data"""
        self._load_called = True
        if self.mock_data is not None:
            return self.mock_data.copy()
        
        # Default mock data for Alzheimer prediction
        data = {
            'age': [65, 72, 58],
            'gender': ['M', 'F', 'M'],
            'education_level': [16, 12, 18],
            'mmse_score': [28, 24, 29],
            'cdr_score': [0.0, 0.5, 0.0],
            'apoe_genotype': ['E3/E3', 'E3/E4', 'E2/E3'],
            'diagnosis': ['Normal', 'MCI', 'Normal']
        }
        return pd.DataFrame(data)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Mock validation always returns True"""
        self._validate_called = True
        return df is not None and not df.empty
    
    @property
    def load_called(self) -> bool:
        """Check if load_data was called"""
        return self._load_called
    
    @property
    def validate_called(self) -> bool:
        """Check if validate_data was called"""
        return self._validate_called


def create_data_loader(loader_type: str, **config) -> DataLoader:
    """Factory function to create appropriate data loader"""
    if loader_type == "kaggle":
        return KaggleDataLoader(
            dataset_name=config["dataset_name"],
            file_path=config["file_path"]
        )
    elif loader_type == "csv":
        return CSVDataLoader(file_path=config["file_path"])
    elif loader_type == "mock":
        return MockDataLoader(mock_data=config.get("mock_data"))
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")