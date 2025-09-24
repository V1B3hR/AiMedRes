"""
Unit tests for data loader abstractions
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import tempfile
import os

from scripts.data_loaders import (
    DataLoader, KaggleDataLoader, CSVDataLoader, MockDataLoader, 
    create_data_loader
)


class TestMockDataLoader:
    """Unit tests for MockDataLoader"""
    
    def test_initialization(self):
        """Test MockDataLoader initialization"""
        loader = MockDataLoader()
        assert loader.mock_data is None
        assert not loader.load_called
        assert not loader.validate_called
    
    def test_initialization_with_data(self, sample_alzheimer_data):
        """Test MockDataLoader with custom data"""
        loader = MockDataLoader(mock_data=sample_alzheimer_data)
        assert loader.mock_data is not None
        assert len(loader.mock_data) == len(sample_alzheimer_data)
    
    def test_load_data_default(self):
        """Test loading default mock data"""
        loader = MockDataLoader()
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'diagnosis' in df.columns
        assert loader.load_called
    
    def test_load_data_custom(self, sample_alzheimer_data):
        """Test loading custom mock data"""
        loader = MockDataLoader(mock_data=sample_alzheimer_data)
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_alzheimer_data)
        assert list(df.columns) == list(sample_alzheimer_data.columns)
        assert loader.load_called
    
    def test_validate_data(self):
        """Test data validation"""
        loader = MockDataLoader()
        
        # Valid data
        valid_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert loader.validate_data(valid_df)
        assert loader.validate_called
        
        # Invalid data (empty)
        loader._validate_called = False
        empty_df = pd.DataFrame()
        assert not loader.validate_data(empty_df)
        assert loader.validate_called


class TestCSVDataLoader:
    """Unit tests for CSVDataLoader"""
    
    def test_initialization(self):
        """Test CSVDataLoader initialization"""
        loader = CSVDataLoader("/path/to/file.csv")
        assert loader.file_path == "/path/to/file.csv"
    
    def test_load_data_success(self, temp_csv_file):
        """Test successful CSV loading"""
        loader = CSVDataLoader(str(temp_csv_file))
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'diagnosis' in df.columns
    
    def test_load_data_file_not_found(self):
        """Test CSV loading with non-existent file"""
        loader = CSVDataLoader("/nonexistent/file.csv")
        
        with pytest.raises(Exception):
            loader.load_data()
    
    def test_validate_data(self):
        """Test CSV data validation"""
        loader = CSVDataLoader("/dummy/path")
        
        # Valid data
        valid_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert loader.validate_data(valid_df)
        
        # Invalid data (empty)
        empty_df = pd.DataFrame()
        assert not loader.validate_data(empty_df)
        
        # Invalid data (None)
        assert not loader.validate_data(None)


class TestKaggleDataLoader:
    """Unit tests for KaggleDataLoader"""
    
    def test_initialization(self):
        """Test KaggleDataLoader initialization"""
        loader = KaggleDataLoader("dataset/name", "file.csv")
        assert loader.dataset_name == "dataset/name"
        assert loader.file_path == "file.csv"
    
    @patch('builtins.__import__')
    def test_load_data_success(self, mock_import, sample_alzheimer_data):
        """Test successful Kaggle loading with mocked kagglehub"""
        # Setup mock kagglehub module
        mock_kagglehub = MagicMock()
        mock_kagglehub.load_dataset.return_value = sample_alzheimer_data
        mock_kagglehub.KaggleDatasetAdapter.PANDAS = "pandas"
        
        def mock_import_func(name, *args, **kwargs):
            if name == 'kagglehub':
                return mock_kagglehub
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_func
        
        loader = KaggleDataLoader("test/dataset", "test.csv")
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_alzheimer_data)
        mock_kagglehub.load_dataset.assert_called_once()
    
    @patch('builtins.__import__')
    def test_load_data_import_error(self, mock_import):
        """Test Kaggle loading when kagglehub is not available"""
        def mock_import_func(name, *args, **kwargs):
            if name == 'kagglehub':
                raise ImportError("No module named 'kagglehub'")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = mock_import_func
        
        loader = KaggleDataLoader("test/dataset", "test.csv")
        
        with pytest.raises(ImportError, match="kagglehub is not available"):
            loader.load_data()
    
    def test_validate_data(self):
        """Test Kaggle data validation"""
        loader = KaggleDataLoader("test/dataset", "test.csv")
        
        # Valid data
        valid_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        assert loader.validate_data(valid_df)
        
        # Invalid data (empty)
        empty_df = pd.DataFrame()
        assert not loader.validate_data(empty_df)


class TestDataLoaderFactory:
    """Unit tests for data loader factory function"""
    
    def test_create_kaggle_loader(self):
        """Test creating Kaggle loader"""
        loader = create_data_loader(
            "kaggle", 
            dataset_name="test/dataset", 
            file_path="test.csv"
        )
        assert isinstance(loader, KaggleDataLoader)
        assert loader.dataset_name == "test/dataset"
        assert loader.file_path == "test.csv"
    
    def test_create_csv_loader(self):
        """Test creating CSV loader"""
        loader = create_data_loader("csv", file_path="/path/to/file.csv")
        assert isinstance(loader, CSVDataLoader)
        assert loader.file_path == "/path/to/file.csv"
    
    def test_create_mock_loader(self, sample_alzheimer_data):
        """Test creating mock loader"""
        loader = create_data_loader("mock", mock_data=sample_alzheimer_data)
        assert isinstance(loader, MockDataLoader)
        assert loader.mock_data is not None
    
    def test_create_unknown_loader(self):
        """Test creating unknown loader type"""
        with pytest.raises(ValueError, match="Unknown loader type"):
            create_data_loader("unknown_type")