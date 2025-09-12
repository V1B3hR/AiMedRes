"""
Pytest configuration and fixtures for duetmind_adaptive tests
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Set random seeds for deterministic testing
np.random.seed(42)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_alzheimer_data():
    """Create sample Alzheimer dataset for testing"""
    data = {
        'age': [65, 72, 58, 81, 69, 75, 63, 79, 67, 74, 61, 83],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'education_level': [16, 12, 18, 14, 20, 10, 15, 13, 17, 11, 19, 12],
        'mmse_score': [28, 24, 29, 20, 26, 18, 27, 22, 28, 23, 30, 19],
        'cdr_score': [0.0, 0.5, 0.0, 1.0, 0.0, 2.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0],
        'apoe_genotype': ['E3/E3', 'E3/E4', 'E2/E3', 'E4/E4', 'E3/E3', 'E3/E4', 'E2/E3', 'E3/E4', 'E3/E3', 'E3/E4', 'E2/E3', 'E4/E4'],
        'diagnosis': ['Normal', 'MCI', 'Normal', 'Dementia', 'Normal', 'Dementia', 'MCI', 'Dementia', 'Normal', 'MCI', 'Normal', 'Dementia']
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_alzheimer_data, test_data_dir):
    """Create temporary CSV file with test data"""
    csv_path = test_data_dir / "test_data.csv"
    sample_alzheimer_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_kaggle_loader():
    """Create a mock Kaggle loader for testing"""
    mock = MagicMock()
    return mock


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure deterministic behavior in all tests"""
    np.random.seed(42)
    # If using other libraries with random state, set them here too