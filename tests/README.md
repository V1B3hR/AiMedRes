# DuetMind Adaptive Test Suite

This directory contains a comprehensive, structured test suite for the duetmind_adaptive project using pytest.

## Test Organization

The tests are organized into three main categories:

### Unit Tests (`unit/`)
Tests for individual components in isolation:
- `test_data_loaders.py` - Tests for data loader abstractions (MockDataLoader, CSVDataLoader, KaggleDataLoader)
- `test_training.py` - Tests for training module components (AlzheimerTrainer, TrainingConfig)

### Integration Tests (`integration/`)
Tests for component interactions and end-to-end workflows:
- `test_training_pipeline.py` - Tests for complete training workflows and data loader integration

### Regression Tests (`regression/`)
Tests to ensure backwards compatibility and prevent regressions:
- `test_backwards_compatibility.py` - Tests to ensure existing APIs continue to work

## Test Configuration

- `conftest.py` - Pytest fixtures and configuration
- `pytest.ini` - Pytest configuration in project root
- `__init__.py` - Package initialization files

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only  
pytest tests/integration/

# Regression tests only
pytest tests/regression/
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

## Key Features

### Deterministic Testing
All tests use fixed random seeds to ensure reproducible results across runs.

### Mockable Architecture
The data loader abstraction allows for easy mocking of external dependencies like Kaggle API.

### Comprehensive Fixtures
Common test data and configurations are provided through pytest fixtures in `conftest.py`.

### Backwards Compatibility
Regression tests ensure that existing functionality continues to work while new features are added.

## Test Data

Test fixtures provide:
- Sample Alzheimer dataset for training tests
- Temporary CSV files for file I/O tests  
- Mock data loaders for isolated testing
- Consistent random seeds for deterministic behavior

## Adding New Tests

When adding new functionality:

1. **Unit tests** - Add to appropriate file in `unit/` directory
2. **Integration tests** - Add end-to-end tests to `integration/`
3. **Regression tests** - Add compatibility tests to `regression/`
4. **Fixtures** - Add reusable test data/mocks to `conftest.py`

Follow the existing naming conventions and test structure for consistency.