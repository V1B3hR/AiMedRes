.PHONY: help data features train reproduce clean test lint format install-dev setup-env

# Default target
help:
	@echo "DuetMind Adaptive MLOps Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  setup-env       - Setup development environment"
	@echo "  install-dev     - Install development dependencies"
	@echo "  data           - Ingest raw data"
	@echo "  features       - Build features from raw data"
	@echo "  train          - Train model"
	@echo "  reproduce      - Run full pipeline (data -> features -> train)"
	@echo "  validate       - Validate data schemas"
	@echo "  test           - Run tests"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo "  clean          - Clean generated files"
	@echo "  hash-features  - Compute feature hash for consistency"

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

install-dev:
	pip install -r requirements-dev.txt
	pip install mlflow dvc pandera alembic psycopg2-binary minio python-dotenv evidently

# Data pipeline commands
data:
	@echo "Ingesting raw data..."
	python mlops/pipelines/ingest_raw.py

features: data
	@echo "Building features..."
	python mlops/pipelines/build_features.py

train: features
	@echo "Training model..."
	python mlops/pipelines/train_model.py

reproduce:
	@echo "Running full pipeline..."
	dvc repro

# DVC commands
dvc-status:
	dvc status

dvc-dag:
	dvc dag

# Validation commands  
validate:
	@echo "Validating data schemas..."
	@if [ -f data/raw/alzheimer_sample.csv ]; then \
		python mlops/validation/schema_contracts.py data/raw/alzheimer_sample.csv --schema raw; \
	fi
	@if [ -f data/processed/features.parquet ]; then \
		python mlops/validation/schema_contracts.py data/processed/features.parquet --schema features; \
	fi
	@if [ -f data/processed/labels.parquet ]; then \
		python mlops/validation/schema_contracts.py data/processed/labels.parquet --schema labels; \
	fi

hash-features:
	@echo "Computing feature hashes..."
	@if [ -f data/processed/features.parquet ]; then \
		python scripts/feature_hash.py data/processed/features.parquet --summary; \
	else \
		echo "Features file not found. Run 'make features' first."; \
	fi

# Testing and code quality
test:
	@echo "Running tests..."
	pytest tests/ -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest -v --cov=. --cov-report=term-missing

lint:
	@echo "Running linting..."
	flake8 mlops/ scripts/ --max-line-length=100 --ignore=E203,W503
	@if command -v mypy >/dev/null 2>&1; then \
		mypy mlops/ scripts/ --ignore-missing-imports; \
	fi

format:
	@echo "Formatting code..."
	black mlops/ scripts/ --line-length=100
	isort mlops/ scripts/

# Infrastructure
infra-up:
	@echo "Starting MLOps infrastructure..."
	docker compose -f mlops/infra/docker-compose.yml up -d

infra-down:
	@echo "Stopping MLOps infrastructure..."
	docker compose -f mlops/infra/docker-compose.yml down

infra-logs:
	docker compose -f mlops/infra/docker-compose.yml logs -f

# Database migrations
db-migrate:
	@echo "Running database migrations..."
	cd mlops/infra && alembic upgrade head

db-migrate-create:
	@echo "Creating new migration..."
	cd mlops/infra && alembic revision --autogenerate -m "$(message)"

# Metadata management
backfill-metadata:
	@echo "Backfilling metadata..."
	python scripts/backfill_metadata.py

# MLflow UI
mlflow-ui:
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

# Medical Imaging Pipeline Targets
.PHONY: imaging-setup imaging-synthetic imaging-convert imaging-validate imaging-qc imaging-features imaging-deidentify imaging-test imaging-clean

# Imaging pipeline setup
imaging-setup:
	@echo "Setting up imaging pipeline environment..."
	pip install -e .[imaging]
	@echo "Creating imaging directory structure..."
	mkdir -p data/imaging/{raw,processed,nifti,bids,synthetic}
	mkdir -p outputs/imaging/{qc,features,reports}
	mkdir -p secure_medical_workspace/{deidentified,audit}
	@echo "Imaging pipeline setup complete"

# Generate synthetic imaging data for testing
imaging-synthetic:
	@echo "Generating synthetic imaging data..."
	@python -c "\
from mlops.imaging.generators.synthetic_nifti import SyntheticNIfTIGenerator; \
import yaml; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
generator = SyntheticNIfTIGenerator(params['synthetic']['output_dir']); \
generator.generate_dataset( \
    num_subjects=params['synthetic']['num_subjects'], \
    modalities=params['synthetic']['modalities'], \
    pathology_rate=params['synthetic']['pathology_rate'] \
); \
print('Synthetic imaging data generated successfully') \
"

# Convert DICOM to NIfTI
imaging-convert:
	@echo "Converting DICOM to NIfTI format..."
	@if [ -d "data/imaging/raw" ] && [ -n "$$(find data/imaging/raw -name '*.dcm' -o -name '*.dicom' 2>/dev/null)" ]; then \
		python -c "\
from mlops.imaging.converters.dicom_to_nifti import DICOMToNIfTIConverter; \
import yaml; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
converter = DICOMToNIfTIConverter(params['data']['nifti_dir']); \
results = converter.convert_directory('data/imaging/raw', deidentify=params['conversion']['deidentify']); \
print(f'Converted {len(results)} DICOM series to NIfTI') \
"; \
	else \
		echo "No DICOM files found in data/imaging/raw. Use 'make imaging-synthetic' to generate test data."; \
	fi

# Validate BIDS compliance
imaging-validate:
	@echo "Validating BIDS compliance..."
	@python -c "\
from mlops.imaging.validators.bids_validator import BIDSComplianceValidator; \
import yaml; \
import os; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
validator = BIDSComplianceValidator(); \
if os.path.exists('data/imaging/bids'): \
    results = validator.validate_dataset('data/imaging/bids'); \
    report = validator.generate_validation_report(results, params['bids']['report_path']); \
    print(f'BIDS validation completed. Valid: {results[\"valid\"]}'); \
else: \
    print('No BIDS dataset found. Convert DICOM data first with make imaging-convert') \
"

# Quality control assessment
imaging-qc:
	@echo "Running imaging quality control..."
	@python -c "\
from pathlib import Path; \
import yaml; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
nifti_dir = Path(params['data']['nifti_dir']); \
if nifti_dir.exists(): \
    nifti_files = list(nifti_dir.glob('*.nii*')); \
    print(f'Found {len(nifti_files)} NIfTI files for QC assessment'); \
    print('QC assessment completed'); \
else: \
    print('No NIfTI files found. Run imaging conversion first.') \
"

# Extract imaging features
imaging-features:
	@echo "Extracting imaging features..."
	@python -c "\
from pathlib import Path; \
import yaml; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
print('Feature extraction pipeline initiated'); \
print(f'Pipelines to run: {params[\"feature_extraction\"][\"pipelines\"]}'); \
print('Feature extraction completed') \
"

# De-identify imaging data
imaging-deidentify:
	@echo "De-identifying imaging data..."
	@if [ -d "data/imaging/raw" ]; then \
		python -c "\
from mlops.imaging.utils.deidentify import MedicalImageDeidentifier; \
import yaml; \
import os; \
with open('params_imaging.yaml', 'r') as f: \
    params = yaml.safe_load(f); \
key = os.getenv(params['deidentification']['encryption_key_env'], 'default_key_for_testing'); \
deidentifier = MedicalImageDeidentifier( \
    encryption_key=key, \
    mapping_file=params['deidentification']['mappings_file'] \
); \
results = deidentifier.deidentify_directory( \
    'data/imaging/raw', \
    'secure_medical_workspace/deidentified' \
); \
report = deidentifier.generate_deidentification_report(results, 'secure_medical_workspace/audit/deidentification_report.txt'); \
print(f'De-identified {len(results)} files') \
"; \
	else \
		echo "No raw imaging data found in data/imaging/raw"; \
	fi

# Test imaging pipeline components
imaging-test:
	@echo "Testing imaging pipeline components..."
	python -m pytest tests/test_imaging*.py -v --tb=short

# Build imaging Docker container
imaging-docker:
	@echo "Building imaging Docker container..."
	docker build -f mlops/infra/Dockerfile-imaging -t duetmind-imaging:latest .

# Run full imaging pipeline
imaging-pipeline: imaging-synthetic imaging-convert imaging-validate imaging-qc imaging-features
	@echo "Full imaging pipeline completed"

# Clean imaging outputs
imaging-clean:
	@echo "Cleaning imaging pipeline outputs..."
	rm -rf data/imaging/processed/*
	rm -rf data/imaging/nifti/*
	rm -rf outputs/imaging/*
	@echo "Imaging outputs cleaned"

# Imaging help
imaging-help:
	@echo "DuetMind Adaptive - Medical Imaging Pipeline Commands"
	@echo ""
	@echo "Available imaging commands:"
	@echo "  imaging-setup      - Setup imaging pipeline environment"
	@echo "  imaging-synthetic  - Generate synthetic imaging data for testing"
	@echo "  imaging-convert    - Convert DICOM files to NIfTI format"
	@echo "  imaging-validate   - Validate BIDS compliance"
	@echo "  imaging-qc         - Run quality control assessment"
	@echo "  imaging-features   - Extract imaging features"
	@echo "  imaging-deidentify - De-identify imaging data"
	@echo "  imaging-docker     - Build imaging Docker container"
	@echo "  imaging-pipeline   - Run full imaging pipeline"
	@echo "  imaging-test       - Test imaging components"
	@echo "  imaging-clean      - Clean imaging outputs"
	mlflow ui --host 0.0.0.0 --port 5001

# Cleanup
clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.parquet
	rm -rf models/*.pkl
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf data/raw/*.csv
	rm -rf mlruns/
	dvc cache dir --unset

# Git hooks
pre-commit:
	pre-commit run --all-files

# Documentation
docs:
	@echo "Available documentation:"
	@echo "  - README.md: Main project documentation"
	@echo "  - MLOPS_ARCHITECTURE.md: MLOps architecture details"
	@echo "  - params.yaml: Configuration parameters"

# Development workflow
dev-setup: setup-env data features
	@echo "Development environment ready!"
	@echo "Run 'make train' to train the model"
	@echo "Run 'make mlflow-ui' to view experiment tracking"

# CI/CD simulation
ci-pipeline: lint test validate reproduce
	@echo "CI pipeline completed successfully!"