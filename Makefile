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
	pytest tests/ -v --cov=. --cov-report=html

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
	@echo "  - docs/MLOPS_ARCHITECTURE.md: MLOps architecture details"
	@echo "  - params.yaml: Configuration parameters"

# Development workflow
dev-setup: setup-env data features
	@echo "Development environment ready!"
	@echo "Run 'make train' to train the model"
	@echo "Run 'make mlflow-ui' to view experiment tracking"

# CI/CD simulation
ci-pipeline: lint test validate reproduce
	@echo "CI pipeline completed successfully!"