# Requirements

This document describes software, system, environment, and dependency requirements for developing, running, and optionally deploying the DuetMind Adaptive project.

Sources used:
- pyproject.toml
- setup.py
- requirements-dev.txt
- .env.example
- mlops/config/.env.example

## 1) Supported Platforms

- OS: Linux, macOS, Windows
- CPU: x86_64 (ARM64 should work for most Python wheels; some imaging extras may vary)
- GPU (optional): NVIDIA GPU with recent CUDA/cuDNN for acceleration of Sentence-Transformers; not required

## 2) Python Version

- Recommended: Python 3.10+
- Notes on repo metadata:
  - pyproject.toml: `requires-python = ">=3.10"`
  - setup.py: `python_requires=">=3.8"`
  To avoid conflicts, prefer Python 3.10 or newer.

## 3) Core Python Dependencies

From pyproject.toml:

- SQLAlchemy >= 2.0.20
- psycopg2-binary >= 2.9.9
- pgvector >= 0.2.5
- sentence-transformers >= 3.0.0
- PyYAML >= 6.0.1

These will be installed automatically when you install the package.

## 4) Optional Extras

Defined in setup.py:

- Visualization (`viz`):
  - matplotlib >= 3.4.0
  - plotly >= 5.0.0
- Web (`web`):
  - streamlit >= 1.0.0
- Imaging (`imaging`):
  - nibabel >= 5.0.0
  - pydicom >= 2.4.0
  - simpleitk >= 2.3.0
  - pyradiomics >= 3.0.0
  - nipype >= 1.8.0
  - nilearn >= 0.10.0
  - bids-validator >= 1.13.0
  - pybids >= 0.15.0

Install example:
```bash
pip install ".[viz]" ".[web]" ".[imaging]"
```

## 5) Development and QA Dependencies

You can use either the extras in setup.py or the pinned tools in requirements-dev.txt.

- requirements-dev.txt includes:
  - pytest >= 6.2.0
  - pytest-cov >= 2.12.0
  - pytest-mock >= 3.6.0
  - black >= 21.0.0
  - flake8 >= 3.9.0
  - isort >= 5.9.0
  - mypy >= 0.910

- pyproject.toml also defines newer “dev” optional deps:
  - pytest >= 8.2.0
  - black >= 24.8.0
  - isort >= 5.13.2
  - mypy >= 1.11.1

Recommendation:
- For a stable local setup, prefer the pinned set in requirements-dev.txt:
  ```bash
  pip install -r requirements-dev.txt
  ```
- Or use the extras:
  ```bash
  pip install ".[dev]"
  ```

## 6) System Services and External Tooling

- PostgreSQL (for application DB and MLflow tracking if used)
  - Ensure the `pgvector` extension is installed on the server.
  - Example (as a superuser):
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
- Git LFS (for large files, datasets, and imaging pointers)
  - Install Git LFS and run:
    ```bash
    git lfs install
    git lfs pull
    ```
- Optional MLOps stack (as referenced by mlops/config/.env.example):
  - MLflow
  - S3-compatible storage (e.g., MinIO) for artifacts
  - These are optional but required to run the full MLflow pipelines.

## 7) Environment Configuration

Create and populate a `.env` file from provided examples.

Application (.env.example):
- LOG_LEVEL=INFO
- EMBEDDING_MODEL=all-MiniLM-L6-v2
- EMBEDDING_DIM=384
- DATABASE_URL=postgresql://duetmind:duetmind_secret@localhost:5432/duetmind
- SENTENCE_TRANSFORMERS_HOME=.cache/sentence-transformers

MLOps (mlops/config/.env.example):
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SCHEMA
- MLFLOW_TRACKING_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT, MLFLOW_S3_ENDPOINT_URL
- MINIO endpoint and credentials (or alternative S3)
- DVC_REMOTE_URL, DVC_REMOTE_ENDPOINT_URL
- VECTOR_EMBEDDING_MODEL, VECTOR_DIMENSION (384)
- SECRET_KEY, JWT_SECRET
- ENVIRONMENT, DEBUG
- DRIFT_MONITORING_ENABLED, DRIFT_THRESHOLD, DRIFT_WINDOW_DAYS
- MODEL_REGISTRY_ENABLED, AUTO_MODEL_PROMOTION

Copy examples and adjust:
```bash
cp .env.example .env
cp mlops/config/.env.example mlops/config/.env
```

## 8) Installation

Using a virtual environment is recommended.

```bash
# Clone
git clone https://github.com/V1B3hR/duetmind_adaptive.git
cd duetmind_adaptive

# Python 3.10+ environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
pip install --upgrade pip
pip install -e .

# Optionally install extras
pip install ".[dev]" ".[viz]" ".[web]" ".[imaging]"
# or dev tools via pinned file
pip install -r requirements-dev.txt
```

## 9) Database Setup (PostgreSQL with pgvector)

1) Create a database and enable pgvector:
```sql
CREATE DATABASE duetmind;
\c duetmind
CREATE EXTENSION IF NOT EXISTS vector;
```

2) Configure `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/duetmind
```

3) Ensure network access and credentials match your environment.

## 10) Data and MLflow Tracking

- MLflow is used for experiment tracking and model versioning
- Set MLflow environment variables for tracking and artifact storage

Basic commands:
```bash
# Configure environment variables (see Section 7)
# MLflow UI (optional)
mlflow ui
```

## 11) Running and Testing

- Unit tests:
  ```bash
  pytest
  ```
- Lint/format:
  ```bash
  black .
  isort .
  flake8
  mypy .
  ```
- Example scripts exist in the repository (e.g., main.py, clinical_decision_support_main.py, etc.). Ensure `.env` is set and DB is reachable before running.

## 12) Known Inconsistencies to Resolve

- Python version requirement:
  - pyproject.toml: Python >= 3.10
  - setup.py: Python >= 3.8
  Recommendation: use 3.10+ and align both files to the same version.

- Package name metadata:
  - pyproject.toml [project].name: `duetmind-adaptive`
  - setup.py name: `duetmind_adaptive`
  Consider standardizing to one canonical name to avoid confusion.

## 13) Troubleshooting

- `ModuleNotFoundError` on imports: verify you installed with `pip install -e .` and that `src` is used as the package root.
- Database connection errors: re-check `DATABASE_URL` and that PostgreSQL and `pgvector` extension are available.
- Slow embedding: optionally enable GPU and ensure PyTorch detects CUDA; set `SENTENCE_TRANSFORMERS_HOME` for faster cache reuse.

---
Last updated based on repository state at time of writing.
