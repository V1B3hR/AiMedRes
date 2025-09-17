# Quickstart: Run all tests (pytest) and reproduce the full MLOps pipeline (DVC)

This guide shows exactly how to:
- Run all tests with pytest (per pytest.ini)
- Reproduce the full DVC pipeline (per dvc.yaml)
- Use Makefile shortcuts for both

Always run these commands from the repository root so pytest and DVC resolve paths correctly.

## Prerequisites
- Python ≥ 3.10
- Recommended: virtual environment

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
# Prefer editable install with dev extras if available, otherwise requirements
pip install -e .[dev] || pip install -r requirements.txt

# Ensure these tools are present
pip install pytest dvc
```

References:
- Test config: [pytest.ini](./pytest.ini)
- Pipeline: [dvc.yaml](./dvc.yaml)
- Shortcuts: [Makefile](./Makefile)

---

## Run all tests (pytest)

Run the entire test suite:
```bash
pytest
```

Useful variants:
- With coverage:
  ```bash
  pytest --cov=. --cov-report=term-missing
  ```
- Only unit tests:
  ```bash
  pytest -m unit
  ```
- Only integration tests:
  ```bash
  pytest -m integration
  ```
- Exclude slow tests:
  ```bash
  pytest -m "not slow"
  ```

Markers available (from pytest.ini):
- unit
- integration
- regression
- slow

---

## Full MLOps pipeline (DVC)

Reproduce the entire pipeline:
```bash
dvc repro
```

Visualize the pipeline DAG:
```bash
dvc dag
```

Check pipeline status:
```bash
dvc status
```

Key stages and outputs (from dvc.yaml):
- ingest_raw → outputs data/raw/alzheimer_sample.csv
- ingest_imaging → outputs data/imaging/raw/
- preprocess_imaging → outputs data/imaging/processed/, outputs/imaging/qc/
- extract_features → outputs outputs/imaging/features/, metrics: outputs/imaging/features/feature_summary.json
- build_features → outputs data/processed/features.parquet, data/processed/labels.parquet
- train_model → metrics: mlruns/, outputs: models/

Note: Some outputs are marked with `cache: false` in dvc.yaml and won’t use DVC cache.

---

## Makefile shortcuts

From the repo root:
```bash
# Tests
make test
make test-coverage

# DVC pipeline
make reproduce        # runs `dvc repro`
make dvc-dag          # runs `dvc dag`
make dvc-status       # runs `dvc status`
```

Additional helpful targets (see Makefile):
- make data           # ingest raw data
- make features       # build features
- make train          # train model
- make validate       # data schema validation
- make ci-pipeline    # lint + test + validate + reproduce
