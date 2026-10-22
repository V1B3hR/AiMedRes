# AiMedRes Debugging and Validation

## Maintained validation path

The supported validation path is the reproducible Parkinson's research package
and the unit-test gate. Run these commands from the repository root:

```powershell
# Windows: keep test artefacts in a writable repository-local temporary directory.
$env:TEMP = (Join-Path (Get-Location) '.test-tmp')
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -m pytest tests/unit/ -q

# Runs the leakage-safe PPMI pipeline against the versioned synthetic fixture.
python scripts/reproduce_ppmi_benchmark.py --output-dir .test-tmp/ppmi_benchmark
```

The test gate covers the maintained research, security, packaging, deployment,
and selected training surfaces. A passing run is evidence only for the tested
research prototype; it is not clinical validation, production authorization, or
evidence that every historical module in the repository remains supported.

## Benchmark interpretation

`scripts/reproduce_ppmi_benchmark.py` uses the repository's synthetic PPMI
fixture by default. Its metrics prove deterministic execution and leakage
controls on that fixture. They must not be presented as prospective clinical
performance or as results from an external validation cohort.

Authorized holders of real PPMI data may provide a local dataset path:

```powershell
python scripts/reproduce_ppmi_benchmark.py --data-path data/ppmi/ppmi_sbr_longitudinal.csv
```

Follow the data-governance requirements in
[`ParkinsonDatasets/DATA_GOVERNANCE.md`](../ParkinsonDatasets/DATA_GOVERNANCE.md)
before running against controlled data.

## Legacy phase artefacts

Only `phase10_final_validation.py` is retained in this directory. It trains
generic models on synthetic data and is preserved as a historical exploratory
tool, not as a release gate for the active PPMI pipeline:

```powershell
python debug/phase10_final_validation.py --data-source synthetic
```

The former Phase 1–9 scripts are not present in this checkout. Their generated
JSON files, plots, and historical summaries do not demonstrate that those
phases can be rerun today. Use the maintained commands above for debugging and
CI evidence.

## Clean-environment verification

Create a fresh virtual environment and install the declared package extras
before testing a release candidate:

```powershell
python -m venv .test-env
.\.test-env\Scripts\python -m pip install --upgrade pip
.\.test-env\Scripts\python -m pip install -e ".[dev]"
.\.test-env\Scripts\python -m pytest tests/unit/ -q
```

`pyproject.toml` is the package metadata source. The `requirements*.txt` files
remain compatible installation lists for the repository CI workflow.
