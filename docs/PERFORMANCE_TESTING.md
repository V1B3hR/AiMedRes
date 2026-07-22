# Performance Testing Guide

## Overview

AiMedRes now includes a roadmap-aligned performance validation baseline:

- **k6 API load testing** via `tests/performance/k6_api_load.js`
- **Lighthouse CI** via `frontend/lighthouserc.json`
- **GitHub Actions workflow** via `.github/workflows/performance-validation.yml`

## Run k6 Locally

Prerequisite: install [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/).

```bash
cd AiMedRes
API_BASE_URL=http://localhost:8080 HEALTH_PATH=/health K6_VUS=10 K6_DURATION=30s k6 run tests/performance/k6_api_load.js
```

Environment variables:

- `API_BASE_URL` (default: `http://localhost:8080`)
- `HEALTH_PATH` (default: `/health`)
- `K6_VUS` (default: `10`)
- `K6_DURATION` (default: `30s`)

## Run Lighthouse Locally

```bash
cd AiMedRes/frontend
npm ci --legacy-peer-deps
npm run build
npx @lhci/cli@0.14.x autorun --config=./lighthouserc.json
```

Artifacts are written to `frontend/.lighthouseci/`.

## Run in GitHub Actions

Use the **Performance Validation** workflow from the Actions tab:

- Configure optional k6 inputs (`api_base_url`, `health_path`, `vus`, `duration`)
- Workflow uploads:
  - `k6-summary` artifact (`performance-results/k6-summary.json`)
  - `lighthouse-report` artifact (`frontend/.lighthouseci/`)

## Scope and Next Improvements

This baseline validates health endpoint latency and frontend quality signals.
Future expansion can add:

- Additional k6 endpoint scenarios (predict, agent, status APIs)
- Stress/soak profiles beyond baseline thresholds
- PR gating based on stricter Lighthouse and k6 budgets
