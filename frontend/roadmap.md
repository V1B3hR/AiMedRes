### Site access note
I could not load the AiMedRes repository page because the URL requires signing in to GitHub; the page returned a sign-in/error page when accessed.

---

### Prompt — Generate a runnable, cheap, efficient Webapp GUI for AiMedRes

Goal
- Produce a complete, runnable single-repo web application that provides a fast, low-cost GUI front end for AiMedRes. Prioritise minimal infra, low memory/CPU footprint, quick response for typical medical-research workflows, and clear extendability for advanced features.

High-level constraints
- Must run locally and in a low-cost cloud container (single small instance).
- Use a lightweight backend framework and static-first frontend (fast cold start).
- Keep persistent storage simple (file + embedded DB) with optional migration to managed DB.
- Provide configuration for safe, auditable model/service endpoints (do not embed proprietary model weights in the web UI).
- Include automated dev start, simple Dockerfile, and a documented deploy path (e.g., Docker -> single small VM or container host).

Non-functional requirements
- Performance: sub-second UI interactions for control actions; async background tasks for long-running model ops with progress UI.
- Cost: minimal runtime memory and CPU; default to CPU inference or remote model APIs.
- Security: authentication (basic but pluggable), TLS-ready, input validation, logging of actions/audit trail.
- Accessibility: keyboard navigable, semantic HTML, responsive layout for desktop and tablet.
- Observability: health endpoint, simple metrics, log rotation/basic audit log.

Suggested tech stack (one-line rationale)
- Backend: Python FastAPI (async, small, well-documented) or Node.js + Express if preferred.
- Frontend: React + Vite or SvelteKit (fast dev builds, small bundles).
- DB: SQLite for local runs; optional PostgreSQL for production.
- Task queue: built-in asyncio + Redis optional for scale; simple background worker for PoC.
- Containerization: Dockerfile with small base image (python:3.x-slim or node:18-slim).
- Deployment: single container on small host (e.g., low-tier cloud container service).

Core features (each with dedicated options)
- Patient / Case Management
  - Create/edit patient/case record; tags for cohorting; import/export CSV; view timeline.
  - Options: anonymize toggle; schema mapping for custom fields.
- Data Upload and Preprocessing
  - Upload clinical text, images, structured CSVs; preview, simple validation, sanitise; run built-in preprocessing pipelines.
  - Options: batch size, trimming rules, anonymization level, replace missing strategy.
- Model Inference Console
  - Select model endpoint (local, remote API, or containerized service); send a case for inference; show structured outputs and confidence metadata.
  - Options: model selection dropdown; request timeout; beam/temperature for generative models; cached responses toggle.
- Interactive Explanation & Evidence Viewer
  - Show model rationale, provenance links to source data, highlighted text spans, and supporting evidence list.
  - Options: explanation mode (saliency, LIME-like, attention viz), toggle raw tokens, confidence threshold filter.
- Clinician Review Workflow
  - Assign cases to reviewers, annotate model outputs, accept/reject suggestions, add notes; audit log of decisions.
  - Options: reviewer roles, required sign-off, export annotated cases.
- Evaluation & Metrics Dashboard
  - Track throughput, latency, per-model metrics, and clinician acceptance rates; downloadable evaluation CSV.
  - Options: rolling window, per-cohort breakouts, custom metric upload.
- Secure Export & Reporting
  - Export anonymised reports (PDF/CSV) and datasets for research; selective field redaction.
  - Options: include audit trail, export template selection.
- Settings & Admin
  - Configure model endpoints, API keys, storage paths, retention policy, and user accounts.
  - Options: import/export configuration, DB backup, retention policy controls.

UI/UX — pages and key interactions
- Landing / Dashboard
  - Quick system health, queued tasks, recent cases, quick action buttons.
- Cases List
  - Filter by status, cohort, model used, reviewer; multi-select and batch actions (run model, export).
- Case Detail (primary instrument)
  - Left: case metadata and timeline; center: model outputs with confidence and explanation panels; right: annotation tools and reviewer comments.
  - Inline controls: re-run inference with altered params, add evidence link, flag for escalation.
- Model Console (dev mode)
  - Raw request builder and JSON response view; ability to save common payloads.
- Admin / Settings
  - API endpoint management, feature flags, user roles, logs.

Background jobs and UX for long tasks
- Queue submission returns job id.
- Polling endpoint / WebSocket push for updates.
- Progress indicator + incremental partial results display (if supported).

Developer DX and runnable deliverables
- Repository structure proposal:
  - /app/backend (FastAPI)
  - /app/frontend (Vite React)
  - /app/workers (background tasks)
  - /infra/Dockerfile, docker-compose.yml, Makefile
  - /migrations, /data (sqlite by default), /config (env examples)
- Commands (examples)
  - Local dev: make dev -> starts backend, frontend, optional redis
  - Build: make build
  - Run container: docker build -t aimedres-ui .; docker run -p 8000:80 aimedres-ui
- Health checks: GET /health, GET /metrics
- Example minimal Dockerfile and docker-compose that run in small cloud containers.

Extensibility hooks (developer-focused)
- Plugin API for new preprocessing modules, new model adapters (wrap any REST/gRPC model), and new explanation methods.
- Webhook callbacks for audit events.
- Config-driven forms for custom case schemas.

Sample prompt to give to a UI dev / code generator (copy-paste ready)
- Provide a single-paragraph summary of the project, then a prioritized checklist, then required files to produce. Example:
  - Summary (1–2 sentences): "Build a small, fast webapp to manage cases, run AiMedRes model inferences, review outputs, and export annotated datasets. Prioritise a runnable Dockerized repo that works locally with SQLite and minimal memory footprint."
  - Checklist (priority order): implement backend API (FastAPI) with job queue; implement frontend (React+Vite) with Cases List and Case Detail; implement model adapter pattern; authentication; Dockerfile and docker-compose; README with run steps.
  - Required deliverables: working repo with start scripts, Dockerfile, example config, sample dataset (anonymised), a short test suite (basic route tests), and README showing how to run locally and deploy to a single container host.
- Non-functional acceptance criteria: startup under 30s on low-tier VM, memory <1.5GB baseline, response latencies <300ms for UI operations (excluding model runtime), background jobs reported and cancellable.

Optional advanced ideas (pick as stretch goals)
- Lightweight offline-first mode (service worker caching for UI with queueing).
- Model sandboxing via ephemeral containers to isolate expensive inference.
- Role-based redaction rules per user group.
- Differential privacy export mode for dataset releases.

Deliverable checklist for me to produce next (I went ahead and picked the most useful)
- A runnable starter repo scaffold: minimal FastAPI backend + React frontend + Dockerfile + docker-compose + example data and README.
- Minimal implementation plan for model adapters and explanation hooks.
- UI wireframe suggestions for Case Detail page (component list with responsibilities).

If you want, I can now:
- Generate the starter repo scaffold (file list and content) for immediate cloning and running,
- Or produce the concrete FastAPI + React code for the core flows (Cases List, Case Detail, model call stub, Dockerfile, README).

Would you like me to generate the runnable starter code now?
