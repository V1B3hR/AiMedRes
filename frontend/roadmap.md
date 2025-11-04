Phase 0 — Prep and alignment (1 week)
Goal: Agree scope, repo layout, and minimal acceptance criteria so the first commit is runnable and deployable.

Milestones

Finalize minimal feature set (MVP): auth, Cases List, Case Detail, model call stub, background job stub, SQLite persistence, Dockerfile, README.

Lock tech stack: FastAPI backend, React + Vite frontend, SQLite, optional Redis.

Create issue tracker with prioritized tasks and owner assignments.

Deliverables

Project brief, acceptance tests, and repo skeleton tickets.

Success criteria

Team sign-off on MVP checklist and time estimates.

Risks

Scope creep; mitigate with strict MVP checklist.

Phase 1 — Minimal runnable MVP (2–3 weeks)
Goal: Shipping a local, Dockerised app that runs end-to-end with dummy model responses and sample data.

Milestones

Backend: FastAPI app with API routes: auth, /cases, /cases/:id, /inference job endpoints, /health, /metrics.

Persistence: SQLite schema and simple migrations; sample anonymized dataset.

Background worker: lightweight asyncio job queue with job ID/poll API (Redis optional).

Frontend: React + Vite pages: Login, Cases List, Case Detail with annotation panel and model output area.

Dev ergonomics: Makefile / npm scripts, docker-compose for local dev (backend, frontend, optional redis).

Deliverables

Working repo that starts with one command and exposes frontend at :3000 and backend at :8080; README with run steps.

Success criteria

Full user flow: login → view case list → open case → run inference → view job status → annotate and export sample CSV.

Risks

Integration gaps between frontend expectations and backend API; mitigate by using the API contract in the frontend roadmap as spec.

Phase 2 — Core features and UX polish (3–4 weeks)
Goal: Harden core clinical workflows, add explainability UI, clinician review loop, and secure exports.

Milestones

Patient/Case Management: create/edit, import/export CSV, anonymize toggle.

Data upload & preprocessing UI: file upload, validation preview, sanitize options.

Model Inference Console: model endpoint selector, param controls (timeout, temperature), cached responses toggle.

Explainability Panel: show rationale, highlighted spans, selectable explanation modes.

Clinician Review Workflow: assign reviewers, annotate outputs, accept/reject with audit log.

Secure Export: anonymised CSV/PDF export templates and redaction controls.

Deliverables

Frontend components wired to backend endpoints; audit logs persisted; export templates implemented.

Success criteria

Clinician can complete review workflow end-to-end and produce an anonymized export.

Risks

PHI handling errors; mitigate with strict validation, de-identification tests, and config-driven redaction.

Phase 3 — Performance, security, and observability (2–3 weeks)
Goal: Make the webapp cheap to run, resilient, observable, and secure for small-scale deployments.

Milestones

Optimize frontend bundles (Vite code-splitting, lazy loads, virtualized lists).

Add auth hardening (JWT sessions, role-based UI controls, TLS-ready config).

Health and metrics: /health, basic Prometheus-friendly metrics, simple request logging and audit rotation.

Background jobs: retry policy, cancellation, job progress push via WebSocket or server-sent events.

Add basic CI: build/test (unit + simple route tests).

Deliverables

Production Dockerfile (slim base), small memory footprint profile, documented resource expectations.

Success criteria

Cold start under 30s on a low-tier container; baseline memory <1.5 GB; UI interactions <300 ms excluding model runtime.

Risks

Unexpected memory use in image/DICOM components; mitigate with conditional lazy-loading and service limits.

Phase 4 — Evaluation, metrics dashboard, and admin (2 weeks)
Goal: Provide operators and researchers visibility into system performance, model behavior, and clinician acceptance.

Milestones

Metrics dashboard: latency, throughput, per-model acceptance rates, cohort breakdowns.

Evaluation export: downloadable CSV of recent runs and annotations.

Admin settings: manage model endpoints, API keys, retention policy, backup/export config.

Deliverables

Dashboard pages, export endpoints, admin UI with config import/export.

Success criteria

Admin can view rolling metrics and export evaluation data for a given period.

Risks

Data sensitivity in metrics; minimize PHI in metrics and apply aggregation.

Phase 5 — Optional stretch goals and extensibility (ongoing)
Goal: Add advanced, optional features that improve safety, isolation, and developer extensibility.

Candidate items

Model sandboxing via ephemeral containers for heavy inference.

Offline-first UI caching and job queueing (service workers).

Plugin API for preprocessing modules and model adapters (REST/gRPC wrappers).

Explainability library pluggable adapters (saliency, LIME-like, attention).

Role-based redaction rules and differential privacy export mode.

Deliverables

Plugin spec, example plugin, sandbox orchestration scripts.

Success criteria

Third-party module can be added without backend changes via plugin contract.

Minimal acceptance tests (run with CI)
Start app with docker-compose and confirm:

GET /health returns OK.

Auth login returns JWT and protected endpoints reject missing token.

Create case, run inference job, poll job status to completion.

Export anonymized CSV contains no raw PHI fields.

Metrics endpoint returns recent request counts.

Automated E2E: simple Cypress test for login → approve case → export, using sample dataset (aligns to frontend test examples).

Timeline summary (approximate)
Week 0: Prep and alignment (1 week)

Weeks 1–3: MVP (2–3 weeks)

Weeks 4–6: Core workflows and UX (3–4 weeks)

Weeks 7–8: Performance, security, observability (2–3 weeks)

Weeks 9–10: Metrics, admin, and docs (2 weeks)

Week 11+: Stretch goals and polishing (ongoing)

Total time to robust MVP: ~6–8 weeks depending on team size and parallelization.

Roles, team composition, and suggested sprint cadence
Core team (recommended for 8-week timeline)

1 full-stack engineer (FastAPI + React)

1 frontend engineer (UX, accessibility, performance)

1 backend/infra engineer (jobs, Docker, security)

1 UX/product lead (wireframes, acceptance tests, clinical workflow)

Sprint cadence: 2-week sprints with clear sprint goals and demos; prioritize MVP stories in sprint 1 and 2.

Key risks and mitigations
PHI leakage: enforce de-identification, forbid PHI in logs, and add export redaction tests.

Cost blowout from heavy inference: default to remote model APIs and CPU inference; provide per-request timeout and caching.

Integration mismatch: publish an API contract (OpenAPI) and run contract tests during CI.

Regulatory concerns: design audit trail and role separation early.

Next-step checklist for me to produce
Generate the runnable starter code (FastAPI backend + React frontend + Dockerfile + docker-compose + sample dataset and README).

Or produce the concrete FastAPI + React code for the core flows (Cases List, Case Detail, model call stub, Dockerfile, README). Pick one and I’ll produce the scaffold and files in the requested format.
