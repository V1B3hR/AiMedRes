# 🧭 AiMedRes: roadmapa gotowości badawczej (2026–2027)

> **Status dokumentu:** żywa, nadrzędna roadmapa i jedyne źródło prawdy o kolejności prac.
> **Aktualizacja:** 25 października 2026.
> **Pozycjonowanie:** AiMedRes jest prototypem badawczym. Nie jest certyfikowanym wyrobem medycznym, narzędziem diagnostycznym ani systemem gotowym do użycia klinicznego.

---

## 1. Executive Summary & Strategic Reset

AiMedRes has developed extensive scaffolding across conventional disease-model training pipelines, an adaptive "DuetMind" neural agent framework, clinical decision support (CDS) interfaces, a Flask API, and React/React Native frontends. While this scaffolding represents substantial software engineering and interface development, an objective scientific audit reveals that **the repository does not yet substantiate claims of validated, production-ready, or clinically usable AI**.

To transition from an exploratory multi-component prototype to an institutionally credible medical research platform, this **Consolidated Roadmap** executes a fundamental strategic reset:

1. **Containment & Truthfulness First**: Immediately reconcile documentation, licensing, and capability claims with actual verified code.
2. **Singular Research Wedge**: Halt parallel attempts to simultaneously mature seven disease models, EHR write-back integrations, multi-agent clinical diagnostic loops, and mobile apps. Instead, focus all rigorous AI/ML efforts on a single, high-value clinical target: **Retrospective Parkinson’s Disease progression and severity analysis using the Parkinson’s Progression Markers Initiative (PPMI)**.
3. **Leakage-Proof Scientific Rigor**: Redesign the Parkinson’s pipeline from the ground up to eliminate circular prediction tasks, participant-level leakage across longitudinal visits, and pre-cross-validation scaling/imputation.
4. **Software & Security Hardening**: Decouple research analysis code from production inference services, implement real identity and secrets management, enforce strict CI gates, and prepare for external institutional review.

---

## 2. Current Baseline & Honest Capability Audit

The table below establishes an unvarnished audit of the repository, categorizing every major area into three reality tiers:
- **Tier 1 — Implemented Prototype Code**: Genuinely present executable code, tests, and functional scripts.
- **Tier 2 — Partially Implemented or Heuristic Simulation**: Modules that function via static rules, heuristic approximations, or in-memory mock logic rather than verified clinical models.
- **Tier 3 — Documentation or Planning Only**: Documents, templates, or roadmaps describing future milestones without corresponding operational reality or third-party validation.

### 2.1 Capability Audit Matrix

| Area | Tier | What is Genuinely Present | What is Missing, Inconsistent, or Flawed |
|---|:---:|---|---|
| **Training Infrastructure** | Tier 1 | 7 default jobs in `src/aimedres/cli/train.py`; orchestrator records Git SHA, GPU details, logs, JSON summaries. Classical ML and PyTorch. | DVC has no configured cloud remote; long-running distributed GPU training requires local setup. |
| **Parkinson's / PPMI Work** | Tier 1 | Decoupled research module in `src/aimedres/research/parkinsons/`, synthetic fixture, participant-grouped `StratifiedGroupKFold` on `PATNO`, independent target, bootstrap-CI and provenance-manifest tooling. | The current evidence is synthetic-fixture only. Controlled PPMI access, external cohorts and independent methodological review have not yet occurred. |
| **Adaptive NN & Agents** | Tier 2 | `AdaptiveNeuralNetwork`, `DuetMindAgent`, specialist agent roles, state handling, memory store, and safety heuristics in `src/aimedres/core/` and `src/aimedres/agents/`. Quarantined in `src/aimedres/simulation/` with runtime isolation gates (`assert_simulation_allowed()`). | Experimental simulation code only. `specialized_medical_agents.py` generates heuristic diagnoses, simulated peer review, and synthetic outcome accuracy. Not validated clinical reasoning. |
| **Clinical Decision Support (CDS)** | Tier 2 | Risk scoring logic, intervention recommendation structures, FHIR/HL7 conversion helpers, human-in-the-loop audit stubs in `src/aimedres/clinical/`. | Risk engine is hand-authored static heuristic rules. EHR "sync" methods are explicitly simulated mocks. Not connected to live partner hospital EHRs. |
| **API & Applications** | Tier 1 | Flask research API in `src/aimedres/api/server.py` and decoupled `src/aimedres/service/server.py`, technical controls, health endpoint and research-model metadata. | Online inference and heuristic-agent output return unavailable. FHIR, case review, explainability, DICOM and P3 modules are deliberately not registered by the maintained API; UI/mobile work is exploratory. |
| **Privacy & Security** | Tier 1 | Regex-based PHI scrubbing, DICOM tag clearing, encryption helpers (AES-256 / Kyber-768 hybrid prototype), database-backed `AuthDatabaseStore` (SQLite), instant token & key revocation, PBKDF2 hashing with salt, security audit logging, and fail-closed production secret validation. | Pattern scrubbing is not a formal HIPAA de-identification certification or data-use authorization system. DICOM stripper clears limited fields. No third-party penetration testing or external SOC audit completed. |
| **Deployment & MLOps** | Tier 1 | Dockerfiles, Docker Compose, Kubernetes/Helm/monitoring configuration assets, local experiment tracking, a CI workflow, SBOM generator and local CI runner. | No container-image build, deployment, TLS, production WSGI, public registry or cloud environment was verified in this P0 pass. |
| **Documentation & Compliance** | Tier 3 | Extensive documentation: training guides, security policies, model cards, FDA Q-Sub generator (`src/aimedres/compliance/`). | Claims overstate maturity. Generates draft FDA/QMS skeletons, but no formal FDA submission has been made, no clinical pilot has run with live patients, and no certified QMS exists. |
| **Packaging & Governance** | Tier 1 | `pyproject.toml`, `setup.py`, requirements files, MIT `LICENSE`, `ParkinsonDatasets/DATA_GOVERNANCE.md`. | Licensing, data-access and DUA implications need owner review before any controlled-data or institutional use. |
| **Testing & CI** | Tier 1 | Focused P0 unit tests cover research configuration, provenance, route isolation, authentication controls and the supported launch path. | This pass did not execute the full repository suite, frontend/e2e tests, deployment validation, CVE scan or any test using real clinical data. |

---

## 3. P0 — gotowość do przedstawienia i audyt

**Status: w toku.** P0 jest warunkiem rozmów o finansowaniu badań. Nie przechodzimy dalej, dopóki jego kryteria wyjścia nie zostaną udokumentowane.

| Priorytet | Działanie | Dowód ukończenia |
|---|---|---|
| P0.1 | **Jedna wersja prawdy.** Uzgodnić README, roadmapę, model card, katalog danych, ograniczenia i materiały wdrożeniowe. Historyczne/planujące treści oznaczyć zamiast przedstawiać je jako wdrożone. | Rejestr claimów: każde twierdzenie o funkcji, metryce lub zgodności ma źródło, status i właściciela. |
| P0.2 | **Zamrożony baseline techniczny.** Dokończyć bieżącą konfigurację PPMI, manifest i runner; uruchomić właściwe testy oraz benchmark ze śledzonym commitem i środowiskiem. | Powtarzalny raport techniczny; wynik pozytywny, negatywny lub niejednoznaczny zapisany bez selekcji. |
| P0.3 | **Audyt bezpieczeństwa i jakości.** Przejrzeć utrzymywany API/service/auth/config/data flow; zweryfikować sekrety, debug, CORS, autoryzację, logowanie, zależności, deserializację i wywołania powłoki. Ryzyka klasyfikować jako P0/P1/P2. | Rejestr ryzyk, zakres wspieranego kodu, dowody testowe i plan remediacji. Ryzyko P0 blokuje prezentację. |
| P0.4 | **Audyt naukowy.** Sprawdzić endpoint, definicję kohorty, podział po uczestnikach/czasie, przetwarzanie w foldach, kalibrację, bootstrap i ograniczenia podgrup. | Checklist niezależnego biostatystyka lub metodologa oraz zaktualizowany model card. |
| P0.5 | **Evidence Pack.** Złożyć pytanie badawcze, manifest danych, protokół, komendę odtworzenia, wyniki, ograniczenia i demonstrację syntetyczną. | Pakiet może zostać odtworzony przez inną osobę bez dostępu do danych pacjentów. |

### Stan bieżącej iteracji

- **P0.1 — częściowo wykonane:** roadmapa przejęła plan gotowości; zduplikowany dokument został usunięty. README, model card, katalog danych, OpenAPI i kluczowe materiały kliniczne/bezpieczeństwa mają jawne granice statusu; pozostałe historyczne materiały nadal wymagają claim review.
- **P0.2 — częściowo wykonane:** 25 października ukierunkowane testy konfiguracji i pipeline'u przeszły lokalnie. Dwa pełne reruny synthetic fixture z identyczną konfiguracją dały identyczne metryki klasyczne i neuronowe w tym samym środowisku Windows. Trzeci kontrolny rerun jest zapisany w wersjonowanym [Evidence Pack](../evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md) wraz z hashami danych, konfiguracji i plików źródłowych. Git worktree był celowo oznaczony jako dirty, dlatego ten pakiet nie zastępuje commitu/release ani niezależnego rerunu.
- **P0.3 — częściowo wykonane:** pierwszy statyczny audyt naprawił ekspozycję klucza API w query string i niespójną ścieżkę uruchamiania. Historyczne karty modeli, fallback heurystycznej predykcji i mock/prototype routes zostały wycofane z utrzymywanego API; online inference pozostaje zablokowane do czasu zatwierdzenia artefaktu badawczego. Po advisory Render z 25 października dodano pin Node 24.22.1, wymaganie `certifi>=2026.10.19`, odświeżanie CA w Dockerfile i usunięto opcjonalną usługę PHP Adminer z domyślnego MLOps compose. Nadal brakuje potwierdzenia build/redeploy oraz testu z działającej usługi Render. Otwarte obserwacje oraz granice audytu są w [Known Limitations](KNOWN_LIMITATIONS.md).
- **P0.4 — rozpoczęte:** 20 testów PPMI/research lokalnie przeszło 25 października, w tym kontrolę leakage po `PATNO`, izolację preprocessingu, niezależność etykiety i wykonanie end-to-end. Lokalna kontrola potwierdza grouping po `PATNO`, fold-isolated preprocessing, bootstrap i Brier score wyłącznie na fixture syntetycznym. Brakuje zatwierdzonego endpointu/kohorty, planu podgrup i niezależnej oceny metodologicznej.
- **P0.5 — częściowo wykonane:** minimalny, bezpieczny do wersjonowania Evidence Pack istnieje dla fixture syntetycznego; nie zawiera danych pacjentów ani artefaktów modeli.

### Rejestr kluczowych claimów

| Claim | Dowód w repozytorium | Dopuszczalny opis zewnętrzny | Niedopuszczalny skrót |
|---|---|---|---|
| Pipeline Parkinsona stosuje kontrolę leakage po uczestniku. | [`configs/parkinsons_ppmi_study.yml`](../configs/parkinsons_ppmi_study.yml), runner i testy PPMI. | „Lokalnie odtwarzalny benchmark na synthetic fixture z `StratifiedGroupKFold` po `PATNO`.” | „Zweryfikowany model PPMI” lub „zero leakage na kohorcie klinicznej”. |
| Benchmark osiąga ROC-AUC 0,907. | Lokalne artefakty synthetic fixture oraz [model card](MODEL_CARD_PPMI_PARKINSONS.md). | „Wynik kontrolny na synthetic fixture; nie jest metryką pacjentów.” | „AUC modelu klinicznego” lub „wynik PPMI”. |
| API, modele i agenci są dostępne. | [`src/aimedres/service/`](../src/aimedres/service/) oraz testy bezpieczeństwa. | „API jest prototypową powierzchnią badawczą; online inference i heuristic agent output są zablokowane.” | „Działająca usługa predykcyjna” lub „clinical decision support”. |
| Mechanizmy auth/secrets istnieją. | [`src/aimedres/security/auth.py`](../src/aimedres/security/auth.py) i testy fail-closed. | „Lokalnie przetestowane kontrolki techniczne; bez zewnętrznego pentestu lub certyfikacji.” | „HIPAA/GDPR compliant”, „secure by certification” lub „production hardened”. |
| Docker, K8s, Helm i monitoring są w repozytorium. | `Dockerfile`, `k8s/`, `helm/`, `monitoring/`. | „Aktywa konfiguracyjne wymagające środowiskowej walidacji.” | „Wdrożona platforma”, „operacyjny monitoring” lub „gotowe dla NHS”. |
| Materiały QMS, SaMD i kliniczne są dostępne. | `docs/compliance/`, `docs/clinical/`. | „Szablony i materiały planistyczne do adaptacji instytucjonalnej.” | „Certyfikowany QMS”, „IRB/HRA approval” lub „regulatory clearance”. |

### Kryteria wyjścia z P0

1. Nie ma sprzecznych twierdzeń o gotowości klinicznej, bezpieczeństwie ani wdrożeniu.
2. Wersjonowany pipeline uruchamia się na syntetycznym fixture z udokumentowaną komendą i wynikami.
3. Ryzyka P0 są usunięte albo jednoznacznie blokują użycie poza lokalnym środowiskiem badawczym.
4. Istnieje rozdział między utrzymywanym kodem a komponentami eksperymentalnymi/legacy.
5. Materiał zewnętrzny mówi wyłącznie „funding-ready for research”, nigdy „clinical-ready”, „market-ready” lub „SaMD-ready”.

### Kolejne bramki

- **P1 — niezależna replikacja i partnerstwo:** prerejestrowane pytanie, rerun na danych z uprawnionego źródła, niezależna ocena metodyczna, clinical PI, biostatystyk, właściciel danych i PPI.
- **P2 — wniosek o finansowanie:** concept note, budżet oparty na work packages, demonstracja syntetyczna oraz uzgodnione role, IP i governance danych. Szczegóły konkursów są w [strategii finansowania UK i Polski](FUNDING_STRATEGY_UK_AND_POLAND.md).
- **P3 — walidacja instytucjonalna i decyzja regulacyjna:** intended purpose, wymagania HRA/REC/NHS/MHRA oraz QMS proporcjonalny do rzeczywistego celu użycia — wyłącznie po finansowaniu i wynikach walidacji.

### Standard Evidence Pack

Pakiet musi zawierać pytanie i endpoint ustalone przed analizą; manifest danych (źródło, wersja, podstawa dostępu, SHA-256 i DUA); commit/release, środowisko i komendę odtworzenia; metryki z przedziałami ufności; wyniki podgrup, wykluczenia i znane porażki; raport niezależnego rerunu oraz demonstrację syntetyczną.

**Kontrolny pakiet syntetyczny (25 października 2026):** [EVIDENCE_PACK.md](../evidence/ppmi_synthetic_fixture/2026-10-25/EVIDENCE_PACK.md) i [machine-readable manifest](../evidence/ppmi_synthetic_fixture/2026-10-25/evidence_manifest.json). To kontrola odtwarzalności, nie dowód kliniczny. Generator przyjmie do wersjonowanego pakietu wyłącznie committed synthetic fixture i kanoniczną konfigurację study; odmówi użycia danych kontrolowanych/pacjenckich oraz override'ów parametrów.

Aktualna komenda kontrolna dla synthetic fixture:

```powershell
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml
```

Aby utworzyć nowy, pusty katalog pakietu po własnym rerunie:

```powershell
python scripts/reproduce_ppmi_benchmark.py --config configs/parkinsons_ppmi_study.yml --evidence-dir evidence/ppmi_synthetic_fixture/<nowy-pusty-katalog>
```

Niepotwierdzony wynik jest wynikiem projektu: raportujemy rozbieżność i zawężamy hipotezę, zamiast zmieniać claimy po fakcie.

### P0.4 — checklista metodologiczna przed niezależnym rerunem

| Kontrola | Obecny stan | Co blokuje zamknięcie |
|---|---|---|
| Pytanie, endpoint i horyzont czasu | `motor_progression_label` jest wyłącznie etykietą synthetic fixture. | Właściciel danych i clinical PI muszą zatwierdzić definicję realnego endpointu, indeks czasu i wykluczenia przed analizą. |
| Kohorta i governance | Fixture ma 170 obserwacji/60 syntetycznych uczestników; hash i konfiguracja są w Evidence Pack. | DUA, data dictionary, kryteria kohorty oraz manifest uprawnionych danych PPMI lub innej kohorty. |
| Leakage i preprocessing | Testy i kontrolny rerun stosują `StratifiedGroupKFold` po `PATNO`; imputation/scaling są fitowane w foldzie treningowym. | Niezależny przegląd kodu i plan rozdziału czasowego/zewnętrznego dla realnej kohorty. |
| Metryki i niepewność | ROC-AUC, PR-AUC, Brier oraz 95% bootstrap CI są raportowane dla synthetic fixture. | Prerejestrowane primary metric, plan missing data, kalibracji, podgrup i reguła interpretacji CI. |
| Reprodukcja | Dwa lokalne deterministyczne reruny oraz wersjonowany control run. | Rerun przez niezależną osobę, na osobnym środowisku, z zapisanym wynikiem pozytywnym lub negatywnym. |

## 4. Archiwalny plan docelowy i inwentarz implementacji

Poniższe sekcje zachowują historyczny plan i wskazują artefakty obecne w repozytorium. **Nie są potwierdzeniem ukończenia ani niezależnej walidacji.** Oznaczenia `[x]` pochodzą z wcześniejszej wersji dokumentu i do zakończenia P0 oznaczają wyłącznie deklarowane/istniejące artefakty wymagające weryfikacji.

## 4. The 5-Phase Production-Ready AI/ML Roadmap

```mermaid
flowchart TD
    subgraph P0["Phase 0: Immediate Containment (Weeks 0-2)"]
        P0A[PPMI Data Governance & Restricted Data Removal] --> P0B[Freeze Unsupported Claims in Docs & README]
        P0B --> P0C[Consolidate Single Research Focus: Parkinson's]
        P0C --> P0D[Reconcile Packaging, Licenses MIT & Dependencies]
    end

    subgraph P1["Phase 1: Research-Grade PPMI Pipeline (Months 1-2)"]
        P1A[Pre-Registered Clinical Study Protocol] --> P1B[Defensible Clinical Target Definition]
        P1B --> P1C[Leakage-Proof PATNO GroupKFold Pipeline]
        P1C --> P1D[Data Manifest & Synthetic CI Benchmark Fixture]
    end

    subgraph P2["Phase 2: Reproducibility & Scientific Quality (Months 2-4)"]
        P2A[Typed Pydantic Config & Pinned Lockfiles] --> P2B[Leakage & Scientific Regression Test Suite]
        P2B --> P2C[MLflow Tracking & Model Registry]
        P2C --> P2D[Honest Model Card & Peer Review Signoff]
    end

    subgraph P3["Phase 3: Software Hardening & Decoupling (Months 4-8)"]
        P3A[Decouple research/ service/ ui/] --> P3B[Serve Registered Model via Flask API]
        P3B --> P3C[OAuth2/OIDC Auth & DB Credential Store]
        P3C --> P3D[Hardened Non-Clinical Research Sandbox K8s]
    end

    subgraph P4["Phase 4: Institutional Pilot Readiness (Months 8-15)"]
        P4A[Institutional IRB & Intended-Use Specification] --> P4B[Temporal & External Multi-Site Validation]
        P4B --> P4C[Clinician-in-the-Loop Usability Testing]
        P4C --> P4D[Quality Management & SaMD Traceability]
    end

    P0 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
```

---

### Phase 0: Immediate Containment, Data Governance & Truthfulness (Weeks 0–2)

**Goal**: Establish an honest baseline, eliminate legal and data-governance vulnerabilities, reconcile package metadata, and freeze speculative distractions.

- [x] **0.1 PPMI & Data-Use Governance Enforcement**
  - **Action**: Audit `ParkinsonDatasets/`. Under PPMI Data Use Agreements (DUA), raw patient records cannot be redistributed in public open-source repositories.
  - **Deliverable**: Isolate raw PPMI files from public distribution, add `ParkinsonDatasets/DATA_GOVERNANCE.md` detailing credential requirements, and introduce a verified, non-PHI synthetic fixture (`synthetic_pet_sbr_fixture.csv`) for open-source CI/CD execution.
- [x] **0.2 Claim Containment & Repository Truthfulness**
  - **Action**: Update `README.md`, `docs/KNOWN_LIMITATIONS.md`, and model cards to remove premature claims of "FDA submission readiness", "certified HIPAA compliance", and "multi-hospital network operational".
  - **Deliverable**: Explicit classification markers across all docs designating components as *Prototype*, *Simulated*, *Planned*, or *Validated*.
- [x] **0.3 Single Research Focus & Scope Freeze**
  - **Action**: Concentrate core AI/ML development exclusively on retrospective Parkinson’s disease progression and severity modeling using PPMI data.
  - **Deliverable**: Freeze feature expansion on the other six disease models, the Expo mobile app, and simulated multi-agent consensus until the primary research pipeline achieves scientific peer-review readiness.
- [x] **0.4 Packaging, Metadata & License Reconciliation**
  - **Action**: Resolve license divergence. The repository license is **MIT** (per `LICENSE` and `pyproject.toml`). Remove all stale references to GPL-3.0 in `README.md`, `setup.py`, templates, and documentation.
  - **Deliverable**: Clean `setup.py` without placeholder values (`your-email@example.com`), aligned with `pyproject.toml` and standard requirements files.

---

### Phase 1: Research-Grade PPMI Pipeline (Months 1–2)

**Goal**: Transform the Parkinson’s ML script into a scientifically defensible, statistically sound, leakage-free predictive pipeline.

- [x] **1.1 Pre-Registered Clinical Study Protocol & Target Definition**
  - **Objective**: Author a formal study protocol prior to model optimization.
  - **Specification**: Adopted independent motor progression milestone ($\Delta\text{MDS-UPDRS Part III} \ge 5.0$ points), verified in [`docs/MODEL_CARD_PPMI_PARKINSONS.md`](MODEL_CARD_PPMI_PARKINSONS.md).
- [x] **1.2 Leakage-Proof Participant-Level Modeling**
  - **Objective**: Redesign model validation to prevent all forms of data leakage.
  - **Specification**: Enforced `StratifiedGroupKFold` on `PATNO`, fold-encapsulated `Pipeline` scaling/imputation, out-of-fold calibration (Brier Score = 0.108), and 95% non-parametric bootstrap confidence intervals in [`src/aimedres/training/train_parkinsons.py`](../src/aimedres/training/train_parkinsons.py).
- [x] **1.3 Data Manifest & Provenance Tracking**
  - **Objective**: Create a complete, non-PHI experiment manifest.
  - **Specification**: Published [`docs/DATASET_CATALOG.md`](DATASET_CATALOG.md) detailing LONI IDA PPMI ingestion rules and open-access alternatives (OASIS-2, PRO-ACT, PTB-XL).
- [x] **1.4 Synthetic PPMI Benchmark Fixture**
  - **Objective**: Maintain a 100% synthetic dataset that mimics the schema, covariance structure, and longitudinal visit hierarchy of PPMI.
  - **Specification**: Deployed [`ParkinsonDatasets/synthetic_pet_sbr_fixture.csv`](../ParkinsonDatasets/synthetic_pet_sbr_fixture.csv).
- [x] **1.5 One-Command Reproduction Runner**
  - **Objective**: Provide a single executable script for independent verification.
  - **Specification**: Deployed [`scripts/reproduce_ppmi_benchmark.py`](../scripts/reproduce_ppmi_benchmark.py) generating full metrics JSON and summary tables in < 20 seconds.

---

### Phase 2: Reproducibility & Scientific Quality (Months 2–4)

**Goal**: Deliver clean packaging, deterministic rerunability, automated scientific regression testing, and transparent model reporting.

- [x] **2.1 Modularized, Pinned & Deterministic Pipeline Package**
  - **Objective**: Package the Parkinson’s research pipeline into an independent, typed module (`src/aimedres/research/parkinsons/`).
  - **Specification**: Deployed [`src/aimedres/research/parkinsons/`](../src/aimedres/research/parkinsons/) with typed Pydantic configuration (`ParkinsonsPipelineConfig`), deterministic random seeding, and cryptographic provenance tracking (`provenance_manifest.json`). Verified in `tests/unit/test_research_parkinsons.py`.
- [x] **2.2 Scientific Rigor & Leakage Regression Test Suite**
  - **Objective**: Implement unit tests dedicated to catching methodological errors in CI.
  - **Specification**: 41 automated unit tests across research configuration, PPMI reproducibility, open-access datasets, and security components passing cleanly.
- [x] **2.3 Live Experiment Tracking & Model Lineage**
  - **Objective**: Implement verifiable experiment provenance, parameter logging, metric tracking, and artifact checksums.
  - **Specification**: Deployed [`src/aimedres/research/tracking.py`](../src/aimedres/research/tracking.py) featuring `LocalExperimentTracker`, SQLite relational run store (`experiments.db`), JSON run cards (`runs/<run_id>.json`), parameter sweeps, metric curves, and cryptographic SHA-256 artifact hashing. Integrated into `ParkinsonsResearchPipeline`.
- [x] **2.4 Evidence-Based Model Card & External Methodological Signoff**
  - **Objective**: Publish a comprehensive, honest model card adhering to FDA Good Machine Learning Practice (GMLP) and TRIPOD-AI principles.
  - **Specification**: Published [`docs/MODEL_CARD_PPMI_PARKINSONS.md`](MODEL_CARD_PPMI_PARKINSONS.md) with ROC-AUC CIs, PR-AUC, calibration curves, demographic caveats, and out-of-scope boundaries.

---

### Phase 3: Software Hardening & Architecture Decoupling (Months 4–8)

**Goal**: Decouple research analytics from production services, harden security boundaries, and eliminate architectural anti-patterns.

- [x] **3.1 Domain-Driven Architectural Decoupling**
  - **Objective**: Restructure the repository into clear architectural boundaries:
    ```text
    src/aimedres/
      research/       # Reproducible scientific pipelines (PPMI, OASIS, PRO-ACT, datasets, tracking)
      service/        # Production inference API, model serving, security middleware
      simulation/     # Quarantined experimental agent logic and cognitive simulations
      ui/             # Clinical decision-support dashboard and client interfaces
    ```
  - **Specification**:
    - Established decoupled [`src/aimedres/service/`](../src/aimedres/service/) exporting production `create_app`, `SecureAPIServer`, and `ModelRegistry`.
    - Quarantined heuristic multi-agent rules into [`src/aimedres/simulation/`](../src/aimedres/simulation/) with runtime quarantine checks (`assert_simulation_allowed()`) preventing unvalidated cognitive simulation execution in production clinical mode.
    - Centralized open dataset loaders and tracking in [`src/aimedres/research/`](../src/aimedres/research/).
- [x] **3.2 Registered Model Serving API**
  - **Objective**: Replace on-the-fly random/generic network initialization in `src/aimedres/api/server.py` with validated model artifact serving.
  - **Specification**:
    - Integrated `ModelRegistry` serving verified model checkpoints (`ppmi_parkinsons_v1`) from disk with full confidence and provenance metrics.
    - Registered clinical blueprints in `SecureAPIServer`: `/api/v1/model`, `/api/v1/fhir`, `/api/v1/cases`, and `/api/v1/explain`.
    - Main prediction endpoint routes to registered model artifacts when `model_id` is specified, falling back gracefully to adaptive network simulation with explicit disclosure.
- [x] **3.3 Enterprise Security & Identity Hardening**
  - **Objective**: Replace in-memory credentials and dev-secret fallbacks with production-grade security.
  - **Specification**:
    - Implemented fail-closed security gates: raises `SecurityConfigurationError` in production mode (`AIMEDRES_ENV=production` or `STRICT_SECURITY=true`) if `FLASK_SECRET_KEY` or `JWT_SECRET` are missing, weak (< 32 chars), or using default values.
    - Deployed `AuthDatabaseStore` in `src/aimedres/security/auth.py` providing SQLite relational persistence for API keys, PBKDF2 hash verification with salt, instant key and session revocation (`revoke_api_key`, `revoke_session`), and audit logging (`auth_audit_log`).
    - Integrated OIDC provider scaffolding, MFA manager, and role-based access control (RBAC).
- [x] **3.4 Production CI/CD Gateways**
  - **Objective**: Establish strict gating workflows before merge.
  - **Specification**:
    - Deployed GitHub Actions CI workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) running automated scientific unit tests, PPMI reproducibility benchmarks, and SBOM generation.
    - Implemented standardized CycloneDX 1.5 Software Bill of Materials (SBOM) generator in [`scripts/generate_sbom.py`](../scripts/generate_sbom.py) capturing runtime environment, package versions, and licenses into `reports/sbom.json`.
    - Deployed one-command local CI verification runner in [`scripts/run_ci_checks.py`](../scripts/run_ci_checks.py).
- [x] **3.5 Non-Clinical Research Sandbox Deployment**
  - **Objective**: Deploy and validate the decoupled platform in an isolated Kubernetes and Helm research sandbox.
  - **Specification**:
    - Implemented sandbox validator in [`scripts/validate_sandbox_deployment.py`](../scripts/validate_sandbox_deployment.py) verifying Kubernetes manifests ([`k8s/`](../k8s/)), Helm chart templates ([`helm/aimedres/`](../helm/aimedres/)), and Prometheus/Alertmanager monitoring ([`monitoring/`](../monitoring/)).
    - Enforced least-privilege securityContext (`runAsNonRoot: true`), Prometheus scraping annotations (`prometheus.io/scrape: true`), and health endpoints. Verified in `tests/unit/test_cicd_and_deployment.py`.

---

### Phase 4: Institutional Pilot Readiness & Clinical Evaluation (Months 8–15)

**Goal**: Prepare for ethical, institutional evaluation under formal oversight and rigorous clinical validation.

- [x] **4.1 Institutional IRB & Intended-Use Definition**
  - **Objective**: Establish legally and clinically defensible regulatory classification and ethical study governance.
  - **Specification**:
    - Authored formal Intended Use and Indications for Use (IFU) document ([`docs/clinical/INTENDED_USE_SPECIFICATION.md`](clinical/INTENDED_USE_SPECIFICATION.md)) establishing Non-Device Clinical Decision Support (CDS) qualification under FDA 21st Century Cures Act § 3060 (FD&C Act § 520(o)(1)(E)) and EU MDR 2017/745 Rule 11.
    - Formulated institutional clinical study protocol ([`docs/clinical/IRB_CLINICAL_STUDY_PROTOCOL.md`](clinical/IRB_CLINICAL_STUDY_PROTOCOL.md)) specifying multi-cohort retrospective validation (PPMI, OASIS-2, UCI Telemetry) and prospective observational workflows with 45 CFR 46.116(f) consent waiver justification.
- [x] **4.2 Real-World Cohort Benchmarking & Open Dataset Integration**
  - **Objective**: Validate the platform against real-world, open-access patient cohorts with zero-leakage cross-validation.
  - **Specification**:
    - Built automated clinical data downloader ([`scripts/download_real_world_datasets.py`](../scripts/download_real_world_datasets.py)) and training pipeline ([`scripts/train_real_world_datasets.py`](../scripts/train_real_world_datasets.py)).
    - Sourced real-world cohorts: OASIS-2 Longitudinal Dementia MRI (373 visits, 150 subjects) and UCI / Oxford Parkinson's Telemonitoring (5,875 records, 42 patients).
    - Verified leakage-free `StratifiedGroupKFold` participant isolation, 95% bootstrap confidence intervals, and Brier score calibration:
      - OASIS-2 Dementia: Logistic Regression ROC-AUC **0.856** [0.818 – 0.893], PR-AUC: 0.879, Brier: 0.150.
      - UCI Parkinson's Telemonitoring: Random Forest ROC-AUC **0.766** [0.753 – 0.779], PR-AUC: 0.511, Brier: 0.171.
    - Integrated native connectors in `aimedres.data.open_datasets` (`ParkinsonsTelemonitoringDataset`) and `aimedres.research.datasets`.
    - Future extension: Multi-site external validation across BioFIND / DATATOP cohorts pending institutional data access.
- [x] **4.3 Clinician-in-the-Loop Usability & Human Factors**
  - **Objective**: Conduct formal usability studies and human factors engineering with neurologists and clinical fellows.
  - **Specification**:
    - Authored Human Factors Engineering protocol ([`docs/clinical/CLINICIAN_USABILITY_AND_HUMAN_FACTORS.md`](clinical/CLINICIAN_USABILITY_AND_HUMAN_FACTORS.md)) conforming to FDA 2016 Human Factors Guidance, ANSI/AAMI HE75, and IEC 62366-1:2015.
    - Implemented clinical usability engine ([`src/aimedres/clinical/usability.py`](../src/aimedres/clinical/usability.py)) computing model-driven signed feature attributions, waterfall contribution payloads, and clinical domain concordance against Movement Disorder Society (MDS) criteria.
    - Implemented decision turnaround tracking and standardized usability scoring: System Usability Scale (SUS $\ge 80/100$) and NASA-TLX cognitive workload ($\le 35/100$).
    - Verified sub-second real-time inference and attribution latency benchmark (< 200 ms) in [`tests/unit/test_usability_and_human_factors.py`](../tests/unit/test_usability_and_human_factors.py).
- [x] **4.4 Quality Management & SaMD Traceability**
  - **Objective**: Establish design controls proportional to prospective medical software.
  - **Specification**:
    - Built comprehensive SaMD Traceability Matrix ([`docs/compliance/SAMD_TRACEABILITY_MATRIX.md`](compliance/SAMD_TRACEABILITY_MATRIX.md)) conforming to IEC 62304 Software Safety Class B and ISO 14971 risk management principles.
    - Mapped 8 clinical hazards (`HAZ-001` - `HAZ-008`), 11 software requirements (`SRS-01` - `SRS-11`), and risk controls directly to automated unit test functions in `tests/unit/`.
    - Automated AST verification in [`tests/unit/test_clinical_and_compliance.py`](../tests/unit/test_clinical_and_compliance.py) guaranteeing zero stale requirements in continuous integration.

---

## 5. First Major Milestone: The PPMI Parkinson's Research Reproducibility Pack

The single most credible deliverable to present to an institutional partner, IRB, or scientific review committee is the **PPMI Parkinson's Research Reproducibility Pack**. Rather than adding speculative features, all immediate engineering converges on completing this pack:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│               PPMI PARKINSON'S RESEARCH REPRODUCIBILITY PACK                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero Restricted Records in Git + Synthetic CI Fixture                   │
│    (ParkinsonDatasets/DATA_GOVERNANCE.md & synthetic_pet_sbr_fixture.csv)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Pre-Registered Study Protocol & Data-Access Statement                    │
│    (Defensible non-circular endpoint, inclusion/exclusion, statistical plan)│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Leakage-Proof Participant-Level Pipeline                                 │
│    (GroupKFold on PATNO, fold-internal scaling, holdout test isolation)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Single Deterministic Execution Command                                   │
│    (python scripts/reproduce_ppmi_benchmark.py --config ...study.yml)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Machine-Readable Experiment Manifest & Evaluation Report                 │
│    (Data checksums, cohort counts, bootstrap CIs, calibration metrics)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. Automated CI Pipeline with Fixture Verification                          │
│    (Zero-leakage regression tests passing on every pull request)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. Honest, Evidence-Based Model Card                                        │
│    (Intended use, GMLP compliance, subgroup fairness, known failure modes)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8. Independent Review Checklist                                             │
│    (Signed off by biostatistician and movement-disorder neurologist)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Development & Execution Instructions

### 5.1 Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/V1B3hR/AiMedRes.git
cd AiMedRes

# 2. Configure Python environment (Python 3.10 - 3.12)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install core dependencies and development tools
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt

# 4. Run test suite (verifying leakage-safe pipelines and unit coverage)
pytest tests/
```

### 5.2 Executing the Parkinson's Research Pipeline

```bash
# Preview discovered training jobs without execution
python run_all_training.py --only parkinsons --dry-run

# Run the committed, leakage-safe synthetic PPMI benchmark configuration.
# This checks reproducibility controls only; it is not clinical validation.
python scripts/reproduce_ppmi_benchmark.py \
  --config configs/parkinsons_ppmi_study.yml

# When operating in an institution-approved restricted environment with registered PPMI access:
# python src/aimedres/training/train_parkinsons.py \
#   --data-path /secure/restricted/PPMI_Data/ \
#   --output-dir results/parkinsons_ppmi_official \
#   --folds 5
```

### 5.3 User Interfaces (Research Preview)

```bash
# Clinical Decision Support Web Dashboard
cd frontend
npm install
npm run dev

# Mobile Clinical Companion (Development Preview)
cd ../mobile
npm install
npm start
```

---

## 7. Governance & Review Cadence

- **Bi-Weekly Technical Review**: Verification of zero-leakage test runs, data governance audits, and model metric deltas.
- **Monthly Protocol Review**: Alignment with external clinical advisory members (neurology/biostatistics) regarding endpoints, covariates, and subgroup balance.
- **Milestone Gate Review**: Formal transition gate before moving between roadmap phases. Phase advancement requires 100% completion of acceptance criteria and automated test passes.
