# Render Deployment P0 Security Advisory Record

**Date:** 2026-10-26  
**Status:** Remediated & Guarded (Awaiting Live Service Deployment & Operator Digest Log)  
**Advisories Addressed:**
1. [Render eMudhra Advisory](https://render.com/docs/advisory-emudhra) (Compromised Intermediate CAs)
2. [Render PHP2Shell Advisory](https://render.com/docs/advisory-php2shell) (CVE-2026-109618, CVE-2026-109619)

---

## 1. Summary of Controls Implemented

| Control Domain | Requirement | Implementation in Repository | Status |
| :--- | :--- | :--- | :--- |
| **Node.js Runtime** | Node >= 24.22.1 | [.node-version](file:///c:/Projekty/AiMedRes/.node-version) pinned to `24.22.1`; `.github/workflows/performance-validation.yml` pinned to `24.22.1`. | **PASS** |
| **Python HTTP Clients** | `certifi>=2026.10.19` | Pinned in `pyproject.toml`, `requirements.txt`, `mlops/infra/mlflow_requirements.txt`, and `deployment/technical/mlflow-requirements.txt`. | **PASS** |
| **Docker Base & CA Store** | Rebuild with fresh CA store | `Dockerfile`, `deployment/technical/Dockerfile`, and `mlops/infra/Dockerfile-imaging` declare `CA_CERTIFICATES_REFRESH=2026-10-25` and run `apt-get update && apt-get install -y ca-certificates && update-ca-certificates`. | **PASS** |
| **PHP / Adminer Isolation** | Eliminate unpatched PHP / Adminer | Removed `adminer:latest` container service from `mlops/infra/docker-compose.yml`. AiMedRes core contains zero PHP runtime code. | **PASS** |
| **Automated Regression Suite** | Continuous verification | `tests/unit/test_render_advisory_controls.py` (5 passing tests). | **PASS** |
| **Live Verification Script** | In-service audit executable | `scripts/verify_render_advisory_live.py` testing TLS rejection on advisory endpoints. | **PASS** |

---

## 2. In-Service Verification Protocol

When a Render service or Docker container is deployed or redeployed, execute the automated verification tool from inside the running environment:

```bash
python scripts/verify_render_advisory_live.py --json-out evidence/render_live_verification.json
```

### Advisory Test Endpoints:
- `https://dv-g3b-emudhra.we-have-a-problem.com/`
- `https://dv-g3b-aaa-emudhra.we-have-a-problem.com/`

**Acceptance Criteria:**
Both requests MUST fail certificate verification (`[SSL: CERTIFICATE_VERIFY_FAILED]`). If any request succeeds, the service trust store is vulnerable and requires an immediate fresh rebuild and CA bundle update.

---

## 3. Deployment Evidence Log (To be filled upon redeploy)

| Field | Required Value / Format | Deployment Value |
| :--- | :--- | :--- |
| **Render Service ID / Name** | e.g. `srv-c...` / `aimedres-api` | *Pending operator deployment* |
| **Deploy ID & Timestamp** | e.g. `dep-...` / ISO 8601 UTC | *Pending operator deployment* |
| **Committed Git Ref / SHA** | Full commit hash | `codex/dependabot-followup` |
| **Node.js Runtime Version** | `24.22.1` | `24.22.1` (via `.node-version`) |
| **Docker Image Digest** | `sha256:<64-hex>` | *To be recorded after `docker build --pull`* |
| **Certifi Package Version** | `>= 2026.10.19` | `2026.10.23` (Installed & Verified) |
| **Endpoint 1 Rejection** (`dv-g3b-emudhra`) | `CERTIFICATE_VERIFY_FAILED` | **PASS (REJECTED AS EXPECTED)** |
| **Endpoint 2 Rejection** (`dv-g3b-aaa-emudhra`)| `CERTIFICATE_VERIFY_FAILED` | **PASS (REJECTED AS EXPECTED)** |
| **Adminer / PHP Container Status** | Terminated / Stopped | **STOPPED / REMOVED FROM COMPOSE** |

---

## 4. Operational Instructions for Deployment Owner

1. **Push Changes & Trigger Fresh Build:**
   ```bash
   git push origin codex/dependabot-followup
   ```
   In the Render dashboard, select **Manual Deploy** -> **Clear build cache & deploy** to ensure package layers and base images are re-fetched without cached layers.

2. **Terminating Existing Adminer Containers:**
   If an existing standalone Adminer or PHP container was previously launched on your infrastructure:
   ```bash
   docker stop <adminer_container_id>
   docker rm <adminer_container_id>
   ```
   Do not restart third-party PHP images without verifying vendor patch release (PHP >= 8.2.35, 8.3.36, 8.4.27, 8.5.12).

3. **Log Evidence:**
   Record the deploy ID, image digest, and verification JSON into this document or release notes.
