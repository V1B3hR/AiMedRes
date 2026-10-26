# Known Limitations and Future Improvements

## P0 Audit Status — 25 October 2026

**Scope of this initial pass:** maintained research API launch path, authentication decorator, public-bind safety gates, PPMI reproducibility runner, and their focused regression tests. This is a local code review and test run; it is **not** a penetration test, external security assessment, clinical validation, or dependency vulnerability scan.

### Resolved in this pass

- API keys are accepted only through `X-API-Key`; query-string credentials are rejected to avoid retention in URL histories, referrer headers, and proxy/access logs. The behaviour has a regression test.
- Direct execution of `src/aimedres/api/server.py` no longer enables Flask debug mode.
- `aimedres serve`, `aimedres-serve`, the root compatibility wrapper, and Docker now route to `aimedres.cli.secure_serve`, which starts the maintained `SecureAPIServer` path. A non-loopback bind requires production/strict-security mode; debug is rejected there.
- The Docker image now declares production mode and intentionally fails closed until a runtime `FLASK_SECRET_KEY` and `JWT_SECRET` are supplied.
- The PPMI runner now forwards its versioned seed, bootstrap, batch-size, and learning-rate parameters through the research wrapper. Two local runs of the synthetic fixture with the same configuration produced identical classical and neural summary metrics on the same Windows environment.
- Historical Alzheimer/ALS/Parkinson cards and heuristic fallback predictions have been removed from the maintained model API. The remaining synthetic-fixture card is metadata only; API inference is deliberately disabled.
- The maintained API now also rejects heuristic agent output; `src/aimedres/api/mock_server.py` is local-only and runs without Flask debug mode.
- The maintained API exposes model metadata only. FHIR, case-review, explainability, DICOM, and P3 route modules are not registered because they are still mock or prototype scaffolding.
- Legacy `SecureMedicalDataProcessor` inference paths now refuse to deserialize encrypted Python pickles. Encryption is not treated as a trust boundary for executable serialization.
- The 25 October Render/eMudhra response pins Node.js `24.22.1` for Render discovery and the frontend CI workflow, requires `certifi>=2026.10.19` for Python environments, and refreshes `ca-certificates` in all project Dockerfiles during a fresh rebuild. The optional PHP-based Adminer service was removed from the MLOps compose file; an already running container still needs an operator-led patch/redeploy.

### Open risks and required follow-up

1. **P1 — trusted model artifacts:** model serving is disabled in the maintained API. Legacy `pickle` deserialization remains in `src/aimedres/core/production_agent.py`, `src/aimedres/training/automl.py`, and `src/aimedres/training/train_als.py`; none may receive untrusted artifacts. Before enabling any artifact, restrict its write path, verify a signed digest against a reviewed manifest, and migrate where feasible to a safer interchange format.
2. **P1 — legacy surface:** `src/aimedres/core/production_agent.py` remains importable for compatibility and contains a permissive CORS fallback plus legacy cache deserialization. It is not the supported API launch path. Quarantine, remove, or harden it before exposing any service; do not cite it as production capability.
3. **P2 — legacy command execution:** the known `shell=True` usage in `src/aimedres/core/verify_training.py` was converted to explicit argument lists and covered by a regression test. Other legacy/example scripts remain outside the maintained API scope and need review before reuse with untrusted input.
4. **P1 — deployment evidence:** the new Docker command was statically and CLI-tested, but no container image build, TLS termination test, or deployed-environment test has been performed in this pass. The built-in Flask server remains a research/development runtime, not a substitute for a reviewed production WSGI and network deployment.
5. **P1 — independent assurance:** no external penetration test, formal privacy assessment, SBOM/CVE audit, or review of real patient-data handling has been performed. These remain prerequisites for any institutional or clinical use.
6. **P1 — reproducibility scope:** local repeatability on one synthetic fixture does not establish reproducibility across hardware, Python/PyTorch versions, datasets, or independent teams. Those checks belong to the Evidence Pack and partner-led validation.
7. **P2 — dependency forward compatibility:** the local PPMI test run emits scikit-learn's deprecation warning for `SVC(probability=True)`. Before upgrading to the announced removal version, replace it with an explicitly validated calibration design that preserves participant-group isolation; do not change it only to silence the warning.
8. **P1 — local training artifacts:** the Parkinson's runner writes `.pkl` model artifacts into its ignored local output directory. They are excluded from the public Evidence Pack and must not be loaded from an untrusted location. Any future supported serving workflow needs signed provenance and a reviewed interchange format.
9. **P0 — deployment confirmation after eMudhra advisory:** no Render service, deployed image digest, runtime Node version, fresh build, redeploy, or in-service certificate-rejection check was available in this local workspace. The source changes reduce the risk for the next build but do not prove that any existing deployment is patched.

The P0 exit gate remains open until these items are dispositioned and the claims register, independent methodological review, and Evidence Pack are complete.

## Current Limitations

### 1. Cornerstone.js Version
**Status**: Using cornerstone-core@2.6.1

**Note**: This is a stable version but not the latest. The Cornerstone.js ecosystem has moved to Cornerstone3D for better performance and features.

**Future Improvement**: Consider migrating to @cornerstonejs/core (Cornerstone3D) for:
- Better performance with GPU acceleration
- Volume rendering support
- Better multi-planar reconstruction (MPR)
- Improved streaming capabilities

**Migration Path**:
```bash
npm uninstall cornerstone-core cornerstone-wado-image-loader
npm install @cornerstonejs/core @cornerstonejs/tools @cornerstonejs/dicom-image-loader
```

### 2. 3D Brain Visualization Performance
**Current**: Continuous rotation runs even when not visible

**Implemented**: Added `isRotating` state for pausing (currently unused)

**Future Improvement**: 
- Add pause/resume controls
- Implement visibility detection (pause when off-screen)
- Add performance mode for low-end devices

### 3. API Mocking for E2E Tests
**Current**: Tests assume live backend

**Future Improvement**:
- Add `cy.intercept()` for API mocking
- Create fixture data for consistent testing
- Reduce test execution time

### 4. PHI Scrubbing in Tests
**Current**: Basic pattern matching

**Future Improvement**:
- More sophisticated PHI detection
- Integration with actual PHI scrubber module
- Automated PII scanner in CI/CD

## Security Considerations

### 1. Rate Limiting
**Status**: ✅ Implemented in `src/aimedres/api/server.py` (Redis-backed sliding window with local fallback)

Current limits: `/health` → 30 req/min, `/api/v1/predict` → 60 req/min, `/api/v1/agent/think` → 30 req/min, `/api/v1/status` → 20 req/min

**Remaining work**: The legacy `secure_api_server.py` wrapper and `src/aimedres/cli/serve.py` do not apply these rate limits. Ensure all public-facing endpoints route through `src/aimedres/api/server.py` in production, or apply the same `RateLimiter` to remaining endpoints.

### 2. CORS Configuration
**Status**: ✅ Implemented in `src/aimedres/api/server.py` — origins are read from `self.config.security.allowed_origins`

**Remaining work**: Ensure `allowed_origins` is set explicitly in your production config (do **not** leave it as `["*"]`). The example configuration:
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://aimedres.example.com"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 3. Input Validation
**Status**: Basic validation in place

**Recommendation**: Add comprehensive validation middleware
- Use marshmallow or pydantic for request validation
- Add JSON schema validation
- Implement request size limits

## Performance Optimizations

### 1. Frontend Bundle Size
**Current**: All dependencies bundled

**Future**: 
- Code splitting for routes
- Lazy loading for heavy components (3D viewer, dashboards)
- Tree shaking for unused code

### 2. DICOM Streaming
**Current**: Basic streaming implementation

**Future**:
- Progressive loading for large series
- Image caching
- WebWorker for image processing
- HTTP/2 server push

### 3. Dashboard Performance
**Current**: Full re-render on updates

**Future**:
- React.memo for expensive components
- Virtual scrolling for large lists
- Debounced auto-refresh

## Testing Improvements

### 1. Unit Tests
**Status**: Frontend Vitest API unit tests and targeted pytest coverage are implemented

**Future**:
- Expand Vitest coverage from API modules into more frontend components
- Add broader pytest coverage for additional backend integration surfaces
- Add utility function tests

### 2. Integration Tests
**Status**: API security tests only

**Future**:
- Add full API integration tests
- Add database integration tests
- Add external service integration tests

### 3. Performance Tests
**Status**: Baseline implemented (`k6` + Lighthouse CI)

**Implemented**:
- GitHub Actions workflow: `.github/workflows/performance-validation.yml`
- k6 health-endpoint load test: `tests/performance/k6_api_load.js`
- Lighthouse CI configuration: `frontend/lighthouserc.json`

**Remaining work**:
- Expand k6 scenarios to additional production endpoints
- Add stress/soak profiles for higher sustained traffic
- Enforce stricter performance budgets as merge-gating policy

## Deployment Considerations

### 1. Container Configuration
**Status**: Root `Dockerfile` exists; K8s manifests and Helm chart are available for deployment orchestration

**Remaining work**:
- Publish production-ready container images through CI/CD
- Add a dedicated frontend container image if the UI is deployed independently
- Harden image provenance/signing for regulated environments

### 2. Kubernetes Deployment
**Status**: Kubernetes manifests exist in `k8s/` and Helm chart exists in `helm/aimedres/`

**Remaining work**:
- Add cloud-provider-specific production overlays
- Validate multi-region deployment and residency controls in target environments
- Automate release promotion through CI/CD

### 3. CI/CD Pipeline
**Recommendation**: Add GitHub Actions workflows
- Lint on PR
- Run tests on PR
- Build and push Docker images
- Deploy to staging/production
- Run security scans

## Monitoring & Observability

### 1. Application Monitoring
**Status**: Prometheus, Grafana, and Alertmanager configs are present in `monitoring/`

**Remaining work**:
- Add Loki/Jaeger or equivalent centralized logs and tracing
- Wire production dashboards and alerts into hosted infrastructure
- Add automated validation for observability configs

### 2. Frontend Monitoring
**Recommendation**: Add frontend monitoring
- Sentry for error tracking
- Google Analytics or Plausible for usage
- Web Vitals monitoring

### 3. Security Monitoring
**Status**: AI security monitoring is implemented in `src/aimedres/security/ai_security_monitoring.py`

**Remaining work**:
- Integrate hosted SIEM pipelines
- Add automated vulnerability scanning in CI/CD
- Conduct external penetration testing
- Expand long-term audit log analytics

## Documentation Improvements

### 1. API Documentation
**Future**: Add OpenAPI/Swagger documentation
- Auto-generate API docs
- Interactive API explorer
- Code generation for clients

### 2. Component Documentation
**Future**: Add Storybook for components
- Visual component documentation
- Interactive component playground
- Accessibility testing

### 3. User Documentation
**Future**: Add user guides
- Getting started guide
- Feature tutorials
- Troubleshooting guide
- FAQ section

## Compliance Enhancements

### 1. HIPAA Compliance
**Current**: Basic compliance measures

**Future**:
- Formal HIPAA compliance audit
- Business Associate Agreement templates
- Incident response procedures
- Regular compliance reviews

### 2. GDPR Compliance
**Current**: Basic data protection

**Future**:
- Data Protection Impact Assessment
- Privacy by design audit
- Right to be forgotten implementation
- Data portability features

### 3. FDA Compliance
**Current**: Research use only

**Future** (if targeting clinical use):
- 510(k) submission preparation
- Clinical validation studies
- Quality management system
- Risk management documentation

---

**Note**: These improvements should be prioritized based on:
1. Production requirements
2. User feedback
3. Security assessments
4. Compliance needs
5. Performance metrics

For immediate production deployment, focus on:
- Security hardening (rate limiting, CORS)
- Monitoring setup
- Container configuration
- CI/CD pipeline
