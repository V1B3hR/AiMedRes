# Known Limitations and Future Improvements

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
**Status**: Not implemented

**Future**:
- Add Lighthouse CI for frontend
- Add load testing with k6
- Add stress testing for APIs

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
