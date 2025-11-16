# AiMedRes System Validation Guide

**Step 5: Initial System Validation**

This guide provides comprehensive procedures for validating the AiMedRes system before production deployment. All validation steps must be completed successfully before proceeding to production deployment (Step 6).

## Overview

System validation consists of three critical phases:

1. **Dry Run / Smoke Test** - Verify basic functionality with test data
2. **Model Verification** - Confirm models are loaded and performing correctly
3. **User Acceptance Testing (UAT)** - Validate system meets clinical requirements

## Prerequisites

- Completed Steps 1-4 of the Healthcare Deployment Plan
- System deployed in staging/test environment
- Access to test data (de-identified, no PHI)
- Test user accounts created with appropriate roles

---

## Phase 5a: Dry Run / Smoke Test

### Purpose
Verify that AiMedRes is properly installed and basic functionality works with test data (no PHI).

### Test Data Preparation

**Important**: All test data must be completely de-identified with NO PHI.

```bash
# Example test data creation
cd /path/to/aimedres/deployment/validation

# Use provided test data samples
python generate_test_data.py --output test_data/ --samples 100 --no-phi
```

### CLI Smoke Test

Test the command-line interface to ensure basic operations work:

```bash
# 1. Verify AiMedRes installation
aimedres --version
# Expected: aimedres version number

# 2. List available models
aimedres model list
# Expected: List of available trained models

# 3. Run model info command
aimedres model info alzheimer_v1
# Expected: Model metadata including version, metrics, status

# 4. Test data validation (with test data)
aimedres validate --data test_data/sample_clinical_data.json
# Expected: Validation results with PHI check status

# 5. Run smoke test script
python smoke_test_cli.py
# Expected: All tests pass
```

### API Smoke Test

Test the REST API endpoints:

```bash
# 1. Start API server (if not already running)
aimedres serve --host 127.0.0.1 --port 5000 &
# Wait for server to start
sleep 5

# 2. Check health endpoint
curl http://127.0.0.1:5000/health
# Expected: {"status": "healthy"}

# 3. Authenticate (get API token)
curl -X POST http://127.0.0.1:5000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user", "password": "test_password"}'
# Expected: {"token": "...", "expires_in": 3600}

# 4. List models via API
curl http://127.0.0.1:5000/api/v1/model/list \
  -H "Authorization: Bearer YOUR_TOKEN"
# Expected: {"models": [...], "total": N}

# 5. Run comprehensive API smoke test
python smoke_test_api.py --host 127.0.0.1 --port 5000
# Expected: All API tests pass
```

### Log Review

After running smoke tests, review system logs:

```bash
# Check application logs
tail -n 100 /var/log/aimedres/app.log

# Check for errors
grep -i "error\|exception\|failed" /var/log/aimedres/app.log

# Verify audit logging (if enabled)
tail -n 50 /var/log/aimedres/audit.log
```

**Checklist - Log Review:**
- [ ] No critical errors in application logs
- [ ] All test requests logged properly
- [ ] Audit trail complete for data access
- [ ] No unauthorized access attempts
- [ ] PHI scrubber activations logged (if PHI detected)

### System Resource Utilization

Monitor resource usage during smoke tests:

```bash
# Run resource monitor during tests
python monitor_resources.py --duration 300 --output resource_report.json

# Check resource report
cat resource_report.json
```

**Metrics to Review:**
- CPU utilization: Should be < 70% during normal operations
- Memory usage: Should not exceed 80% of available RAM
- Disk I/O: Monitor for bottlenecks
- Network latency: API response times < 500ms for most requests
- GPU utilization: If applicable, check model inference GPU usage

**Checklist - Dry Run / Smoke Test:**
- [ ] CLI commands execute successfully
- [ ] API endpoints respond correctly
- [ ] Authentication and authorization work
- [ ] Test data processed without errors
- [ ] Logs generated and accessible
- [ ] Resource utilization within acceptable limits
- [ ] No PHI in test data confirmed
- [ ] Results files created in correct locations

---

## Phase 5b: Model Verification

### Purpose
Confirm correct models are loaded, operational, and performing within expected parameters.

### List Available Models

**Via CLI:**
```bash
# List all models
aimedres model list

# Get detailed information for specific model
aimedres model info alzheimer_v1
aimedres model info parkinsons_v1
aimedres model info als_v1
```

**Via API:**
```bash
# List models via API
curl http://127.0.0.1:5000/api/v1/model/list \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get model card
curl http://127.0.0.1:5000/api/v1/model/card?model_version=alzheimer_v1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Model Verification Script

Run the comprehensive model verification:

```bash
# Run model verification
python model_verification.py --all-models --validation-data validation_datasets/

# Expected output:
# ✓ Alzheimer model v1.0.0: LOADED
# ✓ Parkinson model v1.0.0: LOADED
# ✓ ALS model v1.0.0: LOADED
# ✓ All models validation PASSED
```

### Benchmark Against Validation Datasets

Test models with known validation datasets:

```bash
# Run benchmark tests
python benchmark_models.py \
  --models alzheimer_v1,parkinsons_v1,als_v1 \
  --validation-data validation_datasets/ \
  --output benchmark_report.json

# Review benchmark report
cat benchmark_report.json
```

**Expected Performance Thresholds:**

| Model | Metric | Expected | Acceptable Range |
|-------|--------|----------|------------------|
| Alzheimer v1 | Accuracy | 0.89 | 0.85 - 0.93 |
| Alzheimer v1 | AUC-ROC | 0.93 | 0.90 - 0.95 |
| Parkinson v1 | R² Score | 0.82 | 0.78 - 0.86 |
| Parkinson v1 | MAE | 0.12 | 0.10 - 0.15 |
| ALS v1 | Accuracy | 0.85 | 0.82 - 0.88 |
| ALS v1 | Sensitivity | 0.88 | 0.85 - 0.91 |

### Model Output Validation

Verify model outputs are formatted correctly:

```bash
# Test model inference output format
python test_model_outputs.py \
  --model alzheimer_v1 \
  --test-cases test_data/alzheimer_test_cases.json

# Verify output structure
python validate_output_schema.py --output test_results/
```

**Output Requirements:**
- Predictions include confidence scores
- Results contain required metadata (model version, timestamp)
- Output format matches API specification
- Error handling works correctly for invalid inputs

**Checklist - Model Verification:**
- [ ] All expected models loaded successfully
- [ ] Model versions match deployment specification
- [ ] Validation metrics within acceptable ranges
- [ ] Model inference produces expected output format
- [ ] Edge cases handled correctly
- [ ] Performance benchmarks meet requirements
- [ ] No model loading errors or warnings
- [ ] Model cards accessible and complete

---

## Phase 5c: User Acceptance Testing (UAT)

### Purpose
Involve clinical stakeholders to validate that the system meets real-world requirements using scenario-based testing with de-identified data.

### UAT Preparation

**1. Identify UAT Participants:**
- Clinicians (physicians, nurses)
- Clinical researchers
- IT staff
- Compliance officer

**2. Prepare Test Environment:**
```bash
# Create UAT environment
python setup_uat_environment.py \
  --users uat_participants.json \
  --test-data uat_test_datasets/

# Verify UAT data is de-identified
python verify_deidentified_data.py --data uat_test_datasets/
```

**3. Create Test Scenarios:**
See `uat_scenarios.md` for detailed test scenarios.

### UAT Test Scenarios

#### Scenario 1: Alzheimer's Early Detection
**Objective:** Assess system's ability to process clinical data for Alzheimer's risk assessment

**Test Data:** De-identified patient records with cognitive assessment scores

**Steps:**
1. Log in to AiMedRes with clinician credentials
2. Upload test patient data (CSV or via API)
3. Select Alzheimer detection model
4. Review model predictions and confidence scores
5. Assess clinical relevance of results
6. Export results for documentation

**Acceptance Criteria:**
- [ ] System accepts valid clinical data
- [ ] Risk assessment completes within 30 seconds
- [ ] Results include clear risk level and confidence
- [ ] Clinician can understand and interpret output
- [ ] Results can be exported in standard format

#### Scenario 2: Parkinson's Progression Tracking
**Objective:** Test longitudinal tracking capabilities

**Test Data:** De-identified patient with multiple assessment timepoints

**Steps:**
1. Load historical patient assessments
2. Run progression analysis
3. Review progression metrics and visualizations
4. Compare with clinical expectations
5. Generate progression report

**Acceptance Criteria:**
- [ ] System handles multiple timepoints correctly
- [ ] Progression trends displayed clearly
- [ ] Metrics align with clinical understanding
- [ ] Visualization aids interpretation
- [ ] Report suitable for clinical review

#### Scenario 3: Multi-Model Assessment
**Objective:** Test workflow with multiple disease models

**Test Data:** Complex case requiring assessment across models

**Steps:**
1. Load patient data
2. Run assessments on multiple relevant models
3. Review integrated results
4. Compare cross-model findings
5. Document clinical decision support

**Acceptance Criteria:**
- [ ] Multiple models run efficiently
- [ ] Results integrated coherently
- [ ] No conflicts in UI or workflow
- [ ] Clinical decision-making supported
- [ ] Performance remains acceptable

#### Scenario 4: Error Handling and Edge Cases
**Objective:** Validate system robustness with problematic data

**Test Cases:**
- Incomplete data
- Out-of-range values
- Unsupported data formats
- PHI in input (to test scrubber)

**Acceptance Criteria:**
- [ ] System rejects invalid data gracefully
- [ ] Clear error messages provided
- [ ] PHI scrubber activates when needed
- [ ] No system crashes or freezes
- [ ] Audit logs capture all events

### UAT Feedback Collection

**Feedback Template:**

```json
{
  "scenario_id": "scenario_1",
  "participant": "Dr. Smith (Neurologist)",
  "date": "2025-01-15",
  "ratings": {
    "ease_of_use": 4,
    "clinical_relevance": 5,
    "performance": 4,
    "output_clarity": 5
  },
  "comments": [
    "Results are clinically meaningful",
    "Would like to see confidence intervals",
    "Performance is acceptable"
  ],
  "issues": [],
  "acceptance": "approved"
}
```

**Collect Feedback:**
```bash
# Aggregate UAT feedback
python collect_uat_feedback.py \
  --feedback-dir uat_feedback/ \
  --output uat_summary_report.json

# Generate UAT report
python generate_uat_report.py \
  --feedback uat_summary_report.json \
  --output uat_final_report.pdf
```

### UAT Sign-Off

All UAT participants must review and sign off:

**Sign-Off Checklist:**
- [ ] All test scenarios completed successfully
- [ ] System meets clinical requirements
- [ ] Performance acceptable for clinical use
- [ ] Security controls verified
- [ ] No blocking issues identified
- [ ] Recommendations documented
- [ ] Training needs identified
- [ ] Ready for production deployment

**Sign-Off Form:**
- Participant name and role
- Date of testing
- Scenarios tested
- Overall assessment (Approved / Approved with conditions / Not approved)
- Signature

---

## Validation Summary Report

After completing all validation phases, generate a comprehensive report:

```bash
# Generate full validation report
python generate_validation_report.py \
  --smoke-test-results smoke_test_results.json \
  --model-verification model_verification_results.json \
  --benchmark benchmark_report.json \
  --uat-feedback uat_summary_report.json \
  --resource-monitoring resource_report.json \
  --output deployment_validation_report.pdf
```

**Report Contents:**
1. Executive Summary
2. Smoke Test Results
3. Model Verification Results
4. Performance Benchmarks
5. UAT Findings and Feedback
6. Resource Utilization Analysis
7. Issues and Resolutions
8. Recommendations
9. Sign-Off Documentation

---

## Troubleshooting Common Issues

### Issue: Models Not Loading
**Symptoms:** Model list returns empty or errors
**Solutions:**
```bash
# Check model files exist
ls -la /path/to/models/

# Verify model registry
python verify_model_registry.py

# Rebuild model index
python rebuild_model_index.py
```

### Issue: API Authentication Failing
**Symptoms:** 401 Unauthorized errors
**Solutions:**
```bash
# Verify auth configuration
cat .env | grep AUTH

# Reset test user
python reset_test_users.py

# Check JWT secret
python verify_auth_config.py
```

### Issue: High Resource Usage
**Symptoms:** Slow response times, high CPU/memory
**Solutions:**
```bash
# Review resource limits
docker stats aimedres

# Check for resource leaks
python check_memory_leaks.py

# Optimize configuration
python optimize_config.py --profile production
```

### Issue: PHI Detected in Test Data
**Symptoms:** PHI scrubber warnings in logs
**Solutions:**
```bash
# Re-sanitize test data
python sanitize_test_data.py --input test_data/ --output test_data_clean/

# Verify de-identification
python verify_deidentified_data.py --data test_data_clean/

# Regenerate test data
python generate_test_data.py --no-phi --strict
```

---

## Validation Completion Criteria

All criteria must be met before proceeding to production:

### Technical Validation
- [x] All smoke tests pass (CLI and API)
- [x] Models load and respond correctly
- [x] Performance benchmarks within thresholds
- [x] Resource utilization acceptable
- [x] No critical errors in logs
- [x] Security controls functioning

### Clinical Validation
- [x] UAT scenarios completed successfully
- [x] Clinical stakeholders approve
- [x] Output clinically relevant and interpretable
- [x] Workflow meets clinical needs
- [x] Training plan developed

### Compliance Validation
- [x] PHI scrubber tested and working
- [x] Audit logging complete
- [x] Access controls verified
- [x] Data de-identification confirmed
- [x] Documentation complete

---

## Next Steps

Upon successful validation:

1. **Document Results:** Archive all validation artifacts
2. **Address Feedback:** Resolve any non-blocking issues from UAT
3. **Finalize Training:** Complete end-user training
4. **Update Documentation:** Ensure all docs reflect validated system
5. **Proceed to Step 6:** Production Deployment (per Healthcare Deployment Plan)

---

## Validation Artifacts

Save all validation artifacts for audit and compliance:

```
deployment/validation/
├── smoke_test_results.json
├── model_verification_results.json
├── benchmark_report.json
├── resource_report.json
├── uat_summary_report.json
├── uat_final_report.pdf
├── deployment_validation_report.pdf
├── sign_off_forms/
│   ├── clinical_sign_off.pdf
│   ├── technical_sign_off.pdf
│   └── compliance_sign_off.pdf
└── logs/
    ├── validation_app.log
    └── validation_audit.log
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-16  
**Next Review:** Before each deployment

---

## References

- Healthcare Deployment Plan: `/healthcaredeploymentplan.md`
- API Documentation: `/docs/API_REFERENCE.md`
- Model Documentation: `/docs/models/`
- Security Guide: `/SECURITY.md`
- Test Data Generators: `deployment/validation/generate_test_data.py`
