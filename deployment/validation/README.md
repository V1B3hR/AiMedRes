# AiMedRes System Validation

This directory contains comprehensive validation tools and documentation for Step 5 (Initial System Validation) of the Healthcare Deployment Plan.

## Overview

System validation must be completed successfully before production deployment. The validation process consists of three phases:

1. **Dry Run / Smoke Test** - Basic functionality verification
2. **Model Verification** - Model loading and performance validation
3. **User Acceptance Testing (UAT)** - Clinical stakeholder validation

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install psutil requests

# Ensure AiMedRes is installed
pip install -e /path/to/aimedres
```

### 1. Generate Test Data

```bash
# Generate synthetic de-identified test data
python generate_test_data.py --output test_data/ --samples 100 --no-phi
```

### 2. Run Smoke Tests

```bash
# CLI smoke test
python smoke_test_cli.py --verbose

# API smoke test (requires API server running)
aimedres serve --host 127.0.0.1 --port 5000 &
python smoke_test_api.py --host 127.0.0.1 --port 5000 --verbose
```

### 3. Verify Models

```bash
# Verify all models
python model_verification.py --all-models --verbose
```

### 4. Monitor Resources

```bash
# Monitor system resources during testing
python monitor_resources.py --duration 300 --output resource_report.json
```

### 5. Prepare for UAT

```bash
# Review UAT scenarios
cat uat_scenarios.md

# Generate UAT test data
python generate_test_data.py \
  --output uat_test_datasets/ \
  --samples 50 \
  --longitudinal 10 \
  --no-phi
```

## Files

### Documentation
- **`system_validation_guide.md`** - Complete validation procedures and requirements
- **`uat_scenarios.md`** - Detailed user acceptance testing scenarios
- **`README.md`** - This file

### Scripts

#### Smoke Testing
- **`smoke_test_cli.py`** - Automated CLI smoke tests
  ```bash
  python smoke_test_cli.py [--verbose]
  ```
  Tests: version, help, commands, module imports

- **`smoke_test_api.py`** - Automated API smoke tests
  ```bash
  python smoke_test_api.py --host HOST --port PORT [--verbose]
  ```
  Tests: health check, endpoints, authentication, error handling

#### Model Verification
- **`model_verification.py`** - Model loading and validation
  ```bash
  python model_verification.py --all-models [--verbose]
  python model_verification.py --models MODEL1,MODEL2 [--verbose]
  ```
  Verifies: model registry, metadata, validation metrics, status

#### Resource Monitoring
- **`monitor_resources.py`** - System resource monitoring
  ```bash
  python monitor_resources.py --duration SECONDS --interval SECONDS --output FILE
  ```
  Monitors: CPU, memory, disk, network, processes

#### Test Data Generation
- **`generate_test_data.py`** - Synthetic test data generator
  ```bash
  python generate_test_data.py --output DIR --samples N --longitudinal M --no-phi
  ```
  Generates: De-identified clinical data for all conditions

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   Step 5: System Validation                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐           ┌───────▼────────┐
         │ 5a. Dry Run │           │ 5b. Model      │
         │  Smoke Test │           │  Verification  │
         └──────┬──────┘           └───────┬────────┘
                │                           │
         ┌──────▼──────────────────────────▼────────┐
         │                                           │
    ┌────▼─────┐  ┌─────────┐  ┌──────────┐  ┌─────▼─────┐
    │   CLI    │  │   API   │  │  Model   │  │ Resource  │
    │  Tests   │  │  Tests  │  │  Tests   │  │ Monitor   │
    └────┬─────┘  └────┬────┘  └────┬─────┘  └─────┬─────┘
         │             │             │              │
         └─────────────┴─────────────┴──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  5c. User          │
                    │  Acceptance        │
                    │  Testing (UAT)     │
                    └─────────┬──────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          ┌──────▼──────┐         ┌───────▼────────┐
          │  Clinical   │         │   Feedback     │
          │  Scenarios  │         │   Collection   │
          └──────┬──────┘         └───────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Validation        │
                    │  Summary Report    │
                    │  & Sign-Off        │
                    └────────────────────┘
```

## Expected Results

### Smoke Tests
- **CLI Tests:** 7 tests, all passing
- **API Tests:** 6+ tests, all passing
- **Execution Time:** < 2 minutes

### Model Verification
- **Models Checked:** 3 (Alzheimer, Parkinson, ALS)
- **Verification Items:** Registry, metadata, metrics, status
- **Expected Result:** All models loaded, metrics within range

### Resource Monitoring
- **CPU:** < 70% during normal operations
- **Memory:** < 80% of available RAM
- **Response Time:** < 500ms for API requests

### UAT
- **Scenarios:** 6 comprehensive test scenarios
- **Participants:** 5 roles (clinical, researcher, IT, compliance)
- **Duration:** 2-5 days depending on scope
- **Outcome:** Sign-off from all stakeholders

## Validation Artifacts

All validation activities generate artifacts that must be saved for audit and compliance:

```
deployment/validation/
├── results/
│   ├── smoke_test_results.json
│   ├── api_smoke_test_results.json
│   ├── model_verification_results.json
│   ├── resource_report.json
│   ├── uat_summary_report.json
│   └── deployment_validation_report.pdf
├── test_data/
│   ├── alzheimer_test_data.json
│   ├── parkinsons_test_data.json
│   ├── als_test_data.json
│   └── README.md
├── uat_feedback/
│   ├── participant1_feedback.json
│   ├── participant2_feedback.json
│   └── ...
└── logs/
    ├── validation_app.log
    └── validation_audit.log
```

## Troubleshooting

### Smoke Tests Fail
- Verify AiMedRes is installed: `pip list | grep aimedres`
- Check Python version: `python --version` (requires 3.8+)
- Review error logs in output

### API Tests Fail
- Ensure API server is running: `ps aux | grep aimedres`
- Check port availability: `netstat -tuln | grep PORT`
- Verify firewall settings

### Model Verification Fails
- Check model files exist in expected locations
- Verify model registry configuration
- Review model loading logs

### Resource Monitoring Shows High Usage
- Check for other processes consuming resources
- Review system requirements in deployment plan
- Consider scaling resources or optimizing configuration

## Validation Completion Criteria

Before proceeding to production deployment, verify:

- [x] All smoke tests passed
- [x] All models verified and performing within thresholds
- [x] Resource utilization acceptable
- [x] UAT scenarios completed
- [x] Clinical stakeholders signed off
- [x] Security controls verified
- [x] All validation artifacts archived
- [x] Issues documented and resolved

## Next Steps

After successful validation:

1. Generate comprehensive validation report
2. Obtain all required sign-offs
3. Archive validation artifacts
4. Update deployment documentation
5. Proceed to **Step 6: Production Deployment**

## Support

For issues or questions:
- Review `system_validation_guide.md` for detailed procedures
- Check main deployment plan: `/healthcaredeploymentplan.md`
- Review application logs
- Contact technical support team

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-16  
**Next Review:** Before each deployment cycle
