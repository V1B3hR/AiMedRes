# API Performance SLOs (Service Level Objectives)

**Version**: 1.0.0 | **Last Updated**: November 2025

## Overview

This document defines the performance Service Level Objectives for the AiMedRes API. These SLOs are used to ensure the system meets required performance standards for clinical use.

## Latency SLOs

### General API Endpoints

| Endpoint Type | p50 (50th percentile) | p95 (95th percentile) | p99 (99th percentile) |
|---------------|----------------------|----------------------|----------------------|
| Health Check  | ≤ 50 ms              | ≤ 200 ms             | ≤ 500 ms             |
| Authentication| ≤ 100 ms             | ≤ 300 ms             | ≤ 800 ms             |
| Case Listing  | ≤ 200 ms             | ≤ 500 ms             | ≤ 1000 ms            |
| Case Detail   | ≤ 150 ms             | ≤ 400 ms             | ≤ 900 ms             |

### Model Inference Endpoints

| Endpoint Type           | p50          | p95          | p99          |
|------------------------|--------------|--------------|--------------|
| Model Inference        | ≤ 500 ms     | ≤ 2000 ms    | ≤ 5000 ms    |
| Explainability         | ≤ 300 ms     | ≤ 1000 ms    | ≤ 2500 ms    |
| Model Card Retrieval   | ≤ 100 ms     | ≤ 300 ms     | ≤ 800 ms     |

### FHIR Integration Endpoints

| Endpoint Type         | p50          | p95          | p99          |
|----------------------|--------------|--------------|--------------|
| Patient List         | ≤ 300 ms     | ≤ 800 ms     | ≤ 1500 ms    |
| Patient Detail       | ≤ 200 ms     | ≤ 600 ms     | ≤ 1200 ms    |
| Observations         | ≤ 400 ms     | ≤ 1000 ms    | ≤ 2000 ms    |

## Throughput SLOs

### Requests Per Second (RPS)

| Environment | Sustained RPS | Peak RPS | Notes |
|-------------|--------------|----------|-------|
| Development | 10 RPS       | 20 RPS   | Single instance |
| Staging     | 50 RPS       | 100 RPS  | 2 instances |
| Production  | 200 RPS      | 500 RPS  | Auto-scaling enabled |

## Availability SLOs

| Metric | Target | Measurement Period |
|--------|--------|-------------------|
| Uptime | ≥ 99.5% | Monthly |
| API Availability | ≥ 99.9% | Monthly |
| Maximum Unplanned Downtime | ≤ 3.6 hours | Monthly |

## Error Rate SLOs

| Error Type | Target | Notes |
|-----------|--------|-------|
| 5xx Errors | ≤ 0.1% | Server errors |
| 4xx Errors | ≤ 5% | Client errors (excluding 401/403) |
| Timeout Rate | ≤ 0.5% | Request timeouts |

## Data Consistency SLOs

| Metric | Target | Notes |
|--------|--------|-------|
| Audit Log Write Success | ≥ 99.99% | Critical for compliance |
| Data Replication Lag | ≤ 1 second | For multi-region deployments |
| Backup Completion | 100% | Daily backups must complete |

## Load Test Requirements

### Baseline Load Test

- **Concurrent Users**: 20
- **Test Duration**: 10 minutes
- **Request Distribution**:
  - 40% read operations (GET)
  - 30% model inference (POST)
  - 20% case management (GET/POST)
  - 10% explainability (POST)

### Stress Test

- **Concurrent Users**: 100
- **Test Duration**: 5 minutes
- **Expected Behavior**: System should gracefully degrade without crashes

### Spike Test

- **Baseline**: 20 concurrent users
- **Spike**: 200 concurrent users for 1 minute
- **Expected Behavior**: System should handle spike and return to normal

## Monitoring and Alerting

### Critical Alerts (Page immediately)

- p95 latency > 2x SLO for 5 minutes
- Error rate > 1% for 5 minutes
- Availability < 99.9% for 5 minutes
- Audit log failures > 0

### Warning Alerts (Notify on-call)

- p95 latency > 1.5x SLO for 10 minutes
- Error rate > 0.5% for 10 minutes
- Throughput < 50% of expected for 10 minutes

## Testing Schedule

| Test Type | Frequency | Owner |
|-----------|-----------|-------|
| Performance Regression Tests | Every release | QA Team |
| Load Tests | Weekly | DevOps Team |
| Stress Tests | Monthly | DevOps Team |
| Full System Load Test | Quarterly | Engineering Team |

## SLO Review Process

- **Review Frequency**: Quarterly
- **Review Team**: Engineering, Product, Clinical
- **Success Criteria**: 95% of measurements meet SLOs
- **Remediation**: If SLOs not met for 2 consecutive quarters, initiate performance improvement project

## Pilot-Specific SLOs

For the initial clinical pilot phase:

| Metric | Pilot Target | Production Target |
|--------|-------------|------------------|
| Concurrent Users | 5-10 | 100+ |
| Daily API Calls | < 1,000 | < 100,000 |
| p50 Latency | ≤ 200 ms | ≤ 100 ms |
| p95 Latency | ≤ 1000 ms | ≤ 500 ms |
| Uptime | ≥ 99% | ≥ 99.9% |

## Notes

1. **Clinical Context**: Some latency is acceptable given the non-emergency nature of the application. Priority is accuracy over speed.

2. **Staging Environment**: Staging SLOs may be 2-3x higher due to resource constraints.

3. **Cold Start**: First request after idle may exceed SLOs (model loading). This is acceptable.

4. **Network Latency**: SLOs include network latency from client to API gateway.

5. **Data Volume**: SLOs assume typical case data size (< 1MB). Large imaging data may require different SLOs.

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-01-24 | 1.0 | Initial SLO definition | DevOps Team |

## References

- Performance test scripts: `tests/performance/test_api_performance.py`
- Monitoring dashboards: [Link to Grafana dashboards]
- Incident response: [Link to runbook]
