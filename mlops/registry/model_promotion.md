# Model Promotion Guidelines

This document outlines the guidelines and procedures for promoting models through different stages in the DuetMind Adaptive MLOps pipeline.

## Model Lifecycle Stages

### 1. Staging
- **Purpose**: Initial deployment for testing and validation
- **Requirements**: 
  - Model has passed all validation tests
  - Performance metrics meet minimum thresholds
  - Schema compatibility verified
- **Access**: Development and testing teams only
- **Duration**: Typically 7-14 days

### 2. Production
- **Purpose**: Live serving for end users
- **Requirements**:
  - Successful staging period with no critical issues
  - Performance equal or better than current production model
  - Business stakeholder approval
  - Drift monitoring configured
- **Access**: Production system and authorized personnel
- **Monitoring**: Continuous performance and drift monitoring

### 3. Archived
- **Purpose**: Deprecated models kept for compliance and rollback
- **Trigger**: Model replaced by newer version or deprecated
- **Retention**: Minimum 1 year for compliance

## Promotion Criteria

### Performance Thresholds

#### Alzheimer Classification Model
- **Minimum Accuracy**: 85%
- **Minimum ROC AUC**: 0.80
- **Maximum Prediction Latency**: 100ms
- **Feature Schema**: Must match approved feature view

#### General Requirements
- Model size < 100MB for deployment efficiency
- Memory usage < 1GB during inference
- CPU utilization < 80% under load

## Promotion Process

### Automated Checks
1. **Schema Validation**: Feature hash must match registered feature view
2. **Performance Validation**: Metrics must exceed thresholds
3. **Data Quality**: Training data must pass quality checks
4. **Model Artifacts**: All required artifacts present and valid

### Manual Approval Gates
1. **Data Science Review**: Model architecture and training approach
2. **DevOps Review**: Deployment and infrastructure readiness  
3. **Business Review**: Model meets business requirements
4. **Security Review**: Model artifacts scanned for vulnerabilities

## Promotion Commands

### CLI Commands
```bash
# Promote model from staging to production
python mlops/registry/promote_model.py \
  --model-name alzheimer_classifier \
  --version 1.2.0 \
  --from-stage staging \
  --to-stage production \
  --approver "john.doe@company.com"

# Check promotion eligibility
python mlops/registry/check_promotion.py \
  --model-name alzheimer_classifier \
  --version 1.2.0
```

### Database Updates
```sql
-- Promote model version to production
UPDATE model_version_detail 
SET status = 'production', 
    promoted_at = NOW() 
WHERE model_id = (SELECT id FROM model_registry WHERE name = 'alzheimer_classifier')
  AND version = '1.2.0';

-- Archive previous production version
UPDATE model_version_detail 
SET status = 'archived', 
    deprecated_at = NOW()
WHERE model_id = (SELECT id FROM model_registry WHERE name = 'alzheimer_classifier')
  AND status = 'production'
  AND version != '1.2.0';
```

## Rollback Procedures

### Immediate Rollback
If critical issues are detected in production:

1. **Automated Rollback**: Monitor triggers automatic rollback if:
   - Prediction accuracy drops > 10%
   - Error rate exceeds 5%
   - Response time exceeds 500ms

2. **Manual Rollback**: 
   ```bash
   python mlops/registry/rollback_model.py \
     --model-name alzheimer_classifier \
     --to-version 1.1.0 \
     --reason "Performance degradation detected"
   ```

### Rollback Validation
After rollback:
- Verify service health metrics
- Confirm prediction quality
- Update monitoring dashboards
- Notify stakeholders

## Monitoring and Alerts

### Production Model Monitoring
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Infrastructure Metrics**: Latency, throughput, error rates
- **Data Drift**: Feature distribution changes
- **Concept Drift**: Model performance degradation

### Alert Thresholds
- **Critical**: Accuracy drop > 5%, error rate > 2%
- **Warning**: Accuracy drop > 2%, latency > 200ms
- **Info**: Minor performance fluctuations

### Alert Channels
- **Critical**: PagerDuty + Slack + Email
- **Warning**: Slack + Email
- **Info**: Slack channel only

## Compliance and Audit

### Model Registry Records
All promotions must maintain:
- Promotion timestamp and approver
- Performance metrics snapshot
- Feature schema version
- Training data lineage
- Model artifact checksums

### Audit Trail
- All promotion decisions logged
- Performance history maintained
- Rollback events documented
- Compliance reports generated monthly

## Best Practices

### Pre-Promotion Checklist
- [ ] Model trained on latest approved dataset
- [ ] Feature schema matches production requirements
- [ ] Performance meets or exceeds thresholds
- [ ] A/B testing completed (if applicable)
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Rollback plan verified

### Post-Promotion Tasks
- [ ] Update model serving endpoint
- [ ] Configure monitoring dashboards
- [ ] Notify stakeholders of deployment
- [ ] Schedule performance review
- [ ] Archive previous version artifacts

### Continuous Improvement
- Monthly review of promotion criteria
- Quarterly assessment of model performance
- Annual review of promotion process
- Feedback incorporation from incidents

## Contact Information

- **Data Science Team**: ds-team@duetmind.ai
- **MLOps Team**: mlops@duetmind.ai
- **DevOps Team**: devops@duetmind.ai
- **On-Call**: +1-555-ML-ALERT

## Related Documentation

- [Model Training Guide](../TRAINING_GUIDE.md)
- [Deployment Architecture](../../docs/MLOPS_ARCHITECTURE.md)
- [Monitoring Setup](../monitoring/README.md)
- [Incident Response](../security/INCIDENT_RESPONSE.md)