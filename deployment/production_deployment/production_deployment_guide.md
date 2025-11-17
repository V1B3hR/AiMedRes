# Production Deployment Guide for AiMedRes

This guide provides comprehensive instructions for deploying AiMedRes to a production healthcare environment with enterprise-grade reliability, monitoring, and disaster recovery.

## Table of Contents

1. [Deployment to Production](#1-deployment-to-production)
2. [Monitoring & Support](#2-monitoring--support)
3. [Backups & Disaster Recovery](#3-backups--disaster-recovery)

---

## 1. Deployment to Production

### Overview

This section covers deploying AiMedRes containers/services to production with monitored rollout strategies including blue/green and canary deployments.

### 1.1 Pre-Deployment Checklist

Before deploying to production, ensure all validation steps from Section 5 are complete:

- [ ] All smoke tests passed (CLI and API)
- [ ] Models loaded and verified
- [ ] Performance benchmarks within thresholds
- [ ] UAT completed and signed off
- [ ] Security controls validated
- [ ] Network and firewall rules configured
- [ ] SSL/TLS certificates installed and valid
- [ ] Database migrations tested
- [ ] Backup systems verified
- [ ] Monitoring dashboards configured
- [ ] Alerting rules configured
- [ ] Runbooks documented
- [ ] Change control approval obtained

### 1.2 Production Environment Configuration

#### Environment Variables

Create a production environment file based on the template:

```bash
# Copy and customize production environment
cp deployment/technical/.env.healthcare.template /etc/aimedres/.env.production

# Edit with production values
nano /etc/aimedres/.env.production
```

**Critical Production Settings:**

```bash
# Application Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database (use production DB credentials)
DATABASE_URL=postgresql://aimedres_prod:SECURE_PASSWORD@prod-db.internal:5432/aimedres_prod
DATABASE_SSL_MODE=require
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis Cache
REDIS_URL=redis://prod-redis.internal:6379/0
REDIS_PASSWORD=SECURE_REDIS_PASSWORD
REDIS_SSL=true

# Security
SECRET_KEY=GENERATE_SECURE_KEY_HERE
JWT_SECRET_KEY=GENERATE_SECURE_JWT_KEY_HERE
API_KEY_SALT=GENERATE_SECURE_SALT_HERE
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Strict

# HTTPS/TLS
TLS_ENABLED=true
TLS_CERT_PATH=/etc/ssl/certs/aimedres.crt
TLS_KEY_PATH=/etc/ssl/private/aimedres.key
TLS_MIN_VERSION=1.2

# Model Storage
MODEL_STORAGE_PATH=/var/aimedres/models
MODEL_REGISTRY_URL=https://mlflow.internal.hospital.org
MLFLOW_TRACKING_URI=postgresql://mlflow:PASSWORD@prod-db.internal:5432/mlflow

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_URL=https://grafana.internal.hospital.org
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Audit Logging
AUDIT_LOG_PATH=/var/log/aimedres/audit.log
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years for HIPAA
SIEM_INTEGRATION_ENABLED=true
SYSLOG_SERVER=syslog.internal.hospital.org:514

# Backup
BACKUP_ENABLED=true
BACKUP_PATH=/var/backup/aimedres
BACKUP_RETENTION_DAYS=90
BACKUP_ENCRYPTION_KEY_ID=arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012

# Resource Limits
MAX_WORKERS=4
MAX_REQUESTS_PER_CHILD=1000
MEMORY_LIMIT=4G
CPU_LIMIT=2.0
```

#### Docker Compose Production Configuration

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  aimedres-web:
    image: aimedres:${VERSION:-latest}
    container_name: aimedres-web-prod
    restart: always
    env_file:
      - /etc/aimedres/.env.production
    ports:
      - "8080:8080"
    volumes:
      - /var/aimedres/models:/var/aimedres/models:ro
      - /var/log/aimedres:/var/log/aimedres
      - /etc/ssl/certs:/etc/ssl/certs:ro
      - /etc/ssl/private:/etc/ssl/private:ro
    networks:
      - aimedres-internal
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "production,aimedres,web"

  aimedres-worker:
    image: aimedres:${VERSION:-latest}
    container_name: aimedres-worker-prod
    restart: always
    command: celery -A aimedres.tasks worker --loglevel=info
    env_file:
      - /etc/aimedres/.env.production
    volumes:
      - /var/aimedres/models:/var/aimedres/models:ro
      - /var/log/aimedres:/var/log/aimedres
    networks:
      - aimedres-internal
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  nginx:
    image: nginx:1.25-alpine
    container_name: aimedres-nginx-prod
    restart: always
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/ssl/certs:/etc/ssl/certs:ro
      - /etc/ssl/private:/etc/ssl/private:ro
      - /var/log/nginx:/var/log/nginx
    networks:
      - aimedres-internal
    depends_on:
      - aimedres-web
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  aimedres-internal:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### 1.3 Deployment Strategies

AiMedRes supports multiple deployment strategies based on your update frequency and risk tolerance.

#### Strategy 1: Direct Deployment (Low-Frequency Updates)

Use for infrequent updates (monthly or less) in stable environments.

```bash
#!/bin/bash
# deploy_direct.sh - Direct production deployment

set -e

VERSION=${1:-latest}
BACKUP_DIR="/var/backup/aimedres/pre-deploy-$(date +%Y%m%d-%H%M%S)"

echo "Starting direct deployment of version $VERSION"

# 1. Create backup
echo "Creating pre-deployment backup..."
./backup.sh full "$BACKUP_DIR"

# 2. Health check current system
echo "Checking current system health..."
curl -f http://localhost:8080/health || echo "Warning: System not responding"

# 3. Pull new image
echo "Pulling new image..."
docker pull aimedres:$VERSION

# 4. Stop current services
echo "Stopping current services..."
docker-compose -f docker-compose.production.yml down

# 5. Start new services
echo "Starting new services..."
VERSION=$VERSION docker-compose -f docker-compose.production.yml up -d

# 6. Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# 7. Health check new system
echo "Checking new system health..."
for i in {1..10}; do
  if curl -f http://localhost:8080/health; then
    echo "System is healthy!"
    break
  fi
  echo "Attempt $i failed, retrying..."
  sleep 10
done

# 8. Verify models loaded
echo "Verifying models..."
curl -f http://localhost:8080/api/v1/models/list | jq .

# 9. Run smoke tests
echo "Running smoke tests..."
cd deployment/validation
python smoke_test_api.py --host localhost --port 8080

echo "Deployment complete!"
```

#### Strategy 2: Blue/Green Deployment (Medium-Frequency Updates)

Use for monthly/quarterly updates when you need instant rollback capability.

**Setup:**

```bash
#!/bin/bash
# setup_blue_green.sh - Setup blue/green deployment

# Create blue environment
docker network create aimedres-blue
docker network create aimedres-green

# Deploy blue (current production)
COMPOSE_PROJECT_NAME=aimedres-blue \
  docker-compose -f docker-compose.production.yml up -d

# Configure load balancer (nginx) to point to blue
# Update upstream in nginx.conf to blue backend
```

**Deployment Script:**

```bash
#!/bin/bash
# deploy_blue_green.sh - Blue/Green deployment

set -e

VERSION=${1:-latest}
CURRENT_ENV=${2:-blue}  # Current production environment
TARGET_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "Deploying version $VERSION to $TARGET_ENV environment"

# 1. Deploy to target environment
echo "Deploying to $TARGET_ENV..."
COMPOSE_PROJECT_NAME=aimedres-$TARGET_ENV \
  VERSION=$VERSION \
  docker-compose -f docker-compose.production.yml up -d

# 2. Wait for health
echo "Waiting for $TARGET_ENV to be healthy..."
sleep 30

TARGET_PORT=$([ "$TARGET_ENV" = "blue" ] && echo "8081" || echo "8082")
for i in {1..20}; do
  if curl -f http://localhost:$TARGET_PORT/health; then
    echo "$TARGET_ENV is healthy!"
    break
  fi
  sleep 10
done

# 3. Run smoke tests on target
echo "Running smoke tests on $TARGET_ENV..."
cd deployment/validation
python smoke_test_api.py --host localhost --port $TARGET_PORT

# 4. Switch traffic to target
echo "Switching traffic to $TARGET_ENV..."
./switch_traffic.sh $TARGET_ENV

# 5. Monitor for 5 minutes
echo "Monitoring new environment for 5 minutes..."
sleep 300

# 6. Check for errors
ERROR_COUNT=$(grep "ERROR" /var/log/aimedres/app.log | tail -n 100 | wc -l)
if [ $ERROR_COUNT -gt 10 ]; then
  echo "ERROR: Too many errors detected, rolling back!"
  ./switch_traffic.sh $CURRENT_ENV
  exit 1
fi

# 7. Success - old environment can be stopped
echo "Deployment successful! Old environment ($CURRENT_ENV) can be stopped."
echo "To stop old environment: docker-compose -f docker-compose.production.yml -p aimedres-$CURRENT_ENV down"
```

#### Strategy 3: Canary Deployment (High-Frequency Updates)

Use for weekly updates or continuous deployment with gradual rollout.

**Implementation using existing pipeline:**

```python
#!/usr/bin/env python3
"""
Production canary deployment script using AiMedRes MLOps pipeline.
"""

import sys
import os
sys.path.insert(0, '/var/aimedres/src')

from mlops.pipelines.canary_deployment import (
    CanaryPipeline, CanaryConfig, DeploymentStatus
)
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_canary(model_id: str, model_version: str, auto_promote: bool = False):
    """Deploy model using canary strategy."""
    
    # Configure canary deployment
    config = CanaryConfig(
        shadow_duration_hours=24,  # 24 hours shadow mode
        canary_stages=[5.0, 10.0, 25.0, 50.0, 100.0],  # Gradual rollout
        stage_duration_hours=2,  # 2 hours per stage
        min_accuracy=0.85,
        max_error_rate=0.05,
        auto_rollback_enabled=True,
        rollback_on_validation_failure=True
    )
    
    # Initialize canary pipeline
    pipeline = CanaryPipeline(config, storage_path="/var/aimedres/deployments")
    
    # Register new model version
    logger.info(f"Registering model {model_id} version {model_version}")
    pipeline.register_model(
        model_id=model_id,
        version=model_version,
        model_artifact_path=f"/var/aimedres/models/{model_id}/{model_version}",
        metadata={'deployment_date': '2024-01-15'}
    )
    
    # Deploy in shadow mode first
    logger.info("Starting shadow deployment...")
    deployment = pipeline.deploy_shadow(
        model_id=model_id,
        model_version=model_version
    )
    
    # Check shadow validation results
    if deployment.status == DeploymentStatus.DEPLOYING:
        logger.info("Shadow validation passed! Starting canary rollout...")
        
        # Deploy canary
        success = pipeline.deploy_canary(
            deployment_id=deployment.deployment_id,
            auto_promote=auto_promote
        )
        
        if success:
            logger.info("Canary deployment initiated successfully!")
            logger.info(f"Deployment ID: {deployment.deployment_id}")
            logger.info(f"Initial traffic: {deployment.traffic_percentage}%")
            
            # Print deployment status
            status = pipeline.get_deployment_status(deployment.deployment_id)
            print("\nDeployment Status:")
            print(f"  Mode: {status['mode']}")
            print(f"  Status: {status['status']}")
            print(f"  Traffic: {status['traffic_percentage']}%")
            print(f"\nValidation Tests:")
            for test in status['validation_tests']:
                print(f"  - {test['test_name']}: {'PASS' if test['passed'] else 'FAIL'} "
                      f"(score: {test['score']:.3f}, threshold: {test['threshold']:.3f})")
            
            return 0
        else:
            logger.error("Canary deployment failed!")
            return 1
    else:
        logger.error(f"Shadow validation failed! Status: {deployment.status.value}")
        logger.error("Review validation test results:")
        for test in deployment.validation_tests:
            if not test.passed:
                logger.error(f"  - {test.test_name}: FAILED (score: {test.score:.3f}, "
                           f"threshold: {test.threshold:.3f})")
        return 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python deploy_canary.py <model_id> <model_version> [auto_promote]")
        sys.exit(1)
    
    model_id = sys.argv[1]
    model_version = sys.argv[2]
    auto_promote = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
    
    sys.exit(deploy_canary(model_id, model_version, auto_promote))
```

**Usage:**

```bash
# Deploy with manual promotion (recommended for first deployment)
python deployment/production_deployment/deploy_canary.py alzheimer_classifier v2.1 false

# Deploy with automatic promotion (for trusted updates)
python deployment/production_deployment/deploy_canary.py alzheimer_classifier v2.1 true

# Monitor deployment progress
watch -n 30 'curl -s http://localhost:8080/api/v1/deployments/status'
```

### 1.4 Kubernetes Deployment (Enterprise Scale)

For large-scale deployments, use Kubernetes with Helm charts.

**Helm Chart:** `deployment/production_deployment/helm/aimedres/values.yaml`

```yaml
# AiMedRes Helm Chart Values - Production

replicaCount: 3

image:
  repository: aimedres
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
  hosts:
    - host: aimedres.hospital.org
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: aimedres-tls
      hosts:
        - aimedres.hospital.org

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 20
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

persistence:
  enabled: true
  storageClass: fast-ssd
  accessMode: ReadWriteMany
  size: 100Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

secrets:
  DATABASE_URL: "postgresql://user:pass@db:5432/aimedres"
  SECRET_KEY: "CHANGE_ME"
  JWT_SECRET_KEY: "CHANGE_ME"
```

**Deploy with Helm:**

```bash
# Install/upgrade production deployment
helm upgrade --install aimedres-prod \
  ./deployment/production_deployment/helm/aimedres \
  --namespace aimedres-production \
  --create-namespace \
  --values values.production.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl -n aimedres-production get pods
kubectl -n aimedres-production get svc
kubectl -n aimedres-production get ingress

# Check rollout status
kubectl -n aimedres-production rollout status deployment/aimedres-prod

# View logs
kubectl -n aimedres-production logs -l app=aimedres-prod --tail=100 -f
```

### 1.5 Rollback Procedures

#### Quick Rollback (Docker Compose)

```bash
#!/bin/bash
# rollback.sh - Quick rollback to previous version

set -e

PREVIOUS_VERSION=${1:-}

if [ -z "$PREVIOUS_VERSION" ]; then
  echo "Error: Please specify previous version"
  echo "Usage: ./rollback.sh <previous_version>"
  exit 1
fi

echo "Rolling back to version $PREVIOUS_VERSION"

# Stop current services
docker-compose -f docker-compose.production.yml down

# Deploy previous version
VERSION=$PREVIOUS_VERSION docker-compose -f docker-compose.production.yml up -d

# Wait and verify
sleep 30
curl -f http://localhost:8080/health

echo "Rollback complete!"
```

#### Kubernetes Rollback

```bash
# Rollback to previous revision
kubectl -n aimedres-production rollout undo deployment/aimedres-prod

# Rollback to specific revision
kubectl -n aimedres-production rollout undo deployment/aimedres-prod --to-revision=2

# Check rollback status
kubectl -n aimedres-production rollout status deployment/aimedres-prod
```

### 1.6 Post-Deployment Verification

After any deployment, run these verification steps:

```bash
#!/bin/bash
# verify_deployment.sh - Post-deployment verification

set -e

echo "Running post-deployment verification..."

# 1. Health check
echo "1. Checking system health..."
if ! curl -f http://localhost:8080/health; then
  echo "ERROR: Health check failed!"
  exit 1
fi

# 2. Model verification
echo "2. Verifying models..."
MODELS=$(curl -s http://localhost:8080/api/v1/models/list | jq -r '.models | length')
if [ "$MODELS" -lt 1 ]; then
  echo "ERROR: No models loaded!"
  exit 1
fi

# 3. API smoke test
echo "3. Running API smoke tests..."
cd deployment/validation
python smoke_test_api.py --host localhost --port 8080

# 4. Check logs for errors
echo "4. Checking logs for errors..."
ERROR_COUNT=$(docker logs aimedres-web-prod 2>&1 | grep "ERROR" | wc -l)
if [ $ERROR_COUNT -gt 5 ]; then
  echo "WARNING: Found $ERROR_COUNT errors in logs"
fi

# 5. Monitor resource usage
echo "5. Checking resource usage..."
docker stats --no-stream aimedres-web-prod

echo "Verification complete!"
```

---

## 2. Monitoring & Support

### Overview

Comprehensive monitoring ensures system health, performance, and early detection of issues.

### 2.1 System Monitoring Setup

AiMedRes includes built-in monitoring using the production monitor from `mlops/monitoring/production_monitor.py`.

#### Configure Production Monitoring

```python
#!/usr/bin/env python3
"""
Setup production monitoring for AiMedRes.
"""

import sys
sys.path.insert(0, '/var/aimedres/src')

from mlops.monitoring.production_monitor import ProductionMonitor, AlertConfig
import pandas as pd
import numpy as np

def setup_monitoring():
    """Setup production monitoring with alerting."""
    
    # Configure alert thresholds
    alert_config = AlertConfig(
        critical_accuracy_drop=0.05,  # 5% drop triggers critical alert
        warning_accuracy_drop=0.02,   # 2% drop triggers warning
        critical_error_rate=0.02,     # 2% error rate critical
        warning_error_rate=0.01,      # 1% error rate warning
        critical_latency_ms=500.0,    # 500ms critical
        warning_latency_ms=200.0,     # 200ms warning
        drift_threshold=0.1,          # 10% drift threshold
        data_quality_threshold=0.95   # 95% data quality minimum
    )
    
    # Initialize monitor for each model
    models = ['alzheimer_classifier', 'parkinsons_predictor', 'als_classifier']
    
    for model_name in models:
        print(f"Setting up monitoring for {model_name}...")
        
        monitor = ProductionMonitor(
            model_name=model_name,
            mlflow_tracking_uri="postgresql://mlflow:PASSWORD@prod-db:5432/mlflow",
            redis_host="prod-redis.internal",
            redis_port=6379,
            alert_config=alert_config
        )
        
        # Load baseline data for drift detection
        # In production, load actual validation dataset
        baseline_data = pd.read_csv(f'/var/aimedres/data/baseline/{model_name}_validation.csv')
        baseline_labels = baseline_data['label']
        baseline_features = baseline_data.drop('label', axis=1)
        
        monitor.set_baseline_metrics(baseline_features, baseline_labels)
        
        # Start continuous monitoring (5-minute intervals)
        monitor.start_monitoring(interval_seconds=300)
        
        print(f"Monitoring started for {model_name}")
    
    print("\nAll monitoring systems active!")
    print("View metrics at: http://grafana.internal.hospital.org")
    print("Alerts configured for Slack/Email/PagerDuty")

if __name__ == "__main__":
    setup_monitoring()
```

### 2.2 Metrics Collection

#### Prometheus Configuration

Create `/etc/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s
  external_labels:
    environment: 'production'
    cluster: 'aimedres'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Scrape configurations
scrape_configs:
  - job_name: 'aimedres-web'
    static_configs:
      - targets: ['aimedres-web:9090']
        labels:
          service: 'aimedres-web'
    
  - job_name: 'aimedres-worker'
    static_configs:
      - targets: ['aimedres-worker:9090']
        labels:
          service: 'aimedres-worker'
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
        labels:
          service: 'system'
  
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
        labels:
          service: 'database'
  
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
        labels:
          service: 'cache'
  
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['dcgm-exporter:9400']
        labels:
          service: 'gpu'
```

#### Grafana Dashboard

Import pre-configured dashboard: `deployment/production_deployment/grafana_dashboard.json`

**Key Metrics to Monitor:**

1. **System Resources:**
   - CPU utilization (alert if > 80% for 5 minutes)
   - Memory usage (alert if > 85% for 5 minutes)
   - Disk space (alert if < 15% free)
   - GPU utilization (if applicable, alert if > 90%)
   - GPU memory (if applicable, alert if > 85%)

2. **Application Performance:**
   - Request rate (requests per second)
   - Response time (p50, p95, p99 latencies)
   - Error rate (alert if > 1%)
   - Active connections
   - Queue depth (for async tasks)

3. **Model Performance:**
   - Prediction latency (alert if > 500ms p95)
   - Accuracy score (alert if drop > 5%)
   - Drift score (alert if > 10%)
   - Predictions per minute
   - Model load time

4. **Database:**
   - Connection pool utilization
   - Query latency (p95)
   - Active transactions
   - Database size
   - Replication lag (if applicable)

### 2.3 Alerting Configuration

#### AlertManager Configuration

Create `/etc/alertmanager/alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  
  # Slack configuration
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  
  # PagerDuty configuration
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

# Alert routing
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default-receiver'
  
  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    
    # All alerts to Slack
    - match_re:
        severity: warning|critical
      receiver: 'slack-alerts'
    
    # Model performance alerts to ML team
    - match:
        category: model_performance
      receiver: 'ml-team-email'

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'devops@hospital.org'
        from: 'alertmanager@hospital.org'
        smarthost: 'smtp.hospital.org:587'
        auth_username: 'alertmanager@hospital.org'
        auth_password: 'SMTP_PASSWORD'
  
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .CommonAnnotations.description }}'
  
  - name: 'slack-alerts'
    slack_configs:
      - channel: '#aimedres-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
  
  - name: 'ml-team-email'
    email_configs:
      - to: 'ml-team@hospital.org'
        from: 'alertmanager@hospital.org'
        smarthost: 'smtp.hospital.org:587'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

#### Alert Rules

Create `/etc/prometheus/rules/aimedres.yml`:

```yaml
groups:
  - name: aimedres_system
    interval: 30s
    rules:
      - alert: HighCPUUsage
        expr: rate(process_cpu_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 80% for 5 minutes"
      
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.85
        for: 5m
        labels:
          severity: warning
          category: system
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 85% for 5 minutes"
      
      - alert: DiskSpaceLow
        expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) > 0.85
        for: 10m
        labels:
          severity: critical
          category: system
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Disk usage is above 85%"
  
  - name: aimedres_application
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
          category: application
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 1% for 5 minutes"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
          category: application
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is above 500ms"
      
      - alert: ServiceDown
        expr: up{job="aimedres-web"} == 0
        for: 1m
        labels:
          severity: critical
          category: application
        annotations:
          summary: "AiMedRes service is down"
          description: "The AiMedRes web service is not responding"
  
  - name: aimedres_models
    interval: 60s
    rules:
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.80
        for: 10m
        labels:
          severity: critical
          category: model_performance
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"
      
      - alert: DataDriftDetected
        expr: data_drift_score > 0.1
        for: 30m
        labels:
          severity: warning
          category: model_performance
        annotations:
          summary: "Data drift detected"
          description: "Drift score for {{ $labels.model_name }} is {{ $value }}"
      
      - alert: ModelPredictionLatencyHigh
        expr: histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m])) > 1.0
        for: 10m
        labels:
          severity: warning
          category: model_performance
        annotations:
          summary: "Model prediction latency is high"
          description: "95th percentile prediction latency is {{ $value }}s"
```

### 2.4 Log Management

#### Centralized Logging with ELK Stack

**Filebeat Configuration:** `/etc/filebeat/filebeat.yml`

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/aimedres/app.log
      - /var/log/aimedres/audit.log
    fields:
      service: aimedres
      environment: production
    multiline:
      pattern: '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
      negate: true
      match: after
  
  - type: container
    enabled: true
    paths:
      - '/var/lib/docker/containers/*/*.log'

processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~
  - add_docker_metadata: ~

output.elasticsearch:
  hosts: ["elasticsearch.internal.hospital.org:9200"]
  username: "filebeat"
  password: "${ELASTICSEARCH_PASSWORD}"
  index: "aimedres-logs-%{+yyyy.MM.dd}"
  ssl:
    enabled: true
    certificate_authorities: ["/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem"]

setup.ilm:
  enabled: true
  policy_name: aimedres-logs
  rollover_alias: aimedres-logs
  pattern: "{now/d}-000001"

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

#### SIEM Integration

For HIPAA compliance, integrate with institutional SIEM (e.g., Splunk, QRadar):

```python
# deployment/production_deployment/siem_integration.py
"""
SIEM Integration for AiMedRes audit logs.
"""

import syslog
import json
from datetime import datetime
from typing import Dict, Any

class SIEMLogger:
    """Send audit events to SIEM via syslog."""
    
    def __init__(self, siem_host: str, siem_port: int = 514, facility: int = syslog.LOG_LOCAL0):
        """Initialize SIEM logger."""
        self.siem_host = siem_host
        self.siem_port = siem_port
        self.facility = facility
        
        # Open syslog connection
        syslog.openlog(
            ident='aimedres',
            logoption=syslog.LOG_PID,
            facility=facility
        )
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event to SIEM."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'application': 'aimedres',
            'event_type': event_type,
            'details': details
        }
        
        # Send to syslog (which forwards to SIEM)
        syslog.syslog(syslog.LOG_INFO, json.dumps(event))
    
    def log_access(self, user: str, resource: str, action: str, result: str):
        """Log data access event."""
        self.log_audit_event('DATA_ACCESS', {
            'user': user,
            'resource': resource,
            'action': action,
            'result': result
        })
    
    def log_model_inference(self, user: str, model: str, patient_id: str):
        """Log model inference event."""
        self.log_audit_event('MODEL_INFERENCE', {
            'user': user,
            'model': model,
            'patient_id': patient_id
        })
    
    def close(self):
        """Close syslog connection."""
        syslog.closelog()
```

### 2.5 Health Check Endpoints

Implement comprehensive health checks:

```python
# src/aimedres/api/health.py (reference implementation)
"""
Health check endpoints for monitoring.
"""

from flask import Blueprint, jsonify
import psutil
import torch
import time

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check - service is up."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()}), 200

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check - service is ready to accept traffic."""
    checks = {
        'database': check_database(),
        'redis': check_redis(),
        'models': check_models_loaded(),
        'disk': check_disk_space()
    }
    
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    
    return jsonify({
        'ready': all_ready,
        'checks': checks,
        'timestamp': time.time()
    }), status_code

@health_bp.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus-compatible metrics endpoint."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    gpu_metrics = ""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_metrics = f"""
# GPU metrics
gpu_memory_used_gb {gpu_memory:.2f}
gpu_memory_total_gb {gpu_memory_total:.2f}
gpu_utilization_percent {torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0}
"""
    
    metrics_text = f"""
# System metrics
cpu_utilization_percent {cpu_percent}
memory_used_percent {memory.percent}
memory_used_gb {memory.used / 1024**3:.2f}
memory_total_gb {memory.total / 1024**3:.2f}
disk_used_percent {disk.percent}
disk_used_gb {disk.used / 1024**3:.2f}
disk_total_gb {disk.total / 1024**3:.2f}
{gpu_metrics}
"""
    
    return metrics_text, 200, {'Content-Type': 'text/plain'}
```

---

## 3. Backups & Disaster Recovery

### Overview

Comprehensive backup and disaster recovery procedures to ensure data protection and business continuity.

### 3.1 Backup Strategy

#### Backup Components

1. **Models** - ML model files and metadata
2. **Configuration** - Application config, environment variables, secrets
3. **Database** - Production database (encrypted)
4. **Results** - Generated reports, predictions, assessments
5. **Audit Logs** - 7-year retention for HIPAA compliance

#### Backup Schedule

| Component | Frequency | Retention | Method |
|-----------|-----------|-----------|--------|
| Database | Every 6 hours | 30 days | Automated pg_dump |
| Models | Daily | 90 days | Rsync to backup storage |
| Config | On change | 365 days | Git + encrypted backup |
| Results | Daily | 90 days (then archive) | Incremental backup |
| Audit Logs | Continuous | 2555 days (7 years) | Write-once storage |

### 3.2 Backup Implementation

Create `/var/aimedres/scripts/backup.sh`:

```bash
#!/bin/bash
# backup.sh - Comprehensive backup script for AiMedRes

set -e

BACKUP_TYPE=${1:-incremental}  # full, incremental, or differential
BACKUP_DIR=${2:-/var/backup/aimedres}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"
S3_BUCKET=${S3_BUCKET:-s3://aimedres-backups-prod}
ENCRYPTION_KEY_ID=${ENCRYPTION_KEY_ID:-arn:aws:kms:us-east-1:123456789012:key/12345678}

echo "Starting $BACKUP_TYPE backup at $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# 1. Backup Database
echo "Backing up database..."
pg_dump -h prod-db.internal -U aimedres_prod -F c -b -v \
  -f "$BACKUP_PATH/database.dump" aimedres_prod

# Encrypt database backup
openssl enc -aes-256-cbc -salt -pbkdf2 \
  -in "$BACKUP_PATH/database.dump" \
  -out "$BACKUP_PATH/database.dump.enc" \
  -pass "pass:$(aws kms decrypt --ciphertext-blob fileb://encryption_key.enc --output text --query Plaintext)"

rm "$BACKUP_PATH/database.dump"  # Remove unencrypted copy

# 2. Backup Models
echo "Backing up models..."
if [ "$BACKUP_TYPE" = "full" ]; then
  rsync -av --delete \
    /var/aimedres/models/ \
    "$BACKUP_PATH/models/"
else
  rsync -av \
    --link-dest="$BACKUP_DIR/latest/models/" \
    /var/aimedres/models/ \
    "$BACKUP_PATH/models/"
fi

# 3. Backup Configuration
echo "Backing up configuration..."
mkdir -p "$BACKUP_PATH/config"
cp -r /etc/aimedres/* "$BACKUP_PATH/config/"
cp /etc/nginx/nginx.conf "$BACKUP_PATH/config/"
cp docker-compose.production.yml "$BACKUP_PATH/config/"

# Encrypt sensitive config files
find "$BACKUP_PATH/config" -type f -name "*.env*" -o -name "*secret*" | while read file; do
  openssl enc -aes-256-cbc -salt -pbkdf2 \
    -in "$file" \
    -out "$file.enc" \
    -pass "pass:$(aws kms decrypt --ciphertext-blob fileb://encryption_key.enc --output text --query Plaintext)"
  rm "$file"
done

# 4. Backup Results (last 30 days)
echo "Backing up recent results..."
find /var/aimedres/results -type f -mtime -30 | \
  rsync -av --files-from=- / "$BACKUP_PATH/results/"

# 5. Backup Audit Logs
echo "Backing up audit logs..."
rsync -av \
  /var/log/aimedres/audit.log* \
  "$BACKUP_PATH/audit_logs/"

# 6. Create manifest
echo "Creating backup manifest..."
cat > "$BACKUP_PATH/manifest.json" <<EOF
{
  "backup_type": "$BACKUP_TYPE",
  "timestamp": "$TIMESTAMP",
  "components": [
    "database",
    "models",
    "configuration",
    "results",
    "audit_logs"
  ],
  "encryption": "AES-256-CBC",
  "kms_key_id": "$ENCRYPTION_KEY_ID",
  "size_bytes": $(du -sb "$BACKUP_PATH" | cut -f1)
}
EOF

# 7. Calculate checksums
echo "Calculating checksums..."
find "$BACKUP_PATH" -type f -exec sha256sum {} \; > "$BACKUP_PATH/checksums.txt"

# 8. Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
  echo "Uploading to S3..."
  aws s3 sync "$BACKUP_PATH" "$S3_BUCKET/$TIMESTAMP/" \
    --storage-class STANDARD_IA \
    --server-side-encryption aws:kms \
    --ssekms-key-id "$ENCRYPTION_KEY_ID"
fi

# 9. Update 'latest' symlink
ln -sfn "$BACKUP_PATH" "$BACKUP_DIR/latest"

# 10. Cleanup old backups (keep last 30 days locally)
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} \;

# 11. Verify backup integrity
echo "Verifying backup integrity..."
sha256sum -c "$BACKUP_PATH/checksums.txt"

echo "Backup completed successfully: $BACKUP_PATH"
echo "Backup size: $(du -sh "$BACKUP_PATH" | cut -f1)"
```

#### Automated Backup Scheduling

Create cron jobs:

```bash
# /etc/cron.d/aimedres-backup

# Full backup weekly (Sunday 2 AM)
0 2 * * 0 root /var/aimedres/scripts/backup.sh full /var/backup/aimedres >> /var/log/aimedres/backup.log 2>&1

# Incremental backup every 6 hours
0 */6 * * * root /var/aimedres/scripts/backup.sh incremental /var/backup/aimedres >> /var/log/aimedres/backup.log 2>&1

# Audit log backup (continuous to write-once storage)
*/15 * * * * root rsync -av /var/log/aimedres/audit.log* /mnt/worm-storage/aimedres/audit/ >> /var/log/aimedres/audit-backup.log 2>&1
```

### 3.3 Disaster Recovery Procedures

#### Recovery Time Objective (RTO) and Recovery Point Objective (RPO)

| Component | RTO | RPO | Priority |
|-----------|-----|-----|----------|
| Critical Services | 4 hours | 6 hours | P0 |
| Database | 4 hours | 6 hours | P0 |
| Models | 8 hours | 24 hours | P1 |
| Historical Data | 24 hours | 24 hours | P2 |

#### Restore Script

Create `/var/aimedres/scripts/restore.sh`:

```bash
#!/bin/bash
# restore.sh - Disaster recovery restore script

set -e

BACKUP_PATH=${1:-}
RESTORE_TARGET=${2:-/var/aimedres}

if [ -z "$BACKUP_PATH" ]; then
  echo "Usage: ./restore.sh <backup_path> [restore_target]"
  echo "Example: ./restore.sh /var/backup/aimedres/20240115-020000"
  exit 1
fi

echo "Starting restore from $BACKUP_PATH to $RESTORE_TARGET"
echo "WARNING: This will overwrite existing data!"
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Restore cancelled"
  exit 0
fi

# Verify backup integrity
echo "Verifying backup integrity..."
cd "$BACKUP_PATH"
sha256sum -c checksums.txt || {
  echo "ERROR: Backup integrity check failed!"
  exit 1
}

# 1. Stop services
echo "Stopping services..."
docker-compose -f /var/aimedres/docker-compose.production.yml down

# 2. Restore Database
echo "Restoring database..."
# Decrypt database backup
openssl enc -aes-256-cbc -d -pbkdf2 \
  -in "$BACKUP_PATH/database.dump.enc" \
  -out /tmp/database.dump \
  -pass "pass:$(aws kms decrypt --ciphertext-blob fileb://encryption_key.enc --output text --query Plaintext)"

# Drop and recreate database
psql -h prod-db.internal -U postgres -c "DROP DATABASE IF EXISTS aimedres_prod;"
psql -h prod-db.internal -U postgres -c "CREATE DATABASE aimedres_prod OWNER aimedres_prod;"

# Restore from dump
pg_restore -h prod-db.internal -U aimedres_prod -d aimedres_prod -v /tmp/database.dump

# Cleanup
rm /tmp/database.dump

# 3. Restore Models
echo "Restoring models..."
rsync -av --delete \
  "$BACKUP_PATH/models/" \
  "$RESTORE_TARGET/models/"

# 4. Restore Configuration
echo "Restoring configuration..."
# Decrypt config files
find "$BACKUP_PATH/config" -name "*.enc" | while read file; do
  decrypted="${file%.enc}"
  openssl enc -aes-256-cbc -d -pbkdf2 \
    -in "$file" \
    -out "$decrypted" \
    -pass "pass:$(aws kms decrypt --ciphertext-blob fileb://encryption_key.enc --output text --query Plaintext)"
done

rsync -av \
  "$BACKUP_PATH/config/" \
  /etc/aimedres/

# 5. Restore Results
echo "Restoring results..."
rsync -av \
  "$BACKUP_PATH/results/" \
  "$RESTORE_TARGET/results/"

# 6. Restore Audit Logs
echo "Restoring audit logs..."
rsync -av \
  "$BACKUP_PATH/audit_logs/" \
  /var/log/aimedres/

# 7. Verify restoration
echo "Verifying restoration..."

# Check database connection
psql -h prod-db.internal -U aimedres_prod -d aimedres_prod -c "SELECT COUNT(*) FROM models;" || {
  echo "ERROR: Database verification failed!"
  exit 1
}

# Check models exist
if [ ! -d "$RESTORE_TARGET/models" ] || [ -z "$(ls -A $RESTORE_TARGET/models)" ]; then
  echo "ERROR: Models directory is empty!"
  exit 1
fi

# 8. Start services
echo "Starting services..."
docker-compose -f /var/aimedres/docker-compose.production.yml up -d

# 9. Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# 10. Run health checks
echo "Running health checks..."
for i in {1..10}; do
  if curl -f http://localhost:8080/health; then
    echo "System is healthy!"
    break
  fi
  echo "Attempt $i failed, retrying..."
  sleep 10
done

# 11. Verify models loaded
echo "Verifying models..."
curl -f http://localhost:8080/api/v1/models/list | jq .

echo "Restore completed successfully!"
echo "Please verify system functionality before resuming production traffic."
```

#### Disaster Recovery Testing

Test disaster recovery procedures quarterly:

```bash
#!/bin/bash
# dr_test.sh - Disaster recovery drill

set -e

echo "Starting DR drill..."

# 1. Create test backup
echo "Creating test backup..."
/var/aimedres/scripts/backup.sh full /tmp/dr-test

# 2. Setup test environment
echo "Setting up test environment..."
# Use separate test infrastructure
export DATABASE_URL="postgresql://test-db.internal:5432/aimedres_test"
export REDIS_URL="redis://test-redis.internal:6379/0"

# 3. Restore to test environment
echo "Restoring to test environment..."
/var/aimedres/scripts/restore.sh /tmp/dr-test /var/aimedres-test

# 4. Verify test environment
echo "Verifying test environment..."
# Run comprehensive tests
cd /var/aimedres-test/deployment/validation
python smoke_test_api.py --host test.internal --port 8080

# 5. Measure RTO
# Time from failure to recovery should be < 4 hours

# 6. Cleanup test environment
echo "Cleaning up test environment..."
docker-compose -f /var/aimedres-test/docker-compose.production.yml down
rm -rf /var/aimedres-test /tmp/dr-test

echo "DR drill completed successfully!"
```

### 3.4 Backup Monitoring

Monitor backup health:

```python
#!/usr/bin/env python3
"""
Backup monitoring script - verify backups are current and valid.
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def check_backup_freshness(backup_dir: str, max_age_hours: int = 12):
    """Check if latest backup is recent enough."""
    latest_link = Path(backup_dir) / "latest"
    
    if not latest_link.exists():
        return False, "No backup found"
    
    latest_backup = latest_link.resolve()
    manifest_file = latest_backup / "manifest.json"
    
    if not manifest_file.exists():
        return False, "Backup manifest missing"
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    backup_time = datetime.strptime(manifest['timestamp'], '%Y%m%d-%H%M%S')
    age_hours = (datetime.now() - backup_time).total_seconds() / 3600
    
    if age_hours > max_age_hours:
        return False, f"Backup is {age_hours:.1f} hours old (max: {max_age_hours})"
    
    return True, f"Backup is {age_hours:.1f} hours old"

def verify_backup_integrity(backup_path: Path):
    """Verify backup checksums."""
    checksum_file = backup_path / "checksums.txt"
    
    if not checksum_file.exists():
        return False, "Checksum file missing"
    
    try:
        result = subprocess.run(
            ['sha256sum', '-c', str(checksum_file)],
            cwd=backup_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False, f"Integrity check failed: {result.stderr}"
        
        return True, "Integrity verified"
    except Exception as e:
        return False, f"Integrity check error: {e}"

def check_s3_sync(backup_dir: str, s3_bucket: str):
    """Check if backups are synced to S3."""
    latest_backup = (Path(backup_dir) / "latest").resolve()
    backup_name = latest_backup.name
    
    # Check if backup exists in S3
    result = subprocess.run(
        ['aws', 's3', 'ls', f'{s3_bucket}/{backup_name}/'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return False, "Backup not found in S3"
    
    return True, "S3 sync verified"

if __name__ == "__main__":
    backup_dir = "/var/backup/aimedres"
    s3_bucket = "s3://aimedres-backups-prod"
    
    print("=== Backup Health Check ===\n")
    
    # Check freshness
    fresh, msg = check_backup_freshness(backup_dir)
    print(f"Backup Freshness: {'✓' if fresh else '✗'} {msg}")
    
    # Check integrity
    latest_backup = (Path(backup_dir) / "latest").resolve()
    valid, msg = verify_backup_integrity(latest_backup)
    print(f"Backup Integrity: {'✓' if valid else '✗'} {msg}")
    
    # Check S3 sync
    synced, msg = check_s3_sync(backup_dir, s3_bucket)
    print(f"S3 Sync Status: {'✓' if synced else '✗'} {msg}")
    
    # Overall status
    all_good = fresh and valid and synced
    print(f"\nOverall Status: {'✓ HEALTHY' if all_good else '✗ ISSUES DETECTED'}")
    
    exit(0 if all_good else 1)
```

Add to cron for automated monitoring:

```bash
# Check backup health every hour
0 * * * * root /var/aimedres/scripts/check_backup_health.py >> /var/log/aimedres/backup-monitor.log 2>&1
```

---

## Summary

This production deployment guide provides:

1. **Deployment Strategies**: Direct, blue/green, canary, and Kubernetes deployments
2. **Comprehensive Monitoring**: System metrics, application performance, model health
3. **Alerting**: Multi-channel alerts for critical issues
4. **Backup & Recovery**: Automated backups with encryption and tested restore procedures

**Next Steps:**

1. Review and customize configurations for your environment
2. Test deployment scripts in staging environment
3. Configure monitoring dashboards and alerts
4. Schedule and test disaster recovery procedures
5. Proceed to Clinical & Operational Readiness (Section 7)

**Support:**

For deployment assistance, contact: devops@hospital.org
