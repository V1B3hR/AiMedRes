#!/usr/bin/env python3
"""
Monitoring Setup Automation Script
Configures Prometheus, Grafana, and AlertManager for AiMedRes
"""

import os
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_DIR = Path("/opt/aimedres/monitoring")
PROMETHEUS_CONFIG = CONFIG_DIR / "prometheus.yml"
ALERTMANAGER_CONFIG = CONFIG_DIR / "alertmanager.yml"
GRAFANA_DASHBOARD = CONFIG_DIR / "grafana_dashboard.json"


class MonitoringSetup:
    """Setup monitoring infrastructure"""
    
    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def setup_prometheus(self):
        """Configure Prometheus"""
        logger.info("Setting up Prometheus configuration...")
        
        config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'alerting': {
                'alertmanagers': [{
                    'static_configs': [{
                        'targets': ['alertmanager:9093']
                    }]
                }]
            },
            'scrape_configs': [
                {
                    'job_name': 'aimedres',
                    'static_configs': [{
                        'targets': ['aimedres:8002']
                    }]
                },
                {
                    'job_name': 'node_exporter',
                    'static_configs': [{
                        'targets': ['node-exporter:9100']
                    }]
                },
                {
                    'job_name': 'postgres',
                    'static_configs': [{
                        'targets': ['postgres-exporter:9187']
                    }]
                },
                {
                    'job_name': 'redis',
                    'static_configs': [{
                        'targets': ['redis-exporter:9121']
                    }]
                }
            ]
        }
        
        # Write Prometheus config
        import yaml
        try:
            with open(PROMETHEUS_CONFIG, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Prometheus config written to: {PROMETHEUS_CONFIG}")
            return True
        except Exception as e:
            logger.error(f"Failed to write Prometheus config: {e}")
            # Fallback - write as text if PyYAML not available
            with open(PROMETHEUS_CONFIG, 'w') as f:
                f.write("# Prometheus Configuration\n")
                f.write("global:\n")
                f.write("  scrape_interval: 15s\n")
                f.write("  evaluation_interval: 15s\n\n")
                f.write("scrape_configs:\n")
                f.write("  - job_name: 'aimedres'\n")
                f.write("    static_configs:\n")
                f.write("      - targets: ['aimedres:8002']\n")
            logger.info("Basic Prometheus config written (install PyYAML for full config)")
            return True
    
    def setup_alertmanager(self):
        """Configure AlertManager"""
        logger.info("Setting up AlertManager configuration...")
        
        config = {
            'global': {
                'smtp_smarthost': 'smtp.hospital.org:587',
                'smtp_from': 'aimedres-alerts@hospital.org'
            },
            'route': {
                'group_by': ['alertname', 'severity'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'default'
            },
            'receivers': [
                {
                    'name': 'default',
                    'email_configs': [{
                        'to': 'ops-team@hospital.org',
                        'send_resolved': True
                    }]
                },
                {
                    'name': 'critical',
                    'email_configs': [{
                        'to': 'oncall@hospital.org',
                        'send_resolved': True
                    }],
                    'slack_configs': [{
                        'api_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
                        'channel': '#aimedres-alerts'
                    }]
                }
            ]
        }
        
        with open(ALERTMANAGER_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"AlertManager config written to: {ALERTMANAGER_CONFIG}")
        return True
    
    def setup_grafana_dashboard(self):
        """Create Grafana dashboard"""
        logger.info("Setting up Grafana dashboard...")
        
        dashboard = {
            'dashboard': {
                'id': None,
                'title': 'AiMedRes Monitoring',
                'tags': ['aimedres', 'healthcare'],
                'timezone': 'browser',
                'panels': [
                    {
                        'id': 1,
                        'title': 'Request Rate',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'rate(http_requests_total[5m])'
                        }]
                    },
                    {
                        'id': 2,
                        'title': 'Response Time (p95)',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
                        }]
                    },
                    {
                        'id': 3,
                        'title': 'Error Rate',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'rate(http_requests_total{status=~"5.."}[5m])'
                        }]
                    },
                    {
                        'id': 4,
                        'title': 'CPU Usage',
                        'type': 'graph',
                        'targets': [{
                            'expr': '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
                        }]
                    },
                    {
                        'id': 5,
                        'title': 'Memory Usage',
                        'type': 'graph',
                        'targets': [{
                            'expr': '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100'
                        }]
                    },
                    {
                        'id': 6,
                        'title': 'Model Inference Latency',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))'
                        }]
                    }
                ]
            },
            'overwrite': True
        }
        
        with open(GRAFANA_DASHBOARD, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info(f"Grafana dashboard written to: {GRAFANA_DASHBOARD}")
        return True
    
    def setup_alert_rules(self):
        """Configure alert rules"""
        logger.info("Setting up alert rules...")
        
        alert_rules_path = CONFIG_DIR / "alert_rules.yml"
        
        rules = """
groups:
  - name: aimedres_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}%"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"
      
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
      
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service down"
          description: "{{ $labels.job }} is down"
"""
        
        with open(alert_rules_path, 'w') as f:
            f.write(rules)
        
        logger.info(f"Alert rules written to: {alert_rules_path}")
        return True
    
    def verify_setup(self):
        """Verify monitoring setup"""
        logger.info("Verifying monitoring setup...")
        
        required_files = [
            PROMETHEUS_CONFIG,
            ALERTMANAGER_CONFIG,
            GRAFANA_DASHBOARD,
            CONFIG_DIR / "alert_rules.yml"
        ]
        
        all_present = True
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"  ✓ {file_path.name}")
            else:
                logger.error(f"  ✗ {file_path.name} not found")
                all_present = False
        
        return all_present
    
    def run_setup(self):
        """Run complete monitoring setup"""
        logger.info("="*50)
        logger.info("Starting Monitoring Setup")
        logger.info("="*50)
        
        success = True
        
        if not self.setup_prometheus():
            success = False
        
        if not self.setup_alertmanager():
            success = False
        
        if not self.setup_grafana_dashboard():
            success = False
        
        if not self.setup_alert_rules():
            success = False
        
        if not self.verify_setup():
            success = False
        
        if success:
            logger.info("="*50)
            logger.info("Monitoring Setup Completed Successfully!")
            logger.info("="*50)
            logger.info("\nNext steps:")
            logger.info("1. Start Prometheus: docker-compose up -d prometheus")
            logger.info("2. Start Grafana: docker-compose up -d grafana")
            logger.info("3. Import dashboard: curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \\")
            logger.info(f"   -H 'Content-Type: application/json' -d @{GRAFANA_DASHBOARD}")
        else:
            logger.error("Monitoring setup failed")
        
        return success


def main():
    setup = MonitoringSetup()
    success = setup.run_setup()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
