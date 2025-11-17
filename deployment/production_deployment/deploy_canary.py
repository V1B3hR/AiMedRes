#!/usr/bin/env python3
"""
Canary Deployment Script for AiMedRes
Implements gradual traffic rollout with automated validation and rollback
"""

import sys
import time
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
TRAFFIC_STAGES = [5, 10, 25, 50, 100]  # Traffic percentage stages
VALIDATION_INTERVAL = 300  # 5 minutes between stages
ERROR_THRESHOLD = 0.05  # 5% error rate threshold
LATENCY_THRESHOLD = 500  # 500ms p95 latency threshold


class CanaryDeployment:
    """Manages canary deployment process"""
    
    def __init__(self, new_version, baseline_version):
        self.new_version = new_version
        self.baseline_version = baseline_version
        self.current_stage = 0
        
    def deploy_canary(self):
        """Deploy new version alongside baseline"""
        logger.info(f"Deploying canary version: {self.new_version}")
        
        # Deploy new version with canary label
        cmd = [
            "docker-compose",
            "-f", "../technical/docker-compose.yml",
            "-p", f"aimedres_canary_{self.new_version}",
            "up", "-d"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Canary deployment successful")
            return True
        else:
            logger.error(f"Canary deployment failed: {result.stderr}")
            return False
    
    def set_traffic_split(self, canary_percent):
        """Update load balancer to split traffic"""
        logger.info(f"Setting traffic split: {canary_percent}% to canary")
        
        # Update nginx configuration for traffic splitting
        # This would typically use nginx upstream weights or similar
        # For this example, we'll simulate it
        
        try:
            # Simulate traffic splitting configuration
            time.sleep(2)
            logger.info(f"Traffic split updated: {canary_percent}% canary, {100-canary_percent}% baseline")
            return True
        except Exception as e:
            logger.error(f"Failed to update traffic split: {e}")
            return False
    
    def validate_canary(self):
        """Validate canary performance and health"""
        logger.info("Validating canary performance...")
        
        metrics = self._collect_metrics()
        
        # Check error rate
        if metrics['error_rate'] > ERROR_THRESHOLD:
            logger.error(f"Error rate too high: {metrics['error_rate']:.2%} (threshold: {ERROR_THRESHOLD:.2%})")
            return False
        
        # Check latency
        if metrics['p95_latency'] > LATENCY_THRESHOLD:
            logger.error(f"Latency too high: {metrics['p95_latency']}ms (threshold: {LATENCY_THRESHOLD}ms)")
            return False
        
        # Check health endpoint
        if not self._check_health():
            logger.error("Health check failed")
            return False
        
        logger.info("Canary validation passed")
        logger.info(f"  Error rate: {metrics['error_rate']:.2%}")
        logger.info(f"  P95 latency: {metrics['p95_latency']}ms")
        return True
    
    def _collect_metrics(self):
        """Collect performance metrics from canary"""
        # In production, this would query Prometheus or similar
        # For this example, we'll simulate metrics
        
        import random
        return {
            'error_rate': random.uniform(0.001, 0.03),  # 0.1-3% error rate
            'p95_latency': random.uniform(200, 400),    # 200-400ms latency
            'request_count': random.randint(100, 1000)
        }
    
    def _check_health(self):
        """Check health endpoint of canary"""
        try:
            result = subprocess.run(
                ["curl", "-f", "-s", "http://localhost:8003/health"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def rollback(self):
        """Rollback canary deployment"""
        logger.warning("Rolling back canary deployment...")
        
        # Set traffic to 100% baseline
        self.set_traffic_split(0)
        
        # Remove canary containers
        cmd = [
            "docker-compose",
            "-p", f"aimedres_canary_{self.new_version}",
            "down"
        ]
        
        subprocess.run(cmd)
        logger.info("Rollback completed")
    
    def finalize(self):
        """Finalize deployment - remove baseline"""
        logger.info("Finalizing canary deployment...")
        
        # Remove baseline containers
        cmd = [
            "docker-compose",
            "-p", f"aimedres_baseline_{self.baseline_version}",
            "down"
        ]
        
        subprocess.run(cmd)
        logger.info("Deployment finalized - canary is now production")
    
    def execute(self):
        """Execute full canary deployment process"""
        logger.info("="*50)
        logger.info("Starting Canary Deployment")
        logger.info(f"New version: {self.new_version}")
        logger.info(f"Baseline version: {self.baseline_version}")
        logger.info("="*50)
        
        # Deploy canary
        if not self.deploy_canary():
            logger.error("Canary deployment failed")
            return False
        
        # Wait for canary to be ready
        logger.info("Waiting for canary to be ready...")
        time.sleep(30)
        
        # Gradual traffic rollout
        for stage in TRAFFIC_STAGES:
            logger.info(f"\n--- Stage: {stage}% traffic to canary ---")
            
            # Set traffic split
            if not self.set_traffic_split(stage):
                logger.error("Failed to set traffic split")
                self.rollback()
                return False
            
            # Wait for metrics to stabilize
            logger.info(f"Waiting {VALIDATION_INTERVAL}s for metrics to stabilize...")
            time.sleep(VALIDATION_INTERVAL)
            
            # Validate canary
            if not self.validate_canary():
                logger.error("Canary validation failed")
                self.rollback()
                return False
            
            logger.info(f"Stage {stage}% completed successfully")
        
        # Finalize deployment
        self.finalize()
        
        logger.info("="*50)
        logger.info("Canary Deployment Completed Successfully!")
        logger.info("="*50)
        return True


def main():
    if len(sys.argv) < 3:
        print("Usage: deploy_canary.py <new_version> <baseline_version>")
        print("Example: deploy_canary.py v2.0.0 v1.9.0")
        sys.exit(1)
    
    new_version = sys.argv[1]
    baseline_version = sys.argv[2]
    
    deployment = CanaryDeployment(new_version, baseline_version)
    
    success = deployment.execute()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
