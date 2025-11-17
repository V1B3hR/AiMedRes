#!/usr/bin/env python3
"""
Backup Health Monitoring Script
Monitors backup health, verifies integrity, and alerts on issues
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BACKUP_ROOT = Path("/var/backup/aimedres")
MAX_BACKUP_AGE_HOURS = 24
S3_BUCKET = "s3://aimedres-backups"


class BackupHealthChecker:
    """Monitor backup health and integrity"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def check_backup_age(self):
        """Check if backups are recent"""
        logger.info("Checking backup age...")
        
        if not BACKUP_ROOT.exists():
            self.issues.append("Backup directory does not exist")
            return False
        
        # Find latest backup
        backup_dirs = sorted(BACKUP_ROOT.glob("backup_*"), reverse=True)
        
        if not backup_dirs:
            self.issues.append("No backups found")
            return False
        
        latest_backup = backup_dirs[0]
        backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
        age_hours = (datetime.now() - backup_time).total_seconds() / 3600
        
        logger.info(f"Latest backup: {latest_backup.name}")
        logger.info(f"Backup age: {age_hours:.1f} hours")
        
        if age_hours > MAX_BACKUP_AGE_HOURS:
            self.issues.append(f"Backup is too old: {age_hours:.1f} hours (threshold: {MAX_BACKUP_AGE_HOURS})")
            return False
        elif age_hours > MAX_BACKUP_AGE_HOURS * 0.8:
            self.warnings.append(f"Backup is getting old: {age_hours:.1f} hours")
        
        logger.info("✓ Backup age is acceptable")
        return True
    
    def check_backup_completeness(self):
        """Check if backup contains all required components"""
        logger.info("Checking backup completeness...")
        
        backup_dirs = sorted(BACKUP_ROOT.glob("backup_*"), reverse=True)
        if not backup_dirs:
            return False
        
        latest_backup = backup_dirs[0]
        required_components = ['database', 'models', 'config', 'audit_logs']
        
        missing = []
        for component in required_components:
            component_dir = latest_backup / component
            if not component_dir.exists() or not list(component_dir.glob("*.enc")):
                missing.append(component)
        
        if missing:
            self.issues.append(f"Missing backup components: {', '.join(missing)}")
            return False
        
        logger.info("✓ All backup components present")
        return True
    
    def check_backup_integrity(self):
        """Verify backup checksums"""
        logger.info("Checking backup integrity...")
        
        backup_dirs = sorted(BACKUP_ROOT.glob("backup_*"), reverse=True)
        if not backup_dirs:
            return False
        
        latest_backup = backup_dirs[0]
        
        # Find all checksum files
        checksum_files = list(latest_backup.rglob("*.sha256"))
        
        if not checksum_files:
            self.warnings.append("No checksums found for verification")
            return True
        
        failed = 0
        for checksum_file in checksum_files:
            encrypted_file = Path(str(checksum_file).replace('.sha256', ''))
            
            if not encrypted_file.exists():
                self.issues.append(f"Backup file missing: {encrypted_file.name}")
                failed += 1
                continue
            
            # Verify checksum
            try:
                result = subprocess.run(
                    ['sha256sum', '-c', str(checksum_file)],
                    cwd=checksum_file.parent,
                    capture_output=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    self.issues.append(f"Checksum mismatch: {encrypted_file.name}")
                    failed += 1
            except Exception as e:
                self.warnings.append(f"Could not verify checksum for {encrypted_file.name}: {e}")
        
        if failed > 0:
            logger.error(f"✗ {failed} files failed integrity check")
            return False
        
        logger.info(f"✓ All {len(checksum_files)} checksums verified")
        return True
    
    def check_s3_sync(self):
        """Check if backups are synced to S3"""
        logger.info("Checking S3 sync status...")
        
        try:
            # Check if AWS CLI is available
            result = subprocess.run(['which', 'aws'], capture_output=True)
            if result.returncode != 0:
                self.warnings.append("AWS CLI not available - cannot verify S3 sync")
                return True
            
            # Check S3 bucket
            result = subprocess.run(
                ['aws', 's3', 'ls', S3_BUCKET],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.warnings.append("Cannot access S3 bucket")
                return True
            
            # Get latest backup in S3
            result = subprocess.run(
                ['aws', 's3', 'ls', f"{S3_BUCKET}/", '--recursive'],
                capture_output=True,
                timeout=30,
                text=True
            )
            
            if not result.stdout:
                self.issues.append("No backups found in S3")
                return False
            
            # Check if recent backups are in S3
            lines = result.stdout.strip().split('\n')
            if lines:
                last_modified = lines[-1].split()[0] + ' ' + lines[-1].split()[1]
                logger.info(f"Latest S3 backup: {last_modified}")
                logger.info("✓ S3 backups present")
            
            return True
            
        except subprocess.TimeoutExpired:
            self.warnings.append("S3 check timed out")
            return True
        except Exception as e:
            self.warnings.append(f"S3 check failed: {e}")
            return True
    
    def check_storage_space(self):
        """Check available storage space"""
        logger.info("Checking storage space...")
        
        try:
            result = subprocess.run(
                ['df', '-h', str(BACKUP_ROOT)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    usage_pct = int(parts[4].rstrip('%'))
                    
                    logger.info(f"Storage usage: {usage_pct}%")
                    
                    if usage_pct > 90:
                        self.issues.append(f"Storage critically low: {usage_pct}% used")
                        return False
                    elif usage_pct > 80:
                        self.warnings.append(f"Storage getting low: {usage_pct}% used")
                    
                    logger.info("✓ Storage space acceptable")
                    return True
        except Exception as e:
            self.warnings.append(f"Could not check storage: {e}")
        
        return True
    
    def generate_report(self):
        """Generate health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'HEALTHY' if len(self.issues) == 0 else 'UNHEALTHY',
            'issues': self.issues,
            'warnings': self.warnings
        }
        
        logger.info("\n" + "="*50)
        logger.info("Backup Health Report")
        logger.info("="*50)
        logger.info(f"Status: {report['status']}")
        logger.info(f"Issues: {len(self.issues)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.issues:
            logger.error("\nISSUES:")
            for issue in self.issues:
                logger.error(f"  ✗ {issue}")
        
        if self.warnings:
            logger.warning("\nWARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if not self.issues and not self.warnings:
            logger.info("\n✓ All backup health checks passed")
        
        logger.info("="*50)
        
        return report
    
    def run_checks(self):
        """Run all health checks"""
        logger.info("="*50)
        logger.info("Starting Backup Health Checks")
        logger.info("="*50)
        
        self.check_backup_age()
        self.check_backup_completeness()
        self.check_backup_integrity()
        self.check_s3_sync()
        self.check_storage_space()
        
        report = self.generate_report()
        
        # Save report
        report_file = Path("/var/log/aimedres/backup_health.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved to: {report_file}")
        
        return len(self.issues) == 0


def main():
    checker = BackupHealthChecker()
    success = checker.run_checks()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
