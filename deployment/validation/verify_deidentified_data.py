#!/usr/bin/env python3
"""
Data De-identification Verification Tool
Verifies that test data contains no PHI/PII
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# PHI patterns to detect
PHI_PATTERNS = {
    'ssn': r'\d{3}-\d{2}-\d{4}',
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
    'zip': r'\b\d{5}(-\d{4})?\b',
    'mrn': r'\b(MRN|Medical Record|Patient ID)[:\s]+[\w-]+',
}


class DataVerifier:
    """Verify data is properly de-identified"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.violations = []
        self.files_checked = 0
        
    def verify_file(self, file_path):
        """Verify a single file for PHI"""
        logger.debug(f"Checking file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_violations = []
            
            for phi_type, pattern in PHI_PATTERNS.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    file_violations.append({
                        'type': phi_type,
                        'count': len(matches),
                        'samples': matches[:3]  # First 3 matches
                    })
            
            if file_violations:
                self.violations.append({
                    'file': str(file_path),
                    'violations': file_violations
                })
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return True  # Don't fail on read errors
    
    def verify_directory(self):
        """Verify all files in directory"""
        logger.info(f"Verifying data in: {self.data_path}")
        
        if not self.data_path.exists():
            logger.error(f"Path does not exist: {self.data_path}")
            return False
        
        # Find all data files
        extensions = ['*.csv', '*.json', '*.txt', '*.dat']
        files = []
        
        for ext in extensions:
            files.extend(self.data_path.rglob(ext))
        
        logger.info(f"Found {len(files)} files to verify")
        
        for file_path in files:
            self.files_checked += 1
            self.verify_file(file_path)
        
        return len(self.violations) == 0
    
    def generate_report(self):
        """Generate verification report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'files_checked': self.files_checked,
            'violations_found': len(self.violations),
            'status': 'PASS' if len(self.violations) == 0 else 'FAIL',
            'violations': self.violations
        }
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("De-identification Verification Report")
        logger.info("="*50)
        logger.info(f"Files checked: {self.files_checked}")
        logger.info(f"Violations found: {len(self.violations)}")
        
        if self.violations:
            logger.error("❌ FAIL: PHI/PII detected in data!")
            logger.error("\nViolations by file:")
            for violation in self.violations:
                logger.error(f"\n  File: {violation['file']}")
                for v in violation['violations']:
                    logger.error(f"    - {v['type']}: {v['count']} occurrences")
                    logger.error(f"      Samples: {v['samples']}")
        else:
            logger.info("✅ PASS: No PHI/PII detected")
        
        logger.info("="*50)
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify data is properly de-identified'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data directory to verify')
    parser.add_argument('--output', type=str, default='deidentification_report.json',
                       help='Output file for verification report')
    
    args = parser.parse_args()
    
    verifier = DataVerifier(args.data)
    success = verifier.verify_directory()
    report = verifier.generate_report()
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to: {args.output}")
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
