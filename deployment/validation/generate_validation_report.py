#!/usr/bin/env python3
"""
Comprehensive Validation Report Generator
Consolidates all validation results into a final report
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationReportGenerator:
    """Generate comprehensive validation report"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.report = {
            'generated_at': datetime.now().isoformat(),
            'validation_phases': {},
            'overall_status': 'PENDING',
            'summary': {
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
    
    def load_smoke_test_results(self):
        """Load smoke test results"""
        logger.info("Loading smoke test results...")
        
        cli_results = self.results_dir / "smoke_test_cli.json"
        api_results = self.results_dir / "smoke_test_api.json"
        
        results = {'status': 'NOT_RUN'}
        
        if cli_results.exists() and api_results.exists():
            with open(cli_results) as f:
                cli_data = json.load(f)
            with open(api_results) as f:
                api_data = json.load(f)
            
            results = {
                'status': 'PASS' if cli_data.get('passed') and api_data.get('passed') else 'FAIL',
                'cli_tests': cli_data.get('tests', {}),
                'api_tests': api_data.get('tests', {})
            }
        
        self.report['validation_phases']['smoke_tests'] = results
        self._update_summary(results['status'])
    
    def load_model_verification(self):
        """Load model verification results"""
        logger.info("Loading model verification results...")
        
        verification_file = self.results_dir / "model_verification_results.json"
        
        results = {'status': 'NOT_RUN'}
        
        if verification_file.exists():
            with open(verification_file) as f:
                data = json.load(f)
            
            results = {
                'status': data.get('status', 'UNKNOWN'),
                'models_verified': data.get('models_verified', []),
                'models_failed': data.get('models_failed', [])
            }
        
        self.report['validation_phases']['model_verification'] = results
        self._update_summary(results['status'])
    
    def load_benchmark_results(self):
        """Load benchmark results"""
        logger.info("Loading benchmark results...")
        
        benchmark_file = self.results_dir / "benchmark_report.json"
        
        results = {'status': 'NOT_RUN'}
        
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                data = json.load(f)
            
            results = {
                'status': 'PASS' if data['summary']['failed'] == 0 else 'FAIL',
                'models_tested': data.get('models_tested', 0),
                'passed': data['summary'].get('passed', 0),
                'failed': data['summary'].get('failed', 0)
            }
        
        self.report['validation_phases']['benchmark'] = results
        self._update_summary(results['status'])
    
    def load_uat_results(self):
        """Load UAT results"""
        logger.info("Loading UAT results...")
        
        uat_file = self.results_dir / "uat_summary_report.json"
        
        results = {'status': 'NOT_RUN'}
        
        if uat_file.exists():
            with open(uat_file) as f:
                data = json.load(f)
            
            results = {
                'status': data.get('overall_status', 'UNKNOWN'),
                'scenarios_completed': data.get('scenarios_completed', 0),
                'scenarios_passed': data.get('scenarios_passed', 0),
                'user_satisfaction': data.get('satisfaction_score', 0)
            }
        
        self.report['validation_phases']['uat'] = results
        self._update_summary(results['status'])
    
    def load_resource_monitoring(self):
        """Load resource monitoring results"""
        logger.info("Loading resource monitoring results...")
        
        resource_file = self.results_dir / "resource_report.json"
        
        results = {'status': 'NOT_RUN'}
        
        if resource_file.exists():
            with open(resource_file) as f:
                data = json.load(f)
            
            results = {
                'status': data.get('status', 'UNKNOWN'),
                'avg_cpu': data.get('avg_cpu', 0),
                'avg_memory': data.get('avg_memory', 0),
                'peak_cpu': data.get('peak_cpu', 0),
                'peak_memory': data.get('peak_memory', 0)
            }
        
        self.report['validation_phases']['resource_monitoring'] = results
        self._update_summary(results['status'])
    
    def _update_summary(self, status):
        """Update summary counts"""
        if status == 'PASS':
            self.report['summary']['passed'] += 1
        elif status == 'FAIL':
            self.report['summary']['failed'] += 1
        elif status == 'WARNING':
            self.report['summary']['warnings'] += 1
    
    def determine_overall_status(self):
        """Determine overall validation status"""
        if self.report['summary']['failed'] > 0:
            self.report['overall_status'] = 'FAIL'
        elif self.report['summary']['warnings'] > 0:
            self.report['overall_status'] = 'PASS_WITH_WARNINGS'
        elif self.report['summary']['passed'] > 0:
            self.report['overall_status'] = 'PASS'
        else:
            self.report['overall_status'] = 'INCOMPLETE'
    
    def generate_recommendations(self):
        """Generate recommendations based on results"""
        recommendations = []
        
        # Check each validation phase
        for phase, results in self.report['validation_phases'].items():
            if results.get('status') == 'FAIL':
                recommendations.append(f"⚠ {phase}: Failed - Review and address issues before production deployment")
            elif results.get('status') == 'WARNING':
                recommendations.append(f"⚠ {phase}: Warnings present - Review and consider addressing")
            elif results.get('status') == 'NOT_RUN':
                recommendations.append(f"⚠ {phase}: Not run - Complete this validation phase")
        
        if not recommendations:
            recommendations.append("✓ All validation phases passed - System ready for production deployment")
        
        self.report['recommendations'] = recommendations
    
    def generate_report(self, output_file):
        """Generate complete validation report"""
        logger.info("Generating validation report...")
        
        # Load all validation results
        self.load_smoke_test_results()
        self.load_model_verification()
        self.load_benchmark_results()
        self.load_uat_results()
        self.load_resource_monitoring()
        
        # Determine overall status
        self.determine_overall_status()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Print summary
        self._print_summary()
        
        logger.info(f"\nValidation report saved to: {output_path}")
        
        return self.report
    
    def _print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "="*60)
        logger.info("DEPLOYMENT VALIDATION REPORT")
        logger.info("="*60)
        logger.info(f"Generated: {self.report['generated_at']}")
        logger.info(f"Overall Status: {self.report['overall_status']}")
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Passed:   {self.report['summary']['passed']}")
        logger.info(f"  Failed:   {self.report['summary']['failed']}")
        logger.info(f"  Warnings: {self.report['summary']['warnings']}")
        logger.info("")
        logger.info("Validation Phases:")
        
        for phase, results in self.report['validation_phases'].items():
            status = results.get('status', 'UNKNOWN')
            symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠" if status == "WARNING" else "○"
            logger.info(f"  {symbol} {phase.replace('_', ' ').title()}: {status}")
        
        logger.info("")
        logger.info("Recommendations:")
        for rec in self.report['recommendations']:
            logger.info(f"  {rec}")
        
        logger.info("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate comprehensive validation report'
    )
    parser.add_argument('--smoke-test-results', type=str, help='Path to smoke test results')
    parser.add_argument('--model-verification', type=str, help='Path to model verification results')
    parser.add_argument('--benchmark', type=str, help='Path to benchmark report')
    parser.add_argument('--uat-feedback', type=str, help='Path to UAT feedback')
    parser.add_argument('--resource-monitoring', type=str, help='Path to resource monitoring')
    parser.add_argument('--output', type=str, default='deployment_validation_report.json',
                       help='Output file for validation report')
    
    args = parser.parse_args()
    
    # Use current directory if no paths specified
    results_dir = Path('.')
    
    generator = ValidationReportGenerator(results_dir)
    report = generator.generate_report(args.output)
    
    # Exit with error code if validation failed
    if report['overall_status'] in ['FAIL', 'INCOMPLETE']:
        logger.error("\n❌ Validation FAILED - System not ready for production")
        return 1
    elif report['overall_status'] == 'PASS_WITH_WARNINGS':
        logger.warning("\n⚠ Validation PASSED with warnings - Review before production")
        return 0
    else:
        logger.info("\n✅ Validation PASSED - System ready for production deployment")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
