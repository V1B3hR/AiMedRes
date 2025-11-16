#!/usr/bin/env python3
"""
Model Verification Script for AiMedRes System Validation

This script verifies that all expected models are loaded and ready,
and validates their performance against expected metrics.

Usage:
    python model_verification.py [--all-models] [--models MODEL1,MODEL2] [--verbose]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class ModelVerification:
    """Model verification and validation"""
    
    # Expected models with their metadata
    EXPECTED_MODELS = {
        'alzheimer_v1': {
            'name': 'Alzheimer Early Detection',
            'version': 'v1.0.0',
            'type': 'classification',
            'expected_metrics': {
                'accuracy': {'min': 0.85, 'target': 0.89, 'max': 0.93},
                'auc_roc': {'min': 0.90, 'target': 0.93, 'max': 0.95},
                'sensitivity': {'min': 0.88, 'target': 0.92, 'max': 0.95},
                'specificity': {'min': 0.84, 'target': 0.87, 'max': 0.91}
            },
            'required_files': ['model.pkl', 'scaler.pkl', 'config.json']
        },
        'parkinsons_v1': {
            'name': 'Parkinson Disease Progression',
            'version': 'v1.0.0',
            'type': 'regression',
            'expected_metrics': {
                'r2_score': {'min': 0.78, 'target': 0.82, 'max': 0.86},
                'mae': {'min': 0.10, 'target': 0.12, 'max': 0.15},
                'mse': {'min': 0.12, 'target': 0.15, 'max': 0.20}
            },
            'required_files': ['model.pkl', 'scaler.pkl', 'config.json']
        },
        'als_v1': {
            'name': 'ALS Risk Assessment',
            'version': 'v1.0.0',
            'type': 'classification',
            'expected_metrics': {
                'accuracy': {'min': 0.82, 'target': 0.85, 'max': 0.88},
                'sensitivity': {'min': 0.85, 'target': 0.88, 'max': 0.91},
                'specificity': {'min': 0.80, 'target': 0.83, 'max': 0.87}
            },
            'required_files': ['model.pkl', 'scaler.pkl', 'config.json']
        }
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def verify_model_registry(self) -> bool:
        """Verify model registry is accessible"""
        print(f"\n{Colors.BLUE}Verifying Model Registry...{Colors.RESET}")
        
        try:
            # Try to import model registry
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
            from aimedres.api.model_routes import model_registry
            
            models = model_registry.list_models()
            
            if models:
                print(f"  {Colors.GREEN}✓ Model registry accessible{Colors.RESET}")
                print(f"  Found {len(models)} models")
                return True
            else:
                print(f"  {Colors.YELLOW}⚠ Model registry is empty{Colors.RESET}")
                return False
                
        except ImportError as e:
            print(f"  {Colors.RED}✗ Cannot import model registry: {e}{Colors.RESET}")
            return False
        except Exception as e:
            print(f"  {Colors.RED}✗ Error accessing model registry: {e}{Colors.RESET}")
            return False
    
    def verify_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Verify a single model.
        
        Args:
            model_id: Model identifier
            model_info: Expected model information
            
        Returns:
            True if verification passed
        """
        print(f"\n{Colors.BLUE}Verifying Model:{Colors.RESET} {model_info['name']} ({model_id})")
        
        checks = {
            'registry': False,
            'metadata': False,
            'metrics': False,
            'status': False
        }
        
        try:
            # Import model registry
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
            from aimedres.api.model_routes import model_registry
            
            # Check if model exists in registry
            model_card = model_registry.get_model_card(model_id)
            
            if not model_card:
                print(f"  {Colors.RED}✗ Model not found in registry{Colors.RESET}")
                self.results.append({
                    'model_id': model_id,
                    'name': model_info['name'],
                    'success': False,
                    'checks': checks,
                    'error': 'Model not found in registry'
                })
                self.failed += 1
                return False
            
            checks['registry'] = True
            print(f"  {Colors.GREEN}✓ Found in registry{Colors.RESET}")
            
            # Verify metadata
            if model_card.get('name') and model_card.get('version'):
                checks['metadata'] = True
                print(f"  {Colors.GREEN}✓ Metadata complete{Colors.RESET}")
                print(f"    Name: {model_card['name']}")
                print(f"    Version: {model_card['version']}")
                print(f"    Type: {model_card.get('type', 'N/A')}")
            else:
                print(f"  {Colors.RED}✗ Incomplete metadata{Colors.RESET}")
            
            # Verify metrics
            validation_metrics = model_card.get('validation_metrics', {})
            if validation_metrics:
                checks['metrics'] = True
                print(f"  {Colors.GREEN}✓ Validation metrics present{Colors.RESET}")
                
                # Check if metrics are within expected ranges
                expected_metrics = model_info['expected_metrics']
                all_metrics_ok = True
                
                for metric_name, expected_range in expected_metrics.items():
                    actual_value = validation_metrics.get(metric_name)
                    
                    if actual_value is not None:
                        min_val = expected_range['min']
                        max_val = expected_range['max']
                        target_val = expected_range['target']
                        
                        if min_val <= actual_value <= max_val:
                            status = Colors.GREEN + "✓" + Colors.RESET
                        else:
                            status = Colors.RED + "✗" + Colors.RESET
                            all_metrics_ok = False
                        
                        print(f"    {status} {metric_name}: {actual_value:.3f} (target: {target_val:.3f}, range: {min_val:.3f}-{max_val:.3f})")
                    else:
                        print(f"    {Colors.YELLOW}⚠{Colors.RESET} {metric_name}: Not available")
                        all_metrics_ok = False
                
                if not all_metrics_ok:
                    checks['metrics'] = False
            else:
                print(f"  {Colors.RED}✗ No validation metrics{Colors.RESET}")
            
            # Verify status
            status = model_card.get('status', 'unknown')
            if status == 'active':
                checks['status'] = True
                print(f"  {Colors.GREEN}✓ Status: {status}{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}⚠ Status: {status}{Colors.RESET}")
            
            # Overall success
            success = all(checks.values())
            
            self.results.append({
                'model_id': model_id,
                'name': model_info['name'],
                'success': success,
                'checks': checks,
                'model_card': model_card
            })
            
            if success:
                self.passed += 1
                print(f"\n  {Colors.GREEN}✓ Model verification PASSED{Colors.RESET}")
            else:
                self.failed += 1
                print(f"\n  {Colors.RED}✗ Model verification FAILED{Colors.RESET}")
                failed_checks = [k for k, v in checks.items() if not v]
                print(f"  Failed checks: {', '.join(failed_checks)}")
            
            return success
            
        except Exception as e:
            print(f"  {Colors.RED}✗ Verification error: {e}{Colors.RESET}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            self.results.append({
                'model_id': model_id,
                'name': model_info['name'],
                'success': False,
                'checks': checks,
                'error': str(e)
            })
            self.failed += 1
            return False
    
    def print_summary(self):
        """Print verification summary"""
        total = self.passed + self.failed
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Model Verification Summary{Colors.RESET}")
        print(f"{'='*60}")
        print(f"Total models: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}✓ All models verified successfully!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Some models failed verification{Colors.RESET}")
            print(f"\nFailed models:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['name']} ({result['model_id']})")
        
        print(f"{'='*60}\n")
    
    def save_results(self, output_file: str = 'model_verification_results.json'):
        """Save results to JSON file"""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': self.passed + self.failed,
                    'passed': self.passed,
                    'failed': self.failed,
                    'success_rate': self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0
                },
                'models': self.results
            }, f, indent=2)
        print(f"Results saved to: {output_path.absolute()}")


def main():
    """Run model verification"""
    parser = argparse.ArgumentParser(description='AiMedRes Model Verification')
    parser.add_argument('--all-models', action='store_true', help='Verify all expected models')
    parser.add_argument('--models', type=str, help='Comma-separated list of model IDs to verify')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}AiMedRes Model Verification{Colors.RESET}")
    print("="*60)
    
    verifier = ModelVerification(verbose=args.verbose)
    
    # Verify model registry
    if not verifier.verify_model_registry():
        print(f"\n{Colors.RED}✗ Cannot proceed - model registry not accessible{Colors.RESET}")
        sys.exit(1)
    
    # Determine which models to verify
    if args.all_models:
        models_to_verify = verifier.EXPECTED_MODELS
    elif args.models:
        model_ids = [m.strip() for m in args.models.split(',')]
        models_to_verify = {
            mid: verifier.EXPECTED_MODELS[mid]
            for mid in model_ids
            if mid in verifier.EXPECTED_MODELS
        }
        if not models_to_verify:
            print(f"{Colors.RED}Error: No valid models specified{Colors.RESET}")
            print(f"Available models: {', '.join(verifier.EXPECTED_MODELS.keys())}")
            sys.exit(1)
    else:
        # Default: verify all models
        models_to_verify = verifier.EXPECTED_MODELS
    
    print(f"\nVerifying {len(models_to_verify)} model(s)...")
    
    # Verify each model
    for model_id, model_info in models_to_verify.items():
        verifier.verify_model(model_id, model_info)
    
    # Print summary
    verifier.print_summary()
    
    # Save results
    verifier.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if verifier.failed == 0 else 1)


if __name__ == '__main__':
    main()
