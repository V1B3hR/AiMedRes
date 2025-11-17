#!/usr/bin/env python3
"""
CLI Smoke Test for AiMedRes System Validation

This script performs basic smoke tests on the AiMedRes CLI to verify
that the system is properly installed and functioning correctly.

Usage:
    python smoke_test_cli.py [--verbose]
"""

import subprocess
import sys
import json
import time
from typing import Dict, List, Tuple
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class CLISmokeTest:
    """CLI smoke test runner"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def run_test(self, name: str, command: List[str], 
                 expect_success: bool = True,
                 check_output: callable = None) -> bool:
        """
        Run a single test command.
        
        Args:
            name: Test name
            command: Command to execute as list
            expect_success: Whether to expect successful exit code
            check_output: Optional function to validate output
            
        Returns:
            True if test passed
        """
        print(f"\n{Colors.BLUE}Testing:{Colors.RESET} {name}")
        
        if self.verbose:
            print(f"  Command: {' '.join(command)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            duration = time.time() - start_time
            
            # Check exit code
            success = (result.returncode == 0) == expect_success
            
            # Check output if validator provided
            if success and check_output:
                try:
                    success = check_output(result.stdout, result.stderr)
                except Exception as e:
                    success = False
                    print(f"  {Colors.RED}Output validation error:{Colors.RESET} {e}")
            
            # Store result
            self.results.append({
                'name': name,
                'command': ' '.join(command),
                'success': success,
                'exit_code': result.returncode,
                'duration': duration,
                'stdout': result.stdout if self.verbose else result.stdout[:200],
                'stderr': result.stderr if self.verbose else result.stderr[:200]
            })
            
            if success:
                self.passed += 1
                print(f"  {Colors.GREEN}✓ PASSED{Colors.RESET} ({duration:.2f}s)")
            else:
                self.failed += 1
                print(f"  {Colors.RED}✗ FAILED{Colors.RESET}")
                print(f"  Exit code: {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
            
            if self.verbose and result.stdout:
                print(f"  Output: {result.stdout[:500]}")
            
            return success
            
        except subprocess.TimeoutExpired:
            self.failed += 1
            self.results.append({
                'name': name,
                'command': ' '.join(command),
                'success': False,
                'error': 'Timeout after 30s'
            })
            print(f"  {Colors.RED}✗ FAILED{Colors.RESET} - Timeout")
            return False
            
        except Exception as e:
            self.failed += 1
            self.results.append({
                'name': name,
                'command': ' '.join(command),
                'success': False,
                'error': str(e)
            })
            print(f"  {Colors.RED}✗ FAILED{Colors.RESET} - {e}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}✓ All smoke tests passed!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Some tests failed{Colors.RESET}")
            print(f"\nFailed tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['name']}")
        
        print(f"{'='*60}\n")
    
    def save_results(self, output_file: str = 'smoke_test_results.json'):
        """Save results to JSON file"""
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': self.passed + self.failed,
                    'passed': self.passed,
                    'failed': self.failed,
                    'success_rate': self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0
                },
                'tests': self.results
            }, f, indent=2)
        print(f"Results saved to: {output_path.absolute()}")


def main():
    """Run CLI smoke tests"""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print(f"{Colors.BOLD}AiMedRes CLI Smoke Test{Colors.RESET}")
    print("="*60)
    
    tester = CLISmokeTest(verbose=verbose)
    
    # Test 1: Version check
    tester.run_test(
        "Version Check",
        ['python', '-m', 'aimedres', '--version'],
        check_output=lambda out, err: 'aimedres' in out.lower() or 'version' in out.lower()
    )
    
    # Test 2: Help command
    tester.run_test(
        "Help Command",
        ['python', '-m', 'aimedres', '--help'],
        check_output=lambda out, err: 'usage' in out.lower() or 'aimedres' in out.lower()
    )
    
    # Test 3: Train help
    tester.run_test(
        "Train Command Help",
        ['python', '-m', 'aimedres', 'train', '--help'],
        check_output=lambda out, err: 'train' in out.lower()
    )
    
    # Test 4: List training jobs (dry run)
    tester.run_test(
        "List Training Jobs",
        ['python', '-m', 'aimedres', 'train', '--list'],
        expect_success=True  # May succeed or fail depending on setup
    )
    
    # Test 5: Serve help
    tester.run_test(
        "Serve Command Help",
        ['python', '-m', 'aimedres', 'serve', '--help'],
        check_output=lambda out, err: 'serve' in out.lower() or 'server' in out.lower()
    )
    
    # Test 6: Python module check
    tester.run_test(
        "Python Module Import",
        ['python', '-c', 'import aimedres; print("Import successful")'],
        check_output=lambda out, err: 'successful' in out.lower()
    )
    
    # Test 7: Check for required modules
    tester.run_test(
        "Check Dependencies",
        ['python', '-c', 'import flask, numpy, pandas; print("Dependencies OK")'],
        check_output=lambda out, err: 'ok' in out.lower()
    )
    
    # Print summary
    tester.print_summary()
    
    # Save results
    tester.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == '__main__':
    main()
