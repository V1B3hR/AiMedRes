#!/usr/bin/env python3
"""
API Smoke Test for AiMedRes System Validation

This script performs basic smoke tests on the AiMedRes API to verify
that the API server is functioning correctly.

Usage:
    python smoke_test_api.py [--host HOST] [--port PORT] [--verbose]
"""

import requests
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class APISmokeTest:
    """API smoke test runner"""
    
    def __init__(self, base_url: str, verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.results = []
        self.passed = 0
        self.failed = 0
        self.token = None
    
    def run_test(self, name: str, method: str, endpoint: str,
                 data: Optional[Dict] = None,
                 headers: Optional[Dict] = None,
                 expect_status: int = 200,
                 check_response: callable = None) -> bool:
        """
        Run a single API test.
        
        Args:
            name: Test name
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            headers: Request headers
            expect_status: Expected status code
            check_response: Optional function to validate response
            
        Returns:
            True if test passed
        """
        print(f"\n{Colors.BLUE}Testing:{Colors.RESET} {name}")
        
        url = f"{self.base_url}{endpoint}"
        
        if self.verbose:
            print(f"  URL: {url}")
            print(f"  Method: {method}")
        
        try:
            start_time = time.time()
            
            # Make request
            response = requests.request(
                method,
                url,
                json=data,
                headers=headers,
                timeout=10
            )
            
            duration = time.time() - start_time
            
            # Check status code
            success = response.status_code == expect_status
            
            # Check response body if validator provided
            if success and check_response:
                try:
                    response_data = response.json() if response.text else {}
                    success = check_response(response_data)
                except Exception as e:
                    success = False
                    print(f"  {Colors.RED}Response validation error:{Colors.RESET} {e}")
            
            # Store result
            self.results.append({
                'name': name,
                'url': url,
                'method': method,
                'success': success,
                'status_code': response.status_code,
                'duration': duration,
                'response': response.text[:200] if self.verbose else None
            })
            
            if success:
                self.passed += 1
                print(f"  {Colors.GREEN}✓ PASSED{Colors.RESET} ({duration:.2f}s)")
                print(f"  Status: {response.status_code}")
            else:
                self.failed += 1
                print(f"  {Colors.RED}✗ FAILED{Colors.RESET}")
                print(f"  Expected status: {expect_status}, Got: {response.status_code}")
                if response.text:
                    print(f"  Response: {response.text[:200]}")
            
            if self.verbose and response.text:
                print(f"  Response body: {response.text[:500]}")
            
            return success
            
        except requests.exceptions.Timeout:
            self.failed += 1
            self.results.append({
                'name': name,
                'url': url,
                'method': method,
                'success': False,
                'error': 'Request timeout'
            })
            print(f"  {Colors.RED}✗ FAILED{Colors.RESET} - Timeout")
            return False
            
        except requests.exceptions.ConnectionError:
            self.failed += 1
            self.results.append({
                'name': name,
                'url': url,
                'method': method,
                'success': False,
                'error': 'Connection error - Is the server running?'
            })
            print(f"  {Colors.RED}✗ FAILED{Colors.RESET} - Connection error")
            print(f"  Make sure the API server is running on {self.base_url}")
            return False
            
        except Exception as e:
            self.failed += 1
            self.results.append({
                'name': name,
                'url': url,
                'method': method,
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
            print(f"\n{Colors.GREEN}✓ All API smoke tests passed!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Some tests failed{Colors.RESET}")
            print(f"\nFailed tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['name']}")
        
        print(f"{'='*60}\n")
    
    def save_results(self, output_file: str = 'api_smoke_test_results.json'):
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
    """Run API smoke tests"""
    parser = argparse.ArgumentParser(description='AiMedRes API Smoke Test')
    parser.add_argument('--host', default='127.0.0.1', help='API server host')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"{Colors.BOLD}AiMedRes API Smoke Test{Colors.RESET}")
    print("="*60)
    print(f"Testing API at: {base_url}\n")
    
    tester = APISmokeTest(base_url, verbose=args.verbose)
    
    # Test 1: Health check
    tester.run_test(
        "Health Check",
        "GET",
        "/health",
        expect_status=200,
        check_response=lambda r: 'status' in r
    )
    
    # Test 2: API root
    tester.run_test(
        "API Root",
        "GET",
        "/",
        expect_status=200
    )
    
    # Test 3: API version
    tester.run_test(
        "API Version Info",
        "GET",
        "/api/v1",
        # May return 404 or 200 depending on implementation
        expect_status=200
    )
    
    # Test 4: Model list endpoint (without auth - should fail or require auth)
    # We expect this to either fail with 401 or succeed depending on auth config
    result = tester.run_test(
        "Model List (No Auth)",
        "GET",
        "/api/v1/model/list",
        # Could be 401 (auth required) or 200 (auth not enforced in test)
        expect_status=200  # Adjust based on your auth setup
    )
    
    # If we got a response, try to parse models
    if result and tester.results[-1].get('status_code') == 200:
        print(f"  {Colors.YELLOW}Note: Auth may not be enforced in test environment{Colors.RESET}")
    
    # Test 5: Invalid endpoint
    tester.run_test(
        "Invalid Endpoint",
        "GET",
        "/api/v1/invalid_endpoint_xyz",
        expect_status=404
    )
    
    # Test 6: Method not allowed
    tester.run_test(
        "Method Not Allowed",
        "DELETE",
        "/health",
        expect_status=405
    )
    
    # Test 7: Check CORS headers (if enabled)
    print(f"\n{Colors.BLUE}Testing:{Colors.RESET} CORS Headers")
    try:
        response = requests.options(f"{base_url}/api/v1/model/list")
        if 'Access-Control-Allow-Origin' in response.headers:
            print(f"  {Colors.GREEN}✓ CORS enabled{Colors.RESET}")
        else:
            print(f"  {Colors.YELLOW}⚠ CORS not detected{Colors.RESET}")
    except Exception as e:
        print(f"  {Colors.YELLOW}⚠ Could not check CORS: {e}{Colors.RESET}")
    
    # Print summary
    tester.print_summary()
    
    # Save results
    tester.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == '__main__':
    main()
