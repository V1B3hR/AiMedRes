"""
Performance and Load Testing Scripts for AiMedRes API.

Provides:
- Latency benchmarking
- Load testing
- SLO validation
- Performance reporting
"""

import time
import statistics
import concurrent.futures
import requests
import json
from typing import Dict, Any, List
from datetime import datetime


class PerformanceTester:
    """
    Performance testing framework for API endpoints.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = None):
        """
        Initialize performance tester.
        
        Args:
            base_url: Base URL for API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key or "dmk_test_key_for_performance"
        self.headers = {'X-API-Key': self.api_key}
        self.results = []
    
    def measure_latency(self, endpoint: str, method: str = 'GET', data: Dict = None, iterations: int = 100) -> Dict[str, float]:
        """
        Measure endpoint latency.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data for POST/PUT
            iterations: Number of test iterations
        
        Returns:
            Latency statistics (p50, p95, p99, mean, max)
        """
        latencies = []
        url = f"{self.base_url}{endpoint}"
        
        print(f"Testing {method} {endpoint} - {iterations} iterations...")
        
        for i in range(iterations):
            start = time.time()
            
            try:
                if method == 'GET':
                    response = requests.get(url, headers=self.headers, timeout=30)
                elif method == 'POST':
                    response = requests.post(url, headers=self.headers, json=data, timeout=30)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                elapsed = (time.time() - start) * 1000  # Convert to ms
                latencies.append(elapsed)
                
                if response.status_code not in [200, 201]:
                    print(f"  Warning: Request {i+1} returned status {response.status_code}")
                
            except Exception as e:
                print(f"  Error in request {i+1}: {e}")
                continue
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{iterations} requests")
        
        if not latencies:
            return {'error': 'All requests failed'}
        
        latencies.sort()
        
        stats = {
            'p50': latencies[int(len(latencies) * 0.50)],
            'p95': latencies[int(len(latencies) * 0.95)],
            'p99': latencies[int(len(latencies) * 0.99)],
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'max': max(latencies),
            'min': min(latencies),
            'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'total_requests': len(latencies),
            'failed_requests': iterations - len(latencies)
        }
        
        return stats
    
    def load_test(self, endpoint: str, method: str = 'GET', data: Dict = None, 
                  concurrent_users: int = 10, requests_per_user: int = 10) -> Dict[str, Any]:
        """
        Run load test with concurrent users.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user
        
        Returns:
            Load test results
        """
        print(f"\nLoad Testing {method} {endpoint}")
        print(f"Concurrent users: {concurrent_users}, Requests per user: {requests_per_user}")
        
        url = f"{self.base_url}{endpoint}"
        
        def user_session(user_id: int) -> List[float]:
            """Simulate a user session."""
            latencies = []
            
            for i in range(requests_per_user):
                start = time.time()
                
                try:
                    if method == 'GET':
                        response = requests.get(url, headers=self.headers, timeout=30)
                    elif method == 'POST':
                        response = requests.post(url, headers=self.headers, json=data, timeout=30)
                    
                    elapsed = (time.time() - start) * 1000
                    latencies.append(elapsed)
                    
                except Exception as e:
                    print(f"  User {user_id} request {i+1} failed: {e}")
                    continue
            
            return latencies
        
        # Run concurrent user sessions
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session, i) for i in range(concurrent_users)]
            all_latencies = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    latencies = future.result()
                    all_latencies.extend(latencies)
                except Exception as e:
                    print(f"  Session error: {e}")
        
        total_time = time.time() - start_time
        
        if not all_latencies:
            return {'error': 'All requests failed'}
        
        all_latencies.sort()
        
        total_requests = concurrent_users * requests_per_user
        successful_requests = len(all_latencies)
        
        results = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'total_time_seconds': total_time,
            'throughput_rps': successful_requests / total_time,
            'latency_p50': all_latencies[int(len(all_latencies) * 0.50)],
            'latency_p95': all_latencies[int(len(all_latencies) * 0.95)],
            'latency_p99': all_latencies[int(len(all_latencies) * 0.99)],
            'latency_mean': statistics.mean(all_latencies),
            'latency_max': max(all_latencies),
            'concurrent_users': concurrent_users,
            'requests_per_user': requests_per_user
        }
        
        return results
    
    def validate_slos(self, results: Dict[str, Any], slos: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate results against SLOs.
        
        Args:
            results: Performance test results
            slos: SLO thresholds (e.g., {'p50': 100, 'p95': 500})
        
        Returns:
            SLO validation results
        """
        validation = {}
        
        for metric, threshold in slos.items():
            if metric in results:
                validation[metric] = results[metric] <= threshold
        
        return validation
    
    def generate_report(self, test_results: List[Dict[str, Any]], output_file: str = None):
        """
        Generate performance test report.
        
        Args:
            test_results: List of test results
            output_file: Optional file path to save report
        """
        report = {
            'test_date': datetime.now().isoformat(),
            'base_url': self.base_url,
            'results': test_results,
            'summary': {
                'total_tests': len(test_results),
                'passed_tests': sum(1 for r in test_results if r.get('slo_passed', True)),
                'failed_tests': sum(1 for r in test_results if not r.get('slo_passed', True))
            }
        }
        
        # Print summary
        print("\n" + "="*80)
        print("PERFORMANCE TEST REPORT")
        print("="*80)
        print(f"Test Date: {report['test_date']}")
        print(f"Base URL: {self.base_url}")
        print(f"\nSummary:")
        print(f"  Total Tests: {report['summary']['total_tests']}")
        print(f"  Passed: {report['summary']['passed_tests']}")
        print(f"  Failed: {report['summary']['failed_tests']}")
        print("\nDetailed Results:")
        
        for i, result in enumerate(test_results, 1):
            print(f"\n  Test {i}: {result.get('test_name', 'Unnamed')}")
            print(f"    Endpoint: {result.get('endpoint', 'N/A')}")
            if 'latency_p50' in result:
                print(f"    Latency p50: {result['latency_p50']:.2f} ms")
            if 'latency_p95' in result:
                print(f"    Latency p95: {result['latency_p95']:.2f} ms")
            if 'throughput_rps' in result:
                print(f"    Throughput: {result['throughput_rps']:.2f} req/s")
            if 'success_rate' in result:
                print(f"    Success Rate: {result['success_rate']:.1f}%")
            if 'slo_passed' in result:
                status = "✓ PASSED" if result['slo_passed'] else "✗ FAILED"
                print(f"    SLO Status: {status}")
        
        print("\n" + "="*80)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {output_file}")
        
        return report


def run_comprehensive_performance_tests():
    """
    Run comprehensive performance tests on all API endpoints.
    """
    tester = PerformanceTester()
    
    # Define SLOs
    slos = {
        'p50': 100,   # 100ms for p50
        'p95': 500,   # 500ms for p95
        'p99': 1000   # 1000ms for p99
    }
    
    test_results = []
    
    # Test 1: Health endpoint
    print("\n" + "="*80)
    print("Test 1: Health Check Endpoint")
    print("="*80)
    
    health_stats = tester.measure_latency('/health', iterations=100)
    health_slo = tester.validate_slos(health_stats, slos)
    
    test_results.append({
        'test_name': 'Health Check Latency',
        'endpoint': '/health',
        'method': 'GET',
        **health_stats,
        'slo_validation': health_slo,
        'slo_passed': all(health_slo.values())
    })
    
    # Test 2: Model inference endpoint
    print("\n" + "="*80)
    print("Test 2: Model Inference Endpoint")
    print("="*80)
    
    inference_data = {
        'data': {'age': 72, 'mmse': 24},
        'patient_id': 'test-patient'
    }
    
    inference_stats = tester.measure_latency('/api/v1/model/infer', method='POST', data=inference_data, iterations=50)
    inference_slo = tester.validate_slos(inference_stats, slos)
    
    test_results.append({
        'test_name': 'Model Inference Latency',
        'endpoint': '/api/v1/model/infer',
        'method': 'POST',
        **inference_stats,
        'slo_validation': inference_slo,
        'slo_passed': all(inference_slo.values())
    })
    
    # Test 3: Load test on health endpoint
    print("\n" + "="*80)
    print("Test 3: Health Endpoint Load Test")
    print("="*80)
    
    load_results = tester.load_test('/health', concurrent_users=20, requests_per_user=10)
    load_slo = tester.validate_slos(load_results, {'latency_p50': 100, 'latency_p95': 500})
    
    test_results.append({
        'test_name': 'Health Endpoint Load Test',
        'endpoint': '/health',
        'method': 'GET',
        'test_type': 'load',
        **load_results,
        'slo_validation': load_slo,
        'slo_passed': all(load_slo.values())
    })
    
    # Generate report
    report = tester.generate_report(test_results, 'performance_test_report.json')
    
    return report


if __name__ == '__main__':
    print("Starting AiMedRes API Performance Tests...")
    print("Make sure the API server is running on http://localhost:8080")
    print()
    
    try:
        report = run_comprehensive_performance_tests()
        
        # Exit with error code if any tests failed
        if report['summary']['failed_tests'] > 0:
            print("\n⚠️  Some performance tests failed!")
            exit(1)
        else:
            print("\n✓ All performance tests passed!")
            exit(0)
            
    except Exception as e:
        print(f"\n✗ Performance tests failed with error: {e}")
        exit(1)
