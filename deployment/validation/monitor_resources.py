#!/usr/bin/env python3
"""
Resource Monitoring Script for AiMedRes System Validation

Monitors system resources (CPU, Memory, Disk, Network) during validation tests.

Usage:
    python monitor_resources.py [--duration SECONDS] [--interval SECONDS] [--output FILE]
"""

import psutil
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.snapshots = []
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current resource snapshot"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            mem = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process info (for aimedres processes)
            aimedres_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower() or 'aimedres' in proc.info['name'].lower():
                        aimedres_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'per_cpu': psutil.cpu_percent(interval=0, percpu=True) if cpu_count else []
                },
                'memory': {
                    'total_gb': mem.total / (1024**3),
                    'available_gb': mem.available / (1024**3),
                    'used_gb': mem.used / (1024**3),
                    'percent': mem.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': disk.percent
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'processes': aimedres_processes[:5]  # Top 5 processes
            }
            
            return snapshot
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def monitor(self, duration: int):
        """
        Monitor resources for specified duration.
        
        Args:
            duration: Duration in seconds
        """
        print(f"Monitoring resources for {duration} seconds...")
        print(f"Interval: {self.interval} seconds")
        print(f"Expected snapshots: {duration // self.interval}\n")
        
        start_time = time.time()
        snapshot_count = 0
        
        while time.time() - start_time < duration:
            snapshot = self.get_snapshot()
            self.snapshots.append(snapshot)
            snapshot_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            if 'error' not in snapshot:
                print(f"[{snapshot_count}] CPU: {snapshot['cpu']['percent']:.1f}% | "
                      f"Memory: {snapshot['memory']['percent']:.1f}% | "
                      f"Disk: {snapshot['disk']['percent']:.1f}% | "
                      f"Remaining: {remaining:.0f}s")
            
            # Wait for next interval
            time.sleep(self.interval)
        
        print(f"\nMonitoring complete. Collected {snapshot_count} snapshots.")
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze collected snapshots"""
        if not self.snapshots:
            return {'error': 'No snapshots collected'}
        
        # Filter out error snapshots
        valid_snapshots = [s for s in self.snapshots if 'error' not in s]
        
        if not valid_snapshots:
            return {'error': 'No valid snapshots'}
        
        # Calculate statistics
        cpu_values = [s['cpu']['percent'] for s in valid_snapshots]
        mem_values = [s['memory']['percent'] for s in valid_snapshots]
        disk_values = [s['disk']['percent'] for s in valid_snapshots]
        
        analysis = {
            'summary': {
                'total_snapshots': len(self.snapshots),
                'valid_snapshots': len(valid_snapshots),
                'monitoring_period': {
                    'start': valid_snapshots[0]['timestamp'],
                    'end': valid_snapshots[-1]['timestamp']
                }
            },
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'count': valid_snapshots[0]['cpu']['count']
            },
            'memory': {
                'average_percent': sum(mem_values) / len(mem_values),
                'min_percent': min(mem_values),
                'max_percent': max(mem_values),
                'total_gb': valid_snapshots[0]['memory']['total_gb']
            },
            'disk': {
                'average_percent': sum(disk_values) / len(disk_values),
                'min_percent': min(disk_values),
                'max_percent': max(disk_values),
                'total_gb': valid_snapshots[0]['disk']['total_gb']
            },
            'alerts': []
        }
        
        # Generate alerts
        if analysis['cpu']['max'] > 90:
            analysis['alerts'].append({
                'severity': 'high',
                'component': 'cpu',
                'message': f"High CPU usage detected: {analysis['cpu']['max']:.1f}%"
            })
        elif analysis['cpu']['average'] > 70:
            analysis['alerts'].append({
                'severity': 'medium',
                'component': 'cpu',
                'message': f"Elevated average CPU usage: {analysis['cpu']['average']:.1f}%"
            })
        
        if analysis['memory']['max_percent'] > 90:
            analysis['alerts'].append({
                'severity': 'high',
                'component': 'memory',
                'message': f"High memory usage detected: {analysis['memory']['max_percent']:.1f}%"
            })
        elif analysis['memory']['average_percent'] > 80:
            analysis['alerts'].append({
                'severity': 'medium',
                'component': 'memory',
                'message': f"Elevated average memory usage: {analysis['memory']['average_percent']:.1f}%"
            })
        
        if analysis['disk']['average_percent'] > 85:
            analysis['alerts'].append({
                'severity': 'medium',
                'component': 'disk',
                'message': f"High disk usage: {analysis['disk']['average_percent']:.1f}%"
            })
        
        return analysis
    
    def print_report(self):
        """Print resource monitoring report"""
        analysis = self.analyze()
        
        if 'error' in analysis:
            print(f"\nError: {analysis['error']}")
            return
        
        print("\n" + "="*60)
        print("Resource Monitoring Report")
        print("="*60)
        
        print(f"\nMonitoring Summary:")
        print(f"  Total snapshots: {analysis['summary']['total_snapshots']}")
        print(f"  Valid snapshots: {analysis['summary']['valid_snapshots']}")
        
        print(f"\nCPU Usage:")
        print(f"  Average: {analysis['cpu']['average']:.1f}%")
        print(f"  Min: {analysis['cpu']['min']:.1f}%")
        print(f"  Max: {analysis['cpu']['max']:.1f}%")
        print(f"  CPU Cores: {analysis['cpu']['count']}")
        
        print(f"\nMemory Usage:")
        print(f"  Average: {analysis['memory']['average_percent']:.1f}%")
        print(f"  Min: {analysis['memory']['min_percent']:.1f}%")
        print(f"  Max: {analysis['memory']['max_percent']:.1f}%")
        print(f"  Total: {analysis['memory']['total_gb']:.2f} GB")
        
        print(f"\nDisk Usage:")
        print(f"  Average: {analysis['disk']['average_percent']:.1f}%")
        print(f"  Total: {analysis['disk']['total_gb']:.2f} GB")
        
        if analysis['alerts']:
            print(f"\nAlerts:")
            for alert in analysis['alerts']:
                severity_color = '\033[91m' if alert['severity'] == 'high' else '\033[93m'
                print(f"  {severity_color}[{alert['severity'].upper()}]\033[0m {alert['message']}")
        else:
            print(f"\n\033[92m✓ No resource alerts\033[0m")
        
        print("="*60 + "\n")
    
    def save_results(self, output_file: str):
        """Save results to JSON file"""
        output_path = Path(output_file)
        
        data = {
            'analysis': self.analyze(),
            'snapshots': self.snapshots
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {output_path.absolute()}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Monitor system resources during AiMedRes validation')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('--interval', type=int, default=5, help='Snapshot interval in seconds (default: 5)')
    parser.add_argument('--output', type=str, default='resource_report.json', help='Output file path')
    args = parser.parse_args()
    
    print("AiMedRes Resource Monitor")
    print("="*60 + "\n")
    
    monitor = ResourceMonitor(interval=args.interval)
    
    try:
        monitor.monitor(duration=args.duration)
        monitor.print_report()
        monitor.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        if monitor.snapshots:
            monitor.print_report()
            monitor.save_results(args.output)
    
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
