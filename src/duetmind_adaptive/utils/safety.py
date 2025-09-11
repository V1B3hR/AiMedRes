"""
Safety Monitoring and Circuit Breaker System

Provides comprehensive safety monitoring for the DuetMind system
including performance monitoring, error tracking, and automatic
circuit breakers for fault tolerance.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import psutil
import os

logger = logging.getLogger(__name__)

class SafetyState(Enum):
    """Safety monitor states"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class OperationRecord:
    """Record of a system operation"""
    operation: str
    duration: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class SafetyMonitor:
    """
    Comprehensive safety monitoring system with circuit breakers.
    
    Features:
    - Performance monitoring (CPU, memory, response times)
    - Error rate tracking and thresholds
    - Circuit breaker patterns for fault tolerance
    - Resource exhaustion detection
    - Automatic recovery mechanisms
    """
    
    def __init__(self, 
                 max_error_rate: float = 0.1,
                 max_response_time: float = 30.0,
                 max_memory_mb: float = 1024.0,
                 max_cpu_percent: float = 80.0,
                 monitoring_interval: float = 5.0):
        """
        Initialize safety monitor
        
        Args:
            max_error_rate: Maximum acceptable error rate (0-1)
            max_response_time: Maximum acceptable response time in seconds
            max_memory_mb: Maximum acceptable memory usage in MB
            max_cpu_percent: Maximum acceptable CPU usage percentage
            monitoring_interval: Monitoring check interval in seconds
        """
        self.max_error_rate = max_error_rate
        self.max_response_time = max_response_time
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.monitoring_interval = monitoring_interval
        
        # Operation tracking
        self.operation_history: deque = deque(maxlen=1000)
        self.error_count = 0
        self.total_operations = 0
        
        # System resource tracking
        self.cpu_history: deque = deque(maxlen=100)
        self.memory_history: deque = deque(maxlen=100)
        self.response_time_history: deque = deque(maxlen=100)
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_half_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_timeout = timedelta(minutes=5)
        
        # Safety state
        self.safety_state = SafetyState.SAFE
        self.last_safety_check = datetime.now()
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Process reference for resource monitoring
        self.process = psutil.Process(os.getpid())
        
    def start(self):
        """Start background safety monitoring"""
        with self._lock:
            if not self.monitoring:
                self.monitoring = True
                self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitor_thread.start()
                logger.info("Safety monitoring started")
    
    def stop(self):
        """Stop safety monitoring"""
        with self._lock:
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1)
            logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._check_system_resources()
                self._update_safety_state()
                self._check_circuit_breaker()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            self.cpu_history.append(cpu_percent)
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_history.append(memory_mb)
            
            # Log resource warnings
            if cpu_percent > self.max_cpu_percent:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_mb > self.max_memory_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
    
    def _update_safety_state(self):
        """Update overall safety state based on metrics"""
        with self._lock:
            # Calculate current metrics
            error_rate = self.get_error_rate()
            avg_response_time = self.get_average_response_time()
            current_memory = self.memory_history[-1] if self.memory_history else 0
            current_cpu = self.cpu_history[-1] if self.cpu_history else 0
            
            # Determine safety state
            critical_conditions = 0
            warning_conditions = 0
            
            # Check error rate
            if error_rate > self.max_error_rate * 2:
                critical_conditions += 1
            elif error_rate > self.max_error_rate:
                warning_conditions += 1
            
            # Check response time
            if avg_response_time > self.max_response_time * 2:
                critical_conditions += 1
            elif avg_response_time > self.max_response_time:
                warning_conditions += 1
            
            # Check memory usage
            if current_memory > self.max_memory_mb * 1.5:
                critical_conditions += 1
            elif current_memory > self.max_memory_mb:
                warning_conditions += 1
            
            # Check CPU usage
            if current_cpu > self.max_cpu_percent * 1.2:
                critical_conditions += 1
            elif current_cpu > self.max_cpu_percent:
                warning_conditions += 1
            
            # Update safety state
            previous_state = self.safety_state
            
            if critical_conditions >= 2:
                self.safety_state = SafetyState.CRITICAL
            elif critical_conditions >= 1:
                self.safety_state = SafetyState.WARNING
            elif warning_conditions >= 3:
                self.safety_state = SafetyState.WARNING
            else:
                self.safety_state = SafetyState.SAFE
            
            # Log state changes
            if self.safety_state != previous_state:
                logger.warning(f"Safety state changed: {previous_state.value} → {self.safety_state.value}")
            
            self.last_safety_check = datetime.now()
    
    def _check_circuit_breaker(self):
        """Check and update circuit breaker state"""
        with self._lock:
            now = datetime.now()
            
            # Check if circuit breaker should reset
            if (self.circuit_breaker_open and 
                self.circuit_breaker_last_failure and
                now - self.circuit_breaker_last_failure > self.circuit_breaker_timeout):
                
                self.circuit_breaker_half_open = True
                self.circuit_breaker_open = False
                logger.info("Circuit breaker entering half-open state")
    
    def record_operation(self, 
                        operation: str, 
                        duration: float, 
                        success: bool, 
                        error: Optional[str] = None):
        """
        Record an operation for monitoring
        
        Args:
            operation: Operation name/type
            duration: Operation duration in seconds
            success: Whether operation succeeded
            error: Error message if failed
        """
        with self._lock:
            # Get current resource usage
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
            except:
                memory_mb = 0.0
                cpu_percent = 0.0
            
            # Create operation record
            record = OperationRecord(
                operation=operation,
                duration=duration,
                success=success,
                error=error,
                memory_usage=memory_mb,
                cpu_usage=cpu_percent
            )
            
            self.operation_history.append(record)
            self.total_operations += 1
            
            if not success:
                self.error_count += 1
                self.circuit_breaker_failures += 1
                self.circuit_breaker_last_failure = datetime.now()
                
                # Check circuit breaker threshold
                if (self.circuit_breaker_failures >= 5 and
                    not self.circuit_breaker_open):
                    self.circuit_breaker_open = True
                    self.circuit_breaker_half_open = False
                    logger.error("Circuit breaker opened due to failures")
            
            # Track response times
            self.response_time_history.append(duration)
            
            # Reset circuit breaker on successful operation in half-open state
            if success and self.circuit_breaker_half_open:
                self.circuit_breaker_open = False
                self.circuit_breaker_half_open = False
                self.circuit_breaker_failures = 0
                logger.info("Circuit breaker reset after successful operation")
    
    def is_safe_to_operate(self) -> bool:
        """
        Check if system is safe to operate
        
        Returns:
            True if safe to perform operations
        """
        with self._lock:
            # Circuit breaker check
            if self.circuit_breaker_open:
                return False
            
            # Safety state check
            if self.safety_state == SafetyState.CRITICAL:
                return False
            
            # Resource exhaustion check
            if self.memory_history:
                current_memory = self.memory_history[-1]
                if current_memory > self.max_memory_mb * 1.8:  # Emergency threshold
                    return False
            
            if self.cpu_history:
                recent_cpu = list(self.cpu_history)[-5:]  # Last 5 measurements
                if len(recent_cpu) >= 3 and all(cpu > self.max_cpu_percent * 1.5 for cpu in recent_cpu):
                    return False
            
            return True
    
    def get_error_rate(self) -> float:
        """Get current error rate"""
        with self._lock:
            if self.total_operations == 0:
                return 0.0
            return self.error_count / self.total_operations
    
    def get_average_response_time(self) -> float:
        """Get average response time from recent operations"""
        if not self.response_time_history:
            return 0.0
        return sum(self.response_time_history) / len(self.response_time_history)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety monitor status"""
        with self._lock:
            return {
                "safety_state": self.safety_state.value,
                "is_safe": self.is_safe_to_operate(),
                "circuit_breaker": {
                    "open": self.circuit_breaker_open,
                    "half_open": self.circuit_breaker_half_open,
                    "failures": self.circuit_breaker_failures,
                    "last_failure": self.circuit_breaker_last_failure.isoformat() if self.circuit_breaker_last_failure else None
                },
                "metrics": {
                    "total_operations": self.total_operations,
                    "error_count": self.error_count,
                    "error_rate": self.get_error_rate(),
                    "average_response_time": self.get_average_response_time(),
                    "current_memory_mb": self.memory_history[-1] if self.memory_history else 0,
                    "current_cpu_percent": self.cpu_history[-1] if self.cpu_history else 0,
                },
                "thresholds": {
                    "max_error_rate": self.max_error_rate,
                    "max_response_time": self.max_response_time,
                    "max_memory_mb": self.max_memory_mb,
                    "max_cpu_percent": self.max_cpu_percent
                },
                "last_safety_check": self.last_safety_check.isoformat()
            }
    
    def force_circuit_breaker_reset(self):
        """Manually reset circuit breaker (use with caution)"""
        with self._lock:
            self.circuit_breaker_open = False
            self.circuit_breaker_half_open = False
            self.circuit_breaker_failures = 0
            logger.warning("Circuit breaker manually reset")
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()