#!/usr/bin/env python3
"""
High-Performance Medical AI Response Time Monitor

Comprehensive performance monitoring system designed to ensure clinical AI responses
meet the critical <100ms target time for medical applications. Features include:

- Real-time response time tracking with medical-grade precision
- Automated performance optimization recommendations
- Clinical alert system for response time violations
- Performance trend analysis and prediction
- Integration with HIPAA audit logging
- Emergency escalation for critical delays
"""

import time
import threading
import logging
import statistics
import gc
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
from enum import Enum
import queue
import psutil
import contextlib
from functools import wraps

logger = logging.getLogger("duetmind.performance")


class ClinicalPriority(Enum):
    """Clinical priority levels for response time requirements"""

    EMERGENCY = "EMERGENCY"  # <20ms target
    CRITICAL = "CRITICAL"  # <50ms target
    URGENT = "URGENT"  # <100ms target
    ROUTINE = "ROUTINE"  # <200ms target
    ADMINISTRATIVE = "ADMIN"  # <500ms target


class AlertLevel(Enum):
    """Performance alert levels"""

    GREEN = "GREEN"  # Performance within targets
    YELLOW = "YELLOW"  # Performance degraded but acceptable
    ORANGE = "ORANGE"  # Performance concerning, intervention needed
    RED = "RED"  # Critical performance failure


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""

    timestamp: datetime
    operation: str
    response_time_ms: float
    clinical_priority: ClinicalPriority
    success: bool
    error_message: Optional[str]
    user_id: Optional[str]
    patient_id: Optional[str]
    system_load: Dict[str, float]
    memory_usage: Dict[str, float]
    additional_context: Dict[str, Any]


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different clinical priorities"""

    emergency_max_ms: float = 20.0
    critical_max_ms: float = 50.0
    urgent_max_ms: float = 100.0
    routine_max_ms: float = 200.0
    admin_max_ms: float = 500.0

    # Violation thresholds (how many violations trigger alerts)
    violation_count_warning: int = 3  # Per minute
    violation_count_critical: int = 5  # Per minute

    # System resource thresholds
    cpu_warning_threshold: float = 80.0  # %
    memory_warning_threshold: float = 85.0  # %


class ClinicalPerformanceMonitor:
    """
    High-precision performance monitor for clinical AI systems.

    Ensures response times meet medical safety requirements and provides
    real-time optimization recommendations.
    """

    def __init__(
        self,
        thresholds: Optional[PerformanceThresholds] = None,
        enable_audit_integration: bool = True,
        metric_history_size: int = 10000,
    ):
        """
        Initialize clinical performance monitor.

        Args:
            thresholds: Performance thresholds configuration
            enable_audit_integration: Enable HIPAA audit integration
            metric_history_size: Size of metrics history buffer
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.enable_audit_integration = enable_audit_integration

        # Metrics storage
        self.metrics_history = deque(maxlen=metric_history_size)
        self.real_time_metrics = defaultdict(list)  # Last minute of metrics

        # Alert management
        self.active_alerts = {}
        self.alert_callbacks = []
        self.performance_violations = defaultdict(int)

        # Threading
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        self._metrics_queue = queue.Queue()

        # Performance tracking
        self.response_times = {
            ClinicalPriority.EMERGENCY: deque(maxlen=1000),
            ClinicalPriority.CRITICAL: deque(maxlen=1000),
            ClinicalPriority.URGENT: deque(maxlen=1000),
            ClinicalPriority.ROUTINE: deque(maxlen=1000),
            ClinicalPriority.ADMINISTRATIVE: deque(maxlen=1000),
        }

        # System monitoring
        self.system_metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "io_wait": deque(maxlen=100),
        }

        # Auto-optimization settings
        self.auto_optimization_enabled = False
        self.auto_optimization_threshold = 0.1  # 10% performance degradation

        # Monitoring configuration
        self.monitoring_interval_seconds = 0.1  # Default 100ms
        self.metrics_retention_hours = 24  # Default 24 hours

        # Validate configuration
        self._validate_configuration()

        logger.info("Clinical Performance Monitor initialized")

    def _validate_configuration(self) -> None:
        """Validate monitor configuration parameters."""
        if self.thresholds.emergency_max_ms <= 0:
            raise ValueError("Emergency threshold must be positive")

        if self.thresholds.violation_count_warning >= self.thresholds.violation_count_critical:
            logger.warning("Warning violation count should be less than critical count")

        if self.thresholds.cpu_warning_threshold > 100 or self.thresholds.cpu_warning_threshold < 0:
            raise ValueError("CPU threshold must be between 0 and 100")

        if self.thresholds.memory_warning_threshold > 100 or self.thresholds.memory_warning_threshold < 0:
            raise ValueError("Memory threshold must be between 0 and 100")

    def set_monitoring_interval(self, interval_seconds: float) -> None:
        """
        Set monitoring loop interval.

        Args:
            interval_seconds: Interval in seconds (minimum 0.01, maximum 60)
        """
        if interval_seconds < 0.01 or interval_seconds > 60:
            raise ValueError("Monitoring interval must be between 0.01 and 60 seconds")

        self.monitoring_interval_seconds = interval_seconds
        logger.info(f"Monitoring interval set to {interval_seconds}s")

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current monitor configuration.

        Returns:
            Dictionary containing current configuration
        """
        return {
            "thresholds": {
                "emergency_max_ms": self.thresholds.emergency_max_ms,
                "critical_max_ms": self.thresholds.critical_max_ms,
                "urgent_max_ms": self.thresholds.urgent_max_ms,
                "routine_max_ms": self.thresholds.routine_max_ms,
                "admin_max_ms": self.thresholds.admin_max_ms,
                "violation_count_warning": self.thresholds.violation_count_warning,
                "violation_count_critical": self.thresholds.violation_count_critical,
                "cpu_warning_threshold": self.thresholds.cpu_warning_threshold,
                "memory_warning_threshold": self.thresholds.memory_warning_threshold,
            },
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "auto_optimization_threshold": self.auto_optimization_threshold,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "metrics_retention_hours": self.metrics_retention_hours,
            "enable_audit_integration": self.enable_audit_integration,
            "metric_history_size": self.metrics_history.maxlen,
        }

    def start_monitoring(self):
        """Start background performance monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True, name="ClinicalPerformanceMonitor"
            )
            self._monitor_thread.start()
            logger.info("Clinical performance monitoring started")

    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        logger.info("Clinical performance monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Process queued metrics
                self._process_queued_metrics()

                # Update system metrics
                self._update_system_metrics()

                # Check for performance violations
                self._check_performance_violations()

                # Check for auto-optimization triggers
                if self.auto_optimization_enabled:
                    self._check_auto_optimization_triggers()

                # Cleanup old metrics
                self._cleanup_old_metrics()

                time.sleep(self.monitoring_interval_seconds)

            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}", exc_info=True)
                time.sleep(1)  # Wait longer on error

    def _process_queued_metrics(self):
        """Process metrics from the queue with batch processing."""
        processed_count = 0
        batch_size = 100
        metrics_batch = []

        # Collect batch of metrics
        while not self._metrics_queue.empty() and processed_count < batch_size:
            try:
                metric = self._metrics_queue.get_nowait()
                metrics_batch.append(metric)
                processed_count += 1
            except queue.Empty:
                break

        # Process batch with single lock acquisition
        if metrics_batch:
            with self._lock:
                for metric in metrics_batch:
                    self._store_metric_unsafe(metric)

    def _store_metric_unsafe(self, metric: PerformanceMetrics):
        """Store performance metric without lock (caller must hold lock)."""
        # Add to history
        self.metrics_history.append(metric)

        # Add to real-time tracking
        now = datetime.now(timezone.utc)
        minute_key = now.strftime("%Y-%m-%d_%H-%M")
        self.real_time_metrics[minute_key].append(metric)

        # Add to priority-specific tracking
        if metric.clinical_priority in self.response_times:
            self.response_times[metric.clinical_priority].append(metric.response_time_ms)

        # Check for immediate violations (moved outside lock in caller)

    def _store_metric(self, metric: PerformanceMetrics):
        """Store performance metric and trigger analysis (thread-safe)."""
        with self._lock:
            self._store_metric_unsafe(metric)

            # Check for immediate violations
            self._check_immediate_violation(metric)

    def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            with self._lock:
                self.system_metrics["cpu_usage"].append(cpu_percent)
                self.system_metrics["memory_usage"].append(memory.percent)

                # Check system resource alerts
                if cpu_percent > self.thresholds.cpu_warning_threshold:
                    self._trigger_alert("HIGH_CPU_USAGE", f"CPU usage at {cpu_percent:.1f}%", AlertLevel.ORANGE)

                if memory.percent > self.thresholds.memory_warning_threshold:
                    self._trigger_alert(
                        "HIGH_MEMORY_USAGE", f"Memory usage at {memory.percent:.1f}%", AlertLevel.ORANGE
                    )
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def _check_immediate_violation(self, metric: PerformanceMetrics):
        """Check if metric violates performance thresholds immediately."""
        threshold_ms = self._get_threshold_for_priority(metric.clinical_priority)

        if metric.response_time_ms > threshold_ms:
            severity = self._calculate_violation_severity(
                metric.response_time_ms, threshold_ms, metric.clinical_priority
            )

            self._trigger_performance_violation(metric, severity)

    def _get_threshold_for_priority(self, priority: ClinicalPriority) -> float:
        """Get response time threshold for clinical priority."""
        threshold_map = {
            ClinicalPriority.EMERGENCY: self.thresholds.emergency_max_ms,
            ClinicalPriority.CRITICAL: self.thresholds.critical_max_ms,
            ClinicalPriority.URGENT: self.thresholds.urgent_max_ms,
            ClinicalPriority.ROUTINE: self.thresholds.routine_max_ms,
            ClinicalPriority.ADMINISTRATIVE: self.thresholds.admin_max_ms,
        }
        return threshold_map.get(priority, self.thresholds.routine_max_ms)

    def _calculate_violation_severity(
        self, actual_ms: float, threshold_ms: float, priority: ClinicalPriority
    ) -> AlertLevel:
        """Calculate severity of performance violation."""
        violation_ratio = actual_ms / threshold_ms

        # Emergency and critical violations are always RED
        if priority in [ClinicalPriority.EMERGENCY, ClinicalPriority.CRITICAL]:
            return AlertLevel.RED if violation_ratio > 1.5 else AlertLevel.ORANGE

        # Other priorities based on severity
        if violation_ratio > 2.0:
            return AlertLevel.RED
        elif violation_ratio > 1.5:
            return AlertLevel.ORANGE
        else:
            return AlertLevel.YELLOW

    def _trigger_performance_violation(self, metric: PerformanceMetrics, severity: AlertLevel):
        """Handle performance violation."""
        violation_id = f"perf_violation_{int(time.time() * 1000)}"

        alert_data = {
            "violation_id": violation_id,
            "timestamp": metric.timestamp.isoformat(),
            "operation": metric.operation,
            "response_time_ms": metric.response_time_ms,
            "threshold_ms": self._get_threshold_for_priority(metric.clinical_priority),
            "clinical_priority": metric.clinical_priority.value,
            "severity": severity.value,
            "user_id": metric.user_id,
            "patient_id": metric.patient_id,
        }

        self._trigger_alert(
            "PERFORMANCE_VIOLATION",
            f"Response time {metric.response_time_ms:.1f}ms exceeds {self._get_threshold_for_priority(metric.clinical_priority):.1f}ms threshold for {metric.clinical_priority.value} operation",
            severity,
            alert_data,
        )

        # Log HIPAA audit if enabled and patient involved
        if self.enable_audit_integration and metric.patient_id:
            self._log_performance_audit(metric, violation_id)

    def _trigger_alert(self, alert_type: str, message: str, level: AlertLevel, data: Dict[str, Any] = None):
        """Trigger performance alert."""
        alert = {
            "alert_type": alert_type,
            "message": message,
            "level": level.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }

        with self._lock:
            self.active_alerts[alert_type] = alert

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        # Log based on severity
        log_message = f"[{level.value}] {alert_type}: {message}"
        if level in [AlertLevel.RED]:
            logger.error(log_message)
        elif level == AlertLevel.ORANGE:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _log_performance_audit(self, metric: PerformanceMetrics, violation_id: str):
        """Log performance violation to HIPAA audit system."""
        try:
            from .hipaa_audit import get_audit_logger

            audit_logger = get_audit_logger()
            audit_logger._log_audit_event(
                event_type="MODEL_PREDICTION",
                user_id=metric.user_id or "system",
                user_role="ai_system",
                patient_id=metric.patient_id,
                resource="clinical_ai_response",
                purpose="clinical_decision_support",
                outcome="PERFORMANCE_VIOLATION",
                additional_data={
                    "response_time_ms": metric.response_time_ms,
                    "clinical_priority": metric.clinical_priority.value,
                    "violation_id": violation_id,
                    "operation": metric.operation,
                },
            )
        except ImportError:
            logger.warning("HIPAA audit logging not available for performance violation")

    def _check_performance_violations(self):
        """Check for patterns of performance violations."""
        now = datetime.now(timezone.utc)
        minute_key = now.strftime("%Y-%m-%d_%H-%M")

        # Count violations in the last minute
        recent_metrics = self.real_time_metrics.get(minute_key, [])
        violations_this_minute = 0

        for metric in recent_metrics:
            threshold = self._get_threshold_for_priority(metric.clinical_priority)
            if metric.response_time_ms > threshold:
                violations_this_minute += 1

        # Trigger alerts based on violation counts
        if violations_this_minute >= self.thresholds.violation_count_critical:
            self._trigger_alert(
                "CRITICAL_PERFORMANCE_DEGRADATION",
                f"{violations_this_minute} performance violations in the last minute",
                AlertLevel.RED,
                {"violations_count": violations_this_minute, "minute": minute_key},
            )
        elif violations_this_minute >= self.thresholds.violation_count_warning:
            self._trigger_alert(
                "PERFORMANCE_DEGRADATION_WARNING",
                f"{violations_this_minute} performance violations in the last minute",
                AlertLevel.ORANGE,
                {"violations_count": violations_this_minute, "minute": minute_key},
            )

    def _cleanup_old_metrics(self):
        """Clean up old real-time metrics."""
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(minutes=10)

        with self._lock:
            # Remove metrics older than 10 minutes
            keys_to_remove = []
            for minute_key in self.real_time_metrics:
                try:
                    minute_time = datetime.strptime(minute_key, "%Y-%m-%d_%H-%M")
                    minute_time = minute_time.replace(tzinfo=timezone.utc)
                    if minute_time < cutoff_time:
                        keys_to_remove.append(minute_key)
                except ValueError:
                    keys_to_remove.append(minute_key)  # Remove invalid keys

            for key in keys_to_remove:
                del self.real_time_metrics[key]

    def _check_auto_optimization_triggers(self):
        """Check if auto-optimization should be triggered."""
        try:
            # Get recent performance summary
            recent_performance = self.get_performance_summary(hours_back=0.5)  # Last 30 minutes

            if recent_performance["total_operations"] < 10:
                return  # Not enough data

            # Check if violation rate exceeds threshold
            violation_rate = recent_performance["violation_rate_percent"] / 100

            if violation_rate > self.auto_optimization_threshold:
                logger.warning(f"Performance degradation detected: {violation_rate*100:.1f}% violations")

                # Trigger auto-optimization
                optimization_result = self.trigger_auto_optimization()

                # Alert about auto-optimization
                self._trigger_alert(
                    "AUTO_OPTIMIZATION_TRIGGERED",
                    f"Performance degradation ({violation_rate*100:.1f}%) triggered auto-optimization",
                    AlertLevel.YELLOW,
                    {
                        "violation_rate_percent": violation_rate * 100,
                        "actions_taken": len(optimization_result["actions_taken"]),
                        "optimization_timestamp": optimization_result["timestamp"],
                    },
                )

        except Exception as e:
            logger.error(f"Error in auto-optimization check: {e}")

    def record_operation(
        self,
        operation: str,
        response_time_ms: float,
        clinical_priority: ClinicalPriority = ClinicalPriority.ROUTINE,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        additional_context: Dict[str, Any] = None,
    ):
        """
        Record operation performance metrics.

        Args:
            operation: Name of the operation
            response_time_ms: Response time in milliseconds
            clinical_priority: Clinical priority level
            success: Whether operation succeeded
            error_message: Error message if operation failed
            user_id: User performing the operation
            patient_id: Patient ID if applicable
            additional_context: Additional context data
        """
        try:
            # Input validation
            if not operation or not isinstance(operation, str):
                logger.warning("Invalid operation name provided, skipping metric recording")
                return

            if response_time_ms < 0:
                logger.warning(f"Negative response time ({response_time_ms}ms) provided, using absolute value")
                response_time_ms = abs(response_time_ms)

            if response_time_ms > 300000:  # 5 minutes
                logger.warning(
                    f"Unusually high response time ({response_time_ms}ms) recorded for operation: {operation}"
                )

            # Get system metrics with error handling
            try:
                system_load = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                }
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
                system_load = {"cpu_percent": 0.0, "memory_percent": 0.0}

            try:
                memory_info = psutil.virtual_memory()
                memory_usage = {
                    "available_mb": memory_info.available / (1024 * 1024),
                    "used_percent": memory_info.percent,
                }
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
                memory_usage = {"available_mb": 0.0, "used_percent": 0.0}

            metric = PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                operation=operation,
                response_time_ms=response_time_ms,
                clinical_priority=clinical_priority,
                success=success,
                error_message=error_message,
                user_id=user_id,
                patient_id=patient_id,
                system_load=system_load,
                memory_usage=memory_usage,
                additional_context=additional_context or {},
            )

            # Queue for processing with size check
            if self._metrics_queue.qsize() > 1000:
                logger.warning("Metrics queue size exceeds 1000, possible processing bottleneck")

            self._metrics_queue.put(metric, block=False)

        except queue.Full:
            logger.error("Metrics queue is full, dropping metric")
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}", exc_info=True)

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)

    def get_performance_summary(
        self, priority: Optional[ClinicalPriority] = None, hours_back: int = 1
    ) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=hours_back)

        with self._lock:
            # Filter metrics by time and priority
            filtered_metrics = [
                m
                for m in self.metrics_history
                if m.timestamp >= cutoff_time and (priority is None or m.clinical_priority == priority)
            ]

            if not filtered_metrics:
                return {
                    "summary_period_hours": hours_back,
                    "priority_filter": priority.value if priority else "ALL",
                    "total_operations": 0,
                    "avg_response_time_ms": 0,
                    "performance_status": "NO_DATA",
                }

            response_times = [m.response_time_ms for m in filtered_metrics]
            violations = 0

            for metric in filtered_metrics:
                threshold = self._get_threshold_for_priority(metric.clinical_priority)
                if metric.response_time_ms > threshold:
                    violations += 1

            return {
                "summary_period_hours": hours_back,
                "priority_filter": priority.value if priority else "ALL",
                "total_operations": len(filtered_metrics),
                "successful_operations": sum(1 for m in filtered_metrics if m.success),
                "avg_response_time_ms": statistics.mean(response_times),
                "median_response_time_ms": statistics.median(response_times),
                "p95_response_time_ms": self._percentile(response_times, 95),
                "p99_response_time_ms": self._percentile(response_times, 99),
                "max_response_time_ms": max(response_times),
                "min_response_time_ms": min(response_times),
                "violations_count": violations,
                "violation_rate_percent": (violations / len(filtered_metrics)) * 100,
                "performance_status": self._calculate_overall_performance_status(violations, len(filtered_metrics)),
            }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1

        if upper >= len(sorted_data):
            return sorted_data[-1]

        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def _calculate_overall_performance_status(self, violations: int, total: int) -> str:
        """Calculate overall performance status."""
        if total == 0:
            return "NO_DATA"

        violation_rate = violations / total

        if violation_rate == 0:
            return "EXCELLENT"
        elif violation_rate < 0.01:  # Less than 1%
            return "GOOD"
        elif violation_rate < 0.05:  # Less than 5%
            return "ACCEPTABLE"
        elif violation_rate < 0.10:  # Less than 10%
            return "CONCERNING"
        else:
            return "CRITICAL"

    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active performance alerts."""
        with self._lock:
            return self.active_alerts.copy()

    def clear_alert(self, alert_type: str) -> bool:
        """Clear a specific alert."""
        with self._lock:
            return self.active_alerts.pop(alert_type, None) is not None

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []

        # Check system resource usage
        if self.system_metrics["cpu_usage"]:
            avg_cpu = statistics.mean(list(self.system_metrics["cpu_usage"])[-10:])
            if avg_cpu > 80:
                recommendations.append(
                    {
                        "type": "SYSTEM_RESOURCE",
                        "priority": "HIGH",
                        "title": "High CPU Usage Detected",
                        "description": f"Average CPU usage is {avg_cpu:.1f}%. Consider optimizing algorithms or scaling resources.",
                        "suggested_actions": [
                            "Review and optimize AI model inference code",
                            "Implement request queuing and rate limiting",
                            "Consider horizontal scaling",
                            "Profile CPU-intensive operations",
                        ],
                    }
                )

        # Check response time trends by priority
        for priority in ClinicalPriority:
            if self.response_times[priority]:
                recent_times = list(self.response_times[priority])[-100:]
                if len(recent_times) >= 10:
                    avg_time = statistics.mean(recent_times)
                    threshold = self._get_threshold_for_priority(priority)

                    if avg_time > threshold * 0.8:  # Within 80% of threshold
                        recommendations.append(
                            {
                                "type": "RESPONSE_TIME",
                                "priority": "MEDIUM",
                                "title": f"{priority.value} Operations Approaching Threshold",
                                "description": f"Average response time ({avg_time:.1f}ms) is approaching the {threshold}ms threshold.",
                                "suggested_actions": [
                                    "Optimize database queries",
                                    "Implement caching for frequent operations",
                                    "Review AI model complexity",
                                    "Consider precomputing common results",
                                ],
                            }
                        )

        return recommendations

    def trigger_auto_optimization(self) -> Dict[str, Any]:
        """Trigger automatic performance optimization based on current metrics."""
        optimization_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [],
            "performance_before": self.get_performance_summary(hours_back=1),
            "recommendations_applied": [],
        }

        recommendations = self.get_optimization_recommendations()

        for rec in recommendations:
            if rec["priority"] == "HIGH" and rec["type"] == "SYSTEM_RESOURCE":
                # Implement automatic memory cleanup
                gc_collected = self._trigger_garbage_collection()
                optimization_results["actions_taken"].append(
                    {
                        "action": "garbage_collection",
                        "objects_collected": gc_collected,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            elif rec["type"] == "RESPONSE_TIME":
                # Implement automatic cache warming for frequent operations
                cache_warmed = self._warm_performance_cache()
                optimization_results["actions_taken"].append(
                    {
                        "action": "cache_warming",
                        "entries_warmed": cache_warmed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            optimization_results["recommendations_applied"].append(rec)

        # Log optimization actions
        if optimization_results["actions_taken"]:
            logger.info(f"Auto-optimization triggered: {len(optimization_results['actions_taken'])} actions taken")

        return optimization_results

    def _trigger_garbage_collection(self) -> int:
        """Trigger garbage collection and return objects collected."""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        return collected

    def _warm_performance_cache(self) -> int:
        """Warm performance cache with recent operation patterns."""
        # This is a placeholder for cache warming logic
        # In a real implementation, this would pre-load frequently accessed data
        warmed_entries = 0

        # Example: Pre-warm most common operations from recent history
        recent_operations = [m.operation for m in list(self.metrics_history)[-100:]]
        operation_counts = {}

        for op in recent_operations:
            operation_counts[op] = operation_counts.get(op, 0) + 1

        # Sort by frequency and "warm" top operations
        top_operations = sorted(operation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        warmed_entries = len(top_operations)

        logger.info(f"Performance cache warmed with {warmed_entries} operation patterns")
        return warmed_entries

    def enable_auto_optimization(self, performance_threshold: float = 0.1):
        """Enable automatic performance optimization when degradation exceeds threshold."""
        self.auto_optimization_enabled = True
        self.auto_optimization_threshold = performance_threshold
        logger.info(f"Auto-optimization enabled with {performance_threshold*100}% degradation threshold")

    def disable_auto_optimization(self):
        """Disable automatic performance optimization."""
        self.auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot for compatibility."""
        recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []

        if not recent_metrics:
            return {
                "request_count": 0,
                "average_response_time": 0.0,
                "throughput_per_second": 0.0,
                "current_memory_mb": 0.0,
                "peak_memory_mb": 0.0,
                "cpu_usage_avg": 0.0,
                "error_rate": 0.0,
                "uptime_seconds": 0.0,
                "concurrent_requests": 0,
            }

        # Calculate metrics from recent data
        response_times = [m.response_time_ms for m in recent_metrics]
        successful_operations = sum(1 for m in recent_metrics if m.success)

        return {
            "request_count": len(recent_metrics),
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "throughput_per_second": len(recent_metrics) / 3600.0,  # Approximate throughput
            "current_memory_mb": recent_metrics[-1].memory_usage.get("used_percent", 0) * 10 if recent_metrics else 0,
            "peak_memory_mb": (
                max(m.memory_usage.get("used_percent", 0) * 10 for m in recent_metrics) if recent_metrics else 0
            ),
            "cpu_usage_avg": (
                sum(m.system_load.get("cpu_percent", 0) for m in recent_metrics) / len(recent_metrics)
                if recent_metrics
                else 0
            ),
            "error_rate": 1.0 - (successful_operations / len(recent_metrics)) if recent_metrics else 0,
            "uptime_seconds": 3600.0,  # Approximate uptime
            "concurrent_requests": 0,
        }


@contextlib.contextmanager
def monitor_performance(
    monitor: ClinicalPerformanceMonitor,
    operation: str,
    clinical_priority: ClinicalPriority = ClinicalPriority.ROUTINE,
    user_id: Optional[str] = None,
    patient_id: Optional[str] = None,
):
    """
    Context manager for monitoring operation performance.

    Usage:
        with monitor_performance(monitor, 'ai_diagnosis', ClinicalPriority.CRITICAL):
            result = perform_ai_diagnosis(patient_data)
    """
    start_time = time.time()
    success = True
    error_message = None

    try:
        yield
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        monitor.record_operation(
            operation=operation,
            response_time_ms=response_time_ms,
            clinical_priority=clinical_priority,
            success=success,
            error_message=error_message,
            user_id=user_id,
            patient_id=patient_id,
        )


def performance_monitor_decorator(
    monitor: ClinicalPerformanceMonitor,
    operation_name: Optional[str] = None,
    clinical_priority: ClinicalPriority = ClinicalPriority.ROUTINE,
):
    """
    Decorator for monitoring function performance.

    Usage:
        @performance_monitor_decorator(monitor, 'ai_prediction', ClinicalPriority.URGENT)
        def make_prediction(data):
            return model.predict(data)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation = operation_name or func.__name__

            with monitor_performance(monitor, operation, clinical_priority):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global monitor instance
_global_performance_monitor = None


def get_performance_monitor() -> ClinicalPerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = ClinicalPerformanceMonitor()
        _global_performance_monitor.start_monitoring()
    return _global_performance_monitor


def reset_performance_monitor() -> None:
    """Reset global performance monitor instance (useful for testing)."""
    global _global_performance_monitor
    if _global_performance_monitor is not None:
        _global_performance_monitor.stop_monitoring()
        _global_performance_monitor = None


class PerformanceAnalyzer:
    """
    Advanced performance analysis and trend detection.

    Provides statistical analysis of performance metrics over time to identify
    trends, patterns, and anomalies before they become critical issues.
    """

    def __init__(self, monitor: ClinicalPerformanceMonitor):
        """
        Initialize performance analyzer.

        Args:
            monitor: Performance monitor instance to analyze
        """
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def analyze_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Analyze performance trends over specified time period.

        Args:
            hours_back: Number of hours to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        with self.monitor._lock:
            filtered_metrics = [m for m in self.monitor.metrics_history if m.timestamp >= cutoff_time]

        if len(filtered_metrics) < 10:
            return {"error": "Insufficient data for trend analysis", "metrics_count": len(filtered_metrics)}

        # Group metrics by hour
        hourly_stats = defaultdict(lambda: {"response_times": [], "violations": 0, "total": 0})

        for metric in filtered_metrics:
            hour_key = metric.timestamp.strftime("%Y-%m-%d_%H")
            hourly_stats[hour_key]["response_times"].append(metric.response_time_ms)
            hourly_stats[hour_key]["total"] += 1

            threshold = self.monitor._get_threshold_for_priority(metric.clinical_priority)
            if metric.response_time_ms > threshold:
                hourly_stats[hour_key]["violations"] += 1

        # Calculate trends
        hours = sorted(hourly_stats.keys())
        avg_response_times = []
        violation_rates = []

        for hour in hours:
            stats = hourly_stats[hour]
            avg_response_times.append(statistics.mean(stats["response_times"]))
            violation_rates.append(stats["violations"] / stats["total"] if stats["total"] > 0 else 0)

        # Detect trends
        trends = {
            "period_hours": hours_back,
            "data_points": len(hours),
            "response_time_trend": self._detect_trend(avg_response_times),
            "violation_rate_trend": self._detect_trend(violation_rates),
            "avg_response_time": statistics.mean(avg_response_times) if avg_response_times else 0,
            "avg_violation_rate": statistics.mean(violation_rates) if violation_rates else 0,
            "peak_hour": hours[avg_response_times.index(max(avg_response_times))] if avg_response_times else None,
            "best_hour": hours[avg_response_times.index(min(avg_response_times))] if avg_response_times else None,
        }

        return trends

    def _detect_trend(self, values: List[float]) -> str:
        """
        Detect trend direction in time series data.

        Args:
            values: List of numeric values over time

        Returns:
            Trend classification: 'increasing', 'decreasing', or 'stable'
        """
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear regression to detect trend
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Classify based on slope
        if abs(slope) < 0.01:  # Threshold for stability
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def predict_violation_risk(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """
        Predict risk of performance violations in the near future.

        Args:
            hours_ahead: Number of hours to predict ahead

        Returns:
            Risk assessment with confidence level
        """
        trends = self.analyze_trends(hours_back=24)

        if "error" in trends:
            return {"error": "Cannot predict without sufficient historical data"}

        # Simple risk assessment based on trends
        risk_level = "low"
        confidence = 0.5

        if trends["violation_rate_trend"] == "increasing":
            if trends["avg_violation_rate"] > 0.05:
                risk_level = "high"
                confidence = 0.8
            elif trends["avg_violation_rate"] > 0.02:
                risk_level = "medium"
                confidence = 0.7
            else:
                risk_level = "medium"
                confidence = 0.6

        if trends["response_time_trend"] == "increasing":
            if risk_level == "high":
                confidence = min(0.9, confidence + 0.1)
            else:
                risk_level = "medium" if risk_level == "low" else "high"
                confidence = max(0.7, confidence)

        return {
            "hours_ahead": hours_ahead,
            "risk_level": risk_level,
            "confidence": confidence,
            "based_on_trends": trends,
            "recommendations": self._generate_risk_recommendations(risk_level, trends),
        }

    def _generate_risk_recommendations(self, risk_level: str, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk level."""
        recommendations = []

        if risk_level == "high":
            recommendations.append("Enable auto-optimization immediately")
            recommendations.append("Review resource allocation and consider scaling")
            recommendations.append("Investigate recent code changes that may impact performance")

        if risk_level == "medium":
            recommendations.append("Monitor closely for the next few hours")
            recommendations.append("Prepare scaling resources if trend continues")

        if trends.get("response_time_trend") == "increasing":
            recommendations.append("Review and optimize slow operations")
            recommendations.append("Consider implementing caching for frequent queries")

        return recommendations
