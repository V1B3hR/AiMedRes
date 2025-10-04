"""
Security monitoring and alerting system.

Provides real-time security monitoring for:
- Intrusion detection
- Anomaly detection in API usage
- Security event logging
- Alert generation and notification
- Performance impact monitoring
"""

import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Any, List, Callable
import statistics
import logging
import json

security_logger = logging.getLogger("duetmind.security")


class SecurityMonitor:
    """
    Real-time security monitoring and alerting system.

    Features:
    - Intrusion detection and prevention
    - Anomaly detection in API usage patterns
    - Real-time security event monitoring
    - Automated alert generation
    - Security metrics collection
    - Threat intelligence integration
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_enabled = config.get("security_monitoring_enabled", True)

        # Event storage
        self.security_events = deque(maxlen=10000)
        self.api_usage_patterns = defaultdict(lambda: {"requests": deque(maxlen=1000), "patterns": {}})
        self.threat_indicators = {}

        # Monitoring threads
        self.monitor_thread = None
        self.running = False

        # Alert thresholds
        self.alert_thresholds = {
            "failed_auth_rate": config.get("failed_auth_rate_threshold", 10),  # per minute
            "unusual_access_pattern": config.get("unusual_access_threshold", 5),  # std deviations
            "high_request_rate": config.get("high_request_rate_threshold", 1000),  # per minute
            "error_rate": config.get("error_rate_threshold", 0.1),  # 10%
            "response_time_anomaly": config.get("response_time_threshold", 3.0),  # std deviations
        }

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        security_logger.info("Security monitor initialized")

    def start_monitoring(self):
        """Start background security monitoring."""
        if not self.monitoring_enabled:
            security_logger.info("Security monitoring is disabled")
            return

        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            security_logger.info("Security monitoring started")

    def stop_monitoring(self):
        """Stop security monitoring."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        security_logger.info("Security monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._analyze_security_events()
                self._detect_anomalies()
                self._check_threat_indicators()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                security_logger.error(f"Error in security monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "info",
        user_id: str = None,
        ip_address: str = None,
    ):
        """
        Log a security event for monitoring and analysis.

        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity (info, warning, critical)
            user_id: Associated user ID
            ip_address: Source IP address
        """
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "severity": severity,
            "details": details,
            "user_id": user_id,
            "ip_address": ip_address,
        }

        self.security_events.append(event)

        # Log to security logger
        log_message = f"Security Event [{severity.upper()}]: {event_type}"
        if user_id:
            log_message += f" | User: {user_id}"
        if ip_address:
            log_message += f" | IP: {ip_address}"

        if severity == "critical":
            security_logger.critical(log_message)
        elif severity == "warning":
            security_logger.warning(log_message)
        else:
            security_logger.info(log_message)

        # Immediate alert for critical events
        if severity == "critical":
            self._trigger_alert(event)

    def log_api_request(
        self,
        user_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        ip_address: str = None,
        payload_size: int = 0,
    ):
        """
        Log API request for pattern analysis.

        Args:
            user_id: User making the request
            endpoint: API endpoint accessed
            method: HTTP method
            status_code: Response status code
            response_time: Request processing time
            ip_address: Source IP address
            payload_size: Request payload size
        """
        request_data = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "ip_address": ip_address,
            "payload_size": payload_size,
        }

        # Store request pattern
        pattern_key = f"{user_id}_{ip_address}" if ip_address else user_id
        self.api_usage_patterns[pattern_key]["requests"].append(request_data)

        # Update user patterns
        patterns = self.api_usage_patterns[pattern_key]["patterns"]
        patterns[endpoint] = patterns.get(endpoint, 0) + 1

        # Check for immediate security concerns
        if status_code == 401:
            self.log_security_event(
                "failed_authentication",
                {"endpoint": endpoint, "method": method, "user_id": user_id},
                severity="warning",
                user_id=user_id,
                ip_address=ip_address,
            )
        elif status_code >= 500:
            self.log_security_event(
                "server_error",
                {"endpoint": endpoint, "status_code": status_code},
                severity="warning",
                user_id=user_id,
                ip_address=ip_address,
            )

    def _analyze_security_events(self):
        """Analyze recent security events for patterns."""
        if not self.security_events:
            return

        # Analyze events from last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_events = [e for e in self.security_events if e["timestamp"] > cutoff_time]

        if not recent_events:
            return

        # Check for failed authentication patterns
        failed_auths = [e for e in recent_events if e["event_type"] == "failed_authentication"]
        if len(failed_auths) > self.alert_thresholds["failed_auth_rate"]:
            self._trigger_alert(
                {
                    "type": "high_failed_auth_rate",
                    "count": len(failed_auths),
                    "threshold": self.alert_thresholds["failed_auth_rate"],
                    "severity": "warning",
                }
            )

        # Check for critical events
        critical_events = [e for e in recent_events if e["severity"] == "critical"]
        if critical_events:
            self._trigger_alert(
                {
                    "type": "critical_security_events",
                    "count": len(critical_events),
                    "events": [e["event_type"] for e in critical_events],
                    "severity": "critical",
                }
            )

    def _detect_anomalies(self):
        """Detect anomalies in API usage patterns."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Last hour

        for user_pattern_key, data in self.api_usage_patterns.items():
            requests = data["requests"]
            if not requests:
                continue

            # Filter recent requests
            recent_requests = [r for r in requests if r["timestamp"] > cutoff_time]
            if len(recent_requests) < 10:  # Need minimum data for analysis
                continue

            # Analyze request rate
            request_rate = len(recent_requests) / 60  # requests per minute
            if request_rate > self.alert_thresholds["high_request_rate"]:
                self.log_security_event(
                    "high_request_rate", {"user_pattern": user_pattern_key, "rate": request_rate}, severity="warning"
                )

            # Analyze response times
            response_times = [r["response_time"] for r in recent_requests]
            if len(response_times) > 5:
                avg_response_time = statistics.mean(response_times)
                std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0

                # Check for unusual response times
                for rt in response_times[-10:]:  # Check last 10 requests
                    if std_response_time > 0:
                        z_score = abs(rt - avg_response_time) / std_response_time
                        if z_score > self.alert_thresholds["response_time_anomaly"]:
                            self.log_security_event(
                                "response_time_anomaly",
                                {"response_time": rt, "z_score": z_score, "user_pattern": user_pattern_key},
                                severity="warning",
                            )

            # Analyze error rates
            error_requests = [r for r in recent_requests if r["status_code"] >= 400]
            error_rate = len(error_requests) / len(recent_requests) if recent_requests else 0

            if error_rate > self.alert_thresholds["error_rate"]:
                self.log_security_event(
                    "high_error_rate", {"user_pattern": user_pattern_key, "error_rate": error_rate}, severity="warning"
                )

    def _check_threat_indicators(self):
        """Check for known threat indicators."""
        # This would integrate with threat intelligence feeds
        # For now, implement basic checks

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=10)

        # Check for suspicious IP patterns
        ip_requests = defaultdict(int)
        for pattern_key, data in self.api_usage_patterns.items():
            for request in data["requests"]:
                if request.get("ip_address") and request["timestamp"] > cutoff_time.timestamp():
                    ip_requests[request["ip_address"]] += 1

        # Flag IPs with unusual activity
        for ip, count in ip_requests.items():
            if count > 100:  # More than 100 requests in 10 minutes
                self.log_security_event(
                    "suspicious_ip_activity",
                    {"ip_address": ip, "request_count": count},
                    severity="warning",
                    ip_address=ip,
                )

    def _trigger_alert(self, alert_data: Dict[str, Any]):
        """Trigger security alert."""
        alert = {"timestamp": datetime.now(), "alert_data": alert_data, "alert_id": f"alert_{int(time.time())}"}

        security_logger.warning(f"SECURITY ALERT: {json.dumps(alert_data, default=str)}")

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                security_logger.error(f"Error in alert callback: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for security alerts."""
        self.alert_callbacks.append(callback)

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        current_time = datetime.now()

        # Events from last 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        recent_events = [e for e in self.security_events if e["timestamp"] > cutoff_time]

        # Categorize events
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for event in recent_events:
            event_counts[event["event_type"]] += 1
            severity_counts[event["severity"]] += 1

        # API usage statistics
        total_requests = sum(len(data["requests"]) for data in self.api_usage_patterns.values())
        active_users = len(self.api_usage_patterns)

        return {
            "monitoring_status": "active" if self.running else "inactive",
            "summary_period": "24 hours",
            "total_security_events": len(recent_events),
            "events_by_type": dict(event_counts),
            "events_by_severity": dict(severity_counts),
            "api_usage": {"total_requests": total_requests, "active_users": active_users},
            "threat_indicators": len(self.threat_indicators),
            "generated_at": current_time.isoformat(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the security monitoring system.

        Returns:
            Health status including monitoring state and recent activity
        """
        return {
            "healthy": self.running,
            "monitoring_enabled": self.monitoring_enabled,
            "event_queue_size": len(self.security_events),
            "active_patterns": len(self.api_usage_patterns),
            "last_check": datetime.now().isoformat(),
            "status": "operational" if self.running else "stopped",
        }

    def clear_old_data(self, days: int = 30) -> Dict[str, int]:
        """
        Clear old security events and API usage patterns.

        Args:
            days: Number of days to retain

        Returns:
            Count of items cleared
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        events_before = len(self.security_events)

        # Filter events
        self.security_events = deque(
            [e for e in self.security_events if e["timestamp"] > cutoff_time], maxlen=self.security_events.maxlen
        )

        # Clear old API patterns
        patterns_before = len(self.api_usage_patterns)
        for pattern_key, data in list(self.api_usage_patterns.items()):
            # Keep only recent requests
            recent_requests = [r for r in data["requests"] if datetime.fromtimestamp(r["timestamp"]) > cutoff_time]
            if recent_requests:
                self.api_usage_patterns[pattern_key]["requests"] = deque(recent_requests, maxlen=1000)
            else:
                del self.api_usage_patterns[pattern_key]

        return {
            "events_cleared": events_before - len(self.security_events),
            "patterns_cleared": patterns_before - len(self.api_usage_patterns),
        }


class AlertDeduplicator:
    """
    Alert deduplication system to prevent alert fatigue.

    Tracks recent alerts and prevents duplicate alerts within a time window.
    """

    def __init__(self, dedup_window_seconds: int = 300):
        """
        Initialize alert deduplicator.

        Args:
            dedup_window_seconds: Time window for deduplication (default 5 minutes)
        """
        self.dedup_window_seconds = dedup_window_seconds
        self.recent_alerts = {}  # alert_key -> timestamp
        self._lock = threading.Lock()

    def should_send_alert(self, alert_type: str, alert_details: Dict[str, Any]) -> bool:
        """
        Check if alert should be sent or deduplicated.

        Args:
            alert_type: Type of alert
            alert_details: Alert details for generating unique key

        Returns:
            True if alert should be sent, False if deduplicated
        """
        alert_key = self._generate_alert_key(alert_type, alert_details)
        current_time = time.time()

        with self._lock:
            # Check if we've seen this alert recently
            if alert_key in self.recent_alerts:
                last_sent = self.recent_alerts[alert_key]
                if current_time - last_sent < self.dedup_window_seconds:
                    return False

            # Record this alert
            self.recent_alerts[alert_key] = current_time

            # Cleanup old entries
            self._cleanup_old_alerts(current_time)

        return True

    def _generate_alert_key(self, alert_type: str, alert_details: Dict[str, Any]) -> str:
        """Generate unique key for alert deduplication."""
        # Create a key based on alert type and key details
        key_parts = [alert_type]

        # Add relevant details to make key unique but not too specific
        if "user_id" in alert_details:
            key_parts.append(str(alert_details["user_id"]))
        if "ip_address" in alert_details:
            key_parts.append(str(alert_details["ip_address"]))

        return ":".join(key_parts)

    def _cleanup_old_alerts(self, current_time: float) -> None:
        """Remove old alert records outside deduplication window."""
        cutoff_time = current_time - (self.dedup_window_seconds * 2)
        keys_to_remove = [key for key, timestamp in self.recent_alerts.items() if timestamp < cutoff_time]

        for key in keys_to_remove:
            del self.recent_alerts[key]


class SecurityMetricsAggregator:
    """
    Aggregate and analyze security metrics over time.

    Provides statistical analysis and reporting of security events.
    """

    def __init__(self, monitor: SecurityMonitor):
        """
        Initialize metrics aggregator.

        Args:
            monitor: SecurityMonitor instance to aggregate metrics from
        """
        self.monitor = monitor

    def get_hourly_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security metrics aggregated by hour.

        Args:
            hours: Number of hours to analyze

        Returns:
            Hourly aggregated metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.monitor.security_events if e["timestamp"] > cutoff_time]

        hourly_data = defaultdict(lambda: {"events": 0, "critical": 0, "warnings": 0, "info": 0})

        for event in recent_events:
            hour_key = event["timestamp"].strftime("%Y-%m-%d_%H")
            hourly_data[hour_key]["events"] += 1
            hourly_data[hour_key][event["severity"]] += 1

        return {"period_hours": hours, "hourly_breakdown": dict(hourly_data), "total_events": len(recent_events)}

    def get_top_event_types(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequent security event types.

        Args:
            limit: Maximum number of event types to return

        Returns:
            List of event types with counts
        """
        event_counts = defaultdict(int)

        for event in self.monitor.security_events:
            event_counts[event["event_type"]] += 1

        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [{"event_type": event_type, "count": count} for event_type, count in sorted_events]

    def calculate_risk_score(self) -> Dict[str, Any]:
        """
        Calculate overall security risk score based on recent events.

        Returns:
            Risk score and contributing factors
        """
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_events = [e for e in self.monitor.security_events if e["timestamp"] > cutoff_time]

        if not recent_events:
            return {"risk_score": 0, "risk_level": "low", "factors": []}

        # Calculate risk based on severity and frequency
        critical_count = sum(1 for e in recent_events if e["severity"] == "critical")
        warning_count = sum(1 for e in recent_events if e["severity"] == "warning")

        risk_score = (critical_count * 10 + warning_count * 3) / max(len(recent_events), 1)

        # Determine risk level
        if risk_score > 8:
            risk_level = "critical"
        elif risk_score > 5:
            risk_level = "high"
        elif risk_score > 2:
            risk_level = "medium"
        else:
            risk_level = "low"

        factors = []
        if critical_count > 0:
            factors.append(f"{critical_count} critical events in last 24h")
        if warning_count > 10:
            factors.append(f"{warning_count} warnings in last 24h")

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": factors,
            "events_analyzed": len(recent_events),
        }
