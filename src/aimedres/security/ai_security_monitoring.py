"""
AI-Powered Security Monitoring Enhancement (Phase 3).

Implements autonomous threat detection, ML-powered security analytics,
threat intelligence integration, and automated incident response workflows.

Phase 3 Security Roadmap items:
- Autonomous security threat detection
- ML-powered security analytics
- Advanced threat intelligence feed integration
- Automated incident response workflows
"""

import hashlib
import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("aimedres.security.ai_monitoring")


# ---------------------------------------------------------------------------
# Anomaly detection (lightweight, no external ML dependency)
# ---------------------------------------------------------------------------

class RollingStatistics:
    """Welford online algorithm for computing mean and variance incrementally."""

    def __init__(self, window: int = 200):
        self._window = window
        self._buffer: deque = deque(maxlen=window)
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, value: float) -> None:
        if len(self._buffer) == self._window:
            removed = self._buffer[0]
            # Remove oldest value using reverse Welford
            if self._n > 1:
                delta = removed - self._mean
                self._mean -= delta / (self._n - 1)
                self._M2 -= delta * (removed - self._mean)
                self._n -= 1
        self._buffer.append(value)
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._M2 / (self._n - 1) if self._n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def z_score(self, value: float) -> float:
        sd = self.stddev
        return abs(value - self._mean) / sd if sd > 0 else 0.0


class IsolationForest:
    """
    Lightweight isolation forest for anomaly scoring without scikit-learn.

    Uses random feature projections to estimate anomaly scores.  Not a full
    production implementation – intended for early-signal detection that feeds
    into the alert pipeline.
    """

    def __init__(self, n_trees: int = 50, subsample: int = 64, seed: int = 42):
        self._n_trees = n_trees
        self._subsample = subsample
        self._seed = seed
        self._trees: List[Dict] = []
        self._fitted = False
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    def _pseudo_random(self, seed: int) -> float:
        """Deterministic pseudo-random in [0, 1) via SHA-256."""
        h = hashlib.sha256(str(seed).encode()).digest()
        return int.from_bytes(h[:4], "big") / (2 ** 32)

    def _build_tree(self, data: List[List[float]], height_limit: int, seed: int) -> Dict:
        """Recursively build an isolation tree."""
        n = len(data)
        if n <= 1 or height_limit == 0:
            return {"type": "leaf", "size": n}

        n_features = len(data[0])
        feat_idx = int(self._pseudo_random(seed) * n_features)
        col = [row[feat_idx] for row in data]
        min_val, max_val = min(col), max(col)

        if min_val == max_val:
            return {"type": "leaf", "size": n}

        split = min_val + self._pseudo_random(seed + 1) * (max_val - min_val)
        left = [row for row in data if row[feat_idx] < split]
        right = [row for row in data if row[feat_idx] >= split]

        return {
            "type": "internal",
            "feat_idx": feat_idx,
            "split": split,
            "left": self._build_tree(left, height_limit - 1, seed + 2),
            "right": self._build_tree(right, height_limit - 1, seed + 3),
        }

    def _path_length(self, node: Dict, sample: List[float], depth: int) -> float:
        """Return the path length for a sample through a tree."""
        if node["type"] == "leaf":
            n = node["size"]
            # Average path length adjustment for leaf node
            if n <= 1:
                return depth
            h = 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
            return depth + h
        if sample[node["feat_idx"]] < node["split"]:
            return self._path_length(node["left"], sample, depth + 1)
        return self._path_length(node["right"], sample, depth + 1)

    def fit(self, samples: List[List[float]], feature_names: Optional[List[str]] = None) -> None:
        """Fit the isolation forest on a list of feature vectors."""
        if not samples:
            return
        n = min(self._subsample, len(samples))
        height_limit = math.ceil(math.log2(max(n, 2)))
        self._trees = []
        self._feature_names = feature_names or [f"f{i}" for i in range(len(samples[0]))]
        seed = self._seed
        for _ in range(self._n_trees):
            # Subsample
            indices = []
            for j in range(n):
                idx = int(self._pseudo_random(seed + j) * len(samples))
                indices.append(idx)
            seed += n + 1
            subset = [samples[i] for i in indices]
            self._trees.append(self._build_tree(subset, height_limit, seed))
            seed += 1
        self._fitted = True

    def anomaly_score(self, sample: List[float]) -> float:
        """Return anomaly score in [0, 1].  Values >0.6 indicate anomalies."""
        if not self._fitted or not self._trees:
            return 0.0
        avg_path = sum(self._path_length(t, sample, 0) for t in self._trees) / len(self._trees)
        n = self._subsample
        if n <= 1:
            return 0.5
        c_n = 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
        score = 2.0 ** (-avg_path / c_n)
        return float(score)


# ---------------------------------------------------------------------------
# Threat intelligence
# ---------------------------------------------------------------------------

class ThreatIntelligenceFeed:
    """
    Manages threat intelligence indicators and reputation lookups.

    In production this would integrate with external OSINT / CTI feeds
    (e.g. AbuseIPDB, OTX, MISP).  Here we maintain an in-process store
    and expose a pluggable ingest interface.
    """

    def __init__(self):
        self._ip_blocklist: Dict[str, Dict] = {}        # ip -> indicator record
        self._hash_blocklist: Dict[str, Dict] = {}      # sha256 -> indicator record
        self._domain_blocklist: Dict[str, Dict] = {}    # domain -> indicator record
        self._lock = threading.RLock()
        self._feed_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    def ingest_indicator(
        self,
        indicator_type: str,
        value: str,
        confidence: float,
        source: str,
        tags: Optional[List[str]] = None,
        ttl_hours: int = 24,
    ) -> None:
        """
        Ingest a new threat indicator.

        Args:
            indicator_type: One of 'ip', 'hash', 'domain'
            value: Indicator value
            confidence: Confidence score in [0, 1]
            source: Feed name or provider
            tags: Optional classification tags (e.g. ['malware', 'c2'])
            ttl_hours: How long the indicator remains valid
        """
        record = {
            "value": value,
            "type": indicator_type,
            "confidence": confidence,
            "source": source,
            "tags": tags or [],
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat(),
        }
        with self._lock:
            store = {
                "ip": self._ip_blocklist,
                "hash": self._hash_blocklist,
                "domain": self._domain_blocklist,
            }.get(indicator_type)
            if store is not None:
                store[value] = record
        for cb in self._feed_callbacks:
            try:
                cb("indicator_ingested", record)
            except Exception:
                pass

    def check_ip(self, ip: str) -> Optional[Dict]:
        """Return threat record for IP or None if not known malicious."""
        with self._lock:
            record = self._ip_blocklist.get(ip)
        if record and self._is_valid(record):
            return record
        return None

    def check_hash(self, sha256: str) -> Optional[Dict]:
        """Return threat record for file hash or None."""
        with self._lock:
            record = self._hash_blocklist.get(sha256)
        if record and self._is_valid(record):
            return record
        return None

    def check_domain(self, domain: str) -> Optional[Dict]:
        """Return threat record for domain or None."""
        with self._lock:
            record = self._domain_blocklist.get(domain)
        if record and self._is_valid(record):
            return record
        return None

    def register_update_callback(self, callback: Callable) -> None:
        """Register a callback invoked whenever new indicators are ingested."""
        self._feed_callbacks.append(callback)

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "ip_indicators": len(self._ip_blocklist),
                "hash_indicators": len(self._hash_blocklist),
                "domain_indicators": len(self._domain_blocklist),
            }

    @staticmethod
    def _is_valid(record: Dict) -> bool:
        try:
            expires = datetime.fromisoformat(record["expires_at"])
            return datetime.now(timezone.utc) < expires
        except (KeyError, ValueError):
            return True


# ---------------------------------------------------------------------------
# Incident response automation
# ---------------------------------------------------------------------------

class IncidentResponseEngine:
    """
    Automates incident response workflows when threats are detected.

    Supports pluggable response actions (block IP, revoke session, notify SOC,
    quarantine resource).  Each action is executed asynchronously and its
    result is recorded in the incident log.
    """

    SEVERITY_LEVELS = ("low", "medium", "high", "critical")

    def __init__(self):
        self._incidents: deque = deque(maxlen=5000)
        self._response_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._incident_counter = 0

    # ------------------------------------------------------------------
    def register_handler(self, threat_type: str, handler: Callable) -> None:
        """
        Register an async handler function for a specific threat type.

        The handler receives ``(incident: Dict) -> Dict`` and should return
        a result dict with at least ``{"success": bool, "message": str}``.
        """
        self._response_handlers[threat_type].append(handler)

    def create_incident(
        self,
        threat_type: str,
        severity: str,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        auto_respond: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new security incident and optionally trigger response.

        Args:
            threat_type: Classification of the threat
            severity: One of 'low', 'medium', 'high', 'critical'
            details: Contextual details
            source_ip: Originating IP address
            user_id: Associated user identifier
            auto_respond: Whether to automatically trigger response actions

        Returns:
            Incident record including assigned ID and response status
        """
        if severity not in self.SEVERITY_LEVELS:
            severity = "medium"

        with self._lock:
            self._incident_counter += 1
            incident_id = f"INC-{self._incident_counter:06d}"

        incident = {
            "incident_id": incident_id,
            "threat_type": threat_type,
            "severity": severity,
            "details": details,
            "source_ip": source_ip,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "response_actions": [],
            "timeline": [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "incident_created",
                    "actor": "system",
                }
            ],
        }

        self._incidents.append(incident)
        logger.warning(
            "Incident created: %s | type=%s severity=%s", incident_id, threat_type, severity
        )

        if auto_respond:
            self._execute_response(incident)

        return incident

    def _execute_response(self, incident: Dict) -> None:
        """Execute registered response handlers in a background thread."""
        handlers = list(self._response_handlers.get(incident["threat_type"], []))
        # Also run handlers registered for the catch-all key '*'
        handlers += list(self._response_handlers.get("*", []))

        if not handlers:
            return

        def _run():
            for handler in handlers:
                try:
                    result = handler(incident)
                    action_record = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "handler": getattr(handler, "__name__", str(handler)),
                        "result": result,
                    }
                    incident["response_actions"].append(action_record)
                    incident["timeline"].append(
                        {
                            "ts": action_record["ts"],
                            "event": "response_executed",
                            "handler": action_record["handler"],
                            "success": result.get("success", False),
                        }
                    )
                except Exception as exc:
                    logger.error("Response handler %s failed: %s", handler, exc)

            incident["status"] = "responded"
            incident["timeline"].append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "response_complete",
                    "actor": "system",
                }
            )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def get_open_incidents(self, severity: Optional[str] = None) -> List[Dict]:
        """Return open incidents, optionally filtered by severity."""
        with self._lock:
            incidents = list(self._incidents)
        result = [i for i in incidents if i["status"] == "open"]
        if severity:
            result = [i for i in result if i["severity"] == severity]
        return result

    def close_incident(self, incident_id: str, resolution: str, resolved_by: str) -> bool:
        """Mark an incident as resolved."""
        with self._lock:
            for incident in self._incidents:
                if incident["incident_id"] == incident_id:
                    incident["status"] = "closed"
                    incident["resolution"] = resolution
                    incident["resolved_by"] = resolved_by
                    incident["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    incident["timeline"].append(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "event": "incident_closed",
                            "resolution": resolution,
                            "resolved_by": resolved_by,
                        }
                    )
                    return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            incidents = list(self._incidents)
        total = len(incidents)
        by_status: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        for inc in incidents:
            by_status[inc["status"]] += 1
            by_severity[inc["severity"]] += 1
            by_type[inc["threat_type"]] += 1
        return {
            "total_incidents": total,
            "by_status": dict(by_status),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
        }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class AISecurityMonitor:
    """
    Orchestrates AI-powered security monitoring across all subsystems.

    Integrates:
    - ML anomaly detection (IsolationForest + RollingStatistics)
    - Threat intelligence feed lookups
    - Automated incident response
    - Continuous model retraining on fresh behavioural data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._anomaly_threshold: float = cfg.get("anomaly_threshold", 0.65)
        self._retrain_interval: int = cfg.get("retrain_interval_seconds", 3600)
        self._min_samples_for_fit: int = cfg.get("min_samples_for_fit", 30)

        self.threat_intel = ThreatIntelligenceFeed()
        self.incident_engine = IncidentResponseEngine()

        # Per-dimension rolling statistics for fast z-score anomaly detection
        self._rolling_stats: Dict[str, RollingStatistics] = defaultdict(
            lambda: RollingStatistics(window=300)
        )

        # Isolation forest trained on feature vectors
        self._iso_forest = IsolationForest(n_trees=50, subsample=128)
        self._training_buffer: deque = deque(maxlen=5000)

        self._running = False
        self._bg_thread: Optional[threading.Thread] = None
        self._last_retrain = 0.0

        # Prometheus-compatible counters (simple dicts for offline use)
        self.metrics: Dict[str, int] = defaultdict(int)

        logger.info("AISecurityMonitor initialised (threshold=%.2f)", self._anomaly_threshold)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start background monitoring and periodic retraining."""
        if self._running:
            return
        self._running = True
        self._bg_thread = threading.Thread(target=self._run_loop, daemon=True, name="ai-sec-monitor")
        self._bg_thread.start()
        logger.info("AI security monitor started")

    def stop(self) -> None:
        self._running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
        logger.info("AI security monitor stopped")

    # ------------------------------------------------------------------
    def analyse_request(
        self,
        user_id: str,
        endpoint: str,
        ip_address: str,
        response_time_ms: float,
        status_code: int,
        payload_size: int = 0,
        user_agent: str = "",
    ) -> Dict[str, Any]:
        """
        Analyse a single API request for threat signals.

        Returns a dict with 'risk_score', 'anomaly_detected', 'threats', and
        optional 'incident_id' if an incident was raised.
        """
        threats: List[Dict] = []
        risk_score = 0.0

        # --- Threat intel checks ---
        ip_threat = self.threat_intel.check_ip(ip_address)
        if ip_threat:
            threats.append({"type": "malicious_ip", "detail": ip_threat})
            risk_score += 0.8

        # --- Statistical anomaly detection (per dimension) ---
        self._rolling_stats["response_time"].update(response_time_ms)
        rt_z = self._rolling_stats["response_time"].z_score(response_time_ms)
        if rt_z > 4.0:
            threats.append({"type": "response_time_spike", "z_score": rt_z})
            risk_score += min(rt_z / 10.0, 0.3)

        error_flag = 1.0 if status_code >= 400 else 0.0
        self._rolling_stats["error_rate"].update(error_flag)

        self._rolling_stats["payload_size"].update(float(payload_size))
        ps_z = self._rolling_stats["payload_size"].z_score(float(payload_size))
        if ps_z > 5.0:
            threats.append({"type": "unusual_payload_size", "z_score": ps_z})
            risk_score += min(ps_z / 15.0, 0.2)

        # --- Isolation forest anomaly detection ---
        feature_vector = [response_time_ms, float(status_code), float(payload_size), rt_z, ps_z]
        self._training_buffer.append(feature_vector)
        iso_score = 0.0
        if self._iso_forest._fitted:
            iso_score = self._iso_forest.anomaly_score(feature_vector)
            if iso_score > self._anomaly_threshold:
                threats.append({"type": "ml_anomaly", "iso_score": iso_score})
                risk_score += iso_score * 0.5

        risk_score = min(risk_score, 1.0)
        anomaly_detected = bool(threats) or risk_score > 0.5

        self.metrics["requests_analysed"] += 1
        if anomaly_detected:
            self.metrics["anomalies_detected"] += 1

        incident_id = None
        if risk_score >= 0.7:
            severity = "critical" if risk_score >= 0.9 else "high"
            inc = self.incident_engine.create_incident(
                threat_type=threats[0]["type"] if threats else "high_risk_request",
                severity=severity,
                details={
                    "user_id": user_id,
                    "endpoint": endpoint,
                    "ip_address": ip_address,
                    "risk_score": risk_score,
                    "threats": threats,
                    "iso_score": iso_score,
                },
                source_ip=ip_address,
                user_id=user_id,
            )
            incident_id = inc["incident_id"]
            self.metrics["incidents_raised"] += 1

        return {
            "user_id": user_id,
            "endpoint": endpoint,
            "ip_address": ip_address,
            "risk_score": risk_score,
            "anomaly_detected": anomaly_detected,
            "threats": threats,
            "iso_score": iso_score,
            "incident_id": incident_id,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    def analyse_auth_event(
        self,
        user_id: str,
        ip_address: str,
        success: bool,
        event_type: str = "login",
    ) -> Optional[Dict]:
        """
        Analyse an authentication event for credential stuffing / brute-force.
        """
        key = f"auth_fail:{ip_address}"
        self._rolling_stats[key].update(0.0 if success else 1.0)
        fail_rate = self._rolling_stats[key].mean

        if not success and fail_rate > 0.5 and self._rolling_stats[key]._n > 5:
            inc = self.incident_engine.create_incident(
                threat_type="brute_force_auth",
                severity="high",
                details={
                    "user_id": user_id,
                    "ip_address": ip_address,
                    "failure_rate": fail_rate,
                    "event_type": event_type,
                },
                source_ip=ip_address,
                user_id=user_id,
            )
            self.metrics["auth_incidents"] += 1
            return inc
        return None

    # ------------------------------------------------------------------
    def get_security_posture(self) -> Dict[str, Any]:
        """Return a consolidated security posture report."""
        incident_summary = self.incident_engine.get_summary()
        open_critical = len(self.incident_engine.get_open_incidents("critical"))
        open_high = len(self.incident_engine.get_open_incidents("high"))

        if open_critical > 0:
            posture = "critical"
        elif open_high > 3:
            posture = "high_risk"
        elif incident_summary["by_status"].get("open", 0) > 10:
            posture = "elevated"
        else:
            posture = "nominal"

        return {
            "posture": posture,
            "open_incidents": incident_summary["by_status"].get("open", 0),
            "critical_incidents": open_critical,
            "high_incidents": open_high,
            "metrics": dict(self.metrics),
            "threat_intel": self.threat_intel.get_stats(),
            "model_fitted": self._iso_forest._fitted,
            "training_samples": len(self._training_buffer),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """Background loop: periodic retraining and TTL expiry."""
        while self._running:
            try:
                now = time.time()
                if (
                    now - self._last_retrain > self._retrain_interval
                    and len(self._training_buffer) >= self._min_samples_for_fit
                ):
                    self._retrain()
                    self._last_retrain = now
            except Exception as exc:
                logger.error("Background loop error: %s", exc)
            time.sleep(60)

    def _retrain(self) -> None:
        """Retrain the isolation forest on buffered request features."""
        samples = list(self._training_buffer)
        if len(samples) < self._min_samples_for_fit:
            return
        self._iso_forest.fit(
            samples,
            feature_names=["response_time_ms", "status_code", "payload_size", "rt_z", "ps_z"],
        )
        logger.info("IsolationForest retrained on %d samples", len(samples))
        self.metrics["model_retrains"] += 1
