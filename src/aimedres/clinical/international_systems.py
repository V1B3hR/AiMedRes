"""
International Healthcare Systems (P18)

Implements comprehensive international healthcare system capabilities with:
- Multilingual interface & terminology mapping
- Regional clinical guideline adaptation engine
- Low-bandwidth / constrained deployment modes
- Global data collaboration governance framework

This module provides advanced internationalization and regional adaptation
features to enable global deployment of medical AI systems.
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("aimedres.clinical.international_systems")


class Language(Enum):
    """Supported languages for multilingual interface."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    HINDI = "hi"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"


class Region(Enum):
    """Healthcare regions with different guidelines."""

    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"


class TerminologyStandard(Enum):
    """Medical terminology standards."""

    ICD10 = "icd10"
    ICD11 = "icd11"
    SNOMED_CT = "snomed_ct"
    LOINC = "loinc"
    RxNorm = "rxnorm"
    CPT = "cpt"


class DeploymentMode(Enum):
    """Deployment modes for different bandwidth scenarios."""

    FULL_CLOUD = "full_cloud"
    EDGE_COMPUTING = "edge_computing"
    LOW_BANDWIDTH = "low_bandwidth"
    OFFLINE = "offline"
    HYBRID = "hybrid"


class DataGovernanceLevel(Enum):
    """Data governance levels for global collaboration."""

    PUBLIC = "public"
    FEDERATED = "federated"
    RESTRICTED = "restricted"
    SOVEREIGN = "sovereign"


@dataclass
class TerminologyMapping:
    """Represents a terminology mapping between standards."""

    mapping_id: str
    source_standard: TerminologyStandard
    target_standard: TerminologyStandard
    source_code: str
    target_code: str
    source_term: str
    target_term: str
    mapping_confidence: float  # 0-1 scale
    bidirectional: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalGuideline:
    """Represents a regional clinical guideline."""

    guideline_id: str
    region: Region
    condition: str
    guideline_name: str
    version: str
    effective_date: datetime
    recommendations: List[Dict[str, Any]]
    evidence_level: str  # A, B, C, D
    adapted_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationEntry:
    """Represents a translation for UI elements."""

    entry_id: str
    key: str
    source_language: Language
    target_language: Language
    source_text: str
    translated_text: str
    context: str
    translation_quality: float  # 0-1 scale
    verified_by_medical_expert: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfiguration:
    """Configuration for constrained deployment scenarios."""

    config_id: str
    mode: DeploymentMode
    max_bandwidth_mbps: float
    max_latency_ms: int
    storage_limit_gb: float
    compression_enabled: bool
    model_size_mb: float
    feature_flags: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCollaborationPolicy:
    """Policy for global data collaboration."""

    policy_id: str
    region: Region
    governance_level: DataGovernanceLevel
    data_residency_required: bool
    allowed_sharing_regions: List[Region]
    encryption_required: bool
    anonymization_level: str
    audit_requirements: Dict[str, Any]
    compliance_frameworks: List[str]  # e.g., GDPR, HIPAA
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultilingualInterface:
    """
    Manages multilingual interface and terminology translation.
    """

    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.translations: Dict[str, Dict[Language, str]] = {}
        self.terminology_mappings: Dict[str, TerminologyMapping] = {}
        self.supported_languages: Set[Language] = set()
        self.translation_cache: Dict[str, str] = {}

        logger.info(
            f"Initialized MultilingualInterface with default language: {default_language.value}"
        )

    def add_translation(self, translation: TranslationEntry):
        """Add a translation entry."""
        key = translation.key
        if key not in self.translations:
            self.translations[key] = {}

        self.translations[key][translation.target_language] = translation.translated_text
        self.supported_languages.add(translation.target_language)

        # Cache key
        cache_key = f"{key}:{translation.target_language.value}"
        self.translation_cache[cache_key] = translation.translated_text

        logger.debug(f"Added translation: {key} -> {translation.target_language.value}")

    def translate(self, key: str, target_language: Language) -> str:
        """Translate a UI element to target language."""
        cache_key = f"{key}:{target_language.value}"

        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        if key in self.translations and target_language in self.translations[key]:
            return self.translations[key][target_language]

        # Fallback to default language
        if key in self.translations and self.default_language in self.translations[key]:
            logger.warning(
                f"Translation not found for {key} in {target_language.value}, using default"
            )
            return self.translations[key][self.default_language]

        return key  # Return key if no translation found

    def add_terminology_mapping(self, mapping: TerminologyMapping):
        """Add a terminology mapping between standards."""
        mapping_key = f"{mapping.source_standard.value}:{mapping.source_code}"
        self.terminology_mappings[mapping_key] = mapping
        logger.debug(f"Added terminology mapping: {mapping.source_code} -> {mapping.target_code}")

    def map_terminology(
        self,
        source_code: str,
        source_standard: TerminologyStandard,
        target_standard: TerminologyStandard,
    ) -> Optional[TerminologyMapping]:
        """Map terminology from source to target standard."""
        mapping_key = f"{source_standard.value}:{source_code}"

        if mapping_key in self.terminology_mappings:
            mapping = self.terminology_mappings[mapping_key]
            if mapping.target_standard == target_standard:
                return mapping

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            "supported_languages": len(self.supported_languages),
            "translation_keys": len(self.translations),
            "terminology_mappings": len(self.terminology_mappings),
            "cache_size": len(self.translation_cache),
        }


class RegionalGuidelineEngine:
    """
    Adapts clinical guidelines for different healthcare regions.
    """

    def __init__(self):
        self.guidelines: Dict[Region, List[ClinicalGuideline]] = defaultdict(list)
        self.condition_index: Dict[str, List[ClinicalGuideline]] = defaultdict(list)
        self.adaptation_rules: Dict[str, Dict[str, Any]] = {}

        logger.info("Initialized RegionalGuidelineEngine")

    def add_guideline(self, guideline: ClinicalGuideline):
        """Add a regional clinical guideline."""
        self.guidelines[guideline.region].append(guideline)
        self.condition_index[guideline.condition.lower()].append(guideline)
        logger.debug(f"Added guideline: {guideline.guideline_name} for {guideline.region.value}")

    def get_guidelines_for_region(
        self, region: Region, condition: Optional[str] = None
    ) -> List[ClinicalGuideline]:
        """Get guidelines for a specific region and optional condition."""
        region_guidelines = self.guidelines[region]

        if condition:
            region_guidelines = [
                g for g in region_guidelines if g.condition.lower() == condition.lower()
            ]

        return region_guidelines

    def adapt_guideline(
        self, source_guideline: ClinicalGuideline, target_region: Region
    ) -> ClinicalGuideline:
        """Adapt a guideline from one region to another."""
        start_time = time.time()

        # Create adapted guideline
        adapted = ClinicalGuideline(
            guideline_id=str(uuid.uuid4()),
            region=target_region,
            condition=source_guideline.condition,
            guideline_name=f"{source_guideline.guideline_name} (Adapted for {target_region.value})",
            version=f"{source_guideline.version}-adapted",
            effective_date=datetime.now(),
            recommendations=self._adapt_recommendations(
                source_guideline.recommendations, target_region
            ),
            evidence_level=source_guideline.evidence_level,
            adapted_from=source_guideline.guideline_id,
            metadata={
                "original_region": source_guideline.region.value,
                "adaptation_time": time.time() - start_time,
                "adaptation_date": datetime.now().isoformat(),
            },
        )

        logger.info(f"Adapted guideline {source_guideline.guideline_id} for {target_region.value}")
        return adapted

    def _adapt_recommendations(
        self, recommendations: List[Dict[str, Any]], target_region: Region
    ) -> List[Dict[str, Any]]:
        """Adapt recommendations for target region."""
        adapted = []

        for rec in recommendations:
            adapted_rec = rec.copy()
            # Apply region-specific adaptations
            if target_region in self.adaptation_rules:
                rules = self.adaptation_rules[target_region]
                # Apply rules to adapt dosages, protocols, etc.
                if "dosage_adjustment" in rules:
                    adapted_rec["adjusted"] = True

            adapted.append(adapted_rec)

        return adapted

    def compare_guidelines(self, guideline1_id: str, guideline2_id: str) -> Dict[str, Any]:
        """Compare two guidelines."""
        # Find guidelines
        g1 = None
        g2 = None

        for guidelines_list in self.guidelines.values():
            for g in guidelines_list:
                if g.guideline_id == guideline1_id:
                    g1 = g
                if g.guideline_id == guideline2_id:
                    g2 = g

        if not g1 or not g2:
            return {"error": "One or both guidelines not found"}

        return {
            "guideline1": {
                "id": g1.guideline_id,
                "region": g1.region.value,
                "name": g1.guideline_name,
                "recommendations_count": len(g1.recommendations),
            },
            "guideline2": {
                "id": g2.guideline_id,
                "region": g2.region.value,
                "name": g2.guideline_name,
                "recommendations_count": len(g2.recommendations),
            },
            "same_condition": g1.condition == g2.condition,
            "same_evidence_level": g1.evidence_level == g2.evidence_level,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_guidelines": sum(len(gl) for gl in self.guidelines.values()),
            "regions_covered": len(self.guidelines),
            "conditions_covered": len(self.condition_index),
            "adaptation_rules": len(self.adaptation_rules),
        }


class ConstrainedDeploymentManager:
    """
    Manages deployment in low-bandwidth and constrained environments.
    """

    def __init__(self):
        self.configurations: Dict[DeploymentMode, DeploymentConfiguration] = {}
        self.active_deployments: Dict[str, DeploymentConfiguration] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

        logger.info("Initialized ConstrainedDeploymentManager")
        self._initialize_default_configs()

    def _initialize_default_configs(self):
        """Initialize default deployment configurations."""
        # Full cloud configuration
        self.configurations[DeploymentMode.FULL_CLOUD] = DeploymentConfiguration(
            config_id=str(uuid.uuid4()),
            mode=DeploymentMode.FULL_CLOUD,
            max_bandwidth_mbps=1000.0,
            max_latency_ms=100,
            storage_limit_gb=1000.0,
            compression_enabled=False,
            model_size_mb=500.0,
            feature_flags={"all_features": True},
        )

        # Low bandwidth configuration
        self.configurations[DeploymentMode.LOW_BANDWIDTH] = DeploymentConfiguration(
            config_id=str(uuid.uuid4()),
            mode=DeploymentMode.LOW_BANDWIDTH,
            max_bandwidth_mbps=1.0,
            max_latency_ms=5000,
            storage_limit_gb=10.0,
            compression_enabled=True,
            model_size_mb=50.0,
            feature_flags={"minimal_features": True, "compression": True},
        )

        # Edge computing configuration
        self.configurations[DeploymentMode.EDGE_COMPUTING] = DeploymentConfiguration(
            config_id=str(uuid.uuid4()),
            mode=DeploymentMode.EDGE_COMPUTING,
            max_bandwidth_mbps=10.0,
            max_latency_ms=500,
            storage_limit_gb=50.0,
            compression_enabled=True,
            model_size_mb=100.0,
            feature_flags={"edge_optimized": True},
        )

        # Offline configuration
        self.configurations[DeploymentMode.OFFLINE] = DeploymentConfiguration(
            config_id=str(uuid.uuid4()),
            mode=DeploymentMode.OFFLINE,
            max_bandwidth_mbps=0.0,
            max_latency_ms=10,
            storage_limit_gb=20.0,
            compression_enabled=True,
            model_size_mb=30.0,
            feature_flags={"offline_mode": True, "local_only": True},
        )

        logger.info("Initialized default deployment configurations")

    def get_configuration(self, mode: DeploymentMode) -> Optional[DeploymentConfiguration]:
        """Get deployment configuration for a mode."""
        return self.configurations.get(mode)

    def optimize_for_bandwidth(self, available_bandwidth_mbps: float) -> DeploymentConfiguration:
        """Optimize deployment configuration for available bandwidth."""
        start_time = time.time()

        # Select appropriate mode based on bandwidth
        if available_bandwidth_mbps >= 100:
            mode = DeploymentMode.FULL_CLOUD
        elif available_bandwidth_mbps >= 10:
            mode = DeploymentMode.EDGE_COMPUTING
        elif available_bandwidth_mbps >= 1:
            mode = DeploymentMode.LOW_BANDWIDTH
        else:
            mode = DeploymentMode.OFFLINE

        config = self.configurations[mode]

        optimization_time = time.time() - start_time
        logger.info(
            f"Optimized for {available_bandwidth_mbps} Mbps: {mode.value} in {optimization_time*1000:.2f}ms"
        )

        return config

    def deploy(self, deployment_id: str, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Deploy with specific configuration."""
        start_time = time.time()

        self.active_deployments[deployment_id] = config

        deployment_time = time.time() - start_time

        result = {
            "deployment_id": deployment_id,
            "mode": config.mode.value,
            "deployment_time_ms": deployment_time * 1000,
            "estimated_latency_ms": config.max_latency_ms,
            "model_size_mb": config.model_size_mb,
            "features_enabled": config.feature_flags,
        }

        logger.info(f"Deployed {deployment_id} in {config.mode.value} mode")
        return result

    def measure_performance(self, deployment_id: str, latency_ms: float, throughput_mbps: float):
        """Measure deployment performance."""
        if deployment_id not in self.active_deployments:
            logger.warning(f"Deployment {deployment_id} not found")
            return

        self.performance_metrics[deployment_id] = {
            "latency_ms": latency_ms,
            "throughput_mbps": throughput_mbps,
            "timestamp": time.time(),
        }

        logger.debug(
            f"Measured performance for {deployment_id}: {latency_ms}ms, {throughput_mbps}Mbps"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        return {
            "total_configurations": len(self.configurations),
            "active_deployments": len(self.active_deployments),
            "monitored_deployments": len(self.performance_metrics),
        }


class GlobalDataGovernanceFramework:
    """
    Manages global data collaboration and governance policies.
    """

    def __init__(self):
        self.policies: Dict[Region, DataCollaborationPolicy] = {}
        self.collaboration_requests: List[Dict[str, Any]] = []
        self.approved_collaborations: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []

        logger.info("Initialized GlobalDataGovernanceFramework")
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Initialize default regional policies."""
        # North America policy (HIPAA-compliant)
        self.policies[Region.NORTH_AMERICA] = DataCollaborationPolicy(
            policy_id=str(uuid.uuid4()),
            region=Region.NORTH_AMERICA,
            governance_level=DataGovernanceLevel.RESTRICTED,
            data_residency_required=True,
            allowed_sharing_regions=[Region.NORTH_AMERICA],
            encryption_required=True,
            anonymization_level="strict",
            audit_requirements={"full_audit": True, "retention_years": 7},
            compliance_frameworks=["HIPAA", "HITECH"],
        )

        # Europe policy (GDPR-compliant)
        self.policies[Region.EUROPE] = DataCollaborationPolicy(
            policy_id=str(uuid.uuid4()),
            region=Region.EUROPE,
            governance_level=DataGovernanceLevel.SOVEREIGN,
            data_residency_required=True,
            allowed_sharing_regions=[Region.EUROPE],
            encryption_required=True,
            anonymization_level="strict",
            audit_requirements={
                "full_audit": True,
                "retention_years": 10,
                "right_to_be_forgotten": True,
            },
            compliance_frameworks=["GDPR", "eIDAS"],
        )

        # Asia-Pacific policy (Mixed regulations)
        self.policies[Region.ASIA_PACIFIC] = DataCollaborationPolicy(
            policy_id=str(uuid.uuid4()),
            region=Region.ASIA_PACIFIC,
            governance_level=DataGovernanceLevel.FEDERATED,
            data_residency_required=False,
            allowed_sharing_regions=[Region.ASIA_PACIFIC, Region.NORTH_AMERICA],
            encryption_required=True,
            anonymization_level="moderate",
            audit_requirements={"full_audit": True, "retention_years": 5},
            compliance_frameworks=["APEC", "Local"],
        )

        logger.info("Initialized default regional policies")

    def get_policy(self, region: Region) -> Optional[DataCollaborationPolicy]:
        """Get collaboration policy for a region."""
        return self.policies.get(region)

    def request_collaboration(
        self, source_region: Region, target_region: Region, data_type: str, purpose: str
    ) -> Dict[str, Any]:
        """Request data collaboration between regions."""
        request_id = str(uuid.uuid4())
        timestamp = datetime.now()

        source_policy = self.policies.get(source_region)
        target_policy = self.policies.get(target_region)

        if not source_policy or not target_policy:
            return {"approved": False, "reason": "Policy not found for one or both regions"}

        # Check if collaboration is allowed
        approved = target_region in source_policy.allowed_sharing_regions

        request = {
            "request_id": request_id,
            "source_region": source_region.value,
            "target_region": target_region.value,
            "data_type": data_type,
            "purpose": purpose,
            "approved": approved,
            "timestamp": timestamp.isoformat(),
            "encryption_required": source_policy.encryption_required
            and target_policy.encryption_required,
            "anonymization_level": max(
                source_policy.anonymization_level,
                target_policy.anonymization_level,
                key=lambda x: ["none", "moderate", "strict"].index(x),
            ),
        }

        self.collaboration_requests.append(request)

        if approved:
            self.approved_collaborations.append(request)
            logger.info(f"Approved collaboration: {source_region.value} -> {target_region.value}")
        else:
            logger.warning(f"Denied collaboration: {source_region.value} -> {target_region.value}")

        # Audit log
        self._log_audit_event("collaboration_request", request)

        return request

    def verify_compliance(self, region: Region, frameworks: List[str]) -> Dict[str, Any]:
        """Verify compliance with specified frameworks."""
        policy = self.policies.get(region)

        if not policy:
            return {"compliant": False, "reason": "No policy found for region"}

        compliant_frameworks = set(policy.compliance_frameworks)
        required_frameworks = set(frameworks)

        missing = required_frameworks - compliant_frameworks

        result = {
            "region": region.value,
            "compliant": len(missing) == 0,
            "supported_frameworks": list(compliant_frameworks),
            "missing_frameworks": list(missing),
        }

        self._log_audit_event("compliance_verification", result)

        return result

    def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an audit event."""
        audit_entry = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": event_data,
        }
        self.audit_log.append(audit_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self.audit_log[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "total_policies": len(self.policies),
            "collaboration_requests": len(self.collaboration_requests),
            "approved_collaborations": len(self.approved_collaborations),
            "audit_log_entries": len(self.audit_log),
            "approval_rate": (
                (len(self.approved_collaborations) / len(self.collaboration_requests) * 100)
                if self.collaboration_requests
                else 0
            ),
        }


class InternationalHealthcareSystem:
    """
    Main system integrating all international healthcare capabilities.
    """

    def __init__(
        self,
        default_language: Language = Language.ENGLISH,
        default_region: Region = Region.NORTH_AMERICA,
    ):
        self.multilingual = MultilingualInterface(default_language)
        self.guidelines = RegionalGuidelineEngine()
        self.deployment = ConstrainedDeploymentManager()
        self.governance = GlobalDataGovernanceFramework()
        self.default_region = default_region

        logger.info(f"Initialized InternationalHealthcareSystem for {default_region.value}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "multilingual": self.multilingual.get_statistics(),
            "guidelines": self.guidelines.get_statistics(),
            "deployment": self.deployment.get_statistics(),
            "governance": self.governance.get_statistics(),
            "default_region": self.default_region.value,
        }


def create_international_healthcare_system(
    default_language: Language = Language.ENGLISH, default_region: Region = Region.NORTH_AMERICA
) -> InternationalHealthcareSystem:
    """
    Factory function to create an international healthcare system.

    Args:
        default_language: Default interface language
        default_region: Default healthcare region

    Returns:
        Configured InternationalHealthcareSystem instance
    """
    return InternationalHealthcareSystem(default_language, default_region)
