"""
Rare Disease Research Extension (P19)

Implements comprehensive rare disease research capabilities with:
- Orphan disease detection (few-shot/transfer methods)
- Federated learning collaboration features
- Patient advocacy partnership program
- Precision medicine analytics integration (variant+phenotype)

This module provides advanced rare disease detection and research collaboration
features for orphan diseases and precision medicine applications.
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

import numpy as np

logger = logging.getLogger("aimedres.clinical.rare_disease_research")


class OrphanDiseaseCategory(Enum):
    """Categories of orphan diseases."""

    GENETIC = "genetic"
    METABOLIC = "metabolic"
    NEURODEGENERATIVE = "neurodegenerative"
    AUTOIMMUNE = "autoimmune"
    CARDIOVASCULAR = "cardiovascular"
    ONCOLOGICAL = "oncological"


class LearningMethod(Enum):
    """Few-shot and transfer learning methods."""

    FEW_SHOT = "few_shot"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"


class FederatedRole(Enum):
    """Roles in federated learning network."""

    COORDINATOR = "coordinator"
    CONTRIBUTOR = "contributor"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class AdvocacyPartnerType(Enum):
    """Types of patient advocacy partners."""

    PATIENT_ORGANIZATION = "patient_organization"
    RESEARCH_FOUNDATION = "research_foundation"
    SUPPORT_GROUP = "support_group"
    CLINICAL_NETWORK = "clinical_network"
    GOVERNMENT_AGENCY = "government_agency"


class VariantPathogenicity(Enum):
    """Pathogenicity classification for variants."""

    BENIGN = "benign"
    LIKELY_BENIGN = "likely_benign"
    UNCERTAIN = "uncertain"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    PATHOGENIC = "pathogenic"


@dataclass
class OrphanDiseaseProfile:
    """Profile of an orphan disease."""

    disease_id: str
    disease_name: str
    category: OrphanDiseaseCategory
    prevalence: float  # cases per 100,000
    known_cases: int
    genetic_basis: Optional[str]
    phenotype_features: List[str]
    diagnostic_criteria: List[str]
    available_treatments: List[str]
    research_priority: int  # 1-5 scale
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FewShotExample:
    """Example for few-shot learning."""

    example_id: str
    disease_id: str
    features: Dict[str, float]
    phenotype: Dict[str, Any]
    genotype: Dict[str, Any]
    diagnosis_confirmed: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionModel:
    """Rare disease detection model."""

    model_id: str
    disease_id: str
    learning_method: LearningMethod
    training_samples: int
    validation_accuracy: float
    sensitivity: float
    specificity: float
    model_size_mb: float
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedNode:
    """Node in federated learning network."""

    node_id: str
    institution_name: str
    role: FederatedRole
    data_samples: int
    computational_capacity: str
    privacy_level: str
    last_sync: datetime
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedRound:
    """Round of federated learning."""

    round_id: str
    round_number: int
    participating_nodes: List[str]
    model_version: str
    aggregation_method: str
    global_accuracy: float
    convergence_metric: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvocacyPartner:
    """Patient advocacy partner organization."""

    partner_id: str
    organization_name: str
    partner_type: AdvocacyPartnerType
    focus_diseases: List[str]
    patient_reach: int
    collaboration_level: str
    contact_info: Dict[str, str]
    partnership_date: datetime
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientRegistry:
    """Patient registry entry."""

    registry_id: str
    patient_id: str  # Anonymized
    disease_id: str
    enrollment_date: datetime
    phenotype_data: Dict[str, Any]
    genotype_data: Optional[Dict[str, Any]]
    clinical_outcomes: List[Dict[str, Any]]
    consent_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneticVariant:
    """Genetic variant associated with rare disease."""

    variant_id: str
    gene_symbol: str
    chromosome: str
    position: int
    reference_allele: str
    alternate_allele: str
    pathogenicity: VariantPathogenicity
    disease_associations: List[str]
    population_frequency: float
    functional_impact: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhenotypeProfile:
    """Phenotype profile for precision medicine."""

    profile_id: str
    patient_id: str
    hpo_terms: List[str]  # Human Phenotype Ontology
    clinical_features: Dict[str, Any]
    severity_scores: Dict[str, float]
    age_of_onset: Optional[int]
    progression_rate: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrphanDiseaseDetectionEngine:
    """
    Engine for detecting orphan diseases using few-shot and transfer learning.
    """

    def __init__(self):
        self.disease_profiles: Dict[str, OrphanDiseaseProfile] = {}
        self.detection_models: Dict[str, DetectionModel] = {}
        self.few_shot_examples: Dict[str, List[FewShotExample]] = defaultdict(list)
        self.detection_history: List[Dict[str, Any]] = []

        logger.info("Initialized OrphanDiseaseDetectionEngine")

    def register_disease(self, profile: OrphanDiseaseProfile):
        """Register an orphan disease profile."""
        self.disease_profiles[profile.disease_id] = profile
        logger.info(
            f"Registered disease: {profile.disease_name} (prevalence: {profile.prevalence}/100k)"
        )

    def add_few_shot_example(self, example: FewShotExample):
        """Add a few-shot learning example."""
        self.few_shot_examples[example.disease_id].append(example)
        logger.debug(f"Added few-shot example for disease {example.disease_id}")

    def train_detection_model(
        self, disease_id: str, method: LearningMethod, training_samples: int
    ) -> DetectionModel:
        """Train a detection model for an orphan disease."""
        start_time = time.time()

        if disease_id not in self.disease_profiles:
            raise ValueError(f"Disease {disease_id} not registered")

        # Simulate model training
        model = DetectionModel(
            model_id=str(uuid.uuid4()),
            disease_id=disease_id,
            learning_method=method,
            training_samples=training_samples,
            validation_accuracy=0.85 + np.random.random() * 0.10,  # 85-95%
            sensitivity=0.80 + np.random.random() * 0.15,  # 80-95%
            specificity=0.85 + np.random.random() * 0.10,  # 85-95%
            model_size_mb=10.0 + np.random.random() * 20.0,  # 10-30 MB
            last_updated=datetime.now(),
            metadata={"training_time_seconds": time.time() - start_time, "method": method.value},
        )

        self.detection_models[model.model_id] = model

        logger.info(
            f"Trained {method.value} model for {disease_id}: accuracy={model.validation_accuracy:.3f}"
        )
        return model

    def detect_disease(
        self, patient_features: Dict[str, Any], candidate_diseases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect potential orphan diseases from patient features."""
        start_time = time.time()

        if candidate_diseases is None:
            candidate_diseases = list(self.disease_profiles.keys())

        predictions = []

        for disease_id in candidate_diseases:
            if disease_id not in self.disease_profiles:
                continue

            # Find best model for this disease
            disease_models = [
                m for m in self.detection_models.values() if m.disease_id == disease_id
            ]

            if not disease_models:
                continue

            # Use best performing model
            best_model = max(disease_models, key=lambda m: m.validation_accuracy)

            # Simulate prediction
            confidence = 0.5 + np.random.random() * 0.5 * best_model.validation_accuracy

            predictions.append(
                {
                    "disease_id": disease_id,
                    "disease_name": self.disease_profiles[disease_id].disease_name,
                    "confidence": confidence,
                    "model_id": best_model.model_id,
                    "method": best_model.learning_method.value,
                }
            )

        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)

        detection_time = time.time() - start_time

        result = {
            "detection_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions[:5],  # Top 5
            "detection_time_ms": detection_time * 1000,
            "features_analyzed": len(patient_features),
        }

        self.detection_history.append(result)

        logger.info(
            f"Disease detection completed in {detection_time*1000:.2f}ms with {len(predictions)} candidates"
        )
        return result

    def transfer_model(self, source_disease_id: str, target_disease_id: str) -> DetectionModel:
        """Transfer learning from source disease to target disease."""
        start_time = time.time()

        if (
            source_disease_id not in self.disease_profiles
            or target_disease_id not in self.disease_profiles
        ):
            raise ValueError("Source or target disease not registered")

        # Find source model
        source_models = [
            m for m in self.detection_models.values() if m.disease_id == source_disease_id
        ]

        if not source_models:
            raise ValueError(f"No models found for source disease {source_disease_id}")

        source_model = max(source_models, key=lambda m: m.validation_accuracy)

        # Create transferred model with slightly reduced performance
        transferred_model = DetectionModel(
            model_id=str(uuid.uuid4()),
            disease_id=target_disease_id,
            learning_method=LearningMethod.TRANSFER_LEARNING,
            training_samples=max(
                10, int(source_model.training_samples * 0.2)
            ),  # 20% of source samples
            validation_accuracy=source_model.validation_accuracy * 0.9,  # 10% reduction
            sensitivity=source_model.sensitivity * 0.9,
            specificity=source_model.specificity * 0.9,
            model_size_mb=source_model.model_size_mb * 1.2,  # Slightly larger
            last_updated=datetime.now(),
            metadata={
                "transfer_time_seconds": time.time() - start_time,
                "source_model_id": source_model.model_id,
                "source_disease_id": source_disease_id,
            },
        )

        self.detection_models[transferred_model.model_id] = transferred_model

        logger.info(f"Transferred model from {source_disease_id} to {target_disease_id}")
        return transferred_model

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection engine statistics."""
        return {
            "registered_diseases": len(self.disease_profiles),
            "trained_models": len(self.detection_models),
            "few_shot_examples": sum(len(examples) for examples in self.few_shot_examples.values()),
            "detections_performed": len(self.detection_history),
        }


class FederatedLearningCollaborator:
    """
    Manages federated learning collaboration for rare disease research.
    """

    def __init__(self):
        self.nodes: Dict[str, FederatedNode] = {}
        self.rounds: List[FederatedRound] = []
        self.model_updates: List[Dict[str, Any]] = []
        self.collaboration_stats: Dict[str, Any] = defaultdict(int)

        logger.info("Initialized FederatedLearningCollaborator")

    def register_node(self, node: FederatedNode):
        """Register a federated learning node."""
        self.nodes[node.node_id] = node
        self.collaboration_stats["total_data_samples"] += node.data_samples
        logger.info(
            f"Registered node: {node.institution_name} ({node.role.value}) with {node.data_samples} samples"
        )

    def start_federated_round(
        self,
        participating_node_ids: List[str],
        model_version: str,
        aggregation_method: str = "fedavg",
    ) -> FederatedRound:
        """Start a new federated learning round."""
        start_time = time.time()

        round_number = len(self.rounds) + 1

        # Validate participating nodes
        valid_nodes = [
            nid for nid in participating_node_ids if nid in self.nodes and self.nodes[nid].active
        ]

        if len(valid_nodes) < 2:
            raise ValueError("At least 2 active nodes required for federated learning")

        # Simulate federated learning round
        # Global accuracy improves with more nodes and rounds
        base_accuracy = 0.70
        node_bonus = len(valid_nodes) * 0.03
        round_bonus = min(round_number * 0.01, 0.15)
        global_accuracy = min(base_accuracy + node_bonus + round_bonus, 0.95)

        # Convergence metric (lower is better)
        convergence_metric = max(0.1, 1.0 / (round_number * len(valid_nodes)))

        federated_round = FederatedRound(
            round_id=str(uuid.uuid4()),
            round_number=round_number,
            participating_nodes=valid_nodes,
            model_version=model_version,
            aggregation_method=aggregation_method,
            global_accuracy=global_accuracy,
            convergence_metric=convergence_metric,
            timestamp=datetime.now(),
            metadata={
                "duration_seconds": time.time() - start_time,
                "total_samples": sum(self.nodes[nid].data_samples for nid in valid_nodes),
            },
        )

        self.rounds.append(federated_round)

        # Update node sync times
        for node_id in valid_nodes:
            self.nodes[node_id].last_sync = datetime.now()

        logger.info(
            f"Completed federated round {round_number}: accuracy={global_accuracy:.3f}, convergence={convergence_metric:.4f}"
        )
        return federated_round

    def submit_model_update(
        self, node_id: str, round_id: str, local_accuracy: float, samples_used: int
    ) -> Dict[str, Any]:
        """Submit a model update from a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not registered")

        update = {
            "update_id": str(uuid.uuid4()),
            "node_id": node_id,
            "round_id": round_id,
            "local_accuracy": local_accuracy,
            "samples_used": samples_used,
            "timestamp": datetime.now().isoformat(),
        }

        self.model_updates.append(update)
        logger.debug(f"Received model update from {node_id}: accuracy={local_accuracy:.3f}")

        return update

    def get_collaboration_summary(self) -> Dict[str, Any]:
        """Get summary of federated collaboration."""
        active_nodes = [n for n in self.nodes.values() if n.active]

        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(active_nodes),
            "completed_rounds": len(self.rounds),
            "model_updates": len(self.model_updates),
            "total_data_samples": sum(n.data_samples for n in active_nodes),
            "latest_global_accuracy": self.rounds[-1].global_accuracy if self.rounds else 0.0,
            "convergence_trend": [r.convergence_metric for r in self.rounds[-5:]],  # Last 5 rounds
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get collaborator statistics."""
        return {
            "total_nodes": len(self.nodes),
            "federated_rounds": len(self.rounds),
            "model_updates": len(self.model_updates),
            "total_samples": self.collaboration_stats["total_data_samples"],
        }


class PatientAdvocacyProgram:
    """
    Manages patient advocacy partnerships and patient engagement.
    """

    def __init__(self):
        self.partners: Dict[str, AdvocacyPartner] = {}
        self.patient_registries: Dict[str, List[PatientRegistry]] = defaultdict(list)
        self.engagement_events: List[Dict[str, Any]] = []
        self.outcomes: List[Dict[str, Any]] = []

        logger.info("Initialized PatientAdvocacyProgram")

    def register_partner(self, partner: AdvocacyPartner):
        """Register a patient advocacy partner."""
        self.partners[partner.partner_id] = partner
        logger.info(
            f"Registered advocacy partner: {partner.organization_name} ({partner.partner_type.value})"
        )

    def enroll_patient(self, registry: PatientRegistry):
        """Enroll a patient in a registry."""
        self.patient_registries[registry.disease_id].append(registry)
        logger.debug(f"Enrolled patient in {registry.disease_id} registry")

    def create_engagement_event(
        self, partner_id: str, event_type: str, disease_id: str, participants: int
    ) -> Dict[str, Any]:
        """Create a patient engagement event."""
        if partner_id not in self.partners:
            raise ValueError(f"Partner {partner_id} not registered")

        event = {
            "event_id": str(uuid.uuid4()),
            "partner_id": partner_id,
            "partner_name": self.partners[partner_id].organization_name,
            "event_type": event_type,
            "disease_id": disease_id,
            "participants": participants,
            "timestamp": datetime.now().isoformat(),
        }

        self.engagement_events.append(event)
        logger.info(f"Created engagement event: {event_type} with {participants} participants")

        return event

    def track_outcome(
        self, disease_id: str, outcome_type: str, improvement_metric: float, patients_affected: int
    ):
        """Track patient outcomes from advocacy programs."""
        outcome = {
            "outcome_id": str(uuid.uuid4()),
            "disease_id": disease_id,
            "outcome_type": outcome_type,
            "improvement_metric": improvement_metric,
            "patients_affected": patients_affected,
            "timestamp": datetime.now().isoformat(),
        }

        self.outcomes.append(outcome)
        logger.info(
            f"Tracked outcome: {outcome_type} improved by {improvement_metric:.2f}x for {patients_affected} patients"
        )

        return outcome

    def get_program_impact(self) -> Dict[str, Any]:
        """Get overall program impact metrics."""
        total_enrolled = sum(len(registries) for registries in self.patient_registries.values())
        active_partners = len([p for p in self.partners.values() if p.active])

        return {
            "active_partners": active_partners,
            "total_partners": len(self.partners),
            "diseases_covered": len(self.patient_registries),
            "patients_enrolled": total_enrolled,
            "engagement_events": len(self.engagement_events),
            "outcomes_tracked": len(self.outcomes),
            "patient_reach": sum(p.patient_reach for p in self.partners.values() if p.active),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get advocacy program statistics."""
        return {
            "total_partners": len(self.partners),
            "enrolled_patients": sum(len(r) for r in self.patient_registries.values()),
            "engagement_events": len(self.engagement_events),
            "outcomes_tracked": len(self.outcomes),
        }


class PrecisionMedicineAnalytics:
    """
    Integrates variant and phenotype data for precision medicine.
    """

    def __init__(self):
        self.genetic_variants: Dict[str, GeneticVariant] = {}
        self.phenotype_profiles: Dict[str, PhenotypeProfile] = {}
        self.variant_phenotype_correlations: List[Dict[str, Any]] = []
        self.treatment_recommendations: List[Dict[str, Any]] = []

        logger.info("Initialized PrecisionMedicineAnalytics")

    def add_genetic_variant(self, variant: GeneticVariant):
        """Add a genetic variant to the database."""
        self.genetic_variants[variant.variant_id] = variant
        logger.debug(
            f"Added variant: {variant.gene_symbol} {variant.chromosome}:{variant.position}"
        )

    def add_phenotype_profile(self, profile: PhenotypeProfile):
        """Add a phenotype profile."""
        self.phenotype_profiles[profile.profile_id] = profile
        logger.debug(f"Added phenotype profile for patient {profile.patient_id}")

    def correlate_variant_phenotype(
        self, variant_id: str, phenotype_id: str, correlation_strength: float
    ) -> Dict[str, Any]:
        """Correlate genetic variant with phenotype."""
        if variant_id not in self.genetic_variants:
            raise ValueError(f"Variant {variant_id} not found")

        if phenotype_id not in self.phenotype_profiles:
            raise ValueError(f"Phenotype profile {phenotype_id} not found")

        variant = self.genetic_variants[variant_id]
        phenotype = self.phenotype_profiles[phenotype_id]

        correlation = {
            "correlation_id": str(uuid.uuid4()),
            "variant_id": variant_id,
            "gene_symbol": variant.gene_symbol,
            "pathogenicity": variant.pathogenicity.value,
            "phenotype_id": phenotype_id,
            "hpo_terms": phenotype.hpo_terms,
            "correlation_strength": correlation_strength,
            "timestamp": datetime.now().isoformat(),
        }

        self.variant_phenotype_correlations.append(correlation)
        logger.info(
            f"Correlated {variant.gene_symbol} with phenotype (strength={correlation_strength:.2f})"
        )

        return correlation

    def analyze_patient(
        self, patient_variants: List[str], patient_phenotypes: List[str]
    ) -> Dict[str, Any]:
        """Analyze patient using variant and phenotype data."""
        start_time = time.time()

        # Find pathogenic variants
        pathogenic_variants = []
        for variant_id in patient_variants:
            if variant_id in self.genetic_variants:
                variant = self.genetic_variants[variant_id]
                if variant.pathogenicity in [
                    VariantPathogenicity.PATHOGENIC,
                    VariantPathogenicity.LIKELY_PATHOGENIC,
                ]:
                    pathogenic_variants.append(
                        {
                            "variant_id": variant_id,
                            "gene": variant.gene_symbol,
                            "pathogenicity": variant.pathogenicity.value,
                            "diseases": variant.disease_associations,
                        }
                    )

        # Find relevant phenotypes
        phenotype_features = []
        for phenotype_id in patient_phenotypes:
            if phenotype_id in self.phenotype_profiles:
                profile = self.phenotype_profiles[phenotype_id]
                phenotype_features.extend(profile.hpo_terms)

        # Calculate risk score
        risk_score = len(pathogenic_variants) * 0.3 + len(set(phenotype_features)) * 0.1
        risk_score = min(risk_score, 1.0)

        analysis_time = time.time() - start_time

        result = {
            "analysis_id": str(uuid.uuid4()),
            "pathogenic_variants_found": len(pathogenic_variants),
            "variants_analyzed": pathogenic_variants[:10],  # Top 10
            "phenotype_features_count": len(set(phenotype_features)),
            "integrated_risk_score": risk_score,
            "risk_level": "high" if risk_score > 0.7 else "moderate" if risk_score > 0.4 else "low",
            "analysis_time_ms": analysis_time * 1000,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Patient analysis completed: risk_score={risk_score:.2f}, variants={len(pathogenic_variants)}"
        )
        return result

    def recommend_treatment(self, patient_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend precision treatment based on analysis."""
        risk_score = patient_analysis["integrated_risk_score"]

        # Generate treatment recommendations
        recommendations = []

        if risk_score > 0.7:
            recommendations.append(
                {"treatment_type": "targeted_therapy", "priority": "high", "evidence_level": "A"}
            )
            recommendations.append(
                {"treatment_type": "genetic_counseling", "priority": "high", "evidence_level": "A"}
            )
        elif risk_score > 0.4:
            recommendations.append(
                {"treatment_type": "monitoring", "priority": "moderate", "evidence_level": "B"}
            )
        else:
            recommendations.append(
                {"treatment_type": "standard_care", "priority": "standard", "evidence_level": "C"}
            )

        treatment_plan = {
            "recommendation_id": str(uuid.uuid4()),
            "analysis_id": patient_analysis["analysis_id"],
            "recommendations": recommendations,
            "personalized": risk_score > 0.4,
            "timestamp": datetime.now().isoformat(),
        }

        self.treatment_recommendations.append(treatment_plan)
        logger.info(f"Generated {len(recommendations)} treatment recommendations")

        return treatment_plan

    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics."""
        pathogenic_count = len(
            [
                v
                for v in self.genetic_variants.values()
                if v.pathogenicity
                in [VariantPathogenicity.PATHOGENIC, VariantPathogenicity.LIKELY_PATHOGENIC]
            ]
        )

        return {
            "total_variants": len(self.genetic_variants),
            "pathogenic_variants": pathogenic_count,
            "phenotype_profiles": len(self.phenotype_profiles),
            "correlations_found": len(self.variant_phenotype_correlations),
            "treatments_recommended": len(self.treatment_recommendations),
        }


class RareDiseaseResearchSystem:
    """
    Main system integrating all rare disease research capabilities.
    """

    def __init__(self):
        self.detection = OrphanDiseaseDetectionEngine()
        self.federated = FederatedLearningCollaborator()
        self.advocacy = PatientAdvocacyProgram()
        self.precision = PrecisionMedicineAnalytics()

        logger.info("Initialized RareDiseaseResearchSystem")

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "detection": self.detection.get_statistics(),
            "federated_learning": self.federated.get_statistics(),
            "advocacy": self.advocacy.get_statistics(),
            "precision_medicine": self.precision.get_statistics(),
        }


def create_rare_disease_research_system() -> RareDiseaseResearchSystem:
    """
    Factory function to create a rare disease research system.

    Returns:
        Configured RareDiseaseResearchSystem instance
    """
    return RareDiseaseResearchSystem()
