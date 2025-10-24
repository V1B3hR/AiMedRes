#!/usr/bin/env python3
"""
Human-in-Loop Gating and Audit Module

Implements mandatory human oversight for high-risk clinical recommendations
with immutable audit logging.

P0-5 Requirement: Human-in-loop gating enforced end-to-end with mandatory
rationale logging.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for clinical recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Status of human approval"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"  # Admin override


@dataclass
class ClinicalRecommendation:
    """A clinical recommendation requiring review"""
    recommendation_id: str
    recommendation_type: str
    recommendation_text: str
    risk_level: RiskLevel
    confidence_score: float
    patient_id: str
    generated_at: str
    generated_by: str  # System/model identifier
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        return data


@dataclass
class HumanApproval:
    """Record of human approval/rejection"""
    approval_id: str
    recommendation_id: str
    clinician_id: str
    clinician_role: str
    status: ApprovalStatus
    rationale: str
    approved_at: str
    review_duration_seconds: float
    overriding_admin_id: Optional[str] = None
    override_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class AuditEntry:
    """Immutable audit log entry"""
    entry_id: str
    entry_hash: str  # SHA-256 hash of entry for immutability verification
    previous_hash: str  # Hash of previous entry (blockchain-like)
    timestamp: str
    event_type: str
    recommendation: ClinicalRecommendation
    approval: Optional[HumanApproval]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of this entry"""
        # Create deterministic string representation
        data_str = json.dumps({
            'entry_id': self.entry_id,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'recommendation': self.recommendation.to_dict(),
            'approval': self.approval.to_dict() if self.approval else None,
            'metadata': self.metadata
        }, sort_keys=True)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_hash(self) -> bool:
        """Verify the hash of this entry hasn't been tampered with"""
        return self.entry_hash == self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entry_id': self.entry_id,
            'entry_hash': self.entry_hash,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'recommendation': self.recommendation.to_dict(),
            'approval': self.approval.to_dict() if self.approval else None,
            'metadata': self.metadata
        }


class HumanInLoopGatekeeper:
    """
    Enforces human-in-loop approval for high-risk clinical recommendations.
    
    Features:
    - Mandatory human approval for HIGH and CRITICAL risk recommendations
    - Immutable audit trail with cryptographic verification
    - Blockchain-like chaining of audit entries
    - Configurable risk thresholds
    - Support for approval workflows
    """
    
    def __init__(self, 
                 audit_log_path: Optional[Path] = None,
                 require_approval_for: Optional[List[RiskLevel]] = None,
                 allow_admin_override: bool = True):
        """
        Initialize the gatekeeper.
        
        Args:
            audit_log_path: Path to store audit logs (default: ./audit_logs/)
            require_approval_for: Risk levels requiring approval (default: HIGH, CRITICAL)
            allow_admin_override: Whether to allow admin overrides
        """
        self.audit_log_path = audit_log_path or Path("./audit_logs")
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        # Default to requiring approval for HIGH and CRITICAL
        self.require_approval_for = require_approval_for or [
            RiskLevel.HIGH, 
            RiskLevel.CRITICAL
        ]
        
        self.allow_admin_override = allow_admin_override
        
        # In-memory tracking of pending approvals
        self.pending_approvals: Dict[str, ClinicalRecommendation] = {}
        
        # Last audit entry hash for chaining
        self.last_audit_hash = "GENESIS"
        
        # Load existing audit log
        self._load_audit_chain()
    
    def _load_audit_chain(self):
        """Load existing audit chain and verify integrity"""
        audit_file = self.audit_log_path / "audit_chain.jsonl"
        if not audit_file.exists():
            logger.info("No existing audit chain found. Starting fresh.")
            return
        
        with open(audit_file, 'r') as f:
            lines = f.readlines()
            if lines:
                # Get the last hash for chaining
                last_entry = json.loads(lines[-1])
                self.last_audit_hash = last_entry['entry_hash']
                logger.info(f"Loaded audit chain with {len(lines)} entries")
    
    def requires_approval(self, risk_level: RiskLevel) -> bool:
        """Check if a risk level requires human approval"""
        return risk_level in self.require_approval_for
    
    def submit_recommendation(self, 
                            recommendation: ClinicalRecommendation) -> Dict[str, Any]:
        """
        Submit a clinical recommendation for potential approval.
        
        Args:
            recommendation: The clinical recommendation
            
        Returns:
            Dictionary with status and recommendation_id
        """
        # Generate unique ID if not provided
        if not recommendation.recommendation_id:
            recommendation.recommendation_id = str(uuid.uuid4())
        
        # Log submission
        self._create_audit_entry(
            event_type="recommendation_submitted",
            recommendation=recommendation,
            approval=None
        )
        
        if self.requires_approval(recommendation.risk_level):
            # Add to pending approvals
            self.pending_approvals[recommendation.recommendation_id] = recommendation
            logger.warning(
                f"HIGH RISK: Recommendation {recommendation.recommendation_id} "
                f"requires human approval before proceeding"
            )
            return {
                'status': 'pending_approval',
                'recommendation_id': recommendation.recommendation_id,
                'requires_human_approval': True,
                'risk_level': recommendation.risk_level.value,
                'message': 'This recommendation requires review and approval by a licensed clinician'
            }
        else:
            # Auto-approve low/medium risk
            logger.info(
                f"Low risk recommendation {recommendation.recommendation_id} "
                f"auto-approved"
            )
            return {
                'status': 'approved',
                'recommendation_id': recommendation.recommendation_id,
                'requires_human_approval': False,
                'risk_level': recommendation.risk_level.value
            }
    
    def approve_recommendation(self,
                              recommendation_id: str,
                              clinician_id: str,
                              clinician_role: str,
                              rationale: str,
                              review_start_time: datetime) -> Dict[str, Any]:
        """
        Approve a pending recommendation.
        
        Args:
            recommendation_id: ID of recommendation to approve
            clinician_id: ID of approving clinician
            clinician_role: Role/title of clinician
            rationale: Required rationale for approval
            review_start_time: When review started (for duration tracking)
            
        Returns:
            Dictionary with approval status
            
        Raises:
            ValueError: If recommendation not found or rationale missing
        """
        if not rationale or len(rationale.strip()) < 10:
            raise ValueError(
                "Approval rationale is required and must be at least 10 characters"
            )
        
        if recommendation_id not in self.pending_approvals:
            raise ValueError(
                f"Recommendation {recommendation_id} not found in pending approvals"
            )
        
        recommendation = self.pending_approvals[recommendation_id]
        
        # Calculate review duration
        review_duration = (datetime.now(timezone.utc) - review_start_time).total_seconds()
        
        # Create approval record
        approval = HumanApproval(
            approval_id=str(uuid.uuid4()),
            recommendation_id=recommendation_id,
            clinician_id=clinician_id,
            clinician_role=clinician_role,
            status=ApprovalStatus.APPROVED,
            rationale=rationale,
            approved_at=datetime.now(timezone.utc).isoformat(),
            review_duration_seconds=review_duration
        )
        
        # Create immutable audit entry
        self._create_audit_entry(
            event_type="recommendation_approved",
            recommendation=recommendation,
            approval=approval
        )
        
        # Remove from pending
        del self.pending_approvals[recommendation_id]
        
        logger.info(
            f"Recommendation {recommendation_id} approved by {clinician_id} "
            f"after {review_duration:.1f}s review"
        )
        
        return {
            'status': 'approved',
            'recommendation_id': recommendation_id,
            'approval_id': approval.approval_id,
            'approved_by': clinician_id,
            'approved_at': approval.approved_at
        }
    
    def reject_recommendation(self,
                            recommendation_id: str,
                            clinician_id: str,
                            clinician_role: str,
                            rationale: str,
                            review_start_time: datetime) -> Dict[str, Any]:
        """
        Reject a pending recommendation.
        
        Args:
            recommendation_id: ID of recommendation to reject
            clinician_id: ID of rejecting clinician
            clinician_role: Role/title of clinician
            rationale: Required rationale for rejection
            review_start_time: When review started
            
        Returns:
            Dictionary with rejection status
        """
        if not rationale or len(rationale.strip()) < 10:
            raise ValueError(
                "Rejection rationale is required and must be at least 10 characters"
            )
        
        if recommendation_id not in self.pending_approvals:
            raise ValueError(
                f"Recommendation {recommendation_id} not found in pending approvals"
            )
        
        recommendation = self.pending_approvals[recommendation_id]
        review_duration = (datetime.now(timezone.utc) - review_start_time).total_seconds()
        
        approval = HumanApproval(
            approval_id=str(uuid.uuid4()),
            recommendation_id=recommendation_id,
            clinician_id=clinician_id,
            clinician_role=clinician_role,
            status=ApprovalStatus.REJECTED,
            rationale=rationale,
            approved_at=datetime.now(timezone.utc).isoformat(),
            review_duration_seconds=review_duration
        )
        
        self._create_audit_entry(
            event_type="recommendation_rejected",
            recommendation=recommendation,
            approval=approval
        )
        
        del self.pending_approvals[recommendation_id]
        
        logger.info(f"Recommendation {recommendation_id} rejected by {clinician_id}")
        
        return {
            'status': 'rejected',
            'recommendation_id': recommendation_id,
            'rejection_id': approval.approval_id,
            'rejected_by': clinician_id,
            'rejected_at': approval.approved_at
        }
    
    def admin_override(self,
                      recommendation_id: str,
                      admin_id: str,
                      override_reason: str) -> Dict[str, Any]:
        """
        Admin override to force approve a recommendation.
        Should only be used in exceptional circumstances.
        
        Args:
            recommendation_id: ID of recommendation
            admin_id: ID of admin performing override
            override_reason: Required detailed reason for override
            
        Returns:
            Dictionary with override status
        """
        if not self.allow_admin_override:
            raise PermissionError("Admin overrides are not enabled")
        
        if not override_reason or len(override_reason.strip()) < 20:
            raise ValueError(
                "Override reason is required and must be at least 20 characters"
            )
        
        if recommendation_id not in self.pending_approvals:
            raise ValueError(
                f"Recommendation {recommendation_id} not found"
            )
        
        recommendation = self.pending_approvals[recommendation_id]
        
        approval = HumanApproval(
            approval_id=str(uuid.uuid4()),
            recommendation_id=recommendation_id,
            clinician_id="ADMIN_OVERRIDE",
            clinician_role="System Administrator",
            status=ApprovalStatus.OVERRIDDEN,
            rationale=f"ADMIN OVERRIDE: {override_reason}",
            approved_at=datetime.now(timezone.utc).isoformat(),
            review_duration_seconds=0,
            overriding_admin_id=admin_id,
            override_reason=override_reason
        )
        
        self._create_audit_entry(
            event_type="recommendation_admin_override",
            recommendation=recommendation,
            approval=approval,
            metadata={'admin_id': admin_id, 'severity': 'CRITICAL'}
        )
        
        del self.pending_approvals[recommendation_id]
        
        logger.warning(
            f"ADMIN OVERRIDE: Recommendation {recommendation_id} "
            f"overridden by admin {admin_id}"
        )
        
        return {
            'status': 'overridden',
            'recommendation_id': recommendation_id,
            'override_by': admin_id,
            'warning': 'This was an admin override. Requires review.'
        }
    
    def _create_audit_entry(self,
                          event_type: str,
                          recommendation: ClinicalRecommendation,
                          approval: Optional[HumanApproval],
                          metadata: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Create an immutable audit entry.
        
        Args:
            event_type: Type of event being audited
            recommendation: The recommendation
            approval: Optional approval/rejection record
            metadata: Additional metadata
            
        Returns:
            Created audit entry
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create entry without hash first
        entry = AuditEntry(
            entry_id=entry_id,
            entry_hash="",  # Will be calculated
            previous_hash=self.last_audit_hash,
            timestamp=timestamp,
            event_type=event_type,
            recommendation=recommendation,
            approval=approval,
            metadata=metadata or {}
        )
        
        # Calculate and set hash
        entry.entry_hash = entry.calculate_hash()
        
        # Update last hash for chaining
        self.last_audit_hash = entry.entry_hash
        
        # Write to audit log (append-only)
        audit_file = self.audit_log_path / "audit_chain.jsonl"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        logger.info(f"Audit entry created: {entry_id} ({event_type})")
        
        return entry
    
    def verify_audit_chain(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire audit chain.
        
        Returns:
            Dictionary with verification results
        """
        audit_file = self.audit_log_path / "audit_chain.jsonl"
        if not audit_file.exists():
            return {'verified': True, 'entries': 0, 'message': 'No audit chain exists'}
        
        with open(audit_file, 'r') as f:
            entries = [json.loads(line) for line in f]
        
        # Verify each entry
        prev_hash = "GENESIS"
        for i, entry_data in enumerate(entries):
            # Check previous hash chain
            if entry_data['previous_hash'] != prev_hash:
                return {
                    'verified': False,
                    'entries': len(entries),
                    'failed_at': i,
                    'message': f'Hash chain broken at entry {i}'
                }
            
            # Verify entry hash
            entry = AuditEntry(
                entry_id=entry_data['entry_id'],
                entry_hash=entry_data['entry_hash'],
                previous_hash=entry_data['previous_hash'],
                timestamp=entry_data['timestamp'],
                event_type=entry_data['event_type'],
                recommendation=ClinicalRecommendation(**entry_data['recommendation']),
                approval=HumanApproval(**entry_data['approval']) if entry_data['approval'] else None,
                metadata=entry_data.get('metadata', {})
            )
            
            if not entry.verify_hash():
                return {
                    'verified': False,
                    'entries': len(entries),
                    'failed_at': i,
                    'message': f'Hash verification failed at entry {i}'
                }
            
            prev_hash = entry_data['entry_hash']
        
        return {
            'verified': True,
            'entries': len(entries),
            'message': 'Audit chain verified successfully'
        }
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of pending approvals"""
        return [rec.to_dict() for rec in self.pending_approvals.values()]
    
    def get_audit_history(self, 
                         recommendation_id: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit history, optionally filtered by recommendation_id.
        
        Args:
            recommendation_id: Optional filter by recommendation
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries
        """
        audit_file = self.audit_log_path / "audit_chain.jsonl"
        if not audit_file.exists():
            return []
        
        with open(audit_file, 'r') as f:
            entries = [json.loads(line) for line in f]
        
        if recommendation_id:
            entries = [
                e for e in entries 
                if e['recommendation']['recommendation_id'] == recommendation_id
            ]
        
        # Return most recent first
        return list(reversed(entries[-limit:]))


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize gatekeeper
    gatekeeper = HumanInLoopGatekeeper()
    
    # Submit a high-risk recommendation
    recommendation = ClinicalRecommendation(
        recommendation_id="",
        recommendation_type="medication_change",
        recommendation_text="Increase dosage of medication X to 20mg daily",
        risk_level=RiskLevel.HIGH,
        confidence_score=0.85,
        patient_id="SYNTH-12345",
        generated_at=datetime.now(timezone.utc).isoformat(),
        generated_by="AI_Model_v2.3",
        context={'previous_dosage': '10mg', 'indication': 'symptom_progression'}
    )
    
    result = gatekeeper.submit_recommendation(recommendation)
    print(f"\nSubmission result: {result}")
    
    if result['requires_human_approval']:
        rec_id = result['recommendation_id']
        
        # Simulate clinician review
        review_start = datetime.now(timezone.utc)
        
        # Approve the recommendation
        approval_result = gatekeeper.approve_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_SMITH_MD",
            clinician_role="Neurologist",
            rationale="Reviewed patient history and current symptoms. Dosage increase is appropriate given symptom progression and tolerability.",
            review_start_time=review_start
        )
        
        print(f"\nApproval result: {approval_result}")
    
    # Verify audit chain
    verification = gatekeeper.verify_audit_chain()
    print(f"\nAudit chain verification: {verification}")
