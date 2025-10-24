#!/usr/bin/env python3
"""
End-to-End Tests for Human-in-Loop Gating

Tests the complete workflow from recommendation submission through
approval/rejection with audit logging.

Part of P0-5 requirement.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aimedres.security.human_in_loop import (
    HumanInLoopGatekeeper,
    ClinicalRecommendation,
    RiskLevel,
    ApprovalStatus
)


class TestHumanInLoopGatekeeper:
    """Test human-in-loop gating functionality"""
    
    def setup_method(self):
        """Set up test fixtures with temporary audit log"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_path = Path(self.temp_dir) / "test_audit"
        self.gatekeeper = HumanInLoopGatekeeper(
            audit_log_path=self.audit_path
        )
    
    def create_recommendation(self, risk_level: RiskLevel) -> ClinicalRecommendation:
        """Helper to create a test recommendation"""
        return ClinicalRecommendation(
            recommendation_id="",
            recommendation_type="test_recommendation",
            recommendation_text=f"Test recommendation with {risk_level.value} risk",
            risk_level=risk_level,
            confidence_score=0.85,
            patient_id="SYNTH-TEST-001",
            generated_at=datetime.now(timezone.utc).isoformat(),
            generated_by="TestSystem_v1.0",
            context={'test': True}
        )
    
    def test_low_risk_auto_approved(self):
        """Test that low-risk recommendations are auto-approved"""
        rec = self.create_recommendation(RiskLevel.LOW)
        result = self.gatekeeper.submit_recommendation(rec)
        
        assert result['status'] == 'approved'
        assert not result['requires_human_approval']
        assert result['risk_level'] == 'low'
    
    def test_high_risk_requires_approval(self):
        """Test that high-risk recommendations require human approval"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        result = self.gatekeeper.submit_recommendation(rec)
        
        assert result['status'] == 'pending_approval'
        assert result['requires_human_approval']
        assert result['risk_level'] == 'high'
        assert rec.recommendation_id in self.gatekeeper.pending_approvals
    
    def test_critical_risk_requires_approval(self):
        """Test that critical-risk recommendations require human approval"""
        rec = self.create_recommendation(RiskLevel.CRITICAL)
        result = self.gatekeeper.submit_recommendation(rec)
        
        assert result['status'] == 'pending_approval'
        assert result['requires_human_approval']
    
    def test_approve_recommendation(self):
        """Test approving a pending recommendation"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        # Approve the recommendation
        review_start = datetime.now(timezone.utc) - timedelta(seconds=30)
        approval_result = self.gatekeeper.approve_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_TEST",
            clinician_role="Test Clinician",
            rationale="Approved after careful review of patient data and clinical context",
            review_start_time=review_start
        )
        
        assert approval_result['status'] == 'approved'
        assert approval_result['recommendation_id'] == rec_id
        assert approval_result['approved_by'] == "DR_TEST"
        assert 'approval_id' in approval_result
        
        # Should no longer be in pending
        assert rec_id not in self.gatekeeper.pending_approvals
    
    def test_reject_recommendation(self):
        """Test rejecting a pending recommendation"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        # Reject the recommendation
        review_start = datetime.now(timezone.utc) - timedelta(seconds=20)
        rejection_result = self.gatekeeper.reject_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_TEST",
            clinician_role="Test Clinician",
            rationale="Recommendation not appropriate given patient's current condition and history",
            review_start_time=review_start
        )
        
        assert rejection_result['status'] == 'rejected'
        assert rejection_result['recommendation_id'] == rec_id
        assert rejection_result['rejected_by'] == "DR_TEST"
        
        # Should no longer be in pending
        assert rec_id not in self.gatekeeper.pending_approvals
    
    def test_approval_requires_rationale(self):
        """Test that approval requires adequate rationale"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        # Try to approve without rationale
        with pytest.raises(ValueError, match="rationale is required"):
            self.gatekeeper.approve_recommendation(
                recommendation_id=rec_id,
                clinician_id="DR_TEST",
                clinician_role="Test Clinician",
                rationale="",  # Empty rationale
                review_start_time=datetime.now(timezone.utc)
            )
        
        # Try with too short rationale
        with pytest.raises(ValueError, match="rationale is required"):
            self.gatekeeper.approve_recommendation(
                recommendation_id=rec_id,
                clinician_id="DR_TEST",
                clinician_role="Test Clinician",
                rationale="OK",  # Too short
                review_start_time=datetime.now(timezone.utc)
            )
    
    def test_rejection_requires_rationale(self):
        """Test that rejection requires adequate rationale"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        with pytest.raises(ValueError, match="rationale is required"):
            self.gatekeeper.reject_recommendation(
                recommendation_id=rec_id,
                clinician_id="DR_TEST",
                clinician_role="Test Clinician",
                rationale="No",
                review_start_time=datetime.now(timezone.utc)
            )
    
    def test_cannot_approve_nonexistent_recommendation(self):
        """Test that approving non-existent recommendation fails"""
        with pytest.raises(ValueError, match="not found"):
            self.gatekeeper.approve_recommendation(
                recommendation_id="NONEXISTENT",
                clinician_id="DR_TEST",
                clinician_role="Test Clinician",
                rationale="This should fail because recommendation doesn't exist",
                review_start_time=datetime.now(timezone.utc)
            )
    
    def test_admin_override(self):
        """Test admin override functionality"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        override_result = self.gatekeeper.admin_override(
            recommendation_id=rec_id,
            admin_id="ADMIN_TEST",
            override_reason="Emergency override required due to time-sensitive clinical situation requiring immediate action"
        )
        
        assert override_result['status'] == 'overridden'
        assert override_result['recommendation_id'] == rec_id
        assert override_result['override_by'] == "ADMIN_TEST"
        assert 'warning' in override_result
    
    def test_admin_override_requires_detailed_reason(self):
        """Test that admin override requires detailed reason"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        with pytest.raises(ValueError, match="Override reason is required"):
            self.gatekeeper.admin_override(
                recommendation_id=rec_id,
                admin_id="ADMIN_TEST",
                override_reason="Emergency"  # Too short
            )
    
    def test_audit_chain_created(self):
        """Test that audit entries are created"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        self.gatekeeper.submit_recommendation(rec)
        
        # Check audit log file exists
        audit_file = self.audit_path / "audit_chain.jsonl"
        assert audit_file.exists()
        
        # Check audit entry was written
        with open(audit_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse and verify entry
            entry = json.loads(lines[-1])
            assert 'entry_id' in entry
            assert 'entry_hash' in entry
            assert 'previous_hash' in entry
            assert entry['event_type'] == 'recommendation_submitted'
    
    def test_audit_chain_verification(self):
        """Test audit chain integrity verification"""
        # Create several entries
        for i in range(3):
            rec = self.create_recommendation(RiskLevel.HIGH)
            self.gatekeeper.submit_recommendation(rec)
        
        # Verify chain
        verification = self.gatekeeper.verify_audit_chain()
        assert verification['verified']
        assert verification['entries'] >= 3
    
    def test_audit_chain_detects_tampering(self):
        """Test that audit chain detects tampering"""
        # Create an entry
        rec = self.create_recommendation(RiskLevel.HIGH)
        self.gatekeeper.submit_recommendation(rec)
        
        # Tamper with audit log
        audit_file = self.audit_path / "audit_chain.jsonl"
        with open(audit_file, 'r') as f:
            lines = f.readlines()
        
        # Modify an entry
        if lines:
            entry = json.loads(lines[0])
            entry['event_type'] = 'TAMPERED'
            lines[0] = json.dumps(entry) + '\n'
            
            with open(audit_file, 'w') as f:
                f.writelines(lines)
        
        # Verification should fail
        verification = self.gatekeeper.verify_audit_chain()
        assert not verification['verified']
        assert 'failed_at' in verification
    
    def test_get_pending_approvals(self):
        """Test retrieving pending approvals"""
        # Submit multiple high-risk recommendations
        rec1 = self.create_recommendation(RiskLevel.HIGH)
        rec2 = self.create_recommendation(RiskLevel.CRITICAL)
        
        self.gatekeeper.submit_recommendation(rec1)
        self.gatekeeper.submit_recommendation(rec2)
        
        pending = self.gatekeeper.get_pending_approvals()
        assert len(pending) == 2
    
    def test_get_audit_history(self):
        """Test retrieving audit history"""
        rec = self.create_recommendation(RiskLevel.HIGH)
        submit_result = self.gatekeeper.submit_recommendation(rec)
        rec_id = submit_result['recommendation_id']
        
        # Approve it
        self.gatekeeper.approve_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_TEST",
            clinician_role="Test Clinician",
            rationale="Approved after review of clinical data and patient history",
            review_start_time=datetime.now(timezone.utc)
        )
        
        # Get history for this recommendation
        history = self.gatekeeper.get_audit_history(recommendation_id=rec_id)
        assert len(history) >= 2  # Submit + Approve
        
        # Verify events
        events = [entry['event_type'] for entry in history]
        assert 'recommendation_approved' in events
        assert 'recommendation_submitted' in events


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_path = Path(self.temp_dir) / "e2e_audit"
        self.gatekeeper = HumanInLoopGatekeeper(
            audit_log_path=self.audit_path
        )
    
    def test_full_approval_workflow(self):
        """Test complete workflow: submit -> review -> approve"""
        # 1. System generates recommendation
        recommendation = ClinicalRecommendation(
            recommendation_id="",
            recommendation_type="medication_adjustment",
            recommendation_text="Recommend adjusting medication dosage based on lab results",
            risk_level=RiskLevel.HIGH,
            confidence_score=0.92,
            patient_id="SYNTH-E2E-001",
            generated_at=datetime.now(timezone.utc).isoformat(),
            generated_by="ClinicalAI_v3.0",
            context={'lab_results': 'abnormal', 'current_dosage': '10mg'}
        )
        
        # 2. Submit recommendation
        submit_result = self.gatekeeper.submit_recommendation(recommendation)
        assert submit_result['requires_human_approval']
        rec_id = submit_result['recommendation_id']
        
        # 3. Clinician reviews (simulated delay)
        review_start = datetime.now(timezone.utc) - timedelta(seconds=45)
        
        # 4. Clinician approves with rationale
        approval_result = self.gatekeeper.approve_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_SMITH_MD",
            clinician_role="Attending Physician",
            rationale="Reviewed patient's complete medical history, recent lab results showing elevated levels, and current medication regimen. Dosage adjustment is clinically appropriate and within safe parameters.",
            review_start_time=review_start
        )
        
        assert approval_result['status'] == 'approved'
        
        # 5. Verify audit trail
        history = self.gatekeeper.get_audit_history(recommendation_id=rec_id)
        assert len(history) == 2
        assert history[0]['event_type'] == 'recommendation_approved'
        assert history[0]['approval']['clinician_id'] == "DR_SMITH_MD"
        assert history[0]['approval']['review_duration_seconds'] > 0
        
        # 6. Verify audit chain integrity
        verification = self.gatekeeper.verify_audit_chain()
        assert verification['verified']
    
    def test_full_rejection_workflow(self):
        """Test complete workflow: submit -> review -> reject"""
        recommendation = ClinicalRecommendation(
            recommendation_id="",
            recommendation_type="surgical_intervention",
            recommendation_text="Recommend surgical intervention based on imaging findings",
            risk_level=RiskLevel.CRITICAL,
            confidence_score=0.78,
            patient_id="SYNTH-E2E-002",
            generated_at=datetime.now(timezone.utc).isoformat(),
            generated_by="SurgicalAI_v2.1",
            context={'imaging': 'MRI findings', 'symptoms': 'moderate'}
        )
        
        submit_result = self.gatekeeper.submit_recommendation(recommendation)
        rec_id = submit_result['recommendation_id']
        
        review_start = datetime.now(timezone.utc) - timedelta(minutes=2)
        
        rejection_result = self.gatekeeper.reject_recommendation(
            recommendation_id=rec_id,
            clinician_id="DR_JONES_MD",
            clinician_role="Chief of Surgery",
            rationale="After thorough review of imaging, patient's age, comorbidities, and overall health status, the risks of surgical intervention outweigh potential benefits. Recommend conservative management approach instead.",
            review_start_time=review_start
        )
        
        assert rejection_result['status'] == 'rejected'
        
        history = self.gatekeeper.get_audit_history(recommendation_id=rec_id)
        assert history[0]['event_type'] == 'recommendation_rejected'
        assert history[0]['approval']['status'] == 'rejected'
    
    def test_multiple_concurrent_approvals(self):
        """Test handling multiple pending approvals"""
        recommendations = []
        
        # Submit 3 high-risk recommendations
        for i in range(3):
            rec = ClinicalRecommendation(
                recommendation_id="",
                recommendation_type=f"test_type_{i}",
                recommendation_text=f"Test recommendation {i}",
                risk_level=RiskLevel.HIGH,
                confidence_score=0.85,
                patient_id=f"SYNTH-MULTI-{i:03d}",
                generated_at=datetime.now(timezone.utc).isoformat(),
                generated_by="TestSystem",
                context={'index': i}
            )
            result = self.gatekeeper.submit_recommendation(rec)
            recommendations.append(result['recommendation_id'])
        
        # All should be pending
        pending = self.gatekeeper.get_pending_approvals()
        assert len(pending) == 3
        
        # Approve first, reject second, leave third pending
        self.gatekeeper.approve_recommendation(
            recommendation_id=recommendations[0],
            clinician_id="DR_A",
            clinician_role="Clinician A",
            rationale="Approved after clinical review and consultation with patient care team",
            review_start_time=datetime.now(timezone.utc)
        )
        
        self.gatekeeper.reject_recommendation(
            recommendation_id=recommendations[1],
            clinician_id="DR_B",
            clinician_role="Clinician B",
            rationale="Rejected due to contraindications and patient preferences documented in care plan",
            review_start_time=datetime.now(timezone.utc)
        )
        
        # One should remain pending
        pending = self.gatekeeper.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]['recommendation_id'] == recommendations[2]
        
        # Verify audit chain
        verification = self.gatekeeper.verify_audit_chain()
        assert verification['verified']
        assert verification['entries'] >= 5  # 3 submits + 1 approve + 1 reject


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
