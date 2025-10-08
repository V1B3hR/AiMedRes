#!/usr/bin/env python3
"""
Tests for Enhanced Adversarial Robustness and Human Oversight (P11)

Tests the improvements to:
1. Adversarial robustness (target: ≥0.8 from 0.5)
2. Human oversight and override audit workflow (complete from 66.7%)
"""

import pytest
import time
from datetime import datetime, timezone, timedelta

from security.ai_safety import (
    ClinicalAISafetyMonitor,
    RiskLevel,
    SafetyAction,
    SafetyThresholds
)


class TestEnhancedAdversarialRobustness:
    """Test enhanced adversarial robustness improvements"""
    
    def setup_method(self):
        """Set up test environment"""
        self.safety_monitor = ClinicalAISafetyMonitor()
    
    def test_adversarial_robustness_score_improved(self):
        """Test that adversarial robustness score meets ≥0.8 target"""
        # Generate and run standard adversarial tests
        test_cases = self.safety_monitor.generate_standard_adversarial_tests()
        
        assert len(test_cases) >= 10, "Should have comprehensive test cases"
        
        results = self.safety_monitor.run_adversarial_tests(test_cases)
        
        assert results is not None
        assert 'robustness_score' in results
        assert 'total_tests' in results
        assert 'passed_tests' in results
        
        # Main assertion: robustness score should be ≥0.8
        assert results['robustness_score'] >= 0.8, \
            f"Robustness score {results['robustness_score']:.2f} should be ≥0.8"
        
        print(f"✓ Adversarial robustness score: {results['robustness_score']:.2f} (target: ≥0.8)")
    
    def test_input_perturbation_robustness(self):
        """Test robustness against input perturbations"""
        test_cases = [
            tc for tc in self.safety_monitor.generate_standard_adversarial_tests()
            if tc['type'] == 'input_perturbation'
        ]
        
        results = self.safety_monitor.run_adversarial_tests(test_cases)
        
        # Should pass most perturbation tests
        perturbation_score = results['passed_tests'] / results['total_tests']
        assert perturbation_score >= 0.75, \
            f"Input perturbation robustness {perturbation_score:.2f} should be ≥0.75"
    
    def test_boundary_condition_handling(self):
        """Test handling of boundary conditions"""
        test_cases = [
            tc for tc in self.safety_monitor.generate_standard_adversarial_tests()
            if tc['type'] == 'boundary_condition'
        ]
        
        results = self.safety_monitor.run_adversarial_tests(test_cases)
        
        # Should handle boundary conditions well
        boundary_score = results['passed_tests'] / results['total_tests']
        assert boundary_score >= 0.75, \
            f"Boundary condition handling {boundary_score:.2f} should be ≥0.75"
    
    def test_demographic_fairness_improved(self):
        """Test improved demographic fairness"""
        test_cases = [
            tc for tc in self.safety_monitor.generate_standard_adversarial_tests()
            if tc['type'] == 'demographic_fairness'
        ]
        
        results = self.safety_monitor.run_adversarial_tests(test_cases)
        
        # Should show good fairness across demographics
        fairness_score = results['passed_tests'] / results['total_tests']
        assert fairness_score >= 0.75, \
            f"Demographic fairness score {fairness_score:.2f} should be ≥0.75"
    
    def test_comprehensive_test_coverage(self):
        """Test that we have comprehensive adversarial test coverage"""
        test_cases = self.safety_monitor.generate_standard_adversarial_tests()
        
        # Count test types
        test_types = {}
        for tc in test_cases:
            test_type = tc['type']
            test_types[test_type] = test_types.get(test_type, 0) + 1
        
        # Should have all three types
        assert 'input_perturbation' in test_types
        assert 'boundary_condition' in test_types
        assert 'demographic_fairness' in test_types
        
        # Should have multiple tests of each type
        assert test_types['input_perturbation'] >= 3
        assert test_types['boundary_condition'] >= 3
        assert test_types['demographic_fairness'] >= 3
    
    def test_vulnerability_detection_and_reporting(self):
        """Test vulnerability detection and reporting"""
        test_cases = self.safety_monitor.generate_standard_adversarial_tests()
        results = self.safety_monitor.run_adversarial_tests(test_cases)
        
        assert 'vulnerabilities_detected' in results
        assert isinstance(results['vulnerabilities_detected'], list)
        
        # If robustness is high, vulnerabilities should be minimal
        if results['robustness_score'] >= 0.8:
            assert len(results['vulnerabilities_detected']) <= 3, \
                "High robustness score should have few vulnerabilities"


class TestCompleteHumanOversight:
    """Test complete human oversight and override audit workflow"""
    
    def setup_method(self):
        """Set up test environment"""
        self.safety_monitor = ClinicalAISafetyMonitor()
        self.oversight_requests = []
        
        def oversight_callback(request):
            self.oversight_requests.append(request)
        
        self.safety_monitor.add_human_oversight_callback(oversight_callback)
    
    def test_complete_oversight_workflow(self):
        """Test complete human oversight workflow from request to audit"""
        # Create high-risk decision requiring oversight
        clinical_context = {
            'patient_age': 75,
            'primary_condition': 'heart_failure',
            'condition_severity': 'CRITICAL'
        }
        
        ai_recommendation = {
            'primary_recommendation': 'high_risk_medication',
            'treatment_type': 'HIGH_RISK_MEDICATION'
        }
        
        # Step 1: AI decision triggers oversight
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor123',
            patient_id='patient456',
            clinical_context=clinical_context,
            ai_recommendation=ai_recommendation,
            confidence_score=0.82
        )
        
        time.sleep(0.1)  # Allow callbacks to execute
        
        # Verify oversight was requested
        assert len(self.oversight_requests) > 0
        assert decision.human_oversight_required == True
        
        # Step 2: Submit human decision
        human_decision = {
            'final_decision': 'APPROVED_WITH_MODIFICATIONS',
            'modifications': 'Reduce dosage by 50%',
            'rationale': 'Patient age and comorbidities warrant caution'
        }
        
        success = self.safety_monitor.submit_human_decision(
            decision_id=decision.decision_id,
            reviewer_id='senior_physician_123',
            human_decision=human_decision,
            safety_notes='Approved with reduced risk modifications'
        )
        
        assert success == True
        assert decision.human_decision == human_decision
        assert decision.final_outcome == 'APPROVED_WITH_MODIFICATIONS'
        assert decision.reviewer_id == 'senior_physician_123'
        
        print("✓ Complete oversight workflow: Request → Review → Decision → Audit")
    
    def test_oversight_audit_report_generation(self):
        """Test comprehensive oversight audit report generation"""
        # Create multiple oversight decisions
        for i in range(5):
            clinical_context = {
                'patient_age': 70 + i,
                'primary_condition': 'critical_condition',
                'condition_severity': 'HIGH'
            }
            
            decision = self.safety_monitor.assess_ai_decision(
                model_version='v1.0',
                user_id=f'doctor{i}',
                patient_id=f'patient{i}',
                clinical_context=clinical_context,
                ai_recommendation={'primary_recommendation': 'treatment'},
                confidence_score=0.85
            )
            
            if decision.human_oversight_required:
                self.safety_monitor.submit_human_decision(
                    decision_id=decision.decision_id,
                    reviewer_id=f'reviewer{i % 2}',  # 2 reviewers
                    human_decision={
                        'final_decision': 'APPROVED' if i % 2 == 0 else 'MODIFIED',
                        'rationale': f'Review {i}'
                    },
                    safety_notes='Test review'
                )
        
        # Generate audit report
        report = self.safety_monitor.get_oversight_audit_report(hours_back=1)
        
        assert report is not None
        assert 'overview' in report
        assert 'override_outcomes' in report
        assert 'reviewer_performance' in report
        assert 'risk_level_breakdown' in report
        assert 'audit_trail_status' in report
        assert 'workflow_completion_percent' in report
        
        # Verify workflow completion
        assert report['workflow_completion_percent'] == 100.0, \
            "Human oversight workflow should be 100% complete"
        
        assert report['audit_trail_status'] == 'COMPLETE', \
            "Audit trail should be complete"
        
        print(f"✓ Oversight audit report generated successfully")
        print(f"  - Total oversight required: {report['overview']['total_oversight_required']}")
        print(f"  - Completion rate: {report['overview']['completion_rate_percent']:.1f}%")
        print(f"  - Workflow completion: {report['workflow_completion_percent']}%")
    
    def test_oversight_export_functionality(self):
        """Test oversight decision export for compliance"""
        # Create oversight decision
        decision = self.safety_monitor.assess_ai_decision(
            model_version='v1.0',
            user_id='doctor_test',
            patient_id='patient_test',
            clinical_context={'patient_age': 80, 'condition_severity': 'HIGH'},
            ai_recommendation={'primary_recommendation': 'treatment'},
            confidence_score=0.83
        )
        
        if decision.human_oversight_required:
            self.safety_monitor.submit_human_decision(
                decision_id=decision.decision_id,
                reviewer_id='reviewer_test',
                human_decision={'final_decision': 'APPROVED'},
                safety_notes='Test'
            )
        
        # Export decisions
        export_data = self.safety_monitor.export_oversight_decisions(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
            output_format='summary'
        )
        
        assert export_data is not None
        assert 'export_timestamp' in export_data
        assert 'total_decisions' in export_data
        assert 'decisions' in export_data
        assert isinstance(export_data['decisions'], list)
        
        # Also test JSON export
        json_export = self.safety_monitor.export_oversight_decisions(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc),
            output_format='json'
        )
        
        assert isinstance(json_export, str)
        assert len(json_export) > 0
        
        print("✓ Oversight decision export functionality working")
    
    def test_reviewer_performance_tracking(self):
        """Test reviewer performance tracking in audit reports"""
        # Create decisions with multiple reviewers
        reviewers = ['reviewer_a', 'reviewer_b', 'reviewer_c']
        
        for i in range(6):
            decision = self.safety_monitor.assess_ai_decision(
                model_version='v1.0',
                user_id=f'doctor{i}',
                patient_id=f'patient{i}',
                clinical_context={'patient_age': 75, 'condition_severity': 'HIGH'},
                ai_recommendation={'primary_recommendation': 'treatment'},
                confidence_score=0.80
            )
            
            if decision.human_oversight_required:
                self.safety_monitor.submit_human_decision(
                    decision_id=decision.decision_id,
                    reviewer_id=reviewers[i % 3],
                    human_decision={'final_decision': 'APPROVED'},
                    safety_notes='Test'
                )
        
        report = self.safety_monitor.get_oversight_audit_report(hours_back=1)
        
        assert 'reviewer_performance' in report
        reviewer_perf = report['reviewer_performance']
        
        # Should track performance for each reviewer
        for reviewer in reviewers:
            if reviewer in reviewer_perf:
                assert 'total_reviews' in reviewer_perf[reviewer]
                assert 'outcomes' in reviewer_perf[reviewer]
        
        print("✓ Reviewer performance tracking operational")
    
    def test_ai_human_agreement_tracking(self):
        """Test AI-human agreement rate tracking"""
        # Create decisions with varying agreement
        for i in range(4):
            decision = self.safety_monitor.assess_ai_decision(
                model_version='v1.0',
                user_id=f'doctor{i}',
                patient_id=f'patient{i}',
                clinical_context={'patient_age': 70, 'condition_severity': 'MODERATE'},
                ai_recommendation={'primary_recommendation': 'recommend_treatment'},
                confidence_score=0.85
            )
            
            if decision.human_oversight_required:
                # Alternate between agreement and disagreement
                final_decision = 'APPROVED' if i % 2 == 0 else 'REJECTED'
                self.safety_monitor.submit_human_decision(
                    decision_id=decision.decision_id,
                    reviewer_id='reviewer_test',
                    human_decision={'final_decision': final_decision},
                    safety_notes='Test'
                )
        
        report = self.safety_monitor.get_oversight_audit_report(hours_back=1)
        
        assert 'overview' in report
        assert 'ai_human_agreement_rate_percent' in report['overview']
        
        agreement_rate = report['overview']['ai_human_agreement_rate_percent']
        assert 0 <= agreement_rate <= 100
        
        print(f"✓ AI-human agreement tracking: {agreement_rate:.1f}%")
    
    def test_pending_reviews_tracking(self):
        """Test pending reviews are properly tracked"""
        # Create decisions without completing them
        for i in range(3):
            decision = self.safety_monitor.assess_ai_decision(
                model_version='v1.0',
                user_id=f'doctor{i}',
                patient_id=f'patient{i}',
                clinical_context={'patient_age': 80, 'condition_severity': 'HIGH'},
                ai_recommendation={'primary_recommendation': 'treatment'},
                confidence_score=0.75
            )
        
        report = self.safety_monitor.get_oversight_audit_report(hours_back=1)
        
        assert 'overview' in report
        assert 'pending_oversight' in report['overview']
        assert 'pending_reviews_detail' in report
        
        # Should show pending reviews
        pending_count = report['overview']['pending_oversight']
        if pending_count > 0:
            assert len(report['pending_reviews_detail']) > 0
            # Check detail structure
            for detail in report['pending_reviews_detail']:
                assert 'decision_id' in detail
                assert 'risk_level' in detail
                assert 'pending_since' in detail
                assert 'age_hours' in detail
        
        print(f"✓ Pending reviews tracking: {pending_count} pending")


def test_combined_p11_improvements():
    """Test that both P11 improvements work together"""
    safety_monitor = ClinicalAISafetyMonitor()
    
    # Test 1: Adversarial robustness
    test_cases = safety_monitor.generate_standard_adversarial_tests()
    adv_results = safety_monitor.run_adversarial_tests(test_cases)
    
    assert adv_results['robustness_score'] >= 0.8, \
        "P11 objective 1: Adversarial robustness ≥0.8"
    
    # Test 2: Human oversight workflow
    decision = safety_monitor.assess_ai_decision(
        model_version='v1.0',
        user_id='test_doctor',
        patient_id='test_patient',
        clinical_context={'patient_age': 85, 'condition_severity': 'CRITICAL'},
        ai_recommendation={'primary_recommendation': 'high_risk_treatment'},
        confidence_score=0.80
    )
    
    if decision.human_oversight_required:
        safety_monitor.submit_human_decision(
            decision_id=decision.decision_id,
            reviewer_id='test_reviewer',
            human_decision={'final_decision': 'APPROVED'},
            safety_notes='Test'
        )
    
    report = safety_monitor.get_oversight_audit_report(hours_back=1)
    
    assert report['workflow_completion_percent'] == 100.0, \
        "P11 objective 2: Human oversight workflow 100% complete"
    
    print("✓ All P11 objectives achieved:")
    print(f"  1. Adversarial robustness: {adv_results['robustness_score']:.2f} (target: ≥0.8)")
    print(f"  2. Oversight workflow: {report['workflow_completion_percent']}% complete (target: 100%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
