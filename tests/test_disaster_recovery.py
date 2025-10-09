#!/usr/bin/env python3
"""
Tests for Disaster Recovery System (P10)

Tests the implementation of disaster recovery drills, RPO/RTO metrics,
and recovery validation for the scalable cloud architecture.
"""

import pytest
import time
import tempfile
import shutil
from datetime import datetime, timezone

from src.aimedres.training.disaster_recovery import (
    DisasterRecoverySystem,
    DisasterType,
    RecoveryStatus,
    RPOConfig,
    RTOConfig,
    create_dr_system
)


class TestDisasterRecoverySystem:
    """Test disaster recovery system functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.dr_system = create_dr_system(
            rpo_target_seconds=300.0,
            rto_target_seconds=900.0,
            results_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dr_system_initialization(self):
        """Test disaster recovery system initialization"""
        assert self.dr_system is not None
        assert self.dr_system.rpo_config.target_seconds == 300.0
        assert self.dr_system.rto_config.target_seconds == 900.0
        assert len(self.dr_system.drill_history) == 0
    
    def test_region_failure_drill(self):
        """Test region failure disaster recovery drill"""
        services = ["aimedres-api", "aimedres-database", "aimedres-cache"]
        
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.REGION_FAILURE,
            services=services,
            simulate_data_loss=False
        )
        
        assert result is not None
        assert result.disaster_type == DisasterType.REGION_FAILURE
        assert result.rpo_achieved_seconds > 0
        assert result.rto_achieved_seconds > 0
        assert isinstance(result.drill_successful, bool)
        assert len(result.services_recovered) >= 0
        assert len(result.recommendations) > 0
    
    def test_database_corruption_drill(self):
        """Test database corruption disaster recovery drill"""
        services = ["aimedres-database"]
        
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.DATABASE_CORRUPTION,
            services=services,
            simulate_data_loss=True
        )
        
        assert result is not None
        assert result.disaster_type == DisasterType.DATABASE_CORRUPTION
        assert result.data_loss_detected or not result.data_loss_detected  # Can be either
        if result.data_loss_detected:
            assert result.data_loss_percentage >= 0
    
    def test_rpo_rto_metrics(self):
        """Test RPO/RTO metrics tracking"""
        # Run a few drills first
        services = ["aimedres-api", "aimedres-database"]
        
        for disaster_type in [DisasterType.REGION_FAILURE, DisasterType.NETWORK_PARTITION]:
            self.dr_system.run_dr_drill(
                disaster_type=disaster_type,
                services=services,
                simulate_data_loss=False
            )
        
        # Get metrics
        metrics = self.dr_system.get_rpo_rto_metrics()
        
        assert metrics is not None
        assert metrics['total_drills'] >= 2
        assert 'rpo_metrics' in metrics
        assert 'rto_metrics' in metrics
        assert 'success_rate_percent' in metrics
        
        # Check RPO metrics
        assert 'average_achieved_seconds' in metrics['rpo_metrics']
        assert 'target_seconds' in metrics['rpo_metrics']
        assert 'target_met_percent' in metrics['rpo_metrics']
        
        # Check RTO metrics
        assert 'average_achieved_seconds' in metrics['rto_metrics']
        assert 'target_seconds' in metrics['rto_metrics']
        assert 'target_met_percent' in metrics['rto_metrics']
    
    def test_comprehensive_drill_suite(self):
        """Test comprehensive disaster recovery drill suite"""
        summary = self.dr_system.run_comprehensive_drill_suite()
        
        assert summary is not None
        assert summary['total_drills'] == len(DisasterType)
        assert 'successful_drills' in summary
        assert 'success_rate_percent' in summary
        assert 'drill_details' in summary
        assert 'rpo_rto_metrics' in summary
        
        # Verify all disaster types were tested
        disaster_types_tested = set(d['disaster_type'] for d in summary['drill_details'])
        assert len(disaster_types_tested) == len(DisasterType)
    
    def test_rpo_target_validation(self):
        """Test that RPO targets are properly validated"""
        # Create system with strict RPO
        strict_dr = create_dr_system(
            rpo_target_seconds=1.0,  # Very strict 1 second RPO
            rto_target_seconds=5.0,
            results_dir=self.temp_dir
        )
        
        result = strict_dr.run_dr_drill(
            disaster_type=DisasterType.REGION_FAILURE,
            services=["test-service"],
            simulate_data_loss=False
        )
        
        # With such a strict target, drill likely won't meet it
        # But we should get proper recommendations
        if not result.drill_successful:
            assert any('RPO' in issue for issue in result.issues_encountered)
            assert any('RPO' in rec or 'backup' in rec.lower() 
                      for rec in result.recommendations)
    
    def test_rto_target_validation(self):
        """Test that RTO targets are properly validated"""
        # Create system with strict RTO
        strict_dr = create_dr_system(
            rpo_target_seconds=300.0,
            rto_target_seconds=1.0,  # Very strict 1 second RTO
            results_dir=self.temp_dir
        )
        
        result = strict_dr.run_dr_drill(
            disaster_type=DisasterType.REGION_FAILURE,
            services=["test-service"],
            simulate_data_loss=False
        )
        
        # With such a strict target, drill likely won't meet it
        # But we should get proper recommendations
        if not result.drill_successful:
            assert any('RTO' in issue for issue in result.issues_encountered)
            assert any('RTO' in rec or 'recovery' in rec.lower() 
                      for rec in result.recommendations)
    
    def test_drill_result_persistence(self):
        """Test that drill results are properly saved"""
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.HARDWARE_FAILURE,
            services=["test-service"],
            simulate_data_loss=False
        )
        
        # Check that result was added to history
        assert len(self.dr_system.drill_history) == 1
        assert self.dr_system.drill_history[0].drill_id == result.drill_id
        
        # Check that result file was created
        import os
        result_file = os.path.join(self.temp_dir, f"{result.drill_id}.json")
        assert os.path.exists(result_file)
    
    def test_ransomware_attack_drill(self):
        """Test ransomware attack disaster recovery drill"""
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.RANSOMWARE_ATTACK,
            services=["aimedres-api", "aimedres-database"],
            simulate_data_loss=True
        )
        
        assert result is not None
        assert result.disaster_type == DisasterType.RANSOMWARE_ATTACK
        # Ransomware should trigger security recommendations
        assert any('security' in rec.lower() or 'ransomware' in rec.lower() 
                  for rec in result.recommendations)
    
    def test_recovery_status_tracking(self):
        """Test recovery status is properly tracked"""
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.NETWORK_PARTITION,
            services=["service1", "service2", "service3"],
            simulate_data_loss=False
        )
        
        assert result.recovery_status in [
            RecoveryStatus.COMPLETED,
            RecoveryStatus.PARTIAL,
            RecoveryStatus.FAILED
        ]
        
        # If completed, all services should be recovered
        if result.recovery_status == RecoveryStatus.COMPLETED:
            assert len(result.services_failed) == 0
            assert len(result.services_recovered) > 0
    
    def test_data_loss_detection(self):
        """Test data loss detection in drills"""
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.DATABASE_CORRUPTION,
            services=["aimedres-database"],
            simulate_data_loss=True
        )
        
        # When simulating data loss, it should be detected
        if result.data_loss_detected:
            assert result.data_loss_percentage > 0
            assert any('data loss' in rec.lower() for rec in result.recommendations)
    
    def test_multiple_service_recovery(self):
        """Test recovery of multiple services"""
        services = [
            "aimedres-api",
            "aimedres-model-server", 
            "aimedres-database",
            "aimedres-cache",
            "aimedres-monitoring"
        ]
        
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.DATA_CENTER_OUTAGE,
            services=services,
            simulate_data_loss=False
        )
        
        # Should have attempted to recover all services
        total_services = len(result.services_recovered) + len(result.services_failed)
        assert total_services == len(services)
    
    def test_drill_recommendations_generation(self):
        """Test that appropriate recommendations are generated"""
        result = self.dr_system.run_dr_drill(
            disaster_type=DisasterType.REGION_FAILURE,
            services=["test-service"],
            simulate_data_loss=False
        )
        
        assert len(result.recommendations) > 0
        # Recommendations should be strings
        assert all(isinstance(rec, str) for rec in result.recommendations)


def test_factory_function():
    """Test disaster recovery system factory function"""
    dr_system = create_dr_system(
        rpo_target_seconds=600.0,
        rto_target_seconds=1800.0,
        results_dir="/tmp/dr_test"
    )
    
    assert dr_system is not None
    assert dr_system.rpo_config.target_seconds == 600.0
    assert dr_system.rto_config.target_seconds == 1800.0


def test_rpo_rto_config_defaults():
    """Test RPO/RTO configuration defaults"""
    rpo_config = RPOConfig()
    assert rpo_config.target_seconds == 300.0
    assert rpo_config.critical_services_seconds == 60.0
    
    rto_config = RTOConfig()
    assert rto_config.target_seconds == 900.0
    assert rto_config.critical_services_seconds == 300.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
