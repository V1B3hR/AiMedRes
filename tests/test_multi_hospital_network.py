"""
Test Suite for Multi-Hospital Network Launch (P12)

Tests for:
- Partnership management (≥25 institutions)
- Scale processing (10k+ concurrent cases)
- Regional network integration
- Outcome tracking & reporting dashboards
"""

import pytest
import time
from datetime import datetime, timedelta

from aimedres.clinical.multi_hospital_network import (
    MultiHospitalNetwork,
    create_multi_hospital_network,
    InstitutionType,
    PartnershipStatus,
    CaseStatus,
    Institution,
    ClinicalCase
)


class TestPartnershipManagement:
    """Tests for partnership management."""
    
    def test_network_initialization(self):
        """Test multi-hospital network initialization."""
        network = create_multi_hospital_network(
            max_institutions=50,
            default_capacity=500
        )
        
        assert network.max_institutions == 50
        assert network.default_capacity == 500
        assert len(network.institutions) == 0
    
    def test_add_institution(self):
        """Test adding healthcare institutions."""
        network = create_multi_hospital_network()
        
        institution = network.add_institution(
            name="Memorial Hospital",
            institution_type=InstitutionType.COMMUNITY_HOSPITAL,
            region="Northeast",
            capacity=500,
            specialties=["cardiology", "neurology"],
            contact_info={"phone": "555-1234"}
        )
        
        assert institution.name == "Memorial Hospital"
        assert institution.institution_type == InstitutionType.COMMUNITY_HOSPITAL
        assert institution.region == "Northeast"
        assert institution.capacity == 500
        assert institution.partnership_status == PartnershipStatus.PROSPECTIVE
        assert len(institution.specialties) == 2
    
    def test_add_25_institutions(self):
        """Test adding ≥25 institutions (P12 requirement)."""
        network = create_multi_hospital_network(max_institutions=50)
        
        regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
        types = list(InstitutionType)
        
        # Add 30 institutions
        for i in range(30):
            institution = network.add_institution(
                name=f"Hospital {i+1}",
                institution_type=types[i % len(types)],
                region=regions[i % len(regions)],
                capacity=400 + (i * 10)
            )
            assert institution is not None
        
        assert len(network.institutions) == 30
        assert len(network.institutions) >= 25  # P12 requirement
    
    def test_activate_institution(self):
        """Test activating institutions."""
        network = create_multi_hospital_network()
        
        institution = network.add_institution(
            name="Test Hospital",
            institution_type=InstitutionType.ACADEMIC_MEDICAL_CENTER,
            region="East"
        )
        
        assert institution.partnership_status == PartnershipStatus.PROSPECTIVE
        
        success = network.activate_institution(institution.institution_id)
        
        assert success is True
        updated = network.institutions[institution.institution_id]
        assert updated.partnership_status == PartnershipStatus.ACTIVE
        assert updated.onboarded_date is not None
    
    def test_get_active_institutions(self):
        """Test retrieving active institutions."""
        network = create_multi_hospital_network()
        
        # Add 5 institutions
        for i in range(5):
            inst = network.add_institution(
                name=f"Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region="Region1"
            )
            if i < 3:  # Activate first 3
                network.activate_institution(inst.institution_id)
        
        active = network.get_active_institutions()
        assert len(active) == 3
        assert all(inst.partnership_status == PartnershipStatus.ACTIVE for inst in active)
    
    def test_network_at_capacity(self):
        """Test network capacity limit."""
        network = create_multi_hospital_network(max_institutions=5)
        
        # Add 5 institutions
        for i in range(5):
            network.add_institution(
                name=f"Hospital {i}",
                institution_type=InstitutionType.CLINIC,
                region="Test"
            )
        
        # Try to add 6th institution
        with pytest.raises(ValueError, match="Network at capacity"):
            network.add_institution(
                name="Over Capacity Hospital",
                institution_type=InstitutionType.CLINIC,
                region="Test"
            )


class TestScaleProcessing:
    """Tests for scale processing (10k+ concurrent cases)."""
    
    def test_submit_case(self):
        """Test submitting clinical cases."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Test Hospital",
            institution_type=InstitutionType.COMMUNITY_HOSPITAL,
            region="East"
        )
        network.activate_institution(inst.institution_id)
        
        case = network.submit_case(
            institution_id=inst.institution_id,
            patient_id="patient_001",
            condition="diabetes",
            priority=2,
            metadata={"notes": "test case"}
        )
        
        assert case.case_id is not None
        assert case.institution_id == inst.institution_id
        assert case.patient_id == "patient_001"
        assert case.status == CaseStatus.QUEUED
        assert case.priority == 2
    
    def test_process_1000_cases(self):
        """Test processing 1000 cases."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Large Hospital",
            institution_type=InstitutionType.ACADEMIC_MEDICAL_CENTER,
            region="Central",
            capacity=1500
        )
        network.activate_institution(inst.institution_id)
        
        # Submit 1000 cases
        case_ids = []
        for i in range(1000):
            case = network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i:04d}",
                condition="alzheimer" if i % 2 == 0 else "cardiovascular",
                priority=(i % 5) + 1
            )
            case_ids.append(case.case_id)
        
        assert len(case_ids) == 1000
        stats = network.get_processing_stats()
        assert stats["total_cases"] == 1000
        assert stats["queued"] == 1000
    
    def test_process_10000_cases_benchmark(self):
        """Test processing 10k+ concurrent cases (P12 requirement)."""
        network = create_multi_hospital_network()
        
        # Add multiple institutions
        institutions = []
        for i in range(5):
            inst = network.add_institution(
                name=f"Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region=f"Region{i}",
                capacity=3000
            )
            network.activate_institution(inst.institution_id)
            institutions.append(inst)
        
        # Submit 10,000 cases
        start_time = time.time()
        for i in range(10000):
            inst = institutions[i % len(institutions)]
            network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i:05d}",
                condition="condition_" + str(i % 10),
                priority=(i % 5) + 1
            )
        submission_time = time.time() - start_time
        
        stats = network.get_processing_stats()
        assert stats["total_cases"] == 10000
        assert stats["queued"] == 10000
        
        # Should complete in reasonable time (< 5 seconds)
        assert submission_time < 5.0
        
        print(f"\n10k cases submitted in {submission_time:.2f}s")
        print(f"Throughput: {10000/submission_time:.0f} cases/sec")
    
    def test_batch_processing(self):
        """Test batch processing of cases."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Batch Hospital",
            institution_type=InstitutionType.COMMUNITY_HOSPITAL,
            region="West",
            capacity=1000
        )
        network.activate_institution(inst.institution_id)
        
        # Submit 500 cases
        for i in range(500):
            network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test_condition",
                priority=3
            )
        
        # Process batch
        result = network.process_batch(batch_size=100, simulate_processing=True)
        
        assert result["processed"] == 100
        assert result["successful"] >= 90  # ~95% success rate
        assert result["failed"] <= 10
    
    def test_case_lifecycle(self):
        """Test complete case lifecycle."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Lifecycle Hospital",
            institution_type=InstitutionType.CLINIC,
            region="North"
        )
        network.activate_institution(inst.institution_id)
        
        # Submit case
        case = network.submit_case(
            institution_id=inst.institution_id,
            patient_id="patient_lifecycle",
            condition="test",
            priority=2
        )
        assert case.status == CaseStatus.QUEUED
        
        # Process case
        success = network.process_case(case.case_id)
        assert success is True
        
        updated_case = network.get_case_status(case.case_id)
        assert updated_case.status == CaseStatus.PROCESSING
        
        # Complete case
        outcome = {"diagnosis": "confirmed", "risk_score": 0.65}
        success = network.complete_case(case.case_id, outcome, success=True)
        assert success is True
        
        final_case = network.get_case_status(case.case_id)
        assert final_case.status == CaseStatus.COMPLETED
        assert final_case.outcome == outcome
        assert final_case.processing_time_ms is not None
    
    def test_institution_capacity_tracking(self):
        """Test institution capacity tracking."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Capacity Hospital",
            institution_type=InstitutionType.COMMUNITY_HOSPITAL,
            region="South",
            capacity=100
        )
        network.activate_institution(inst.institution_id)
        
        # Submit 50 cases
        for i in range(50):
            network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test",
                priority=3
            )
        
        capacity_info = network.get_institution_capacity(inst.institution_id)
        
        assert capacity_info["total_capacity"] == 100
        assert capacity_info["current_load"] == 50
        assert capacity_info["available"] == 50
        assert capacity_info["utilization"] == 0.5


class TestRegionalNetworkIntegration:
    """Tests for regional network integration."""
    
    def test_get_institutions_by_region(self):
        """Test retrieving institutions by region."""
        network = create_multi_hospital_network()
        
        # Add institutions in different regions
        regions = ["Northeast", "Southeast", "Northeast", "Midwest", "Northeast"]
        for i, region in enumerate(regions):
            network.add_institution(
                name=f"Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region=region
            )
        
        northeast_insts = network.get_institutions_by_region("Northeast")
        assert len(northeast_insts) == 3
        
        southeast_insts = network.get_institutions_by_region("Southeast")
        assert len(southeast_insts) == 1
    
    def test_regional_network_status(self):
        """Test regional network status reporting."""
        network = create_multi_hospital_network()
        
        # Add institutions in a region
        for i in range(5):
            inst = network.add_institution(
                name=f"West Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region="West",
                capacity=500
            )
            if i < 3:
                network.activate_institution(inst.institution_id)
        
        regional_status = network.get_regional_network_status("West")
        
        assert regional_status["region"] == "West"
        assert regional_status["total_institutions"] == 5
        assert regional_status["active_institutions"] == 3
        assert regional_status["total_capacity"] == 2500
    
    def test_get_all_regions(self):
        """Test retrieving all regions."""
        network = create_multi_hospital_network()
        
        regions = ["North", "South", "East", "West"]
        for region in regions:
            network.add_institution(
                name=f"{region} Hospital",
                institution_type=InstitutionType.CLINIC,
                region=region
            )
        
        all_regions = network.get_all_regions()
        assert len(all_regions) == 4
        assert set(all_regions) == set(regions)


class TestOutcomeTracking:
    """Tests for outcome tracking & reporting dashboards."""
    
    def test_calculate_outcome_metrics(self):
        """Test outcome metrics calculation."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Metrics Hospital",
            institution_type=InstitutionType.ACADEMIC_MEDICAL_CENTER,
            region="Central"
        )
        network.activate_institution(inst.institution_id)
        
        # Submit and process cases
        for i in range(20):
            case = network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test_condition",
                priority=2
            )
            network.process_case(case.case_id)
            network.complete_case(
                case.case_id,
                {"outcome": "success"},
                success=(i % 10 != 0)  # 90% success rate
            )
        
        metrics = network.calculate_outcome_metrics(inst.institution_id, period_hours=24)
        
        assert metrics.institution_id == inst.institution_id
        assert metrics.total_cases == 20
        assert metrics.completed_cases >= 15
        assert metrics.avg_processing_time_ms >= 0
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.clinical_agreement_rate <= 1
    
    def test_network_dashboard(self):
        """Test comprehensive network dashboard."""
        network = create_multi_hospital_network()
        
        # Set up network with multiple institutions
        for i in range(5):
            inst = network.add_institution(
                name=f"Dashboard Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region=f"Region{i % 3}",
                capacity=500
            )
            network.activate_institution(inst.institution_id)
            
            # Add some cases
            for j in range(10):
                network.submit_case(
                    institution_id=inst.institution_id,
                    patient_id=f"patient_{i}_{j}",
                    condition="test",
                    priority=2
                )
        
        dashboard = network.get_network_dashboard()
        
        assert "network_status" in dashboard
        assert "processing_stats" in dashboard
        assert "active_institutions" in dashboard
        assert "regional_summary" in dashboard
        
        network_status = dashboard["network_status"]
        assert network_status["total_institutions"] == 5
        assert network_status["active_institutions"] == 5
        assert network_status["total_capacity"] == 2500
        assert network_status["cases_queued"] == 50
    
    def test_export_metrics_report_json(self):
        """Test exporting metrics report in JSON format."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Report Hospital",
            institution_type=InstitutionType.CLINIC,
            region="Test"
        )
        network.activate_institution(inst.institution_id)
        
        # Add some cases
        for i in range(10):
            case = network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test",
                priority=2
            )
            network.process_case(case.case_id)
            network.complete_case(case.case_id, {"success": True}, success=True)
        
        report = network.export_metrics_report(
            institution_id=inst.institution_id,
            format="json"
        )
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Verify it's valid JSON
        import json
        data = json.loads(report)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "institution_id" in data[0]
        assert "metrics" in data[0]
    
    def test_export_metrics_report_summary(self):
        """Test exporting metrics report in summary format."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Summary Hospital",
            institution_type=InstitutionType.COMMUNITY_HOSPITAL,
            region="Test"
        )
        network.activate_institution(inst.institution_id)
        
        # Add some cases
        for i in range(5):
            case = network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test",
                priority=2
            )
            network.process_case(case.case_id)
            network.complete_case(case.case_id, {"success": True}, success=True)
        
        report = network.export_metrics_report(
            institution_id=inst.institution_id,
            format="summary"
        )
        
        assert isinstance(report, str)
        assert "Multi-Hospital Network Metrics Report" in report
        assert "Summary Hospital" in report
        assert "Total Cases" in report


class TestPerformance:
    """Performance and scalability tests."""
    
    def test_high_throughput_submission(self):
        """Test high-throughput case submission."""
        network = create_multi_hospital_network()
        
        inst = network.add_institution(
            name="Throughput Hospital",
            institution_type=InstitutionType.ACADEMIC_MEDICAL_CENTER,
            region="Central",
            capacity=5000
        )
        network.activate_institution(inst.institution_id)
        
        # Submit 5000 cases and measure time
        start_time = time.time()
        for i in range(5000):
            network.submit_case(
                institution_id=inst.institution_id,
                patient_id=f"patient_{i}",
                condition="test",
                priority=3
            )
        elapsed = time.time() - start_time
        
        throughput = 5000 / elapsed
        print(f"\nThroughput: {throughput:.0f} submissions/sec")
        
        # Should handle at least 1000 submissions per second
        assert throughput > 1000
    
    def test_concurrent_institution_operations(self):
        """Test concurrent operations across institutions."""
        import threading
        
        network = create_multi_hospital_network()
        
        # Add 10 institutions
        institutions = []
        for i in range(10):
            inst = network.add_institution(
                name=f"Concurrent Hospital {i}",
                institution_type=InstitutionType.COMMUNITY_HOSPITAL,
                region=f"Region{i % 3}",
                capacity=1000
            )
            network.activate_institution(inst.institution_id)
            institutions.append(inst)
        
        # Submit cases from multiple threads
        def submit_cases(inst_id, count):
            for i in range(count):
                network.submit_case(
                    institution_id=inst_id,
                    patient_id=f"patient_{inst_id}_{i}",
                    condition="test",
                    priority=2
                )
        
        threads = []
        for inst in institutions:
            thread = threading.Thread(
                target=submit_cases,
                args=(inst.institution_id, 100)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        stats = network.get_processing_stats()
        assert stats["total_cases"] == 1000  # 10 institutions * 100 cases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
