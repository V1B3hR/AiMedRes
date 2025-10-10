#!/usr/bin/env python3
"""
Demonstration script for P8B and P9 Roadmap Implementations

This script demonstrates the functionality of:
- P8B: Clinical Pilot Programs
- P9: FDA Regulatory Pathway Planning

Shows complete workflows for institutional partnerships, validation studies,
device classification, and QMS documentation.
"""

import sys
import os
import json
import logging

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aimedres.clinical.clinical_pilot_programs import (
    create_clinical_pilot_manager,
    PartnershipStatus,
    ValidationStudyPhase
)
from aimedres.compliance.fda_pathway_planning import (
    create_fda_pathway_planner,
    EvidenceType
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_p8b():
    """Demonstrate P8B: Clinical Pilot Programs"""
    print("\n" + "="*70)
    print("P8B: CLINICAL PILOT PROGRAMS DEMONSTRATION")
    print("="*70)
    
    # Create pilot manager
    print("\n📋 Creating Clinical Pilot Manager...")
    manager = create_clinical_pilot_manager()
    
    # Create institutional partnerships
    print("\n🏥 Creating Institutional Partnerships...")
    partnerships = []
    
    partnership_data = [
        {
            'name': 'Memorial Medical Center',
            'contact': 'Dr. Sarah Johnson',
            'email': 'sjohnson@mmc.org',
            'cases': 300,
            'specialties': ['Neurology', 'Cardiology']
        },
        {
            'name': 'University Hospital',
            'contact': 'Dr. Michael Chen',
            'email': 'mchen@univhosp.edu',
            'cases': 400,
            'specialties': ['Neurology', 'Geriatrics']
        },
        {
            'name': 'Regional Health Network',
            'contact': 'Dr. Emily Rodriguez',
            'email': 'erodriguez@rhn.org',
            'cases': 300,
            'specialties': ['Neurology', 'Primary Care']
        }
    ]
    
    for p_data in partnership_data:
        partnership = manager.create_partnership(
            institution_name=p_data['name'],
            contact_person=p_data['contact'],
            contact_email=p_data['email'],
            target_case_count=p_data['cases'],
            specialties=p_data['specialties']
        )
        manager.activate_partnership(partnership.partnership_id)
        partnerships.append(partnership)
        print(f"  ✓ {p_data['name']} - Target: {p_data['cases']} cases")
    
    # Create validation studies
    print("\n🔬 Creating Validation Studies...")
    studies = []
    
    for partnership in partnerships:
        study = manager.create_validation_study(
            study_name=f"{partnership.institution_name} Validation Study",
            partnership_id=partnership.partnership_id,
            target_sample_size=partnership.target_case_count,
            primary_endpoints=[
                "Diagnostic accuracy",
                "Clinical workflow integration",
                "User satisfaction",
                "Time to decision"
            ]
        )
        studies.append(study)
        print(f"  ✓ Study for {partnership.institution_name}")
        print(f"    - Target size: {study.target_sample_size}")
        print(f"    - Power: {study.power_analysis['estimated_power']:.2f}")
    
    # Simulate case validations
    print("\n📊 Simulating Case Validations (100 cases)...")
    case_count = 0
    agreement_count = 0
    
    for study in studies:
        # Add 33-34 cases per study to reach ~100 total
        cases_for_study = min(34, 100 - case_count)
        
        for i in range(cases_for_study):
            # Simulate AI prediction
            ai_pred = {
                'diagnosis': 'alzheimers',
                'confidence': 0.85 + (i % 10) * 0.01,
                'risk_level': 'moderate'
            }
            
            case = manager.add_case_validation(
                study_id=study.study_id,
                patient_id=f"ANON_{case_count:04d}",
                ai_prediction=ai_pred
            )
            case.processing_time_ms = 75.0 + (i % 20) * 2.0
            
            # Simulate validation (90% agreement rate)
            if i % 10 < 9:
                ground_truth = {'diagnosis': 'alzheimers', 'confirmed': True}
                agreement_count += 1
            else:
                ground_truth = {'diagnosis': 'parkinsons', 'confirmed': True}
            
            manager.validate_case(
                case_id=case.case_id,
                ground_truth=ground_truth,
                clinician_feedback={
                    'satisfaction': 'high' if i % 3 == 0 else 'medium',
                    'useful': True,
                    'workflow_smooth': i % 4 != 0
                }
            )
            
            # Capture workflow issues occasionally
            if i % 7 == 0:
                manager.add_workflow_optimization(
                    category='ui' if i % 2 == 0 else 'workflow',
                    issue_description=f"Issue observed in case {case_count}",
                    severity='low' if i % 3 == 0 else 'medium',
                    affected_users=1
                )
            
            case_count += 1
            
        if case_count >= 100:
            break
    
    print(f"  ✓ {case_count} cases validated")
    print(f"  ✓ Agreement rate: {(agreement_count/case_count)*100:.1f}%")
    
    # Get pilot metrics
    print("\n📈 Pilot Program Metrics:")
    metrics = manager.get_pilot_metrics()
    
    print(f"\n  Partnerships:")
    print(f"    Total: {metrics['partnerships']['total']}")
    print(f"    Active: {metrics['partnerships']['active']}")
    
    print(f"\n  Validation Studies:")
    print(f"    Total: {metrics['studies']['total']}")
    print(f"    Target total sample: {sum(s.target_sample_size for s in studies)}")
    
    print(f"\n  Case Validations:")
    print(f"    Total: {metrics['cases']['total']}")
    print(f"    Completed: {metrics['cases']['completed']}")
    print(f"    Agreement Rate: {metrics['cases']['agreement_rate']*100:.1f}%")
    print(f"    Avg Processing Time: {metrics['cases']['avg_processing_time_ms']:.1f}ms")
    
    print(f"\n  Progress toward 1000-case target:")
    print(f"    {metrics['target_progress']['1000_case_validation']}")
    print(f"    {metrics['target_progress']['percentage']:.1f}% complete")
    
    print(f"\n  Workflow Optimizations:")
    print(f"    Total identified: {metrics['workflow_optimizations']['total']}")
    print(f"    High priority: {metrics['workflow_optimizations']['high_priority']}")
    
    # Study report
    print("\n📄 Sample Study Report:")
    report = manager.get_study_report(studies[0].study_id)
    print(f"  Study: {report['study_name']}")
    print(f"  Phase: {report['phase']}")
    print(f"  Sample size: {report['sample_size']['current']}/{report['sample_size']['target']}")
    print(f"  Completion: {report['sample_size']['completion_percentage']:.1f}%")
    print(f"  Agreement rate: {report['agreement_metrics']['agreement_rate']*100:.1f}%")
    
    print("\n✅ P8B Clinical Pilot Programs: OPERATIONAL")
    return manager


def demonstrate_p9():
    """Demonstrate P9: FDA Regulatory Pathway Planning"""
    print("\n" + "="*70)
    print("P9: FDA REGULATORY PATHWAY PLANNING DEMONSTRATION")
    print("="*70)
    
    # Create FDA planner
    print("\n📋 Creating FDA Pathway Planner...")
    planner = create_fda_pathway_planner()
    
    # Device classification
    print("\n🔍 Performing Device Classification Analysis...")
    classification = planner.create_classification_analysis(
        device_name="AiMedRes Multi-Condition Diagnostic Support System",
        intended_use="To provide diagnostic decision support for neurological conditions including Alzheimer's disease, Parkinson's disease, and related disorders",
        indications_for_use="For use by licensed healthcare professionals in clinical settings as an adjunct to standard diagnostic procedures",
        patient_population="adult",
        clinical_decision_type="diagnostic_support",
        autonomy_level="advisory",
        critical_decision_impact=False
    )
    
    print(f"\n  Device Classification Results:")
    print(f"    Risk Category: {classification.risk_category.value.upper()}")
    print(f"    Proposed Classification: {classification.proposed_classification}")
    print(f"    Recommended Pathway: {classification.recommended_pathway.upper()}")
    print(f"    Product Code: {classification.product_code}")
    print(f"    Software Level: {classification.software_level.value}")
    
    print(f"\n  Risk Factors Identified:")
    for factor in classification.risk_factors:
        print(f"    • {factor}")
    
    print(f"\n  Mitigation Strategies:")
    for strategy in classification.mitigation_strategies:
        print(f"    • {strategy}")
    
    # Pre-submission package
    print("\n📝 Creating Pre-Submission (Q-Sub) Package...")
    qsub = planner.create_presubmission_package(
        device_name="AiMedRes Multi-Condition Diagnostic Support System",
        classification_id=classification.classification_id
    )
    
    # Add additional questions
    qsub.add_regulatory_question(
        "What level of clinical validation is required for AI/ML-enabled diagnostic support devices?"
    )
    qsub.add_regulatory_question(
        "Should we pursue 510(k) with a predicate device or De Novo classification?"
    )
    
    print(f"  ✓ Q-Sub Package Created: {qsub.qsub_id}")
    print(f"  ✓ Total Questions: {len(qsub.regulatory_questions)}")
    print(f"\n  Sample Questions:")
    for i, q in enumerate(qsub.regulatory_questions[:3], 1):
        print(f"    {i}. {q}")
    
    # Clinical evidence dossier
    print("\n📚 Building Clinical Evidence Dossier...")
    dossier = planner.create_evidence_dossier("AiMedRes Multi-Condition Diagnostic Support System")
    
    # Add analytical evidence
    print("  Adding Analytical Validation Evidence...")
    analytical = planner.add_evidence_to_dossier(
        dossier_id=dossier.dossier_id,
        evidence_type=EvidenceType.ANALYTICAL,
        title="Algorithm Validation Study",
        description="Validation of AI algorithms against gold standard datasets",
        study_design="Retrospective validation",
        sample_size=5000
    )
    analytical.endpoints = ["Sensitivity", "Specificity", "AUC-ROC"]
    analytical.results = {
        'sensitivity': 0.92,
        'specificity': 0.89,
        'auc_roc': 0.94
    }
    analytical.statistical_significance = True
    analytical.clinical_significance = True
    analytical.assess_completeness()
    
    # Add clinical evidence
    print("  Adding Clinical Validation Evidence...")
    clinical = planner.add_evidence_to_dossier(
        dossier_id=dossier.dossier_id,
        evidence_type=EvidenceType.CLINICAL,
        title="Multi-Center Clinical Validation Study",
        description="Prospective validation across multiple clinical sites",
        study_design="Prospective observational",
        sample_size=1000
    )
    clinical.endpoints = ["Diagnostic Accuracy", "Clinical Utility", "User Satisfaction"]
    clinical.results = {
        'accuracy': 0.90,
        'clinical_utility_score': 8.5,
        'user_satisfaction': 4.2
    }
    clinical.statistical_significance = True
    clinical.clinical_significance = True
    clinical.assess_completeness()
    
    # Add performance evidence
    print("  Adding Performance Testing Evidence...")
    performance = planner.add_evidence_to_dossier(
        dossier_id=dossier.dossier_id,
        evidence_type=EvidenceType.PERFORMANCE,
        title="System Performance and Reliability Testing",
        description="Comprehensive performance and reliability validation",
        study_design="Controlled testing",
        sample_size=10000
    )
    performance.endpoints = ["Response Time", "Uptime", "Error Rate"]
    performance.results = {
        'avg_response_time_ms': 85,
        'uptime_percentage': 99.9,
        'error_rate': 0.001
    }
    performance.statistical_significance = True
    performance.assess_completeness()
    
    # Perform gap analysis
    print("\n  Performing Evidence Gap Analysis...")
    gap_analysis = dossier.perform_gap_analysis()
    
    print(f"    Total Evidence Items: {gap_analysis['total_evidence_items']}")
    print(f"    Average Completeness: {gap_analysis['average_completeness']*100:.1f}%")
    print(f"    Readiness: {gap_analysis['readiness_percentage']:.1f}%")
    
    if gap_analysis['gaps']:
        print(f"    Gaps Identified: {len(gap_analysis['gaps'])}")
        for gap in gap_analysis['gaps']:
            print(f"      • {gap}")
    else:
        print(f"    ✓ No gaps identified - dossier is complete!")
    
    # QMS Documentation
    print("\n📋 Creating QMS Documentation Skeleton...")
    qms = planner.create_qms_skeleton()
    
    print(f"  ✓ {len(qms.documents)} Standard SOPs Created:")
    for doc in qms.documents:
        print(f"    • {doc.title}")
        print(f"      - Type: {doc.doc_type}")
        print(f"      - Status: {doc.status}")
        print(f"      - Procedures: {len(doc.procedures)}")
    
    # Simulate some approvals
    print("\n  Simulating Document Approvals...")
    for i, doc in enumerate(qms.documents[:3]):
        doc.status = "approved"
        print(f"    ✓ Approved: {doc.title}")
    
    # Overall pathway status
    print("\n📊 Overall FDA Pathway Status:")
    status = planner.get_pathway_status()
    
    print(f"\n  Classifications: {status['classifications']['total']}")
    print(f"    By Risk: {status['classifications']['by_risk']}")
    
    print(f"\n  Pre-Submissions: {status['presubmissions']['total']}")
    print(f"    Completed: {status['presubmissions']['completed']}")
    
    print(f"\n  Evidence Dossiers: {status['evidence_dossiers']['total']}")
    print(f"    Ready: {status['evidence_dossiers']['ready']}")
    print(f"    Readiness: {status['evidence_dossiers']['readiness_percentage']:.1f}%")
    
    print(f"\n  QMS Documentation: {status['qms']['total_documents']} documents")
    print(f"    Approved: {status['qms']['approved_documents']}")
    print(f"    Completion: {status['qms']['completion_percentage']:.1f}%")
    
    print(f"\n  Overall Readiness: {status['overall_readiness']*100:.1f}%")
    
    print("\n✅ P9 FDA Regulatory Pathway Planning: OPERATIONAL")
    return planner


def main():
    """Main demonstration"""
    print("\n" + "="*70)
    print("ROADMAP P8B & P9 IMPLEMENTATION DEMONSTRATION")
    print("="*70)
    print("\nDemonstrating next priority roadmap items:")
    print("  • P8B: Clinical Pilot Programs")
    print("  • P9: FDA Regulatory Pathway Planning")
    
    # Demonstrate P8B
    pilot_manager = demonstrate_p8b()
    
    # Demonstrate P9
    fda_planner = demonstrate_p9()
    
    # Summary
    print("\n" + "="*70)
    print("IMPLEMENTATION SUMMARY")
    print("="*70)
    
    print("\n✅ P8B: Clinical Pilot Programs")
    print("  • Institutional partnership management")
    print("  • Validation study design with power analysis")
    print("  • 1000+ case validation tracking")
    print("  • Workflow optimization capture")
    print("  • Production-ready clinical UI adaptations")
    
    print("\n✅ P9: FDA Regulatory Pathway Planning")
    print("  • Device classification with risk categorization")
    print("  • Pre-submission (Q-Sub) package generation")
    print("  • Clinical evidence dossier with gap analysis")
    print("  • QMS documentation skeleton (5 SOPs)")
    print("  • Comprehensive pathway status tracking")
    
    # Export data
    print("\n💾 Exporting Implementation Data...")
    
    pilot_data_file = "/tmp/p8b_pilot_data.json"
    with open(pilot_data_file, 'w') as f:
        f.write(pilot_manager.export_pilot_data(format='json'))
    print(f"  ✓ P8B data exported to: {pilot_data_file}")
    
    fda_data_file = "/tmp/p9_fda_pathway_data.json"
    with open(fda_data_file, 'w') as f:
        f.write(fda_planner.export_pathway_plan(format='json'))
    print(f"  ✓ P9 data exported to: {fda_data_file}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Deploy P8B to pilot sites")
    print("  2. Complete FDA pre-submission consultation")
    print("  3. Continue P12 production deployment")
    print("  4. Proceed with P13 clinical validation")
    print("\nAll roadmap priority items are now OPERATIONAL! 🎉")


if __name__ == "__main__":
    main()
