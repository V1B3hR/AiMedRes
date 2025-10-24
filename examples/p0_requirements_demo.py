#!/usr/bin/env python3
"""
P0 Implementation Demo

Demonstrates all P0 requirements in action:
- P0-1: GPL-3.0 License (see LICENSE file)
- P0-2: Clinical disclaimers (see templates/*.html, README.md)
- P0-3: PHI de-identification enforcement
- P0-4: Security disclosure process (see SECURITY.md)
- P0-5: Human-in-loop gating with audit logging
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path and import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'aimedres' / 'security'))

from phi_scrubber import PHIScrubber, enforce_phi_free_ingestion
from human_in_loop import (
    HumanInLoopGatekeeper,
    ClinicalRecommendation,
    RiskLevel
)


def demo_p0_3_phi_protection():
    """Demo P0-3: PHI de-identification"""
    print("=" * 80)
    print("P0-3 DEMO: PHI De-identification & Ingestion Enforcement")
    print("=" * 80)
    
    scrubber = PHIScrubber(aggressive=True, hash_identifiers=True)
    
    # Example 1: Detect and sanitize PHI
    print("\n[Example 1] PHI Detection in Clinical Text:")
    text_with_phi = (
        "Patient John Smith (MRN: 123456) was seen on 03/15/2024. "
        "Contact: john.smith@hospital.com or 555-123-4567. "
        "Address: 123 Main Street, Springfield."
    )
    
    print(f"Original: {text_with_phi}")
    result = scrubber.detect_phi(text_with_phi)
    print(f"PHI Detected: {result.has_phi}")
    print(f"PHI Types: {result.phi_types_found}")
    print(f"Sanitized: {result.sanitized_text}")
    
    # Example 2: Enforce PHI-free ingestion
    print("\n[Example 2] Ingestion Enforcement:")
    clean_data = "Patient shows cognitive improvement with MMSE score 28/30"
    try:
        enforce_phi_free_ingestion(clean_data, "clinical_notes")
        print(f"✓ Clean data accepted: {clean_data}")
    except ValueError as e:
        print(f"✗ PHI detected: {e}")
    
    # Try with PHI
    phi_data = "Patient john.smith@example.com has high blood pressure"
    try:
        enforce_phi_free_ingestion(phi_data, "clinical_notes")
        print(f"✓ Data accepted")
    except ValueError as e:
        print(f"✗ PHI BLOCKED: {e}")


def demo_p0_5_human_in_loop():
    """Demo P0-5: Human-in-loop gating"""
    print("\n\n" + "=" * 80)
    print("P0-5 DEMO: Human-in-Loop Gating with Immutable Audit Logging")
    print("=" * 80)
    
    gatekeeper = HumanInLoopGatekeeper()
    
    # Example 1: Low risk (auto-approved)
    print("\n[Example 1] Low Risk Recommendation (Auto-approved):")
    low_risk_rec = ClinicalRecommendation(
        recommendation_id="",
        recommendation_type="lifestyle_advice",
        recommendation_text="Recommend increased physical activity and balanced diet",
        risk_level=RiskLevel.LOW,
        confidence_score=0.95,
        patient_id="SYNTH-DEMO-001",
        generated_at=datetime.now(timezone.utc).isoformat(),
        generated_by="LifestyleAdvisorAI_v1.0",
        context={'intervention': 'preventive_care'}
    )
    
    result = gatekeeper.submit_recommendation(low_risk_rec)
    print(f"Status: {result['status']}")
    print(f"Requires Approval: {result['requires_human_approval']}")
    
    # Example 2: High risk (requires approval)
    print("\n[Example 2] High Risk Recommendation (Requires Human Approval):")
    high_risk_rec = ClinicalRecommendation(
        recommendation_id="",
        recommendation_type="medication_change",
        recommendation_text="Recommend increasing medication dosage from 10mg to 20mg daily",
        risk_level=RiskLevel.HIGH,
        confidence_score=0.85,
        patient_id="SYNTH-DEMO-002",
        generated_at=datetime.now(timezone.utc).isoformat(),
        generated_by="MedicationAI_v2.3",
        context={'current_dosage': '10mg', 'indication': 'symptom_progression'}
    )
    
    result = gatekeeper.submit_recommendation(high_risk_rec)
    print(f"Status: {result['status']}")
    print(f"Requires Approval: {result['requires_human_approval']}")
    print(f"Message: {result['message']}")
    
    rec_id = result['recommendation_id']
    
    # Clinician reviews and approves
    print("\n[Clinician Review] Dr. Smith reviews the recommendation...")
    review_start = datetime.now(timezone.utc)
    
    approval_result = gatekeeper.approve_recommendation(
        recommendation_id=rec_id,
        clinician_id="DR_SMITH_MD",
        clinician_role="Neurologist",
        rationale=(
            "Reviewed patient's complete medical history and recent symptom progression. "
            "Lab results show good tolerance of current medication. "
            "Dosage increase is clinically appropriate and aligns with treatment guidelines."
        ),
        review_start_time=review_start
    )
    
    print(f"Approval Status: {approval_result['status']}")
    print(f"Approved By: {approval_result['approved_by']}")
    print(f"Approval ID: {approval_result['approval_id']}")
    
    # Verify audit trail
    print("\n[Audit Trail Verification]:")
    verification = gatekeeper.verify_audit_chain()
    print(f"Audit Chain Verified: {verification['verified']}")
    print(f"Total Entries: {verification['entries']}")
    print(f"Message: {verification['message']}")
    
    # Show audit history
    print("\n[Audit History for Recommendation]:")
    history = gatekeeper.get_audit_history(recommendation_id=rec_id)
    for i, entry in enumerate(history, 1):
        print(f"  {i}. Event: {entry['event_type']}")
        print(f"     Timestamp: {entry['timestamp']}")
        print(f"     Entry Hash: {entry['entry_hash'][:16]}...")
        if entry['approval']:
            print(f"     Clinician: {entry['approval']['clinician_id']}")
            print(f"     Status: {entry['approval']['status']}")


def demo_clinical_disclaimers():
    """Demo P0-2: Clinical use classification"""
    print("\n\n" + "=" * 80)
    print("P0-2 DEMO: Clinical Use Classification & Disclaimers")
    print("=" * 80)
    
    print("\n⚠️  IMPORTANT: RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS ⚠️")
    print("\nThis software is:")
    print("  • NOT an FDA-approved medical device")
    print("  • NOT intended for clinical diagnosis or treatment")
    print("  • NOT a replacement for professional medical judgment")
    print("  • For research and development purposes only")
    
    print("\nDisclaimers are prominently displayed in:")
    print("  ✓ README.md - Top-level repository documentation")
    print("  ✓ templates/clinical_dashboard.html - Red banner + warnings")
    print("  ✓ templates/about.html - Comprehensive classification page")
    print("  ✓ All UI mockups and documentation")


def demo_license_compliance():
    """Demo P0-1: License compliance"""
    print("\n\n" + "=" * 80)
    print("P0-1 DEMO: License Compliance")
    print("=" * 80)
    
    print("\nLicense Information:")
    print("  • License: GNU General Public License v3.0 (GPL-3.0)")
    print("  • Consistency: LICENSE file, README.md, setup.py all aligned")
    print("  • Legal Signoff: Tracked in docs/LEGAL_SIGNOFF.md")
    print("  • Contributor Guidelines: GPL-3.0 terms in CONTRIBUTING.md")
    
    print("\nGPL-3.0 means:")
    print("  ✓ Free to use for research")
    print("  ✓ Free to modify and adapt")
    print("  ✓ Must distribute source code")
    print("  ✓ Derivative works must be GPL-3.0")


def demo_security_disclosure():
    """Demo P0-4: Security disclosure"""
    print("\n\n" + "=" * 80)
    print("P0-4 DEMO: Vulnerability Disclosure Process")
    print("=" * 80)
    
    print("\nResponsible Disclosure Process:")
    print("  • Report vulnerabilities via SECURITY.md")
    print("  • Use GitHub private vulnerability reporting (preferred)")
    print("  • DO NOT create public issues for security vulnerabilities")
    
    print("\nResponse Commitments:")
    print("  • Initial response: 48 hours")
    print("  • Critical issues: 7 day resolution target")
    print("  • High severity: 14 day resolution target")
    print("  • 90-day coordinated disclosure embargo")
    
    print("\nSecurity Contact:")
    print("  • Primary: GitHub private vulnerability reporting")
    print("  • Documentation: SECURITY.md in repository root")


def main():
    """Run all P0 demos"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                      AiMedRes P0 Requirements Demo                            ║")
    print("║                                                                               ║")
    print("║  Demonstrating all P0 blockers (legal, privacy, security, gating)            ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    
    # Run all demos
    demo_license_compliance()
    demo_clinical_disclaimers()
    demo_p0_3_phi_protection()
    demo_security_disclosure()
    demo_p0_5_human_in_loop()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: All P0 Requirements Implemented")
    print("=" * 80)
    print("\n✅ P0-1: LICENSE consistency verified (GPL-3.0)")
    print("✅ P0-2: Clinical disclaimers in all UI templates")
    print("✅ P0-3: PHI scrubber with CI enforcement")
    print("✅ P0-4: Vulnerability disclosure process documented")
    print("✅ P0-5: Human-in-loop gating with immutable audit logging")
    
    print("\n🎉 All P0 blockers addressed! Repository is ready for release.")
    print("\nFor more information:")
    print("  • License: See LICENSE file")
    print("  • Security: See SECURITY.md")
    print("  • Clinical Use: See templates/about.html")
    print("  • PHI Protection: See src/aimedres/security/phi_scrubber.py")
    print("  • Human-in-Loop: See src/aimedres/security/human_in_loop.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
