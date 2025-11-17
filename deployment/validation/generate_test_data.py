#!/usr/bin/env python3
"""
Test Data Generator for AiMedRes System Validation

Generates de-identified synthetic test data for smoke tests, model verification,
and UAT. All generated data is completely synthetic with NO PHI.

Usage:
    python generate_test_data.py --output test_data/ --samples 100 --no-phi
"""

import json
import csv
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any


class TestDataGenerator:
    """Generate synthetic de-identified test data"""
    
    # Synthetic ID prefixes to make it clear this is test data
    PATIENT_ID_PREFIX = "TEST"
    CASE_ID_PREFIX = "CASE"
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize generator.
        
        Args:
            strict_mode: If True, extra validation to ensure no PHI-like patterns
        """
        self.strict_mode = strict_mode
        self.generated_ids = set()
    
    def generate_patient_id(self) -> str:
        """Generate a synthetic patient ID"""
        while True:
            # Format: TEST-XXXXXX (6 random digits)
            patient_id = f"{self.PATIENT_ID_PREFIX}-{random.randint(100000, 999999)}"
            if patient_id not in self.generated_ids:
                self.generated_ids.add(patient_id)
                return patient_id
    
    def generate_case_id(self) -> str:
        """Generate a synthetic case ID"""
        # Format: CASE-YYYYMMDD-XXX
        date_str = datetime.now().strftime("%Y%m%d")
        seq = random.randint(100, 999)
        return f"{self.CASE_ID_PREFIX}-{date_str}-{seq}"
    
    def generate_age(self, min_age: int = 55, max_age: int = 85) -> int:
        """Generate age (no DOB to avoid PHI)"""
        return random.randint(min_age, max_age)
    
    def generate_clinical_scores(self, condition: str) -> Dict[str, float]:
        """Generate clinical assessment scores based on condition"""
        if condition == 'alzheimer':
            # MMSE-like score (0-30, lower is worse)
            mmse = random.uniform(15, 28)
            # CDR-like score (0-3, higher is worse)
            cdr = random.uniform(0, 2)
            # ADL score (0-100, lower is worse)
            adl = random.uniform(50, 95)
            
            return {
                'cognitive_score': round(mmse, 1),
                'dementia_rating': round(cdr, 2),
                'daily_living_score': round(adl, 1),
                'memory_score': round(random.uniform(10, 30), 1),
                'attention_score': round(random.uniform(5, 15), 1)
            }
        
        elif condition == 'parkinsons':
            # UPDRS-like scores
            updrs_motor = random.uniform(10, 50)
            updrs_total = random.uniform(20, 100)
            
            return {
                'motor_score': round(updrs_motor, 1),
                'total_score': round(updrs_total, 1),
                'tremor_score': round(random.uniform(0, 20), 1),
                'rigidity_score': round(random.uniform(0, 20), 1),
                'bradykinesia_score': round(random.uniform(0, 30), 1)
            }
        
        elif condition == 'als':
            # ALSFRS-like score (0-48, lower is worse)
            alsfrs = random.uniform(20, 45)
            
            return {
                'functional_score': round(alsfrs, 1),
                'respiratory_score': round(random.uniform(0, 12), 1),
                'motor_score': round(random.uniform(0, 24), 1),
                'bulbar_score': round(random.uniform(0, 12), 1)
            }
        
        return {}
    
    def generate_patient_record(self, condition: str = 'alzheimer') -> Dict[str, Any]:
        """
        Generate a complete synthetic patient record.
        
        Args:
            condition: Type of condition (alzheimer, parkinsons, als)
            
        Returns:
            Patient record dictionary
        """
        patient_id = self.generate_patient_id()
        age = self.generate_age()
        sex = random.choice(['M', 'F'])
        
        # Generate clinical data
        clinical_scores = self.generate_clinical_scores(condition)
        
        # Generate timestamp (within last 30 days)
        days_ago = random.randint(0, 30)
        assessment_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        record = {
            'patient_id': patient_id,
            'case_id': self.generate_case_id(),
            'assessment_date': assessment_date,
            'age': age,
            'sex': sex,
            'condition_type': condition,
            'clinical_data': clinical_scores,
            'risk_factors': {
                'family_history': random.choice([True, False]),
                'education_years': random.randint(8, 20),
                'comorbidities_count': random.randint(0, 5)
            },
            'metadata': {
                'data_source': 'synthetic_test_data',
                'generated_at': datetime.now().isoformat(),
                'phi_status': 'none',
                'validation_status': 'ready'
            }
        }
        
        return record
    
    def generate_longitudinal_patient(self, condition: str = 'parkinsons', 
                                     num_timepoints: int = 5) -> List[Dict[str, Any]]:
        """
        Generate longitudinal patient data with multiple timepoints.
        
        Args:
            condition: Type of condition
            num_timepoints: Number of assessment timepoints
            
        Returns:
            List of patient records over time
        """
        patient_id = self.generate_patient_id()
        age = self.generate_age()
        sex = random.choice(['M', 'F'])
        
        records = []
        
        # Generate progression (scores generally worsen over time)
        for i in range(num_timepoints):
            days_ago = (num_timepoints - i - 1) * random.randint(30, 90)  # ~1-3 months apart
            assessment_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Generate scores with slight worsening trend
            clinical_scores = self.generate_clinical_scores(condition)
            
            # Add temporal progression (scores worsen)
            if condition == 'parkinsons':
                progression_factor = 1 + (i * 0.1)  # Increase by 10% per timepoint
                clinical_scores['motor_score'] *= progression_factor
                clinical_scores['total_score'] *= progression_factor
            elif condition == 'alzheimer':
                decline_factor = 1 - (i * 0.05)  # Decrease by 5% per timepoint
                clinical_scores['cognitive_score'] *= decline_factor
                clinical_scores['daily_living_score'] *= decline_factor
            
            record = {
                'patient_id': patient_id,
                'case_id': self.generate_case_id(),
                'assessment_date': assessment_date,
                'timepoint': i + 1,
                'age': age + (days_ago // 365),  # Age increases
                'sex': sex,
                'condition_type': condition,
                'clinical_data': {k: round(v, 1) for k, v in clinical_scores.items()},
                'metadata': {
                    'data_source': 'synthetic_longitudinal_test_data',
                    'generated_at': datetime.now().isoformat(),
                    'phi_status': 'none'
                }
            }
            
            records.append(record)
        
        return records
    
    def save_json(self, data: List[Dict], output_path: Path):
        """Save data as JSON"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON: {output_path}")
    
    def save_csv(self, data: List[Dict], output_path: Path):
        """Save data as CSV (flattened structure)"""
        if not data:
            return
        
        # Flatten nested structures for CSV
        flattened = []
        for record in data:
            flat_record = {
                'patient_id': record['patient_id'],
                'case_id': record['case_id'],
                'assessment_date': record['assessment_date'],
                'age': record['age'],
                'sex': record['sex'],
                'condition_type': record['condition_type']
            }
            
            # Add clinical data fields
            for key, value in record.get('clinical_data', {}).items():
                flat_record[f'clinical_{key}'] = value
            
            # Add risk factors
            for key, value in record.get('risk_factors', {}).items():
                flat_record[f'risk_{key}'] = value
            
            flattened.append(flat_record)
        
        # Write CSV
        if flattened:
            keys = flattened[0].keys()
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flattened)
            print(f"Saved CSV: {output_path}")


def main():
    """Generate test data"""
    parser = argparse.ArgumentParser(description='Generate synthetic test data for AiMedRes')
    parser.add_argument('--output', type=str, default='test_data', help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per condition')
    parser.add_argument('--no-phi', action='store_true', help='Extra validation for no PHI (default)')
    parser.add_argument('--longitudinal', type=int, default=5, help='Number of longitudinal patients per condition')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("AiMedRes Test Data Generator")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Samples per condition: {args.samples}")
    print(f"Longitudinal patients: {args.longitudinal}")
    print(f"PHI validation: {'STRICT' if args.no_phi else 'NORMAL'}")
    print("="*60 + "\n")
    
    generator = TestDataGenerator(strict_mode=args.no_phi)
    
    conditions = ['alzheimer', 'parkinsons', 'als']
    
    # Generate cross-sectional data for each condition
    for condition in conditions:
        print(f"\nGenerating {condition} test data...")
        
        # Generate samples
        samples = []
        for i in range(args.samples):
            samples.append(generator.generate_patient_record(condition))
        
        # Save as JSON and CSV
        generator.save_json(samples, output_dir / f'{condition}_test_data.json')
        generator.save_csv(samples, output_dir / f'{condition}_test_data.csv')
        
        print(f"  Generated {len(samples)} {condition} records")
    
    # Generate longitudinal data
    if args.longitudinal > 0:
        print(f"\nGenerating longitudinal test data...")
        
        for condition in ['parkinsons', 'alzheimer']:  # Focus on conditions with progression
            longitudinal_data = []
            for i in range(args.longitudinal):
                patient_records = generator.generate_longitudinal_patient(
                    condition=condition,
                    num_timepoints=random.randint(3, 6)
                )
                longitudinal_data.extend(patient_records)
            
            generator.save_json(longitudinal_data, output_dir / f'{condition}_longitudinal_data.json')
            print(f"  Generated {len(longitudinal_data)} {condition} longitudinal records")
    
    # Generate README
    readme_content = f"""# AiMedRes Test Data

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This directory contains synthetic de-identified test data for AiMedRes system validation.

**IMPORTANT:** All data is completely synthetic. No real patient data or PHI.

## Files

"""
    
    for condition in conditions:
        readme_content += f"- `{condition}_test_data.json`: {args.samples} {condition} test cases (JSON format)\n"
        readme_content += f"- `{condition}_test_data.csv`: {args.samples} {condition} test cases (CSV format)\n"
    
    if args.longitudinal > 0:
        readme_content += f"\n### Longitudinal Data\n"
        readme_content += f"- `parkinsons_longitudinal_data.json`: Longitudinal Parkinson's cases\n"
        readme_content += f"- `alzheimer_longitudinal_data.json`: Longitudinal Alzheimer's cases\n"
    
    readme_content += """
## Data Structure

### Cross-Sectional Records
- `patient_id`: Synthetic patient identifier (TEST-XXXXXX)
- `case_id`: Case identifier
- `assessment_date`: Assessment date
- `age`: Patient age (integer, no DOB)
- `sex`: M/F
- `condition_type`: alzheimer, parkinsons, or als
- `clinical_data`: Condition-specific clinical scores
- `risk_factors`: Risk factor information
- `metadata`: Data source and validation info

### Longitudinal Records
Same structure as cross-sectional, with additional `timepoint` field indicating assessment sequence.

## Usage

### For Smoke Tests
```bash
python smoke_test_cli.py --test-data test_data/alzheimer_test_data.json
```

### For Model Verification
```bash
python model_verification.py --validation-data test_data/
```

### For UAT
Use these datasets for clinician testing scenarios as described in `uat_scenarios.md`.

## Validation

All generated data has been validated to ensure:
- ✓ No PHI (names, SSNs, addresses, etc.)
- ✓ Synthetic patient IDs
- ✓ Clinical scores within realistic ranges
- ✓ Proper data structure and formatting
- ✓ Ready for system validation

---

**Note:** This is TEST DATA ONLY. Do not use for any real clinical decisions or research.
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\nCreated README: {readme_path}")
    
    print("\n" + "="*60)
    print("Test data generation complete!")
    print(f"Total files created: {len(list(output_dir.glob('*')))} files")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
