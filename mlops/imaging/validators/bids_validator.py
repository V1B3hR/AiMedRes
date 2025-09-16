"""
BIDS Compliance Validator for DuetMind Adaptive

Validates medical imaging datasets for Brain Imaging Data Structure (BIDS) compliance.
Provides detailed validation reports and suggestions for corrections.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

# BIDS validation (placeholder implementation for now)
# In production, would integrate with bids-validator or pybids


class BIDSComplianceValidator:
    """
    Validates imaging datasets for BIDS compliance.
    
    Features:
    - Dataset structure validation
    - Filename convention checking
    - Required metadata validation
    - JSON sidecar validation
    - Participant data validation
    - Comprehensive validation reporting
    """
    
    def __init__(self):
        """Initialize the BIDS compliance validator."""
        self.logger = logging.getLogger(__name__)
        
        # BIDS filename patterns
        self.filename_patterns = {
            'anat': {
                'T1w': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_ce-[a-zA-Z0-9]+)?(_rec-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_T1w)\.nii(\.gz)?$',
                'T2w': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_ce-[a-zA-Z0-9]+)?(_rec-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_T2w)\.nii(\.gz)?$',
                'FLAIR': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_ce-[a-zA-Z0-9]+)?(_rec-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_FLAIR)\.nii(\.gz)?$',
                'PD': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_ce-[a-zA-Z0-9]+)?(_rec-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_PD)\.nii(\.gz)?$',
            },
            'func': {
                'bold': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_task-[a-zA-Z0-9]+)(_acq-[a-zA-Z0-9]+)?(_ce-[a-zA-Z0-9]+)?(_dir-[a-zA-Z0-9]+)?(_rec-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_echo-[0-9]+)?(_part-[a-zA-Z0-9]+)?(_bold)\.nii(\.gz)?$',
            },
            'dwi': {
                'dwi': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_dir-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_dwi)\.nii(\.gz)?$',
            },
            'fmap': {
                'phasediff': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_phasediff)\.nii(\.gz)?$',
                'magnitude': r'^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_acq-[a-zA-Z0-9]+)?(_run-[0-9]+)?(_magnitude[0-9]+)\.nii(\.gz)?$',
            }
        }
        
        # Required metadata fields for each modality
        self.required_metadata = {
            'anat': {
                'T1w': ['RepetitionTime', 'EchoTime', 'FlipAngle'],
                'T2w': ['RepetitionTime', 'EchoTime', 'FlipAngle'],
                'FLAIR': ['RepetitionTime', 'EchoTime', 'InversionTime'],
            },
            'func': {
                'bold': ['RepetitionTime', 'EchoTime', 'TaskName', 'SliceTiming'],
            },
            'dwi': {
                'dwi': ['RepetitionTime', 'EchoTime', 'BValue', 'BVector'],
            },
            'fmap': {
                'phasediff': ['EchoTime1', 'EchoTime2'],
                'magnitude': ['EchoTime'],
            }
        }
        
        # BIDS dataset descriptor requirements
        self.dataset_descriptor_required = [
            'Name', 'BIDSVersion', 'DatasetType'
        ]
        
        # Common BIDS errors and warnings
        self.validation_messages = {
            'invalid_filename': "Filename does not follow BIDS naming convention",
            'missing_json': "Required JSON sidecar file is missing",
            'missing_metadata': "Required metadata field is missing",
            'invalid_subject_id': "Subject ID does not follow BIDS convention",
            'missing_dataset_description': "dataset_description.json is missing",
            'invalid_directory_structure': "Directory structure does not follow BIDS specification",
            'duplicate_files': "Duplicate files found for the same acquisition",
            'invalid_tsv_format': "TSV file format is invalid",
        }
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate complete BIDS dataset.
        
        Args:
            dataset_path: Path to BIDS dataset root directory
            
        Returns:
            Dictionary containing validation results
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            return {
                'valid': False,
                'errors': [f"Dataset path does not exist: {dataset_path}"],
                'warnings': [],
                'summary': {'total_errors': 1, 'total_warnings': 0}
            }
        
        self.logger.info(f"Validating BIDS dataset: {dataset_path}")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dataset_path': str(dataset_path),
            'validation_date': datetime.now().isoformat(),
            'files_validated': 0,
            'subjects_found': [],
            'sessions_found': [],
            'modalities_found': set(),
            'summary': {}
        }
        
        try:
            # 1. Validate dataset structure
            structure_results = self._validate_dataset_structure(dataset_path)
            validation_results['errors'].extend(structure_results['errors'])
            validation_results['warnings'].extend(structure_results['warnings'])
            
            # 2. Validate dataset_description.json
            desc_results = self._validate_dataset_description(dataset_path)
            validation_results['errors'].extend(desc_results['errors'])
            validation_results['warnings'].extend(desc_results['warnings'])
            
            # 3. Validate participants.tsv (if present)
            participants_results = self._validate_participants_file(dataset_path)
            validation_results['errors'].extend(participants_results['errors'])
            validation_results['warnings'].extend(participants_results['warnings'])
            
            # 4. Validate all imaging files
            files_results = self._validate_imaging_files(dataset_path)
            validation_results['errors'].extend(files_results['errors'])
            validation_results['warnings'].extend(files_results['warnings'])
            validation_results['files_validated'] = files_results['files_validated']
            validation_results['subjects_found'] = files_results['subjects_found']
            validation_results['sessions_found'] = files_results['sessions_found']
            validation_results['modalities_found'] = list(files_results['modalities_found'])
            
            # 5. Cross-validation checks
            cross_results = self._cross_validate_dataset(dataset_path, validation_results)
            validation_results['errors'].extend(cross_results['errors'])
            validation_results['warnings'].extend(cross_results['warnings'])
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed with error: {str(e)}")
        
        # Determine overall validity
        validation_results['valid'] = len(validation_results['errors']) == 0
        
        # Create summary
        validation_results['summary'] = {
            'total_errors': len(validation_results['errors']),
            'total_warnings': len(validation_results['warnings']),
            'files_validated': validation_results['files_validated'],
            'subjects_count': len(validation_results['subjects_found']),
            'sessions_count': len(validation_results['sessions_found']),
            'modalities_count': len(validation_results['modalities_found']),
            'is_valid': validation_results['valid']
        }
        
        self.logger.info(f"BIDS validation completed. Valid: {validation_results['valid']}, "
                        f"Errors: {len(validation_results['errors'])}, "
                        f"Warnings: {len(validation_results['warnings'])}")
        
        return validation_results
    
    def _validate_dataset_structure(self, dataset_path: Path) -> Dict[str, List[str]]:
        """Validate basic BIDS directory structure."""
        errors = []
        warnings = []
        
        # Check for required files
        required_files = ['dataset_description.json']
        for required_file in required_files:
            if not (dataset_path / required_file).exists():
                errors.append(f"Missing required file: {required_file}")
        
        # Check for recommended files
        recommended_files = ['README', 'participants.tsv', 'CHANGES']
        for rec_file in recommended_files:
            if not any((dataset_path / f"{rec_file}{ext}").exists() 
                      for ext in ['', '.txt', '.md']):
                warnings.append(f"Missing recommended file: {rec_file}")
        
        # Check for valid subject directories
        subject_dirs = [d for d in dataset_path.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        
        if not subject_dirs:
            errors.append("No subject directories found (directories starting with 'sub-')")
        
        # Validate subject directory names
        for subject_dir in subject_dirs:
            if not re.match(r'^sub-[a-zA-Z0-9]+$', subject_dir.name):
                errors.append(f"Invalid subject directory name: {subject_dir.name}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_dataset_description(self, dataset_path: Path) -> Dict[str, List[str]]:
        """Validate dataset_description.json file."""
        errors = []
        warnings = []
        
        desc_file = dataset_path / 'dataset_description.json'
        
        if not desc_file.exists():
            errors.append("dataset_description.json is missing")
            return {'errors': errors, 'warnings': warnings}
        
        try:
            with open(desc_file, 'r') as f:
                desc_data = json.load(f)
            
            # Check required fields
            for required_field in self.dataset_descriptor_required:
                if required_field not in desc_data:
                    errors.append(f"Missing required field in dataset_description.json: {required_field}")
            
            # Validate BIDSVersion format
            if 'BIDSVersion' in desc_data:
                bids_version = desc_data['BIDSVersion']
                if not re.match(r'^\d+\.\d+\.\d+$', bids_version):
                    warnings.append(f"BIDSVersion format may be invalid: {bids_version}")
            
            # Check recommended fields
            recommended_fields = ['Authors', 'Acknowledgements', 'HowToAcknowledge', 
                                'Funding', 'ReferencesAndLinks', 'DatasetDOI']
            for rec_field in recommended_fields:
                if rec_field not in desc_data:
                    warnings.append(f"Missing recommended field in dataset_description.json: {rec_field}")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format in dataset_description.json: {e}")
        except Exception as e:
            errors.append(f"Error reading dataset_description.json: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_participants_file(self, dataset_path: Path) -> Dict[str, List[str]]:
        """Validate participants.tsv file if present."""
        errors = []
        warnings = []
        
        participants_file = dataset_path / 'participants.tsv'
        
        if not participants_file.exists():
            warnings.append("participants.tsv file is missing (recommended)")
            return {'errors': errors, 'warnings': warnings}
        
        try:
            with open(participants_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                errors.append("participants.tsv is empty")
                return {'errors': errors, 'warnings': warnings}
            
            # Check header
            header = lines[0].strip().split('\t')
            if 'participant_id' not in header:
                errors.append("participants.tsv must have 'participant_id' column")
            
            # Validate participant IDs
            for i, line in enumerate(lines[1:], 2):
                if line.strip():
                    fields = line.strip().split('\t')
                    if len(fields) != len(header):
                        errors.append(f"participants.tsv line {i}: incorrect number of columns")
                        continue
                    
                    participant_id = fields[header.index('participant_id')]
                    if not re.match(r'^sub-[a-zA-Z0-9]+$', participant_id):
                        errors.append(f"participants.tsv line {i}: invalid participant_id format: {participant_id}")
            
        except Exception as e:
            errors.append(f"Error reading participants.tsv: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_imaging_files(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate all imaging files in the dataset."""
        errors = []
        warnings = []
        files_validated = 0
        subjects_found = set()
        sessions_found = set()
        modalities_found = set()
        
        # Find all NIfTI files
        nifti_files = []
        for ext in ['*.nii', '*.nii.gz']:
            nifti_files.extend(dataset_path.rglob(ext))
        
        for nifti_file in nifti_files:
            try:
                file_results = self._validate_single_file(nifti_file, dataset_path)
                errors.extend(file_results['errors'])
                warnings.extend(file_results['warnings'])
                
                if file_results.get('subject_id'):
                    subjects_found.add(file_results['subject_id'])
                if file_results.get('session_id'):
                    sessions_found.add(file_results['session_id'])
                if file_results.get('modality'):
                    modalities_found.add(file_results['modality'])
                
                files_validated += 1
                
            except Exception as e:
                errors.append(f"Error validating file {nifti_file}: {e}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'files_validated': files_validated,
            'subjects_found': list(subjects_found),
            'sessions_found': list(sessions_found),
            'modalities_found': modalities_found
        }
    
    def _validate_single_file(self, file_path: Path, dataset_root: Path) -> Dict[str, Any]:
        """Validate a single imaging file and its metadata."""
        errors = []
        warnings = []
        
        # Get relative path from dataset root
        rel_path = file_path.relative_to(dataset_root)
        
        # Parse filename
        filename = file_path.name
        filename_base = filename.replace('.nii.gz', '').replace('.nii', '')
        
        # Extract BIDS entities from filename
        entities = self._extract_bids_entities(filename_base)
        
        if not entities.get('subject'):
            errors.append(f"{rel_path}: No subject ID found in filename")
            return {'errors': errors, 'warnings': warnings}
        
        # Validate filename format
        datatype = self._determine_datatype(file_path)
        if datatype:
            modality = entities.get('suffix', 'unknown')
            if datatype in self.filename_patterns and modality in self.filename_patterns[datatype]:
                pattern = self.filename_patterns[datatype][modality]
                if not re.match(pattern, filename):
                    errors.append(f"{rel_path}: Filename does not match BIDS convention")
        
        # Check for corresponding JSON sidecar
        json_path = file_path.with_suffix('').with_suffix('.json')
        if not json_path.exists():
            warnings.append(f"{rel_path}: Missing JSON sidecar file")
        else:
            # Validate JSON metadata
            json_results = self._validate_json_sidecar(json_path, datatype, entities.get('suffix'))
            errors.extend([f"{rel_path}: {error}" for error in json_results['errors']])
            warnings.extend([f"{rel_path}: {warning}" for warning in json_results['warnings']])
        
        return {
            'errors': errors,
            'warnings': warnings,
            'subject_id': entities.get('subject'),
            'session_id': entities.get('session'),
            'modality': entities.get('suffix'),
            'datatype': datatype
        }
    
    def _extract_bids_entities(self, filename: str) -> Dict[str, str]:
        """Extract BIDS entities from filename."""
        entities = {}
        
        # Common BIDS entities
        entity_patterns = {
            'subject': r'sub-([a-zA-Z0-9]+)',
            'session': r'ses-([a-zA-Z0-9]+)',
            'task': r'task-([a-zA-Z0-9]+)',
            'acquisition': r'acq-([a-zA-Z0-9]+)',
            'contrast': r'ce-([a-zA-Z0-9]+)',
            'reconstruction': r'rec-([a-zA-Z0-9]+)',
            'direction': r'dir-([a-zA-Z0-9]+)',
            'run': r'run-([0-9]+)',
            'echo': r'echo-([0-9]+)',
            'part': r'part-([a-zA-Z0-9]+)',
        }
        
        for entity, pattern in entity_patterns.items():
            match = re.search(pattern, filename)
            if match:
                entities[entity] = match.group(1)
        
        # Extract suffix (last part before extension)
        suffix_match = re.search(r'_([a-zA-Z0-9]+)$', filename)
        if suffix_match:
            entities['suffix'] = suffix_match.group(1)
        
        return entities
    
    def _determine_datatype(self, file_path: Path) -> Optional[str]:
        """Determine BIDS datatype from file path."""
        path_parts = file_path.parts
        
        # Look for datatype in path
        datatypes = ['anat', 'func', 'dwi', 'fmap', 'perf', 'pet']
        for part in path_parts:
            if part in datatypes:
                return part
        
        return None
    
    def _validate_json_sidecar(self, json_path: Path, datatype: str, modality: str) -> Dict[str, List[str]]:
        """Validate JSON sidecar metadata."""
        errors = []
        warnings = []
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            if datatype and modality:
                required_fields = self.required_metadata.get(datatype, {}).get(modality, [])
                for field in required_fields:
                    if field not in metadata:
                        warnings.append(f"Missing recommended metadata field: {field}")
            
            # Validate specific field formats
            if 'RepetitionTime' in metadata:
                try:
                    tr = float(metadata['RepetitionTime'])
                    if tr <= 0:
                        warnings.append("RepetitionTime should be positive")
                except (ValueError, TypeError):
                    errors.append("RepetitionTime should be a number")
            
            if 'EchoTime' in metadata:
                try:
                    te = float(metadata['EchoTime'])
                    if te < 0:
                        warnings.append("EchoTime should be non-negative")
                except (ValueError, TypeError):
                    errors.append("EchoTime should be a number")
            
            if 'SliceTiming' in metadata:
                slice_timing = metadata['SliceTiming']
                if not isinstance(slice_timing, list):
                    errors.append("SliceTiming should be an array")
                elif not all(isinstance(x, (int, float)) for x in slice_timing):
                    errors.append("SliceTiming should contain only numbers")
        
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e}")
        except Exception as e:
            errors.append(f"Error reading JSON file: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _cross_validate_dataset(self, dataset_path: Path, validation_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Perform cross-validation checks across the dataset."""
        errors = []
        warnings = []
        
        # Check for consistency between participants.tsv and actual subjects
        participants_file = dataset_path / 'participants.tsv'
        if participants_file.exists():
            try:
                with open(participants_file, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    header = lines[0].strip().split('\t')
                    if 'participant_id' in header:
                        tsv_subjects = set()
                        for line in lines[1:]:
                            if line.strip():
                                fields = line.strip().split('\t')
                                if len(fields) > header.index('participant_id'):
                                    tsv_subjects.add(fields[header.index('participant_id')])
                        
                        found_subjects = set(validation_results['subjects_found'])
                        
                        # Subjects in TSV but not in data
                        missing_data = tsv_subjects - found_subjects
                        for subject in missing_data:
                            warnings.append(f"Subject {subject} listed in participants.tsv but no data found")
                        
                        # Subjects in data but not in TSV
                        missing_tsv = found_subjects - tsv_subjects
                        for subject in missing_tsv:
                            warnings.append(f"Subject {subject} has data but not listed in participants.tsv")
            
            except Exception as e:
                warnings.append(f"Could not cross-validate participants.tsv: {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single imaging file for BIDS compliance.
        
        Args:
            file_path: Path to the imaging file
            
        Returns:
            Dictionary containing validation results for the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'valid': False,
                'errors': [f"File does not exist: {file_path}"],
                'warnings': [],
                'file_path': str(file_path)
            }
        
        # Determine dataset root (go up until we find dataset_description.json)
        dataset_root = file_path.parent
        while dataset_root.parent != dataset_root:
            if (dataset_root / 'dataset_description.json').exists():
                break
            dataset_root = dataset_root.parent
        else:
            dataset_root = file_path.parent
        
        result = self._validate_single_file(file_path, dataset_root)
        result['valid'] = len(result['errors']) == 0
        result['file_path'] = str(file_path)
        result['validation_date'] = datetime.now().isoformat()
        
        return result
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 output_path: str = None) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            validation_results: Results from validate_dataset()
            output_path: Optional path to save the report
            
        Returns:
            Formatted validation report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("BIDS Dataset Validation Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Dataset: {validation_results.get('dataset_path', 'Unknown')}")
        report_lines.append(f"Validation Date: {validation_results.get('validation_date', 'Unknown')}")
        report_lines.append(f"Overall Status: {'VALID' if validation_results['valid'] else 'INVALID'}")
        report_lines.append("")
        
        # Summary
        summary = validation_results.get('summary', {})
        report_lines.append("Summary:")
        report_lines.append(f"  Total Errors: {summary.get('total_errors', 0)}")
        report_lines.append(f"  Total Warnings: {summary.get('total_warnings', 0)}")
        report_lines.append(f"  Files Validated: {summary.get('files_validated', 0)}")
        report_lines.append(f"  Subjects Found: {summary.get('subjects_count', 0)}")
        report_lines.append(f"  Sessions Found: {summary.get('sessions_count', 0)}")
        report_lines.append(f"  Modalities Found: {summary.get('modalities_count', 0)}")
        report_lines.append("")
        
        # Subjects and modalities
        if validation_results.get('subjects_found'):
            report_lines.append("Subjects:")
            for subject in sorted(validation_results['subjects_found']):
                report_lines.append(f"  - {subject}")
            report_lines.append("")
        
        if validation_results.get('modalities_found'):
            report_lines.append("Modalities:")
            for modality in sorted(validation_results['modalities_found']):
                report_lines.append(f"  - {modality}")
            report_lines.append("")
        
        # Errors
        if validation_results.get('errors'):
            report_lines.append("Errors:")
            for i, error in enumerate(validation_results['errors'], 1):
                report_lines.append(f"  {i}. {error}")
            report_lines.append("")
        
        # Warnings
        if validation_results.get('warnings'):
            report_lines.append("Warnings:")
            for i, warning in enumerate(validation_results['warnings'], 1):
                report_lines.append(f"  {i}. {warning}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("Recommendations:")
        if validation_results['valid']:
            report_lines.append("  - Dataset appears to be BIDS compliant!")
            if validation_results.get('warnings'):
                report_lines.append("  - Consider addressing warnings to improve compliance")
        else:
            report_lines.append("  - Please fix all errors before proceeding")
            report_lines.append("  - Refer to BIDS specification: https://bids-specification.readthedocs.io/")
        
        report_text = "\n".join(report_lines)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to: {output_path}")
        
        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = BIDSComplianceValidator()
    
    # Example: Validate a BIDS dataset
    # results = validator.validate_dataset("/path/to/bids/dataset")
    # 
    # print(f"Dataset is valid: {results['valid']}")
    # print(f"Errors: {len(results['errors'])}")
    # print(f"Warnings: {len(results['warnings'])}")
    # 
    # # Generate report
    # report = validator.generate_validation_report(results, "validation_report.txt")
    # print(report)
    
    print("BIDS compliance validator initialized successfully")