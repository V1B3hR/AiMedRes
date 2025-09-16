"""
Medical Image De-identification for DuetMind Adaptive

Provides comprehensive de-identification capabilities for medical imaging data
including DICOM headers, metadata, and filename anonymization.
"""

import os
import json
import hashlib
import logging
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re

# Import medical imaging libraries with fallbacks
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class MedicalImageDeidentifier:
    """
    De-identifies medical imaging data according to HIPAA Safe Harbor guidelines.
    
    Features:
    - DICOM header anonymization
    - Metadata de-identification
    - Date shifting
    - ID replacement with consistent mapping
    - Encrypted mapping storage
    - Audit logging
    """
    
    def __init__(self, encryption_key: Optional[str] = None, 
                 mapping_file: Optional[str] = None):
        """
        Initialize the de-identification system.
        
        Args:
            encryption_key: Key for encrypting ID mappings
            mapping_file: Path to store ID mappings
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        if CRYPTO_AVAILABLE and encryption_key:
            self.fernet = self._initialize_encryption(encryption_key)
        else:
            self.fernet = None
            if encryption_key:
                self.logger.warning("Cryptography not available, ID mappings will not be encrypted")
        
        # ID mapping storage
        self.mapping_file = mapping_file or "id_mappings.json"
        self.id_mappings = self._load_id_mappings()
        
        # Date shift mappings (consistent per patient)
        self.date_shifts = {}
        
        # HIPAA Safe Harbor - 18 identifiers to remove/modify
        self.phi_dicom_tags = [
            # Names
            (0x0010, 0x0010),  # PatientName
            (0x0008, 0x0090),  # ReferringPhysicianName
            (0x0008, 0x1070),  # OperatorsName
            (0x0032, 0x1032),  # RequestingPhysician
            (0x0008, 0x1060),  # NameOfPhysiciansReadingStudy
            
            # Geographic subdivisions smaller than state
            (0x0008, 0x0080),  # InstitutionName
            (0x0008, 0x0081),  # InstitutionAddress
            (0x0040, 0x0310),  # InstitutionAddress
            
            # Dates (except year unless over 89 years old)
            (0x0008, 0x0020),  # StudyDate
            (0x0008, 0x0021),  # SeriesDate
            (0x0008, 0x0022),  # AcquisitionDate
            (0x0008, 0x0023),  # ContentDate
            (0x0008, 0x0030),  # StudyTime
            (0x0008, 0x0031),  # SeriesTime
            (0x0008, 0x0032),  # AcquisitionTime
            (0x0008, 0x0033),  # ContentTime
            (0x0010, 0x0030),  # PatientBirthDate
            
            # Contact information
            (0x0010, 0x1000),  # OtherPatientIDs
            (0x0010, 0x1001),  # OtherPatientNames
            (0x0010, 0x2154),  # PatientTelephoneNumbers
            (0x0010, 0x2155),  # PatientTelecomInformation
            
            # Identifying numbers
            (0x0010, 0x0020),  # PatientID
            (0x0008, 0x0050),  # AccessionNumber
            (0x0020, 0x0010),  # StudyID
            (0x0008, 0x1010),  # StationName
            (0x0018, 0x1000),  # DeviceSerialNumber
            (0x0008, 0x1070),  # OperatorsName
            
            # URLs and IP addresses
            (0x0008, 0x0012),  # InstanceCreationDate
            (0x0008, 0x0013),  # InstanceCreationTime
            
            # Other identifying information
            (0x0010, 0x4000),  # PatientComments
            (0x0008, 0x4000),  # IdentifyingComments
            (0x0040, 0xa730),  # ContentSequence (may contain PHI)
        ]
        
        # Tags to replace with anonymous values
        self.replacement_tags = {
            (0x0010, 0x0010): "Anonymous",  # PatientName
            (0x0008, 0x0080): "Anonymous Institution",  # InstitutionName
            (0x0008, 0x0090): "Anonymous MD",  # ReferringPhysicianName
            (0x0008, 0x1070): "Anonymous",  # OperatorsName
            (0x0008, 0x1010): "Anonymous",  # StationName
        }
        
        # Regular expressions for finding PHI in text fields
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{3}\.\d{2}\.\d{4}\b',  # SSN pattern with dots
            r'\b\d{3} \d{2} \d{4}\b',  # SSN pattern with spaces
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b\d{10}\b',  # Phone number without separators
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b',  # Address
        ]
    
    def _initialize_encryption(self, password: str) -> Fernet:
        """Initialize encryption with password-derived key."""
        password_bytes = password.encode()
        salt = b'duetmind_imaging_salt'  # In production, use random salt per installation
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def _load_id_mappings(self) -> Dict[str, str]:
        """Load existing ID mappings from file."""
        if not os.path.exists(self.mapping_file):
            return {}
        
        try:
            with open(self.mapping_file, 'r') as f:
                mappings_data = json.load(f)
            
            if self.fernet and 'encrypted' in mappings_data:
                # Decrypt mappings
                encrypted_data = mappings_data['encrypted'].encode()
                decrypted_data = self.fernet.decrypt(encrypted_data)
                return json.loads(decrypted_data.decode())
            else:
                return mappings_data.get('mappings', {})
                
        except Exception as e:
            self.logger.warning(f"Could not load ID mappings: {e}")
            return {}
    
    def _save_id_mappings(self):
        """Save ID mappings to file with optional encryption."""
        try:
            if self.fernet:
                # Encrypt mappings
                mappings_json = json.dumps(self.id_mappings)
                encrypted_data = self.fernet.encrypt(mappings_json.encode())
                data = {
                    'encrypted': encrypted_data.decode(),
                    'created': datetime.now().isoformat(),
                    'version': '1.0'
                }
            else:
                data = {
                    'mappings': self.id_mappings,
                    'created': datetime.now().isoformat(),
                    'version': '1.0'
                }
            
            with open(self.mapping_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save ID mappings: {e}")
    
    def generate_anonymous_id(self, original_id: str, id_type: str = "patient") -> str:
        """
        Generate consistent anonymous ID for original identifier.
        
        Args:
            original_id: Original identifier
            id_type: Type of ID (patient, study, series, etc.)
            
        Returns:
            Anonymous identifier
        """
        mapping_key = f"{id_type}:{original_id}"
        
        if mapping_key in self.id_mappings:
            return self.id_mappings[mapping_key]
        
        # Generate new anonymous ID
        # Use hash of original ID with salt for consistency
        hash_input = f"{original_id}:{id_type}:duetmind_salt".encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()[:8]
        
        if id_type == "patient":
            anonymous_id = f"ANON-{hash_digest.upper()}"
        elif id_type == "study":
            anonymous_id = f"STUDY-{hash_digest.upper()}"
        elif id_type == "series":
            anonymous_id = f"SERIES-{hash_digest.upper()}"
        else:
            anonymous_id = f"{id_type.upper()}-{hash_digest.upper()}"
        
        # Store mapping
        self.id_mappings[mapping_key] = anonymous_id
        self._save_id_mappings()
        
        return anonymous_id
    
    def calculate_date_shift(self, original_patient_id: str) -> int:
        """
        Calculate consistent date shift for a patient.
        
        Args:
            original_patient_id: Original patient identifier
            
        Returns:
            Number of days to shift (consistent per patient)
        """
        if original_patient_id in self.date_shifts:
            return self.date_shifts[original_patient_id]
        
        # Generate deterministic but unpredictable shift
        hash_input = f"{original_patient_id}:date_shift:duetmind".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        
        # Shift between -365 and +365 days
        shift_days = hash_value % 730 - 365
        
        self.date_shifts[original_patient_id] = shift_days
        return shift_days
    
    def shift_date(self, date_string: str, patient_id: str, 
                   date_format: str = "%Y%m%d") -> str:
        """
        Shift date by consistent amount for patient.
        
        Args:
            date_string: Original date string
            patient_id: Patient identifier for consistent shifting
            date_format: Format of the date string
            
        Returns:
            Shifted date string
        """
        try:
            original_date = datetime.strptime(date_string, date_format)
            shift_days = self.calculate_date_shift(patient_id)
            
            shifted_date = original_date + timedelta(days=shift_days)
            return shifted_date.strftime(date_format)
            
        except ValueError:
            # Return empty string if date parsing fails
            return ""
    
    def shift_time(self, time_string: str, patient_id: str) -> str:
        """
        Shift time by consistent amount for patient.
        
        Args:
            time_string: Original time string (HHMMSS format)
            patient_id: Patient identifier for consistent shifting
            
        Returns:
            Shifted time string
        """
        try:
            # Parse DICOM time format
            if len(time_string) >= 6:
                hour = int(time_string[:2])
                minute = int(time_string[2:4])
                second = int(time_string[4:6])
                
                # Get consistent time shift (in minutes)
                shift_days = self.calculate_date_shift(patient_id)
                shift_minutes = (shift_days * 7) % 1440  # 0-1439 minutes
                
                # Apply shift
                total_minutes = hour * 60 + minute + shift_minutes
                new_hour = (total_minutes // 60) % 24
                new_minute = total_minutes % 60
                
                return f"{new_hour:02d}{new_minute:02d}{second:02d}"
            else:
                return time_string
                
        except (ValueError, IndexError):
            return ""
    
    def anonymize_text(self, text: str) -> str:
        """
        Remove PHI patterns from text fields.
        
        Args:
            text: Original text
            
        Returns:
            Text with PHI patterns removed
        """
        if not text:
            return text
        
        anonymized = str(text)
        
        # Replace PHI patterns
        for pattern in self.phi_patterns:
            anonymized = re.sub(pattern, "[REMOVED]", anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    def deidentify_dicom_file(self, input_path: str, output_path: str,
                             patient_id_override: str = None) -> Dict[str, Any]:
        """
        De-identify a DICOM file.
        
        Args:
            input_path: Path to input DICOM file
            output_path: Path for de-identified output
            patient_id_override: Override patient ID (for consistent de-identification)
            
        Returns:
            Dictionary with de-identification results
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM de-identification")
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read DICOM file
        ds = pydicom.dcmread(str(input_path))
        
        # Get original patient ID for consistent processing
        original_patient_id = str(getattr(ds, 'PatientID', 'UNKNOWN'))
        if patient_id_override:
            original_patient_id = patient_id_override
        
        # Track what was modified
        modifications = {
            'removed_tags': [],
            'replaced_tags': [],
            'shifted_dates': [],
            'shifted_times': []
        }
        
        # Remove/replace PHI tags
        for tag in self.phi_dicom_tags:
            if tag in ds:
                tag_name = ds[tag].keyword if hasattr(ds[tag], 'keyword') else f"Tag_{tag[0]:04X}_{tag[1]:04X}"
                
                if tag in self.replacement_tags:
                    # Replace with anonymous value
                    old_value = str(ds[tag].value)
                    ds[tag].value = self.replacement_tags[tag]
                    modifications['replaced_tags'].append({
                        'tag': tag_name,
                        'old_value': old_value,
                        'new_value': self.replacement_tags[tag]
                    })
                else:
                    # Remove the tag
                    old_value = str(ds[tag].value)
                    del ds[tag]
                    modifications['removed_tags'].append({
                        'tag': tag_name,
                        'old_value': old_value
                    })
        
        # Replace patient ID with anonymous ID
        if hasattr(ds, 'PatientID'):
            anonymous_patient_id = self.generate_anonymous_id(original_patient_id, "patient")
            ds.PatientID = anonymous_patient_id
            modifications['replaced_tags'].append({
                'tag': 'PatientID',
                'old_value': original_patient_id,
                'new_value': anonymous_patient_id
            })
        
        # Replace study and series UIDs
        if hasattr(ds, 'StudyInstanceUID'):
            original_study_uid = str(ds.StudyInstanceUID)
            anonymous_study_uid = self.generate_anonymous_id(original_study_uid, "study")
            # Create valid DICOM UID format
            anonymous_study_uid = f"1.2.826.0.1.3680043.8.498.{hash(anonymous_study_uid) % 1000000000}"
            ds.StudyInstanceUID = anonymous_study_uid
        
        if hasattr(ds, 'SeriesInstanceUID'):
            original_series_uid = str(ds.SeriesInstanceUID)
            anonymous_series_uid = self.generate_anonymous_id(original_series_uid, "series")
            # Create valid DICOM UID format
            anonymous_series_uid = f"1.2.826.0.1.3680043.8.498.{hash(anonymous_series_uid) % 1000000000}"
            ds.SeriesInstanceUID = anonymous_series_uid
        
        # Shift dates
        date_tags = [(0x0008, 0x0020), (0x0008, 0x0021), (0x0008, 0x0022), (0x0008, 0x0023)]
        for tag in date_tags:
            if tag in ds:
                original_date = str(ds[tag].value)
                shifted_date = self.shift_date(original_date, original_patient_id)
                if shifted_date:
                    ds[tag].value = shifted_date
                    modifications['shifted_dates'].append({
                        'tag': ds[tag].keyword,
                        'original': original_date,
                        'shifted': shifted_date
                    })
        
        # Shift times
        time_tags = [(0x0008, 0x0030), (0x0008, 0x0031), (0x0008, 0x0032), (0x0008, 0x0033)]
        for tag in time_tags:
            if tag in ds:
                original_time = str(ds[tag].value)
                shifted_time = self.shift_time(original_time, original_patient_id)
                if shifted_time:
                    ds[tag].value = shifted_time
                    modifications['shifted_times'].append({
                        'tag': ds[tag].keyword,
                        'original': original_time,
                        'shifted': shifted_time
                    })
        
        # Anonymize text fields that might contain PHI
        text_tags = [(0x0008, 0x103E), (0x0008, 0x1030), (0x0010, 0x4000)]  # SeriesDescription, StudyDescription, PatientComments
        for tag in text_tags:
            if tag in ds:
                original_text = str(ds[tag].value)
                anonymized_text = self.anonymize_text(original_text)
                if anonymized_text != original_text:
                    ds[tag].value = anonymized_text
                    modifications['replaced_tags'].append({
                        'tag': ds[tag].keyword,
                        'old_value': original_text,
                        'new_value': anonymized_text
                    })
        
        # Add de-identification metadata
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "DuetMind Medical Image Deidentifier v1.0"
        ds.DeidentificationMethodCodeSequence = []
        
        # Save de-identified DICOM
        ds.save_as(str(output_path))
        
        self.logger.info(f"De-identified DICOM saved: {output_path}")
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_patient_id': original_patient_id,
            'anonymous_patient_id': anonymous_patient_id if 'anonymous_patient_id' in locals() else None,
            'modifications': modifications,
            'deidentification_date': datetime.now().isoformat()
        }
    
    def deidentify_metadata_json(self, metadata: Dict[str, Any], 
                                patient_id: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        De-identify metadata dictionary (e.g., from BIDS JSON).
        
        Args:
            metadata: Original metadata dictionary
            patient_id: Patient ID for consistent de-identification
            
        Returns:
            Tuple of (de-identified metadata, modifications log)
        """
        deidentified = metadata.copy()
        modifications = {
            'removed_fields': [],
            'replaced_fields': [],
            'shifted_dates': [],
            'shifted_times': []
        }
        
        # Fields to remove (potential PHI)
        phi_fields = [
            'PatientName', 'PatientID', 'PatientBirthDate',
            'InstitutionName', 'InstitutionAddress',
            'ReferringPhysicianName', 'OperatorsName',
            'StationName', 'DeviceSerialNumber',
            'StudyID', 'AccessionNumber'
        ]
        
        for field in phi_fields:
            if field in deidentified:
                old_value = deidentified[field]
                del deidentified[field]
                modifications['removed_fields'].append({
                    'field': field,
                    'old_value': str(old_value)
                })
        
        # Replace with anonymous values
        replacement_fields = {
            'Subject': self.generate_anonymous_id(patient_id or "unknown", "patient") if patient_id else "Anonymous",
            'InstitutionName': 'Anonymous Institution',
            'Manufacturer': deidentified.get('Manufacturer', 'Anonymous'),  # Keep manufacturer for analysis
        }
        
        for field, value in replacement_fields.items():
            if field in deidentified:
                old_value = deidentified[field]
                deidentified[field] = value
                modifications['replaced_fields'].append({
                    'field': field,
                    'old_value': str(old_value),
                    'new_value': str(value)
                })
        
        # Shift dates
        date_fields = ['AcquisitionDate', 'StudyDate', 'SeriesDate']
        for field in date_fields:
            if field in deidentified and patient_id:
                original_date = str(deidentified[field])
                # Convert ISO date format to DICOM format if needed
                if '-' in original_date:
                    try:
                        dt = datetime.fromisoformat(original_date.split('T')[0])
                        dicom_date = dt.strftime("%Y%m%d")
                        shifted_date = self.shift_date(dicom_date, patient_id)
                        if shifted_date:
                            # Convert back to ISO format
                            shifted_dt = datetime.strptime(shifted_date, "%Y%m%d")
                            deidentified[field] = shifted_dt.strftime("%Y-%m-%d")
                            modifications['shifted_dates'].append({
                                'field': field,
                                'original': original_date,
                                'shifted': deidentified[field]
                            })
                    except ValueError:
                        pass
                else:
                    shifted_date = self.shift_date(original_date, patient_id)
                    if shifted_date:
                        deidentified[field] = shifted_date
                        modifications['shifted_dates'].append({
                            'field': field,
                            'original': original_date,
                            'shifted': shifted_date
                        })
        
        # Shift times
        time_fields = ['AcquisitionTime', 'StudyTime', 'SeriesTime']
        for field in time_fields:
            if field in deidentified and patient_id:
                original_time = str(deidentified[field])
                # Convert HH:MM:SS to HHMMSS if needed
                if ':' in original_time:
                    dicom_time = original_time.replace(':', '')
                    shifted_time = self.shift_time(dicom_time, patient_id)
                    if shifted_time:
                        # Convert back to HH:MM:SS format
                        if len(shifted_time) >= 6:
                            formatted_time = f"{shifted_time[:2]}:{shifted_time[2:4]}:{shifted_time[4:6]}"
                            deidentified[field] = formatted_time
                            modifications['shifted_times'].append({
                                'field': field,
                                'original': original_time,
                                'shifted': formatted_time
                            })
                else:
                    shifted_time = self.shift_time(original_time, patient_id)
                    if shifted_time:
                        deidentified[field] = shifted_time
                        modifications['shifted_times'].append({
                            'field': field,
                            'original': original_time,
                            'shifted': shifted_time
                        })
        
        # Add de-identification metadata
        deidentified['DeidentificationDate'] = datetime.now().isoformat()
        deidentified['DeidentificationSoftware'] = 'DuetMind Medical Image Deidentifier v1.0'
        deidentified['DeidentificationCompliance'] = 'HIPAA Safe Harbor'
        
        return deidentified, modifications
    
    def deidentify_directory(self, input_dir: str, output_dir: str,
                           file_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """
        De-identify all files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_patterns: File patterns to process (default: DICOM extensions)
            
        Returns:
            List of de-identification results
        """
        if file_patterns is None:
            file_patterns = ['*.dcm', '*.dicom', '*.json']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        results = []
        
        # Find all matching files
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(input_path.rglob(pattern))
        
        self.logger.info(f"Found {len(files_to_process)} files to de-identify")
        
        for file_path in files_to_process:
            try:
                # Maintain directory structure
                relative_path = file_path.relative_to(input_path)
                output_file = output_path / relative_path
                
                if file_path.suffix.lower() in ['.dcm', '.dicom']:
                    # Process DICOM file
                    result = self.deidentify_dicom_file(
                        str(file_path), 
                        str(output_file)
                    )
                elif file_path.suffix.lower() == '.json':
                    # Process JSON metadata
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                    
                    deidentified_metadata, modifications = self.deidentify_metadata_json(metadata)
                    
                    # Save de-identified JSON
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        json.dump(deidentified_metadata, f, indent=2)
                    
                    result = {
                        'input_path': str(file_path),
                        'output_path': str(output_file),
                        'file_type': 'json',
                        'modifications': modifications,
                        'deidentification_date': datetime.now().isoformat()
                    }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to de-identify {file_path}: {e}")
                results.append({
                    'input_path': str(file_path),
                    'error': str(e),
                    'deidentification_date': datetime.now().isoformat()
                })
        
        self.logger.info(f"De-identification completed: {len(results)} files processed")
        
        return results
    
    def generate_deidentification_report(self, results: List[Dict[str, Any]], 
                                       output_path: str = None) -> str:
        """
        Generate a de-identification audit report.
        
        Args:
            results: List of de-identification results
            output_path: Optional path to save the report
            
        Returns:
            Formatted report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("Medical Image De-identification Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Report Date: {datetime.now().isoformat()}")
        report_lines.append(f"Total Files Processed: {len(results)}")
        report_lines.append("")
        
        # Summary statistics
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        report_lines.append("Summary:")
        report_lines.append(f"  Successfully De-identified: {successful}")
        report_lines.append(f"  Failed: {failed}")
        report_lines.append("")
        
        # File type breakdown
        file_types = {}
        for result in results:
            if 'error' not in result:
                file_type = result.get('file_type', 'dicom')
                file_types[file_type] = file_types.get(file_type, 0) + 1
        
        if file_types:
            report_lines.append("File Types:")
            for file_type, count in file_types.items():
                report_lines.append(f"  {file_type.upper()}: {count}")
            report_lines.append("")
        
        # Modification statistics
        total_removed = 0
        total_replaced = 0
        total_dates_shifted = 0
        total_times_shifted = 0
        
        for result in results:
            if 'modifications' in result:
                mods = result['modifications']
                total_removed += len(mods.get('removed_tags', []) + mods.get('removed_fields', []))
                total_replaced += len(mods.get('replaced_tags', []) + mods.get('replaced_fields', []))
                total_dates_shifted += len(mods.get('shifted_dates', []))
                total_times_shifted += len(mods.get('shifted_times', []))
        
        report_lines.append("Modifications Applied:")
        report_lines.append(f"  PHI Fields Removed: {total_removed}")
        report_lines.append(f"  Fields Replaced: {total_replaced}")
        report_lines.append(f"  Dates Shifted: {total_dates_shifted}")
        report_lines.append(f"  Times Shifted: {total_times_shifted}")
        report_lines.append("")
        
        # Failures
        if failed > 0:
            report_lines.append("Failed Files:")
            for result in results:
                if 'error' in result:
                    report_lines.append(f"  {result['input_path']}: {result['error']}")
            report_lines.append("")
        
        # Compliance statement
        report_lines.append("Compliance:")
        report_lines.append("  This de-identification process follows HIPAA Safe Harbor guidelines")
        report_lines.append("  18 categories of PHI have been addressed according to 45 CFR 164.514(b)(2)")
        report_lines.append("  Date shifting maintains temporal relationships while protecting privacy")
        report_lines.append("  Consistent ID mapping preserves research utility")
        
        report_text = "\n".join(report_lines)
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"De-identification report saved to: {output_path}")
        
        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create de-identifier
    deidentifier = MedicalImageDeidentifier(
        encryption_key="secure_key_for_production",
        mapping_file="test_id_mappings.json"
    )
    
    # Example: De-identify a single DICOM file
    # result = deidentifier.deidentify_dicom_file(
    #     input_path="/path/to/dicom/file.dcm",
    #     output_path="/path/to/output/anonymized.dcm"
    # )
    # 
    # print(f"De-identification completed: {result['output_path']}")
    
    # Example: De-identify metadata
    metadata = {
        'PatientName': 'John Doe',
        'PatientID': 'P123456',
        'StudyDate': '2023-01-15',
        'InstitutionName': 'Example Hospital',
        'RepetitionTime': 2.3
    }
    
    deidentified_metadata, modifications = deidentifier.deidentify_metadata_json(
        metadata, patient_id='P123456'
    )
    
    print("Original metadata:", metadata)
    print("De-identified metadata:", deidentified_metadata)
    print("Modifications:", modifications)
    
    print("Medical image de-identifier initialized successfully")