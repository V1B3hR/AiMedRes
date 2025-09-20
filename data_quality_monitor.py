#!/usr/bin/env python3
"""
Data Validation and Quality Monitoring for duetmind_adaptive
Comprehensive data quality checks and monitoring for training and simulation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
import warnings
from datetime import datetime

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Visualization libraries not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataQuality")

class DataQualityMonitor:
    """Comprehensive data quality monitoring and validation"""
    
    def __init__(self):
        self.quality_report = {}
        self.validation_rules = {}
        self.anomalies = []
        
    def validate_alzheimer_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate Alzheimer's dataset for quality and integrity"""
        logger.info("Starting comprehensive data quality validation...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self._get_dataset_info(df),
            'completeness': self._check_completeness(df),
            'consistency': self._check_consistency(df),
            'validity': self._check_validity(df),
            'uniqueness': self._check_uniqueness(df),
            'medical_validity': self._check_medical_validity(df),
            'anomalies': self._detect_anomalies(df),
            'recommendations': [],
            'overall_score': 0.0
        }
        
        # Calculate overall quality score
        scores = []
        scores.append(report['completeness']['score'])
        scores.append(report['consistency']['score'])
        scores.append(report['validity']['score'])
        scores.append(report['uniqueness']['score'])
        scores.append(report['medical_validity']['score'])
        
        report['overall_score'] = np.mean(scores)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        self.quality_report = report
        logger.info(f"Data quality validation complete. Overall score: {report['overall_score']:.3f}")
        
        return report
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'total_records': len(df),
            'total_features': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage().sum(),
            'shape': df.shape
        }
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Critical columns that shouldn't have missing values
        critical_columns = ['Group', 'Age', 'MMSE', 'CDR']
        critical_missing = missing_percentages[critical_columns].sum()
        
        # Calculate completeness score
        overall_completeness = (1 - (missing_counts.sum() / (len(df) * len(df.columns)))) * 100
        critical_completeness = (1 - (critical_missing / (len(critical_columns) * 100))) * 100
        
        score = (overall_completeness * 0.6 + critical_completeness * 0.4) / 100
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'overall_completeness': overall_completeness,
            'critical_completeness': critical_completeness,
            'score': score,
            'status': 'good' if score >= 0.85 else 'fair' if score >= 0.7 else 'poor'
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_issues = []
        
        # Check for impossible age values
        if 'Age' in df.columns:
            invalid_ages = df[(df['Age'] < 0) | (df['Age'] > 120)]
            if len(invalid_ages) > 0:
                consistency_issues.append(f"Invalid age values: {len(invalid_ages)} records")
        
        # Check MMSE score consistency (should be 0-30)
        if 'MMSE' in df.columns:
            invalid_mmse = df[(df['MMSE'] < 0) | (df['MMSE'] > 30)]
            if len(invalid_mmse) > 0:
                consistency_issues.append(f"Invalid MMSE scores: {len(invalid_mmse)} records")
        
        # Check CDR consistency (should be 0, 0.5, 1, 2, 3)
        if 'CDR' in df.columns:
            valid_cdr_values = [0, 0.5, 1, 2, 3]
            invalid_cdr = df[~df['CDR'].isin(valid_cdr_values) & df['CDR'].notna()]
            if len(invalid_cdr) > 0:
                consistency_issues.append(f"Invalid CDR values: {len(invalid_cdr)} records")
        
        # Check education years consistency
        if 'EDUC' in df.columns:
            invalid_education = df[(df['EDUC'] < 0) | (df['EDUC'] > 25)]
            if len(invalid_education) > 0:
                consistency_issues.append(f"Invalid education values: {len(invalid_education)} records")
        
        score = max(0, 1 - (len(consistency_issues) / 10))  # Normalize to 0-1
        
        return {
            'issues': consistency_issues,
            'issue_count': len(consistency_issues),
            'score': score,
            'status': 'good' if score >= 0.9 else 'fair' if score >= 0.7 else 'poor'
        }
    
    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        validity_issues = []
        
        # Check for required columns
        required_columns = ['Group', 'Age', 'MMSE', 'CDR']
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            validity_issues.extend([f"Missing required column: {col}" for col in missing_required])
        
        # Check data types
        expected_numeric = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
        for col in expected_numeric:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                validity_issues.append(f"Column {col} should be numeric")
        
        # Check Group values
        if 'Group' in df.columns:
            valid_groups = ['Demented', 'Nondemented', 'Converted']
            invalid_groups = df[~df['Group'].isin(valid_groups) & df['Group'].notna()]
            if len(invalid_groups) > 0:
                validity_issues.append(f"Invalid Group values: {len(invalid_groups)} records")
        
        # Check gender values
        if 'M/F' in df.columns:
            valid_genders = ['M', 'F', 1, 0]
            invalid_genders = df[~df['M/F'].isin(valid_genders) & df['M/F'].notna()]
            if len(invalid_genders) > 0:
                validity_issues.append(f"Invalid gender values: {len(invalid_genders)} records")
        
        score = max(0, 1 - (len(validity_issues) / 8))  # Normalize to 0-1
        
        return {
            'issues': validity_issues,
            'issue_count': len(validity_issues),
            'score': score,
            'status': 'good' if score >= 0.9 else 'fair' if score >= 0.7 else 'poor'
        }
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness"""
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        
        # Check for potential identifier duplicates if any exist
        unique_analysis = {}
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['Group', 'M/F']:
                unique_count = df[col].nunique()
                unique_percentage = (unique_count / len(df)) * 100
                unique_analysis[col] = {
                    'unique_count': unique_count,
                    'unique_percentage': unique_percentage
                }
        
        score = max(0, 1 - (duplicate_percentage / 100))
        
        return {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'unique_analysis': unique_analysis,
            'score': score,
            'status': 'good' if score >= 0.95 else 'fair' if score >= 0.9 else 'poor'
        }
    
    def _check_medical_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check medical validity and logical relationships"""
        medical_issues = []
        
        # Check MMSE vs CDR correlation (they should be inversely related)
        if 'MMSE' in df.columns and 'CDR' in df.columns:
            valid_data = df[['MMSE', 'CDR']].dropna()
            if len(valid_data) > 0:
                correlation = valid_data['MMSE'].corr(valid_data['CDR'])
                if correlation > -0.3:  # Should be negative correlation
                    medical_issues.append(f"Weak MMSE-CDR correlation: {correlation:.3f}")
        
        # Check for logical inconsistencies
        if 'Group' in df.columns and 'CDR' in df.columns:
            # Nondemented patients should generally have CDR <= 0.5
            nondemented_high_cdr = df[(df['Group'] == 'Nondemented') & (df['CDR'] > 0.5)]
            if len(nondemented_high_cdr) > 0:
                medical_issues.append(f"Nondemented with high CDR: {len(nondemented_high_cdr)} cases")
            
            # Demented patients should generally have CDR >= 0.5
            demented_low_cdr = df[(df['Group'] == 'Demented') & (df['CDR'] < 0.5)]
            if len(demented_low_cdr) > 0:
                medical_issues.append(f"Demented with low CDR: {len(demented_low_cdr)} cases")
        
        # Check age distribution
        if 'Age' in df.columns:
            age_median = df['Age'].median()
            if age_median < 60:
                medical_issues.append(f"Unusually low median age: {age_median}")
        
        score = max(0, 1 - (len(medical_issues) / 5))
        
        return {
            'issues': medical_issues,
            'issue_count': len(medical_issues),
            'score': score,
            'status': 'good' if score >= 0.8 else 'fair' if score >= 0.6 else 'poor'
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical anomalies in the data"""
        anomalies = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(df)) * 100
                    anomalies.append({
                        'column': col,
                        'outlier_count': len(outliers),
                        'outlier_percentage': outlier_percentage,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    })
        
        return {
            'statistical_outliers': anomalies,
            'total_outlier_columns': len(anomalies)
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        # Completeness recommendations
        if report['completeness']['score'] < 0.8:
            recommendations.append("Consider imputation strategies for missing values")
            if report['completeness']['critical_completeness'] < 90:
                recommendations.append("Critical columns have missing values - investigate data collection")
        
        # Consistency recommendations
        if report['consistency']['score'] < 0.9:
            recommendations.append("Review and correct data entry procedures")
            recommendations.append("Implement validation rules during data collection")
        
        # Validity recommendations
        if report['validity']['score'] < 0.9:
            recommendations.append("Standardize data formats and valid value ranges")
            recommendations.append("Implement data type validation")
        
        # Uniqueness recommendations
        if report['uniqueness']['score'] < 0.9:
            recommendations.append("Remove duplicate records")
            recommendations.append("Implement deduplication procedures")
        
        # Medical validity recommendations
        if report['medical_validity']['score'] < 0.8:
            recommendations.append("Review medical data for logical consistency")
            recommendations.append("Consult medical experts for validation rules")
        
        # Overall score recommendations
        if report['overall_score'] < 0.7:
            recommendations.append("Comprehensive data cleaning required before training")
        elif report['overall_score'] < 0.85:
            recommendations.append("Moderate data quality improvements recommended")
        
        return recommendations
    
    def generate_quality_report_text(self) -> str:
        """Generate a human-readable quality report"""
        if not self.quality_report:
            return "No quality report available. Run validate_alzheimer_dataset() first."
        
        report = self.quality_report
        
        text_report = f"""
DATA QUALITY REPORT
Generated: {report['timestamp']}

=== DATASET OVERVIEW ===
Records: {report['dataset_info']['total_records']:,}
Features: {report['dataset_info']['total_features']}
Memory Usage: {report['dataset_info']['memory_usage'] / 1024:.1f} KB

=== QUALITY SCORES ===
Overall Score: {report['overall_score']:.3f}/1.0
- Completeness: {report['completeness']['score']:.3f} ({report['completeness']['status']})
- Consistency: {report['consistency']['score']:.3f} ({report['consistency']['status']})
- Validity: {report['validity']['score']:.3f} ({report['validity']['status']})
- Uniqueness: {report['uniqueness']['score']:.3f} ({report['uniqueness']['status']})
- Medical Validity: {report['medical_validity']['score']:.3f} ({report['medical_validity']['status']})

=== COMPLETENESS ANALYSIS ===
Overall Completeness: {report['completeness']['overall_completeness']:.1f}%
Critical Columns Completeness: {report['completeness']['critical_completeness']:.1f}%
Missing Data Issues: {len([k for k, v in report['completeness']['missing_percentages'].items() if v > 0])}

=== CONSISTENCY ISSUES ===
Total Issues: {report['consistency']['issue_count']}"""
        
        for issue in report['consistency']['issues']:
            text_report += f"\n- {issue}"
        
        text_report += f"""

=== VALIDITY ISSUES ===
Total Issues: {report['validity']['issue_count']}"""
        
        for issue in report['validity']['issues']:
            text_report += f"\n- {issue}"
        
        text_report += f"""

=== UNIQUENESS ANALYSIS ===
Duplicate Rows: {report['uniqueness']['duplicate_rows']} ({report['uniqueness']['duplicate_percentage']:.2f}%)

=== MEDICAL VALIDITY ===
Medical Issues: {report['medical_validity']['issue_count']}"""
        
        for issue in report['medical_validity']['issues']:
            text_report += f"\n- {issue}"
        
        text_report += f"""

=== ANOMALIES DETECTED ===
Columns with Outliers: {report['anomalies']['total_outlier_columns']}"""
        
        for anomaly in report['anomalies']['statistical_outliers']:
            text_report += f"\n- {anomaly['column']}: {anomaly['outlier_count']} outliers ({anomaly['outlier_percentage']:.1f}%)"
        
        text_report += f"""

=== RECOMMENDATIONS ==="""
        
        for i, rec in enumerate(report['recommendations'], 1):
            text_report += f"\n{i}. {rec}"
        
        text_report += f"""

=== SUMMARY ===
Quality Level: {'Excellent' if report['overall_score'] >= 0.9 else 'Good' if report['overall_score'] >= 0.8 else 'Fair' if report['overall_score'] >= 0.7 else 'Poor'}
Ready for Training: {'Yes' if report['overall_score'] >= 0.7 else 'No - cleaning required'}
"""
        
        return text_report

def validate_training_data():
    """Standalone function to validate training data"""
    from files.training.alzheimer_training_system import load_alzheimer_data
    
    logger.info("Loading dataset for quality validation...")
    df = load_alzheimer_data(file_path="alzheimer.csv")
    
    monitor = DataQualityMonitor()
    report = monitor.validate_alzheimer_dataset(df)
    
    print(monitor.generate_quality_report_text())
    
    return report, monitor

if __name__ == "__main__":
    validate_training_data()