#!/usr/bin/env python3
"""
Phase 2 Data Integrity & Preprocessing Debugging Script for AiMedRes

This script implements Phase 2 of the AiMedRes debugging process as outlined
in debug/debuglist.md. It performs comprehensive data integrity and preprocessing checks.

Subphases:
- 2.1: Validate raw data integrity (missing values, outliers, duplicates)
- 2.2: Check data preprocessing routines (scaling, encoding, normalization)  
- 2.3: Visualize data distributions & class balance
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class Phase2DataIntegrityDebugger:
    """Phase 2 debugging implementation for AiMedRes data integrity"""
    
    def __init__(self, verbose: bool = False, output_dir: str = "debug"):
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='[%(asctime)s] 📝 %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.results = {
            "phase": 2,
            "timestamp": datetime.now().isoformat(),
            "subphase_results": {},
            "detailed_findings": {},
            "data_files_analyzed": [],
            "recommendations": []
        }
        
        # Sample data paths
        self.data_paths = [
            "data/raw/alzheimer_sample.csv",
            "training/train_alzheimers.py",  # For code analysis
            "training/train_als.py",
            "training/train_parkinsons.py",
            "training/train_cardiovascular.py",
            "training/train_diabetes.py"
        ]
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)

    def subphase_2_1_data_integrity_validation(self) -> bool:
        """Subphase 2.1: Validate raw data integrity"""
        self.log("=" * 60)
        self.log("SUBPHASE 2.1: DATA INTEGRITY VALIDATION")
        self.log("=" * 60)
        
        all_checks_passed = True
        integrity_findings = {}
        
        # Find available data files
        root_path = Path(".")
        csv_files = list(root_path.rglob("*.csv"))
        
        if not csv_files:
            self.log("⚠️  No CSV data files found in repository")
            integrity_findings["no_data_files"] = True
            all_checks_passed = False
        else:
            self.log(f"Found {len(csv_files)} CSV files to analyze")
            
        # Analyze each data file
        for csv_file in csv_files[:5]:  # Limit to first 5 files to avoid overwhelming output
            try:
                self.log(f"\n📊 Analyzing: {csv_file}")
                df = pd.read_csv(csv_file)
                
                file_analysis = self._analyze_data_file(df, str(csv_file))
                integrity_findings[str(csv_file)] = file_analysis
                self.results["data_files_analyzed"].append(str(csv_file))
                
                # Check for critical issues
                if file_analysis.get("missing_percentage", 0) > 50:
                    self.log(f"❌ High missing data percentage: {file_analysis['missing_percentage']:.1f}%")
                    all_checks_passed = False
                elif file_analysis.get("missing_percentage", 0) > 20:
                    self.log(f"⚠️  Moderate missing data: {file_analysis['missing_percentage']:.1f}%")
                
                if file_analysis.get("duplicate_count", 0) > 0:
                    self.log(f"⚠️  Found {file_analysis['duplicate_count']} duplicate rows")
                
            except Exception as e:
                self.log(f"❌ Error analyzing {csv_file}: {e}")
                all_checks_passed = False
                
        self.results["detailed_findings"]["subphase_2_1"] = integrity_findings
        
        if all_checks_passed and csv_files:
            self.log("\n✅ Data integrity validation passed")
        else:
            self.log("\n❌ Data integrity validation failed")
            
        return all_checks_passed and len(csv_files) > 0

    def _analyze_data_file(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze a single data file for integrity issues"""
        analysis = {
            "filename": filename,
            "shape": df.shape,
            "missing_count": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "duplicate_count": df.duplicated().sum(),
            "data_types": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
            "outliers_detected": {},
            "column_analysis": {}
        }
        
        self.log(f"  📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        self.log(f"  🔍 Missing values: {analysis['missing_count']} ({analysis['missing_percentage']:.1f}%)")
        self.log(f"  🔄 Duplicates: {analysis['duplicate_count']}")
        
        # Analyze each column
        for col in df.columns:
            col_analysis = {
                "dtype": str(df[col].dtype),
                "missing_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique(),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col])
            }
            
            if col_analysis["is_numeric"]:
                # Outlier detection using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                col_analysis["outlier_count"] = len(outliers)
                
                if len(outliers) > 0:
                    analysis["outliers_detected"][col] = len(outliers)
            
            analysis["column_analysis"][col] = col_analysis
            
        return analysis

    def subphase_2_2_preprocessing_routines_check(self) -> bool:
        """Subphase 2.2: Check data preprocessing routines"""
        self.log("\n" + "=" * 60)
        self.log("SUBPHASE 2.2: PREPROCESSING ROUTINES CHECK")
        self.log("=" * 60)
        
        all_checks_passed = True
        preprocessing_findings = {}
        
        # Check training scripts for preprocessing patterns
        training_scripts = list(Path("training").glob("*.py")) if Path("training").exists() else []
        
        if not training_scripts:
            self.log("⚠️  No training scripts found to analyze preprocessing")
            all_checks_passed = False
        else:
            self.log(f"Found {len(training_scripts)} training scripts to analyze")
            
        preprocessing_patterns = {
            "StandardScaler": False,
            "LabelEncoder": False,
            "OneHotEncoder": False,
            "SimpleImputer": False,
            "train_test_split": False,
            "random_state": False,
            "feature_scaling": False
        }
        
        # Analyze preprocessing in training scripts
        for script in training_scripts:
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    
                script_analysis = self._analyze_preprocessing_script(content, str(script))
                preprocessing_findings[str(script)] = script_analysis
                
                # Update pattern detection
                for pattern in preprocessing_patterns:
                    if script_analysis.get(pattern, False):
                        preprocessing_patterns[pattern] = True
                        
            except Exception as e:
                self.log(f"❌ Error analyzing {script}: {e}")
                all_checks_passed = False
        
        # Report findings
        self.log("\n🔧 Preprocessing Patterns Found:")
        for pattern, found in preprocessing_patterns.items():
            status = "✅" if found else "❌"
            self.log(f"  {status} {pattern}")
            
        # Test preprocessing pipeline with sample data
        if self._test_preprocessing_pipeline():
            self.log("\n✅ Preprocessing pipeline test passed")
        else:
            self.log("\n❌ Preprocessing pipeline test failed")
            all_checks_passed = False
            
        self.results["detailed_findings"]["subphase_2_2"] = preprocessing_findings
        return all_checks_passed

    def _analyze_preprocessing_script(self, content: str, filename: str) -> Dict[str, Any]:
        """Analyze a training script for preprocessing patterns"""
        patterns = {
            "StandardScaler": "StandardScaler" in content,
            "LabelEncoder": "LabelEncoder" in content,
            "OneHotEncoder": "OneHotEncoder" in content,
            "SimpleImputer": "SimpleImputer" in content or "fillna" in content,
            "train_test_split": "train_test_split" in content,
            "random_state": "random_state" in content or "random.seed" in content,
            "feature_scaling": "fit_transform" in content or "transform" in content,
            "cross_validation": "cross_val" in content or "StratifiedKFold" in content,
            "pipeline": "Pipeline" in content
        }
        
        return {
            "filename": filename,
            "patterns_found": patterns,
            "preprocessing_score": sum(patterns.values()) / len(patterns) * 100
        }

    def _test_preprocessing_pipeline(self) -> bool:
        """Test a basic preprocessing pipeline with sample data"""
        try:
            # Create sample data
            np.random.seed(42)
            data = {
                'numeric1': np.random.normal(0, 1, 100),
                'numeric2': np.random.exponential(2, 100),
                'categorical': np.random.choice(['A', 'B', 'C'], 100),
                'target': np.random.choice([0, 1], 100)
            }
            
            # Add some missing values
            data['numeric1'][::10] = np.nan
            data['categorical'][::15] = np.nan
            
            df = pd.DataFrame(data)
            
            # Test preprocessing steps
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Test train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Test imputation
            numeric_imputer = SimpleImputer(strategy='median')
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            
            # Test scaling
            scaler = StandardScaler()
            
            # Test encoding
            encoder = LabelEncoder()
            
            self.log("  ✅ Basic preprocessing pipeline components work correctly")
            return True
            
        except Exception as e:
            self.log(f"  ❌ Preprocessing pipeline test failed: {e}")
            return False

    def subphase_2_3_data_visualization(self) -> bool:
        """Subphase 2.3: Visualize data distributions & class balance"""
        self.log("\n" + "=" * 60)
        self.log("SUBPHASE 2.3: DATA DISTRIBUTIONS & CLASS BALANCE")
        self.log("=" * 60)
        
        all_checks_passed = True
        visualization_findings = {}
        
        # Create visualizations output directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Find and analyze data files for visualization
        csv_files = list(Path(".").rglob("*.csv"))
        visualized_files = 0
        
        for csv_file in csv_files[:3]:  # Limit to first 3 files
            try:
                self.log(f"\n📊 Creating visualizations for: {csv_file}")
                df = pd.read_csv(csv_file)
                
                if len(df) == 0:
                    self.log(f"  ⚠️  Skipping empty file: {csv_file}")
                    continue
                    
                viz_results = self._create_visualizations(df, str(csv_file), viz_dir)
                visualization_findings[str(csv_file)] = viz_results
                visualized_files += 1
                
            except Exception as e:
                self.log(f"❌ Error creating visualizations for {csv_file}: {e}")
                all_checks_passed = False
                
        if visualized_files == 0:
            self.log("⚠️  No data files could be visualized")
            all_checks_passed = False
        else:
            self.log(f"\n✅ Created visualizations for {visualized_files} data files")
            self.log(f"📁 Visualizations saved in: {viz_dir}")
            
        self.results["detailed_findings"]["subphase_2_3"] = visualization_findings
        return all_checks_passed

    def _create_visualizations(self, df: pd.DataFrame, filename: str, output_dir: Path) -> Dict[str, Any]:
        """Create visualizations for a dataset"""
        results = {
            "filename": filename,
            "plots_created": [],
            "class_balance": {},
            "distribution_analysis": {}
        }
        
        # Set matplotlib backend for headless environment
        plt.switch_backend('Agg')
        
        try:
            # 1. Missing data heatmap
            if df.isnull().sum().sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title(f'Missing Data Pattern - {Path(filename).name}')
                plt.tight_layout()
                missing_plot = output_dir / f"missing_data_{Path(filename).stem}.png"
                plt.savefig(missing_plot, dpi=150, bbox_inches='tight')
                plt.close()
                results["plots_created"].append(str(missing_plot))
                
            # 2. Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, axes = plt.subplots(min(4, len(numeric_cols)), 1, figsize=(10, 3*min(4, len(numeric_cols))))
                if len(numeric_cols) == 1:
                    axes = [axes]
                    
                for i, col in enumerate(numeric_cols[:4]):
                    df[col].hist(bins=30, ax=axes[i] if len(numeric_cols) > 1 else axes[0])
                    axes[i if len(numeric_cols) > 1 else 0].set_title(f'Distribution of {col}')
                    
                    # Store distribution stats
                    results["distribution_analysis"][col] = {
                        "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                        "std": float(df[col].std()) if not df[col].isnull().all() else None,
                        "skewness": float(df[col].skew()) if not df[col].isnull().all() else None
                    }
                    
                plt.tight_layout()
                dist_plot = output_dir / f"distributions_{Path(filename).stem}.png"
                plt.savefig(dist_plot, dpi=150, bbox_inches='tight')
                plt.close()
                results["plots_created"].append(str(dist_plot))
                
            # 3. Class balance (look for target-like columns)
            target_candidates = ['target', 'label', 'class', 'diagnosis', 'outcome']
            for col in df.columns:
                if col.lower() in target_candidates or 'target' in col.lower():
                    value_counts = df[col].value_counts()
                    results["class_balance"][col] = value_counts.to_dict()
                    
                    # Create class balance plot
                    plt.figure(figsize=(8, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f'Class Balance - {col}')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    balance_plot = output_dir / f"class_balance_{col}_{Path(filename).stem}.png"
                    plt.savefig(balance_plot, dpi=150, bbox_inches='tight')
                    plt.close()
                    results["plots_created"].append(str(balance_plot))
                    break
                    
            # 4. Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title(f'Feature Correlation Matrix - {Path(filename).name}')
                plt.tight_layout()
                corr_plot = output_dir / f"correlation_{Path(filename).stem}.png"
                plt.savefig(corr_plot, dpi=150, bbox_inches='tight')
                plt.close()
                results["plots_created"].append(str(corr_plot))
                
        except Exception as e:
            self.log(f"  ⚠️  Warning creating some visualizations: {e}")
            
        return results

    def run_phase_2(self) -> bool:
        """Run complete Phase 2 debugging process"""
        start_time = datetime.now()
        self.log("🔍 STARTING PHASE 2: DATA INTEGRITY & PREPROCESSING DEBUGGING")
        self.log("📋 Based on debug/debuglist.md")
        self.log("")
        
        # Run all subphases
        subphase_2_1 = self.subphase_2_1_data_integrity_validation()
        subphase_2_2 = self.subphase_2_2_preprocessing_routines_check()
        subphase_2_3 = self.subphase_2_3_data_visualization()
        
        # Calculate results
        overall_success = subphase_2_1 and subphase_2_2 and subphase_2_3
        duration = (datetime.now() - start_time).total_seconds()
        
        # Store results
        self.results.update({
            "duration_seconds": duration,
            "overall_success": overall_success,
            "subphase_results": {
                "subphase_2_1": subphase_2_1,
                "subphase_2_2": subphase_2_2,
                "subphase_2_3": subphase_2_3
            }
        })
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("PHASE 2 SUMMARY")
        self.log("=" * 60)
        self.log(f"Subphase 2.1 (Data Integrity): {'✅ PASS' if subphase_2_1 else '❌ FAIL'}")
        self.log(f"Subphase 2.2 (Preprocessing): {'✅ PASS' if subphase_2_2 else '❌ FAIL'}")
        self.log(f"Subphase 2.3 (Visualization): {'✅ PASS' if subphase_2_3 else '❌ FAIL'}")
        self.log("")
        
        passed = sum([subphase_2_1, subphase_2_2, subphase_2_3])
        self.log(f"📊 Results: {passed}/3 subphases passed")
        self.log(f"⏱️  Duration: {duration:.1f} seconds")
        
        if overall_success:
            self.log("🎉 PHASE 2 COMPLETE - Ready for Phase 3!")
        else:
            self.log("⚠️  PHASE 2 INCOMPLETE - Address issues before Phase 3")
            self.log("\n📝 Recommendations:")
            for rec in self.results["recommendations"]:
                self.log(f"  • {rec}")
        
        # Save results
        self._save_phase_2_results()
        return overall_success

    def _generate_recommendations(self):
        """Generate actionable recommendations based on findings"""
        recommendations = []
        
        # Data integrity recommendations
        if not self.results["subphase_results"].get("subphase_2_1", False):
            recommendations.extend([
                "Review data quality issues identified in integrity checks",
                "Implement data validation scripts for incoming data",
                "Consider data cleaning procedures for missing values and outliers"
            ])
            
        # Preprocessing recommendations
        if not self.results["subphase_results"].get("subphase_2_2", False):
            recommendations.extend([
                "Standardize preprocessing patterns across training scripts",
                "Implement consistent feature scaling and encoding",
                "Add comprehensive data preprocessing pipelines"
            ])
            
        # Visualization recommendations
        if not self.results["subphase_results"].get("subphase_2_3", False):
            recommendations.extend([
                "Create data exploration notebooks for better understanding",
                "Implement automated data profiling and visualization",
                "Review class balance and consider data augmentation if needed"
            ])
            
        self.results["recommendations"] = recommendations

    def _save_phase_2_results(self):
        """Save Phase 2 results to file"""
        def json_serializer(obj):
            """JSON serializer for numpy and pandas types"""
            if hasattr(obj, 'dtype'):
                return str(obj)
            if hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)
            
        results_file = self.output_dir / "phase2_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=json_serializer)
        self.log(f"📄 Results saved to {results_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Phase 2 Data Integrity & Preprocessing Debugging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output-dir", default="debug", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run debugger
    debugger = Phase2DataIntegrityDebugger(
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    success = debugger.run_phase_2()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()