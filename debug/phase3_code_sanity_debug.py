#!/usr/bin/env python3
"""
PHASE 3: CODE SANITY & LOGICAL ERROR CHECKS
Debug script for comprehensive code validation in AiMedRes
"""

import ast
import os
import sys
import importlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import inspect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase3CodeSanity")


class CodeSanityChecker:
    """Comprehensive code sanity and logical error checker"""
    
    def __init__(self, repo_root: str = "/home/runner/work/AiMedRes/AiMedRes"):
        self.repo_root = Path(repo_root)
        self.python_files = []
        self.issues = {
            'syntax_errors': [],
            'import_errors': [],
            'logical_errors': [],
            'ml_api_issues': [],
            'utility_function_issues': []
        }
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the repository"""
        python_files = []
        
        # Key directories to check
        key_dirs = [
            self.repo_root,
            self.repo_root / "src",
            self.repo_root / "training",
            self.repo_root / "scripts",
            self.repo_root / "tests",
            self.repo_root / "api",
            self.repo_root / "examples"
        ]
        
        for directory in key_dirs:
            if directory.exists():
                python_files.extend(list(directory.rglob("*.py")))
        
        # Remove duplicates and filter out __pycache__, .pyc files
        python_files = list(set([f for f in python_files if "__pycache__" not in str(f)]))
        self.python_files = python_files
        return python_files
    
    def subphase_3_1_syntax_import_logical_errors(self) -> Dict[str, Any]:
        """
        Subphase 3.1: Review code for syntax, import, and logical errors
        """
        logger.info("=== SUBPHASE 3.1: Syntax, Import, and Logical Error Checks ===")
        
        results = {
            'syntax_errors': [],
            'import_errors': [],
            'logical_errors': [],
            'files_checked': 0,
            'clean_files': 0
        }
        
        for py_file in self.python_files:
            results['files_checked'] += 1
            file_issues = self._check_single_file_syntax_imports(py_file)
            
            if file_issues['syntax_errors']:
                results['syntax_errors'].extend(file_issues['syntax_errors'])
            if file_issues['import_errors']:
                results['import_errors'].extend(file_issues['import_errors'])
            if file_issues['logical_errors']:
                results['logical_errors'].extend(file_issues['logical_errors'])
            
            if not any([file_issues['syntax_errors'], file_issues['import_errors'], file_issues['logical_errors']]):
                results['clean_files'] += 1
        
        logger.info(f"Files checked: {results['files_checked']}")
        logger.info(f"Clean files: {results['clean_files']}")
        logger.info(f"Syntax errors found: {len(results['syntax_errors'])}")
        logger.info(f"Import errors found: {len(results['import_errors'])}")
        logger.info(f"Logical errors found: {len(results['logical_errors'])}")
        
        return results
    
    def _check_single_file_syntax_imports(self, file_path: Path) -> Dict[str, List]:
        """Check syntax, imports, and basic logical errors for a single file"""
        issues = {
            'syntax_errors': [],
            'import_errors': [],
            'logical_errors': []
        }
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues['syntax_errors'].append({
                    'file': str(file_path),
                    'line': e.lineno,
                    'error': str(e)
                })
                return issues  # Can't proceed if syntax is invalid
            
            # Check imports by attempting compilation
            try:
                compile(content, str(file_path), 'exec')
            except Exception as e:
                if "import" in str(e).lower():
                    issues['import_errors'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })
            
            # Check for common logical errors
            logical_issues = self._check_logical_errors(content, file_path)
            issues['logical_errors'].extend(logical_issues)
            
        except Exception as e:
            issues['syntax_errors'].append({
                'file': str(file_path),
                'error': f"Failed to read file: {str(e)}"
            })
        
        return issues
    
    def _check_logical_errors(self, content: str, file_path: Path) -> List[Dict]:
        """Check for common logical errors in Python code"""
        logical_issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for common logical errors
            if 'if True:' in line and not line_stripped.startswith('#'):
                logical_issues.append({
                    'file': str(file_path),
                    'line': i,
                    'issue': 'Potential debug code: if True:',
                    'severity': 'warning'
                })
            
            if 'print(' in line and 'main.py' not in str(file_path) and 'debug' not in str(file_path):
                logical_issues.append({
                    'file': str(file_path),
                    'line': i,
                    'issue': 'Print statement in non-main file (potential debug code)',
                    'severity': 'info'
                })
            
            # Check for division by zero risks
            if '/' in line and 'import' not in line and '//' not in line and '/*' not in line:
                if not any(safe_pattern in line for safe_pattern in ['if', 'try', 'except', '!= 0', '> 0']):
                    logical_issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue': 'Potential division by zero risk (no safety check visible)',
                        'severity': 'warning'
                    })
        
        return logical_issues
    
    def subphase_3_2_ml_libraries_apis(self) -> Dict[str, Any]:
        """
        Subphase 3.2: Confirm correct use of ML libraries and APIs
        """
        logger.info("=== SUBPHASE 3.2: ML Libraries and APIs Usage Validation ===")
        
        results = {
            'ml_files_found': [],
            'api_usage_issues': [],
            'library_version_issues': [],
            'deprecated_usage': []
        }
        
        ml_libraries = [
            'sklearn', 'scikit-learn', 'tensorflow', 'torch', 'pytorch',
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'xgboost',
            'lightgbm', 'catboost'
        ]
        
        for py_file in self.python_files:
            if self._file_uses_ml_libraries(py_file, ml_libraries):
                results['ml_files_found'].append(str(py_file))
                
                # Check for common ML API misuse
                api_issues = self._check_ml_api_usage(py_file)
                results['api_usage_issues'].extend(api_issues)
        
        logger.info(f"ML files found: {len(results['ml_files_found'])}")
        logger.info(f"API usage issues: {len(results['api_usage_issues'])}")
        
        return results
    
    def _file_uses_ml_libraries(self, file_path: Path, ml_libraries: List[str]) -> bool:
        """Check if file uses ML libraries"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for lib in ml_libraries:
                if f'import {lib}' in content or f'from {lib}' in content:
                    return True
            return False
        except:
            return False
    
    def _check_ml_api_usage(self, file_path: Path) -> List[Dict]:
        """Check for common ML API misuse patterns"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                # Check for data leakage in preprocessing
                if 'fit_transform' in line and 'test' in line.lower():
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue': 'Potential data leakage: fit_transform on test data',
                        'severity': 'high'
                    })
                
                # Check for missing random state
                if any(pattern in line for pattern in ['train_test_split', 'RandomForest', 'KMeans']) and 'random_state' not in line:
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue': 'Missing random_state for reproducibility',
                        'severity': 'medium'
                    })
                
                # Check for sklearn API misuse
                if '.fit(' in line and '.predict(' in line:
                    issues.append({
                        'file': str(file_path),
                        'line': i,
                        'issue': 'fit() and predict() on same line - check if correct',
                        'severity': 'medium'
                    })
        
        except Exception as e:
            issues.append({
                'file': str(file_path),
                'error': f"Failed to analyze ML API usage: {str(e)}"
            })
        
        return issues
    
    def subphase_3_3_utility_functions(self) -> Dict[str, Any]:
        """
        Subphase 3.3: Validate utility functions (feature engineering, data splitting)
        """
        logger.info("=== SUBPHASE 3.3: Utility Functions Validation ===")
        
        results = {
            'utility_files': [],
            'function_issues': [],
            'feature_engineering_issues': [],
            'data_splitting_issues': []
        }
        
        # Look for utility files and specific utility functions
        utility_patterns = ['util', 'helper', 'preprocessing', 'feature', 'split']
        
        for py_file in self.python_files:
            if any(pattern in str(py_file).lower() for pattern in utility_patterns):
                results['utility_files'].append(str(py_file))
                
                # Analyze utility functions in this file
                function_issues = self._validate_utility_functions(py_file)
                results['function_issues'].extend(function_issues)
        
        # Also check main files for utility function usage
        for py_file in self.python_files:
            if 'train' in str(py_file).lower() or 'main' in str(py_file).lower():
                fe_issues = self._check_feature_engineering_usage(py_file)
                ds_issues = self._check_data_splitting_usage(py_file)
                
                results['feature_engineering_issues'].extend(fe_issues)
                results['data_splitting_issues'].extend(ds_issues)
        
        logger.info(f"Utility files found: {len(results['utility_files'])}")
        logger.info(f"Function issues: {len(results['function_issues'])}")
        logger.info(f"Feature engineering issues: {len(results['feature_engineering_issues'])}")
        logger.info(f"Data splitting issues: {len(results['data_splitting_issues'])}")
        
        return results
    
    def _validate_utility_functions(self, file_path: Path) -> List[Dict]:
        """Validate utility functions for common issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find function definitions
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check function structure
                        func_issues = self._check_function_quality(node, file_path)
                        issues.extend(func_issues)
                        
            except Exception as e:
                issues.append({
                    'file': str(file_path),
                    'error': f"Failed to parse AST: {str(e)}"
                })
        
        except Exception as e:
            issues.append({
                'file': str(file_path),
                'error': f"Failed to read utility file: {str(e)}"
            })
        
        return issues
    
    def _check_function_quality(self, func_node: ast.FunctionDef, file_path: Path) -> List[Dict]:
        """Check individual function for quality issues"""
        issues = []
        
        # Check if function has docstring
        if not ast.get_docstring(func_node):
            issues.append({
                'file': str(file_path),
                'function': func_node.name,
                'issue': 'Function missing docstring',
                'severity': 'low'
            })
        
        # Check if function has type hints
        if not func_node.returns and func_node.name not in ['__init__', '__str__', '__repr__']:
            issues.append({
                'file': str(file_path),
                'function': func_node.name,
                'issue': 'Function missing return type hint',
                'severity': 'low'
            })
        
        # Check for overly complex functions (simple heuristic)
        if len(func_node.body) > 50:
            issues.append({
                'file': str(file_path),
                'function': func_node.name,
                'issue': 'Function potentially too complex (>50 statements)',
                'severity': 'medium'
            })
        
        return issues
    
    def _check_feature_engineering_usage(self, file_path: Path) -> List[Dict]:
        """Check for proper feature engineering practices"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Check for proper scaling
                if 'StandardScaler' in line or 'MinMaxScaler' in line:
                    # Check if scaler is fit only on training data
                    if 'test' in line.lower() and 'fit' in line:
                        issues.append({
                            'file': str(file_path),
                            'line': i,
                            'issue': 'Scaler should not be fit on test data',
                            'severity': 'high'
                        })
                
                # Check for feature selection
                if 'drop(' in line and 'axis=1' in line:
                    # This is generally good practice, just note it
                    pass
        
        except Exception as e:
            issues.append({
                'file': str(file_path),
                'error': f"Failed to check feature engineering: {str(e)}"
            })
        
        return issues
    
    def _check_data_splitting_usage(self, file_path: Path) -> List[Dict]:
        """Check for proper data splitting practices"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Check train_test_split usage
                if 'train_test_split' in line:
                    if 'random_state' not in line:
                        issues.append({
                            'file': str(file_path),
                            'line': i,
                            'issue': 'train_test_split missing random_state for reproducibility',
                            'severity': 'medium'
                        })
                    
                    if 'stratify' not in line and 'classification' in content.lower():
                        issues.append({
                            'file': str(file_path),
                            'line': i,
                            'issue': 'Consider stratify parameter for classification tasks',
                            'severity': 'low'
                        })
        
        except Exception as e:
            issues.append({
                'file': str(file_path),
                'error': f"Failed to check data splitting: {str(e)}"
            })
        
        return issues
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all subphases of Phase 3 code sanity checks"""
        logger.info("=== STARTING PHASE 3: CODE SANITY & LOGICAL ERROR CHECKS ===")
        
        # Find all Python files
        self.find_python_files()
        logger.info(f"Found {len(self.python_files)} Python files to analyze")
        
        results = {
            'phase': 'Phase 3: Code Sanity & Logical Error Checks',
            'timestamp': str(Path(__file__).stat().st_mtime),
            'files_analyzed': len(self.python_files)
        }
        
        # Run all subphases
        results['subphase_3_1'] = self.subphase_3_1_syntax_import_logical_errors()
        results['subphase_3_2'] = self.subphase_3_2_ml_libraries_apis()
        results['subphase_3_3'] = self.subphase_3_3_utility_functions()
        
        # Summary
        total_issues = (
            len(results['subphase_3_1']['syntax_errors']) +
            len(results['subphase_3_1']['import_errors']) +
            len(results['subphase_3_1']['logical_errors']) +
            len(results['subphase_3_2']['api_usage_issues']) +
            len(results['subphase_3_3']['function_issues']) +
            len(results['subphase_3_3']['feature_engineering_issues']) +
            len(results['subphase_3_3']['data_splitting_issues'])
        )
        
        results['summary'] = {
            'total_issues_found': total_issues,
            'phase_status': 'COMPLETE' if total_issues == 0 else f'{total_issues} issues found',
            'recommendation': 'All subphases completed' if total_issues == 0 else 'Review and address identified issues'
        }
        
        logger.info(f"=== PHASE 3 COMPLETE: {total_issues} total issues found ===")
        
        return results


def main():
    """Main execution function"""
    print("PHASE 3: CODE SANITY & LOGICAL ERROR CHECKS")
    print("=" * 50)
    
    checker = CodeSanityChecker()
    results = checker.run_comprehensive_check()
    
    # Print detailed results
    print(f"\nSUMMARY:")
    print(f"Files analyzed: {results['files_analyzed']}")
    print(f"Total issues found: {results['summary']['total_issues_found']}")
    print(f"Phase status: {results['summary']['phase_status']}")
    
    # Print subphase details
    print(f"\nSUBPHASE 3.1 - Syntax, Import, Logical Errors:")
    subphase_1 = results['subphase_3_1']
    print(f"  Syntax errors: {len(subphase_1['syntax_errors'])}")
    print(f"  Import errors: {len(subphase_1['import_errors'])}")
    print(f"  Logical errors: {len(subphase_1['logical_errors'])}")
    
    print(f"\nSUBPHASE 3.2 - ML Libraries and APIs:")
    subphase_2 = results['subphase_3_2']
    print(f"  ML files found: {len(subphase_2['ml_files_found'])}")
    print(f"  API usage issues: {len(subphase_2['api_usage_issues'])}")
    
    print(f"\nSUBPHASE 3.3 - Utility Functions:")
    subphase_3 = results['subphase_3_3']
    print(f"  Utility files: {len(subphase_3['utility_files'])}")
    print(f"  Function issues: {len(subphase_3['function_issues'])}")
    print(f"  Feature engineering issues: {len(subphase_3['feature_engineering_issues'])}")
    print(f"  Data splitting issues: {len(subphase_3['data_splitting_issues'])}")
    
    # Show sample issues if any
    if results['summary']['total_issues_found'] > 0:
        print(f"\nSAMPLE ISSUES (showing first few):")
        
        if subphase_1['syntax_errors']:
            print("  Syntax Errors:")
            for issue in subphase_1['syntax_errors'][:3]:
                print(f"    - {issue['file']}:{issue.get('line', '?')} - {issue['error']}")
        
        if subphase_2['api_usage_issues']:
            print("  ML API Issues:")
            for issue in subphase_2['api_usage_issues'][:3]:
                print(f"    - {issue['file']}:{issue.get('line', '?')} - {issue['issue']}")
        
        if subphase_3['function_issues']:
            print("  Function Issues:")
            for issue in subphase_3['function_issues'][:3]:
                print(f"    - {issue['file']} - {issue['issue']}")
    
    return results


if __name__ == "__main__":
    main()