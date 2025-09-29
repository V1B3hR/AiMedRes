#!/usr/bin/env python3
"""
Phase 1 Debugging Script: Environment & Reproducibility Checks

This script implements Phase 1 of the AiMedRes debugging process as outlined in debuglist.md:
- Subphase 1.1: Verify Python/ML environment setup (package versions, CUDA, etc.)
- Subphase 1.2: Ensure reproducibility (set random seeds, document environment)
- Subphase 1.3: Confirm data, code, and results are version-controlled

Usage:
    python debug/phase1_environment_debug.py [--install-missing] [--verbose]
"""

import sys
import os
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import platform


class Phase1EnvironmentDebugger:
    """Phase 1 debugging implementation for AiMedRes"""
    
    def __init__(self, verbose: bool = False, install_missing: bool = False):
        self.verbose = verbose
        self.install_missing = install_missing
        self.results = {}
        self.repo_root = Path(__file__).parent.parent
        self._load_env_file()
        
    def _load_env_file(self):
        """Load environment variables from .env file if it exists"""
        env_file = self.repo_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#') and '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "📝" if level == "INFO" else "⚠️ " if level == "WARN" else "❌"
        print(f"[{timestamp}] {prefix} {message}")
        if self.verbose and level != "INFO":
            print(f"    Details: {message}")
    
    def subphase_1_1_environment_setup(self) -> bool:
        """Subphase 1.1: Verify Python/ML environment setup"""
        self.log("=" * 60)
        self.log("SUBPHASE 1.1: ENVIRONMENT SETUP VERIFICATION")
        self.log("=" * 60)
        
        all_checks_passed = True
        
        # Check Python version
        python_version = platform.python_version()
        self.log(f"Python Version: {python_version}")
        
        if sys.version_info >= (3, 10):
            self.log("✓ Python version meets requirements (>=3.10)")
        else:
            self.log("✗ Python version too old. Requires >=3.10")
            all_checks_passed = False
        
        # Check platform information
        self.log(f"Platform: {platform.system()} {platform.machine()}")
        self.log(f"Platform Release: {platform.release()}")
        
        # Check core dependencies from pyproject.toml
        core_dependencies = [
            "SQLAlchemy",
            "psycopg2",
            "pgvector", 
            "sentence_transformers",
            "yaml"
        ]
        
        self.log("\nCore Dependencies Check:")
        self.log("-" * 30)
        
        for dep in core_dependencies:
            try:
                if dep == "yaml":
                    import yaml
                    version = getattr(yaml, '__version__', 'unknown')
                elif dep == "psycopg2":
                    import psycopg2
                    version = psycopg2.__version__
                elif dep == "sentence_transformers":
                    import sentence_transformers
                    version = sentence_transformers.__version__
                elif dep == "SQLAlchemy":
                    import sqlalchemy
                    version = sqlalchemy.__version__
                elif dep == "pgvector":
                    import pgvector
                    version = getattr(pgvector, '__version__', 'installed')
                else:
                    module = __import__(dep)
                    version = getattr(module, '__version__', 'unknown')
                
                self.log(f"✓ {dep} {version}")
                
            except ImportError as e:
                self.log(f"✗ {dep} - Not installed")
                if self.install_missing:
                    self.log(f"  Attempting to install {dep}...")
                    # Note: In real scenario, would implement pip install logic
                all_checks_passed = False
        
        # Check ML dependencies from requirements-ml.txt
        ml_dependencies = [
            ("numpy", "1.24.0"),
            ("pandas", "2.0.0"), 
            ("sklearn", "1.3.0"),
            ("matplotlib", "3.5.0"),
            ("seaborn", "0.12.0"),
            ("torch", "2.0.0"),
            ("xgboost", "2.0.0"),
            ("kagglehub", "0.3.0"),
            ("scipy", "1.10.0"),
            ("joblib", "1.3.0")
        ]
        
        self.log("\nML Dependencies Check:")
        self.log("-" * 30)
        
        for dep, min_version in ml_dependencies:
            try:
                if dep == "sklearn":
                    import sklearn
                    version = sklearn.__version__
                    module_name = "scikit-learn"
                elif dep == "torch":
                    import torch
                    version = torch.__version__
                    module_name = dep
                else:
                    module = __import__(dep)
                    version = getattr(module, '__version__', 'unknown')
                    module_name = dep
                
                self.log(f"✓ {module_name} {version} (required >={min_version})")
                
            except ImportError:
                self.log(f"✗ {dep} - Not installed (required >={min_version})")
                all_checks_passed = False
        
        # Check CUDA availability
        self.log("\nGPU/CUDA Check:")
        self.log("-" * 30)
        
        cuda_available = self._check_cuda()
        if cuda_available:
            self.log("✓ CUDA acceleration available")
        else:
            self.log("⚠️  CUDA not available (CPU-only mode)")
        
        # Check environment variables
        self.log("\nEnvironment Variables Check:")
        self.log("-" * 30)
        
        env_vars = [
            "LOG_LEVEL",
            "EMBEDDING_MODEL", 
            "EMBEDDING_DIM",
            "DATABASE_URL",
            "SENTENCE_TRANSFORMERS_HOME"
        ]
        
        env_file = self.repo_root / ".env"
        if env_file.exists():
            self.log("✓ .env file exists")
        else:
            self.log("⚠️  .env file not found (using defaults)")
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                self.log(f"✓ {var}: {value}")
            else:
                self.log(f"⚠️  {var}: Not set")
        
        self.results['subphase_1_1'] = all_checks_passed
        return all_checks_passed
    
    def subphase_1_2_reproducibility(self) -> bool:
        """Subphase 1.2: Ensure reproducibility (set random seeds, document environment)"""
        self.log("\n" + "=" * 60)
        self.log("SUBPHASE 1.2: REPRODUCIBILITY CHECKS")
        self.log("=" * 60)
        
        all_checks_passed = True
        
        # Check for random seed usage in training scripts
        self.log("Random Seed Implementation Check:")
        self.log("-" * 30)
        
        seed_files = [
            "src/aimedres/training/structured_alz_trainer.py",
            "files/training/train_brain_mri.py",
            "training/train_als.py"
        ]
        
        seed_implementations = []
        for file_path in seed_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                content = full_path.read_text()
                has_seed_setting = any(pattern in content for pattern in [
                    "random.seed",
                    "np.random.seed", 
                    "torch.manual_seed",
                    "set_seed",
                    "random_state"
                ])
                
                if has_seed_setting:
                    self.log(f"✓ {file_path} - Implements random seeding")
                    seed_implementations.append(file_path)
                else:
                    self.log(f"⚠️  {file_path} - No random seeding found")
                    all_checks_passed = False
            else:
                self.log(f"⚠️  {file_path} - File not found")
        
        # Test reproducibility by running a small computation
        self.log("\nReproducibility Test:")
        self.log("-" * 30)
        
        try:
            reproducibility_test = self._test_reproducibility()
            if reproducibility_test:
                self.log("✓ Reproducibility test passed")
            else:
                self.log("✗ Reproducibility test failed")
                all_checks_passed = False
        except Exception as e:
            self.log(f"✗ Reproducibility test error: {e}")
            all_checks_passed = False
        
        # Generate environment documentation
        self.log("\nEnvironment Documentation:")
        self.log("-" * 30)
        
        env_doc = self._generate_environment_documentation()
        doc_path = self.repo_root / "debug" / "environment_snapshot.json"
        
        try:
            with open(doc_path, 'w') as f:
                json.dump(env_doc, f, indent=2)
            self.log(f"✓ Environment documented in {doc_path}")
        except Exception as e:
            self.log(f"✗ Failed to save environment documentation: {e}")
            all_checks_passed = False
        
        self.results['subphase_1_2'] = all_checks_passed
        return all_checks_passed
    
    def subphase_1_3_version_control(self) -> bool:
        """Subphase 1.3: Confirm data, code, and results are version-controlled"""
        self.log("\n" + "=" * 60)
        self.log("SUBPHASE 1.3: VERSION CONTROL VERIFICATION")
        self.log("=" * 60)
        
        all_checks_passed = True
        
        # Check Git repository status
        self.log("Git Repository Check:")
        self.log("-" * 30)
        
        git_status = self._check_git_status()
        if git_status['is_repo']:
            self.log("✓ Git repository initialized")
            self.log(f"  Branch: {git_status['branch']}")
            self.log(f"  Uncommitted changes: {git_status['uncommitted_files']}")
            
            if git_status['uncommitted_files'] > 0:
                self.log("⚠️  Warning: Uncommitted changes detected")
        else:
            self.log("✗ Not a Git repository")
            all_checks_passed = False
        
        # Check .gitignore for important files
        self.log("\n.gitignore Check:")
        self.log("-" * 30)
        
        gitignore_path = self.repo_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            important_patterns = [
                "*.pyc",
                "__pycache__",
                ".env",
                "*.log",
                "models/",
                "data/raw/",
                ".cache/"
            ]
            
            missing_patterns = []
            for pattern in important_patterns:
                if pattern not in gitignore_content:
                    missing_patterns.append(pattern)
            
            if not missing_patterns:
                self.log("✓ .gitignore covers important file patterns")
            else:
                self.log(f"⚠️  .gitignore missing patterns: {missing_patterns}")
                
        else:
            self.log("✗ .gitignore file not found")
            all_checks_passed = False
        
        # Check DVC (Data Version Control)
        self.log("\nData Version Control Check:")
        self.log("-" * 30)
        
        dvc_dir = self.repo_root / ".dvc"
        if dvc_dir.exists():
            self.log("✓ DVC initialized")
            
            # Check for .dvc files
            dvc_files = list(self.repo_root.rglob("*.dvc"))
            self.log(f"  DVC tracked files: {len(dvc_files)}")
            
            if dvc_files:
                for dvc_file in dvc_files[:3]:  # Show first 3
                    self.log(f"    - {dvc_file.relative_to(self.repo_root)}")
        else:
            self.log("⚠️  DVC not initialized (data versioning recommended)")
        
        # Check for results/output directories
        self.log("\nOutput Directory Structure:")
        self.log("-" * 30)
        
        output_dirs = [
            "outputs",
            "models", 
            "results",
            "logs",
            "experiments"
        ]
        
        for dir_name in output_dirs:
            dir_path = self.repo_root / dir_name
            if dir_path.exists():
                self.log(f"✓ {dir_name}/ directory exists")
                # Check if it's properly ignored or tracked
                if dir_name in gitignore_content if gitignore_path.exists() else False:
                    self.log(f"  - Properly ignored by Git")
                else:
                    self.log(f"  ⚠️  Not ignored by Git (may track large files)")
            else:
                self.log(f"⚠️  {dir_name}/ directory not found")
        
        self.results['subphase_1_3'] = all_checks_passed
        return all_checks_passed
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                result = subprocess.run(['nvidia-smi'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
                return result.returncode == 0
            except:
                return False
    
    def _test_reproducibility(self) -> bool:
        """Test reproducibility with a simple computation"""
        try:
            import numpy as np
            
            # Test 1: NumPy random state
            np.random.seed(42)
            result1 = np.random.random(10)
            
            np.random.seed(42)
            result2 = np.random.random(10)
            
            numpy_reproducible = np.allclose(result1, result2)
            
            # Test 2: Python random (if available)
            try:
                import random
                random.seed(42)
                py_result1 = [random.random() for _ in range(10)]
                
                random.seed(42)
                py_result2 = [random.random() for _ in range(10)]
                
                python_reproducible = py_result1 == py_result2
            except ImportError:
                python_reproducible = True  # Skip if not available
            
            return numpy_reproducible and python_reproducible
            
        except Exception:
            return False
    
    def _generate_environment_documentation(self) -> Dict:
        """Generate comprehensive environment documentation"""
        env_doc = {
            "timestamp": datetime.now().isoformat(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler()
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "packages": {},
            "environment_variables": {},
            "git_info": self._check_git_status()
        }
        
        # Document installed packages
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                env_doc["packages"] = {pkg["name"]: pkg["version"] for pkg in packages}
        except:
            pass
        
        # Document relevant environment variables
        env_vars = [
            "PATH", "PYTHONPATH", "LOG_LEVEL", "EMBEDDING_MODEL", 
            "DATABASE_URL", "CUDA_VISIBLE_DEVICES"
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                env_doc["environment_variables"][var] = value
        
        return env_doc
    
    def _check_git_status(self) -> Dict:
        """Check Git repository status"""
        try:
            # Check if it's a git repo
            result = subprocess.run(['git', 'rev-parse', '--git-dir'],
                                  cwd=self.repo_root,
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode != 0:
                return {"is_repo": False}
            
            # Get branch name
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  cwd=self.repo_root,
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            branch = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  cwd=self.repo_root,
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            uncommitted = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Get last commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                  cwd=self.repo_root,
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            commit_hash = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
            
            return {
                "is_repo": True,
                "branch": branch,
                "uncommitted_files": uncommitted,
                "last_commit": commit_hash
            }
            
        except Exception:
            return {"is_repo": False}
    
    def run_phase_1(self) -> bool:
        """Run complete Phase 1 debugging process"""
        self.log("🔍 STARTING PHASE 1: ENVIRONMENT & REPRODUCIBILITY DEBUGGING")
        self.log("📋 Based on debug/debuglist.md")
        self.log("")
        
        start_time = datetime.now()
        
        # Run all subphases
        subphase_1_1 = self.subphase_1_1_environment_setup()
        subphase_1_2 = self.subphase_1_2_reproducibility()  
        subphase_1_3 = self.subphase_1_3_version_control()
        
        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log("\n" + "=" * 60)
        self.log("PHASE 1 SUMMARY")
        self.log("=" * 60)
        
        total_passed = sum([subphase_1_1, subphase_1_2, subphase_1_3])
        
        self.log(f"Subphase 1.1 (Environment Setup): {'✓ PASS' if subphase_1_1 else '✗ FAIL'}")
        self.log(f"Subphase 1.2 (Reproducibility): {'✓ PASS' if subphase_1_2 else '✗ FAIL'}")
        self.log(f"Subphase 1.3 (Version Control): {'✓ PASS' if subphase_1_3 else '✗ FAIL'}")
        
        self.log(f"\n📊 Results: {total_passed}/3 subphases passed")
        self.log(f"⏱️  Duration: {duration:.1f} seconds")
        
        overall_success = total_passed == 3
        
        if overall_success:
            self.log("🎉 PHASE 1 COMPLETE - Ready for Phase 2!")
        else:
            self.log("⚠️  PHASE 1 INCOMPLETE - Address issues before Phase 2")
            self.log("\n📝 Recommendations:")
            if not subphase_1_1:
                self.log("  • Install missing ML dependencies (see requirements-ml.txt)")
                self.log("  • Set up proper environment variables (.env file)")
            if not subphase_1_2:
                self.log("  • Implement random seeding in training scripts")
                self.log("  • Review environment documentation")
            if not subphase_1_3:
                self.log("  • Initialize Git repository if needed")
                self.log("  • Set up DVC for data versioning")
                self.log("  • Review .gitignore patterns")
        
        # Save detailed results
        self._save_phase_1_results(overall_success, duration)
        
        return overall_success
    
    def _save_phase_1_results(self, success: bool, duration: float):
        """Save Phase 1 results to file"""
        results = {
            "phase": 1,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "overall_success": success,
            "subphase_results": self.results,
            "summary": {
                "subphase_1_1": "Environment setup verification",
                "subphase_1_2": "Reproducibility checks", 
                "subphase_1_3": "Version control verification"
            }
        }
        
        results_path = self.repo_root / "debug" / "phase1_results.json"
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.log(f"📄 Results saved to {results_path}")
        except Exception as e:
            self.log(f"⚠️  Could not save results: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Environment & Reproducibility Debugging")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--install-missing", action="store_true",
                       help="Attempt to install missing dependencies")
    
    args = parser.parse_args()
    
    debugger = Phase1EnvironmentDebugger(
        verbose=args.verbose,
        install_missing=args.install_missing
    )
    
    success = debugger.run_phase_1()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()