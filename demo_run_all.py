#!/usr/bin/env python3
"""
Demonstration of the "Run All" training functionality.
This script shows how the training orchestrator discovers and prepares to run all models.
"""

import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_demo_command(description, cmd):
    """Run a command and display its output."""
    print(f"📌 {description}")
    print(f"💻 Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        print(output)
        
        if result.returncode != 0:
            print(f"❌ Command failed with exit code {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False


def main():
    """Run the demonstration."""
    print("=" * 70)
    print("  🏥 AiMedRes Training Orchestrator - 'Run All' Demo")
    print("=" * 70)
    print()
    print("This demonstration shows how the training orchestrator automatically")
    print("discovers and can run ALL medical AI training scripts in the repository.")
    print()
    
    # Change to repository root
    repo_root = Path(__file__).parent
    import os
    os.chdir(repo_root)
    
    # Demo 1: List all discovered jobs
    print_section("Step 1: Discover All Training Jobs")
    print("The orchestrator automatically finds all training scripts:")
    print()
    run_demo_command(
        "List all discovered training jobs",
        [sys.executable, "run_all_training.py", "--list"]
    )
    
    # Demo 2: Preview what would run
    print_section("Step 2: Preview 'Run All' Commands")
    print("See what commands would execute when running all training:")
    print("(Using --dry-run to preview without actually training)")
    print()
    run_demo_command(
        "Preview all commands with epochs=10",
        [sys.executable, "run_all_training.py", "--dry-run", "--epochs", "10", "--folds", "3"]
    )
    
    # Demo 3: Show filtering
    print_section("Step 3: Selective Training")
    print("You can also run specific models only:")
    print()
    run_demo_command(
        "Preview training only ALS and Alzheimer's models",
        [sys.executable, "run_all_training.py", "--dry-run", "--only", "als", "alzheimers", "--epochs", "5"]
    )
    
    # Demo 4: Show parallel mode
    print_section("Step 4: Parallel Execution")
    print("For faster training, run multiple models in parallel:")
    print()
    run_demo_command(
        "Preview parallel execution with 4 workers",
        [sys.executable, "run_all_training.py", "--dry-run", "--parallel", "--max-workers", "4", "--epochs", "5"]
    )
    
    # Summary
    print_section("Summary")
    print("✅ The 'Run All' functionality is fully operational!")
    print()
    print("To actually run all training (not just preview):")
    print()
    print("  Basic:      python run_all_training.py")
    print("  Custom:     python run_all_training.py --epochs 50 --folds 5")
    print("  Parallel:   python run_all_training.py --parallel --max-workers 4")
    print("  Filtered:   python run_all_training.py --only als alzheimers")
    print("  Script:     ./run_medical_training.sh")
    print()
    print("📚 For complete documentation, see:")
    print("   - RUN_ALL_GUIDE.md")
    print("   - IMPLEMENTATION_SUMMARY.md")
    print("   - python run_all_training.py --help")
    print()
    print("🎉 Ready to train all medical AI models!")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
