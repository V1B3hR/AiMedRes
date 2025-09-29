#!/usr/bin/env python3
"""
Unified Medical AI Training Runner
Executes training for ALS, Alzheimer's, and Parkinson's disease prediction models
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_training(script_path, output_dir, additional_args=None):
    """Run a training script with specified parameters"""
    cmd = [
        sys.executable, script_path,
        "--output-dir", output_dir,
        "--epochs", "50",
        "--folds", "5"
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🚀 Starting training: {script_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Training completed successfully: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {script_path}")
        print(f"Error: {e}")
        return False

def main():
    """Main training orchestrator"""
    print("=" * 80)
    print("AiMedRes Comprehensive Medical AI Training Pipeline")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get the repository root
    repo_root = Path(__file__).parent
    
    # Define training configurations
    trainings = [
        {
            "name": "ALS (Amyotrophic Lateral Sclerosis)",
            "script": "training/train_als.py",
            "output": "als_comprehensive_results",
            "args": ["--dataset-choice", "als-progression"]
        },
        {
            "name": "Alzheimer's Disease", 
            "script": "files/training/train_alzheimers.py",
            "output": "alzheimer_comprehensive_results",
            "args": []
        },
        {
            "name": "Parkinson's Disease",
            "script": "training/train_parkinsons.py", 
            "output": "parkinsons_comprehensive_results",
            "args": ["--dataset-choice", "vikasukani"]
        }
    ]
    
    success_count = 0
    total_count = len(trainings)
    
    for training in trainings:
        print(f"🧠 {training['name']} Training")
        print("=" * 40)
        
        script_path = repo_root / training["script"]
        if run_training(str(script_path), training["output"], training["args"]):
            success_count += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 Training Pipeline Summary")
    print("=" * 80)
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print()
        print("🎉 All medical AI training pipelines completed successfully!")
        print("🔬 Models are ready for deployment and medical applications!")
        return 0
    else:
        print()
        print("⚠️  Some training pipelines failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())