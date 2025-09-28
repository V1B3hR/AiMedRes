#!/usr/bin/env python3
"""
Brain MRI Training Quick Test

This script runs a quick test of the brain MRI training pipeline with a subset
of the data to demonstrate functionality without waiting for the full training.
"""

import os
import sys
import logging
from pathlib import Path

# Add the files/training directory to Python path
sys.path.append(str(Path(__file__).parent / "files" / "training"))

def quick_test():
    """Run a quick test of the brain MRI training setup"""
    print("=" * 60)
    print("BRAIN MRI TRAINING QUICK TEST")
    print("=" * 60)
    
    try:
        # Import the training pipeline
        from train_brain_mri import BrainMRITrainingPipeline
        print("✓ Successfully imported BrainMRITrainingPipeline")
        
        # Create a test pipeline
        pipeline = BrainMRITrainingPipeline(output_dir='quick_test_output')
        print("✓ Successfully created training pipeline")
        
        # Test dataset loading
        print("\nTesting dataset loading...")
        image_paths, labels = pipeline.load_dataset()
        
        print(f"✓ Successfully loaded dataset:")
        print(f"  - Total images: {len(image_paths)}")
        print(f"  - Total labels: {len(labels)}")
        print(f"  - Unique labels: {len(set(labels))}")
        print(f"  - Label distribution: {dict(zip(*zip(*[(label, labels.count(label)) for label in set(labels)])))}")
        
        # Show some sample paths
        print(f"\nSample image paths:")
        for i, path in enumerate(image_paths[:3]):
            print(f"  {i+1}. {Path(path).name}")
        
        print(f"\nSample labels: {labels[:10]}")
        
        # Test model creation
        print("\nTesting model creation...")
        import torch
        from train_brain_mri import BrainMRICNN
        
        model = BrainMRICNN(num_classes=len(set(labels)))
        print(f"✓ Successfully created 2D CNN model")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # Test model forward pass with dummy data
        dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB, 224x224
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Model forward pass successful: input {list(dummy_input.shape)} -> output {list(output.shape)}")
        
        # Test data transforms
        print("\nTesting data transforms...")
        from PIL import Image
        import torch
        
        # Load a sample image
        sample_img_path = image_paths[0]
        image = Image.open(sample_img_path).convert('RGB')
        original_size = image.size
        
        # Apply transforms
        transformed_image = pipeline.train_transform(image)
        print(f"✓ Data transform successful: {original_size} -> {list(transformed_image.shape)}")
        
        print("\n" + "=" * 60)
        print("✅ QUICK TEST PASSED!")
        print("=" * 60)
        print("The brain MRI training pipeline is working correctly.")
        print("All components (dataset loading, model creation, transforms) are functional.")
        print(f"Ready to train on {len(image_paths)} brain MRI images with {len(set(labels))} classes.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)