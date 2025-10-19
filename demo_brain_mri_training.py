#!/usr/bin/env python3
"""
Brain MRI Training Demo Script

This script demonstrates the brain MRI training pipeline by running
a minimal training session with a small subset of the data.
This is useful for testing and demonstration purposes.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_demo_training(max_samples=100, epochs=2, batch_size=8):
    """
    Run a quick demo of the training pipeline with limited data.
    
    Args:
        max_samples: Maximum number of samples to use (default: 100)
        epochs: Number of training epochs (default: 2)
        batch_size: Batch size for training (default: 8)
    """
    logger.info("=" * 70)
    logger.info("BRAIN MRI TRAINING DEMO")
    logger.info("=" * 70)
    logger.info(f"Demo Configuration:")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("=" * 70)
    
    try:
        # Import required modules
        sys.path.insert(0, str(Path(__file__).parent / "training"))
        from train_brain_mri import BrainMRITrainingPipeline
        import torch
        from torch.utils.data import DataLoader, Subset
        from sklearn.model_selection import train_test_split
        
        # Create output directory
        output_dir = Path('/tmp/brain_mri_demo')
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        logger.info("\n1. Initializing training pipeline...")
        pipeline = BrainMRITrainingPipeline(output_dir=str(output_dir))
        
        logger.info("\n2. Loading dataset...")
        image_paths, labels = pipeline.load_dataset()
        logger.info(f"   Full dataset: {len(image_paths)} images")
        
        # Limit to max_samples for demo
        if len(image_paths) > max_samples:
            import random
            random.seed(42)
            indices = random.sample(range(len(image_paths)), max_samples)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]
            logger.info(f"   Demo subset: {max_samples} images")
        
        logger.info(f"   Classes: {len(set(labels))}")
        logger.info(f"   Label distribution: {dict([(label, labels.count(label)) for label in set(labels)])}")
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"   Training samples: {len(train_paths)}")
        logger.info(f"   Validation samples: {len(val_paths)}")
        
        logger.info("\n3. Creating datasets and data loaders...")
        from train_brain_mri import BrainMRIDataset
        train_dataset = BrainMRIDataset(train_paths, train_labels, pipeline.train_transform)
        val_dataset = BrainMRIDataset(val_paths, val_labels, pipeline.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        logger.info(f"   Training batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        
        logger.info("\n4. Creating model...")
        from train_brain_mri import BrainMRICNN
        import torch.nn as nn
        import torch.optim as optim
        
        model = BrainMRICNN(num_classes=len(set(labels))).to(pipeline.device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Model parameters: {total_params:,}")
        logger.info(f"   Device: {pipeline.device}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        logger.info(f"\n5. Training for {epochs} epochs...")
        logger.info("   (This is a demo - full training would use more epochs)")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels_batch) in enumerate(train_loader):
                images, labels_batch = images.to(pipeline.device), labels_batch.to(pipeline.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(pipeline.device), labels_batch.to(pipeline.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"   Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val Acc: {val_acc:.2f}%")
        
        logger.info("\n6. Saving demo model...")
        model_path = output_dir / "demo_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"   Model saved to: {model_path}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("The brain MRI training pipeline is fully functional.")
        logger.info(f"Demo model saved to: {output_dir}")
        logger.info("\nTo run full training:")
        logger.info("  python training/train_brain_mri.py --epochs 50")
        logger.info("  python training/train_brain_mri.py --epochs 20 --batch-size 32")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain MRI Training Demo - Quick validation with subset of data"
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum number of samples to use (default: 100)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs (default: 2)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    
    args = parser.parse_args()
    
    success = run_demo_training(
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
