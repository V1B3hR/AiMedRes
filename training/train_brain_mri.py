#!/usr/bin/env python3
"""
Brain MRI Images Classification Training Pipeline

Implementation for the brain MRI images dataset with 20 epochs as specified 
in the problem statement: https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images

Features:
- Handles brain MRI image classification 
- Uses 20 epochs for training
- CNN architecture suitable for medical images
- Comprehensive logging and metrics
"""

import os
import sys
import logging
import warnings
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ML and Image Processing Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Kaggle dataset loading
import kagglehub

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BrainMRIDataset(Dataset):
    """Dataset class for brain MRI images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            logger.warning(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a zero tensor if image fails to load
            return torch.zeros(3, 224, 224), self.labels[idx]


class BrainMRICNN(nn.Module):
    """CNN model for brain MRI classification"""
    
    def __init__(self, num_classes=2):
        super(BrainMRICNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrainMRITrainingPipeline:
    """Training pipeline for brain MRI classification"""
    
    def __init__(self, output_dir: str = 'brain_mri_outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """Load and prepare the brain MRI dataset"""
        logger.info("Downloading brain MRI dataset from Kaggle...")
        
        try:
            dataset_path = kagglehub.dataset_download("ashfakyeafi/brain-mri-images")
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # Find all image files
            gan_images_path = Path(dataset_path) / "GAN-Traning Images"
            image_files = list(gan_images_path.glob("*.jpg"))
            
            if not image_files:
                raise ValueError("No image files found in the dataset")
                
            logger.info(f"Found {len(image_files)} brain MRI images")
            
            # For demonstration, create simple labels based on filename patterns
            # In a real scenario, you'd have actual labels or metadata
            image_paths = [str(f) for f in image_files]
            
            # Create labels based on slice orientation or patient ID patterns
            # This is a simplified approach for demonstration
            labels = []
            for path in image_paths:
                filename = Path(path).name
                # Create binary classification based on some criteria
                # Here we use slice orientation as a proxy for classification
                if '_x_slice_' in filename:
                    labels.append(0)  # Class 0 for x-orientation slices
                else:
                    labels.append(1)  # Class 1 for y/z-orientation slices
            
            logger.info(f"Label distribution: Class 0: {labels.count(0)}, Class 1: {labels.count(1)}")
            
            return image_paths, labels
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def train_model(self, epochs: int = 20, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the brain MRI CNN model"""
        logger.info(f"Starting brain MRI classification training with {epochs} epochs...")
        
        # Load dataset
        image_paths, labels = self.load_dataset()
        
        # Split into train/validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=validation_split, 
            random_state=42, stratify=labels
        )
        
        logger.info(f"Training set: {len(train_paths)} images")
        logger.info(f"Validation set: {len(val_paths)} images")
        
        # Create datasets and data loaders
        train_dataset = BrainMRIDataset(train_paths, train_labels, self.train_transform)
        val_dataset = BrainMRIDataset(val_paths, val_labels, self.val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Initialize model
        num_classes = len(set(labels))
        model = BrainMRICNN(num_classes=num_classes).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels_batch) in enumerate(train_loader):
                images, labels_batch = images.to(self.device), labels_batch.to(self.device)
                
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
                    images, labels_batch = images.to(self.device), labels_batch.to(self.device)
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
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.output_dir / "models" / "best_brain_mri_model.pth")
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
        
        # Final evaluation
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        torch.save(model.state_dict(), self.output_dir / "models" / "final_brain_mri_model.pth")
        
        # Save training metrics
        metrics = {
            'epochs': epochs,
            'best_validation_accuracy': best_val_acc,
            'final_train_accuracy': train_acc,
            'final_validation_accuracy': val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'num_classes': num_classes,
            'training_samples': len(train_paths),
            'validation_samples': len(val_paths)
        }
        
        with open(self.output_dir / "metrics" / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to: {self.output_dir}")
        return metrics


def main():
    """Main entry point for brain MRI training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain MRI Images Classification Training Pipeline (20 epochs)"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='brain_mri_outputs',
        help='Output directory for results (default: brain_mri_outputs)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20,
        help='Number of epochs for training (default: 20)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run pipeline
        pipeline = BrainMRITrainingPipeline(output_dir=args.output_dir)
        
        logger.info("=" * 60)
        logger.info("BRAIN MRI CLASSIFICATION TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Train model
        metrics = pipeline.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print(f"Epochs trained: {metrics['epochs']}")
        print(f"Best validation accuracy: {metrics['best_validation_accuracy']:.2f}%")
        print(f"Final validation accuracy: {metrics['final_validation_accuracy']:.2f}%")
        print(f"Training samples: {metrics['training_samples']}")
        print(f"Validation samples: {metrics['validation_samples']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)