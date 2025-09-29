#!/usr/bin/env python3
"""
Brain MRI Images Classification Training Pipeline

Implementation for the brain MRI images dataset with 50 epochs as specified 
in the problem statement: https://www.kaggle.com/datasets/ashfakyeafi/brain-mri-images

Features:
- Handles brain MRI image classification 
- Uses 50 epochs for training
- CNN architecture suitable for medical images
- Comprehensive logging and metrics
- Improved: Model regularization, data augmentation, early stopping, mixed precision training
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

# MLflow integration
import mlflow
import mlflow.pytorch

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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrainMRI3DCNN(nn.Module):
    """3D CNN model for volumetric brain MRI classification"""
    
    def __init__(self, num_classes=2, input_channels=1):
        super(BrainMRI3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EarlyStopping:
    """Early stopping utility to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class BrainMRITrainingPipeline:
    """Training pipeline for brain MRI classification"""
    
    def __init__(self, output_dir: str = 'brain_mri_outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Data transforms (stronger augmentation for training)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
            
            gan_images_path = Path(dataset_path) / "GAN-Traning Images"
            image_files = list(gan_images_path.glob("*.jpg"))
            
            if not image_files:
                raise ValueError("No image files found in the dataset")
                
            logger.info(f"Found {len(image_files)} brain MRI images")
            
            image_paths = [str(f) for f in image_files]
            labels = []
            for path in image_paths:
                filename = Path(path).name
                if '_x_slice_' in filename:
                    labels.append(0)
                else:
                    labels.append(1)
            
            logger.info(f"Label distribution: Class 0: {labels.count(0)}, Class 1: {labels.count(1)}")
            return image_paths, labels
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def train_model(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2, 
                    use_3d: bool = False, mlflow_experiment: str = "brain_mri_classification") -> Dict[str, Any]:
        """Train the brain MRI CNN model with MLflow tracking and improvements"""
        logger.info(f"Starting brain MRI classification training with {epochs} epochs...")
        
        mlflow.set_experiment(mlflow_experiment)
        
        with mlflow.start_run():
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("validation_split", validation_split)
            mlflow.log_param("use_3d", use_3d)
            mlflow.log_param("model_type", "3D_CNN" if use_3d else "2D_CNN")
            
            image_paths, labels = self.load_dataset()
            num_classes = len(set(labels))
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("total_samples", len(image_paths))
            
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels, test_size=validation_split, 
                random_state=42, stratify=labels
            )
            
            mlflow.log_param("train_samples", len(train_paths))
            mlflow.log_param("val_samples", len(val_paths))
            
            logger.info(f"Training set: {len(train_paths)} images")
            logger.info(f"Validation set: {len(val_paths)} images")
            
            train_dataset = BrainMRIDataset(train_paths, train_labels, self.train_transform)
            val_dataset = BrainMRIDataset(val_paths, val_labels, self.val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            
            if use_3d:
                model = BrainMRI3DCNN(num_classes=num_classes).to(self.device)
            else:
                model = BrainMRICNN(num_classes=num_classes).to(self.device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_param("total_parameters", total_params)
            mlflow.log_param("trainable_parameters", trainable_params)
            
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=2e-4)
            mlflow.log_param("optimizer", "AdamW")
            mlflow.log_param("learning_rate", 0.0008)
            mlflow.log_param("weight_decay", 2e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
            
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
            
            train_losses = []
            val_accuracies = []
            best_val_acc = 0.0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (images, labels_batch) in enumerate(train_loader):
                    images, labels_batch = images.to(self.device, non_blocking=True), labels_batch.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels_batch)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels_batch.size(0)
                    train_correct += (predicted == labels_batch).sum().item()
                
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0.0
                
                with torch.no_grad():
                    for images, labels_batch in val_loader:
                        images, labels_batch = images.to(self.device, non_blocking=True), labels_batch.to(self.device, non_blocking=True)
                        if scaler:
                            with torch.cuda.amp.autocast():
                                outputs = model(images)
                                loss = criterion(outputs, labels_batch)
                        else:
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
                
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), self.output_dir / "models" / "best_brain_mri_model.pth")
                    mlflow.log_artifact(str(self.output_dir / "models" / "best_brain_mri_model.pth"))
                
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                                f"Train Loss: {avg_train_loss:.4f}, "
                                f"Train Acc: {train_acc:.2f}%, "
                                f"Val Loss: {avg_val_loss:.4f}, "
                                f"Val Acc: {val_acc:.2f}%")
                
                scheduler.step(avg_val_loss)
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered")
                    break
            
            logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("final_train_accuracy", train_acc)
            mlflow.log_metric("final_val_accuracy", val_acc)
            
            torch.save(model.state_dict(), self.output_dir / "models" / "final_brain_mri_model.pth")
            mlflow.log_artifact(str(self.output_dir / "models" / "final_brain_mri_model.pth"))
            mlflow.pytorch.log_model(model, "model", registered_model_name="brain_mri_classifier")
            
            metrics = {
                'epochs': epoch + 1,
                'best_validation_accuracy': best_val_acc,
                'final_train_accuracy': train_acc,
                'final_validation_accuracy': val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'num_classes': num_classes,
                'training_samples': len(train_paths),
                'validation_samples': len(val_paths),
                'model_type': '3D_CNN' if use_3d else '2D_CNN'
            }
            
            with open(self.output_dir / "metrics" / "training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            mlflow.log_artifact(str(self.output_dir / "metrics" / "training_metrics.json"))
            logger.info(f"Results saved to: {self.output_dir}")
            return metrics


def main():
    """Main entry point for brain MRI training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain MRI Images Classification Training Pipeline (50 epochs, improved)"
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
        default=50,
        help='Number of epochs for training (default: 50)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--use-3d',
        action='store_true',
        help='Use 3D CNN instead of 2D CNN for volumetric data'
    )
    parser.add_argument(
        '--mlflow-experiment',
        type=str,
        default='brain_mri_classification',
        help='MLflow experiment name (default: brain_mri_classification)'
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = BrainMRITrainingPipeline(output_dir=args.output_dir)
        
        logger.info("=" * 60)
        logger.info("BRAIN MRI CLASSIFICATION TRAINING PIPELINE")
        logger.info("=" * 60)
        
        metrics = pipeline.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_3d=args.use_3d,
            mlflow_experiment=args.mlflow_experiment
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
