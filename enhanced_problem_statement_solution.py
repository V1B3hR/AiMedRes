#!/usr/bin/env python3
"""
Deep learning, training.

ENHANCED VERSION: Problem statement implementation with deep learning training capabilities.
This extends the corrected problem statement to include actual deep learning training.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=== Deep Learning, Training - Enhanced Implementation ===")

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "brsdincer/alzheimer-features",  # Working Alzheimer dataset
  "alzheimer.csv",  # Specify the CSV file
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documentation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Enhanced preprocessing for deep learning
print("\n=== Preprocessing for Deep Learning ===")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Encode categorical variables
if 'M/F' in df.columns:
    df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})

# Prepare target variable
if 'Group' in df.columns:
    le = LabelEncoder()
    y = le.fit_transform(df['Group'])
    target_names = le.classes_
    print(f"Target classes: {target_names}")
    print(f"Target distribution: {pd.Series(y).value_counts().sort_index()}")
else:
    raise ValueError("Target column 'Group' not found in dataset")

# Prepare features
feature_columns = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
available_features = [col for col in feature_columns if col in df.columns]
X = df[available_features].values

print(f"Features used: {available_features}")
print(f"Feature matrix shape: {X.shape}")

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Deep Learning Model Training
print("\n=== Deep Learning Model Training ===")

# Neural Network (MLPClassifier for deep learning)
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    verbose=False
)

print("Training deep neural network...")
mlp.fit(X_train, y_train)

# Predictions
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"Deep Learning Model Accuracy: {accuracy_mlp:.3f}")

# Comparison with traditional ML
print("\n=== Comparison with Traditional ML ===")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.3f}")

# Detailed results
print("\n=== Deep Learning Model Results ===")
print("Classification Report:")
print(classification_report(y_test, y_pred_mlp, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_mlp)
print(cm)

# Feature importance (using Random Forest as proxy)
print(f"\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

print("\n=== Training Summary ===")
print(f"✅ Successfully loaded Alzheimer's dataset with {len(df)} samples")
print(f"✅ Trained deep neural network with 3 hidden layers")
print(f"✅ Deep Learning Accuracy: {accuracy_mlp:.1%}")
print(f"✅ Traditional ML Accuracy: {accuracy_rf:.1%}")
print(f"✅ Used {len(available_features)} features for prediction")
print("✅ Deep learning training completed successfully!")

# Model architecture info
print(f"\n=== Model Architecture ===")
print(f"Input features: {X_train.shape[1]}")
print(f"Hidden layers: {mlp.hidden_layer_sizes}")
print(f"Output classes: {len(target_names)}")
print(f"Total parameters: ~{sum([layer * next_layer for layer, next_layer in zip([X_train.shape[1]] + list(mlp.hidden_layer_sizes), list(mlp.hidden_layer_sizes) + [len(target_names)])])}")
print(f"Training iterations: {mlp.n_iter_}")