#!/usr/bin/env python3
"""
Enhanced dementia prediction training system
Integrates the problem statement dataset with the existing training infrastructure
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_dementia_dataset():
    """Load the dementia prediction dataset as specified in problem statement"""
    print("=== Loading Dementia Prediction Dataset ===")
    
    # Set the path to the file you'd like to load
    file_path = ""
    
    # Auto-detect file when empty (required for kagglehub to work)
    if not file_path:
        import os
        dataset_path = kagglehub.dataset_download("shashwatwork/dementia-prediction-dataset")
        files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        file_path = files[0] if files else "dementia_dataset.csv"
        print(f"Auto-detected file: {file_path}")
    
    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "shashwatwork/dementia-prediction-dataset",
      file_path,
      # Provide any additional arguments like 
      # sql_query or pandas_kwargs. See the 
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    
    print("First 5 records:", df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_dementia_data(df):
    """Preprocess the dementia dataset for training"""
    print("\n=== Preprocessing Data ===")
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill any remaining missing values
    df = df.ffill().bfill()
    
    # Encode categorical variables
    le_group = LabelEncoder()
    if 'Group' in df.columns:
        df['Group_encoded'] = le_group.fit_transform(df['Group'])
    
    le_mf = LabelEncoder()
    if 'M/F' in df.columns:
        df['MF_encoded'] = le_mf.fit_transform(df['M/F'])
    
    le_hand = LabelEncoder()
    if 'Hand' in df.columns:
        df['Hand_encoded'] = le_hand.fit_transform(df['Hand'])
    
    # Select features for training
    feature_cols = ['Visit', 'MR Delay', 'MF_encoded', 'Hand_encoded', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df['Group_encoded'] if 'Group_encoded' in df.columns else df.get('Group', df.iloc[:, -1])
    
    print(f"Features used: {available_features}")
    print(f"Target distribution:")
    if hasattr(y, 'value_counts'):
        print(y.value_counts())
    print(f"Preprocessed shape: X={X.shape}, y={y.shape}")
    
    return X, y, le_group

def train_dementia_model(X, y):
    """Train a model on the dementia dataset"""
    print("\n=== Training Dementia Prediction Model ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained on {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))
    
    return clf, X_test, y_test

def run_complete_dementia_training():
    """Run the complete dementia prediction training pipeline"""
    print("=== Dementia Prediction Training System ===")
    
    # Load data
    df = load_dementia_dataset()
    
    # Preprocess
    X, y, label_encoder = preprocess_dementia_data(df)
    
    # Train model
    clf, X_test, y_test = train_dementia_model(X, y)
    
    print("\n=== Training Complete ===")
    print(f"Model ready for predictions")
    print(f"Feature importance (top 5):")
    feature_names = X.columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.3f}")
    
    return clf, df, label_encoder

if __name__ == "__main__":
    clf, df, le = run_complete_dementia_training()