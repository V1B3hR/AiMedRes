#!/usr/bin/env python3
"""
Complete Training Implementation - Problem Statement + Machine Learning
This implements the exact problem statement code and adds comprehensive training
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] scikit-learn matplotlib seaborn
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

def load_problem_statement_data():
    """Load data exactly as specified in the problem statement"""
    print("=== Problem Statement Implementation ===")
    
    # Set the path to the file you'd like to load
    file_path = "oasis_longitudinal.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "jboysen/mri-and-alzheimers",
      file_path,
      # Provide any additional arguments like 
      # sql_query or pandas_kwargs. See the 
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print("First 5 records:", df.head())
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\n=== Data Preprocessing ===")
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill any remaining missing values
    df = df.ffill().bfill()
    
    # Encode categorical variables
    le_group = LabelEncoder()
    df['Group_encoded'] = le_group.fit_transform(df['Group'])
    
    le_mf = LabelEncoder()
    df['MF_encoded'] = le_mf.fit_transform(df['M/F'])
    
    le_hand = LabelEncoder()
    df['Hand_encoded'] = le_hand.fit_transform(df['Hand'])
    
    # Select features for training
    feature_cols = ['Visit', 'MR Delay', 'MF_encoded', 'Hand_encoded', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    X = df[feature_cols]
    y = df['Group_encoded']
    
    print(f"Features: {feature_cols}")
    print(f"Target classes: {le_group.classes_}")
    print(f"Preprocessed shape: X={X.shape}, y={y.shape}")
    
    return X, y, le_group

def train_model(X, y):
    """Train a machine learning model"""
    print("\n=== Model Training ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    
    return rf_model, X_test, y_test, y_pred

def save_model(model, model_path="alzheimer_mri_model.pkl"):
    """Save the trained model"""
    import pickle
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

def main():
    """Main training pipeline"""
    try:
        # Step 1: Load data using exact problem statement code
        df = load_problem_statement_data()
        
        # Step 2: Preprocess data
        X, y, label_encoder = preprocess_data(df)
        
        # Step 3: Train model
        model, X_test, y_test, y_pred = train_model(X, y)
        
        # Step 4: Detailed evaluation
        print("\n=== Model Evaluation ===")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Step 5: Save model
        save_model(model)
        
        print("\n" + "="*50)
        print("✓ Problem statement implementation successful!")
        print("✓ Training completed successfully!")
        print("✓ Model saved and ready for deployment!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)