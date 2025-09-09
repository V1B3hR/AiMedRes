#!/usr/bin/env python3
"""
Enhanced Alzheimer Training System
Supports multiple datasets including the new rabieelkharoua/alzheimers-disease-dataset
Train on real medical data and deployed in realistic collaborative scenarios,
creating a foundation for advanced medical AI research and applications.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] scikit-learn matplotlib seaborn

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

def load_alzheimer_data_new(file_path="alzheimers_disease_data.csv"):
    """
    Load Alzheimer's disease dataset using kagglehub from the new comprehensive dataset.
    This implements the exact problem statement requirements.
    """
    print("Loading comprehensive Alzheimer's disease dataset...")
    
    # Suppress the deprecation warning for exact problem statement compliance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "rabieelkharoua/alzheimers-disease-dataset",
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

def load_alzheimer_data_original(file_path="alzheimer.csv"):
    """
    Load original Alzheimer's disease dataset for comparison and validation.
    Uses the working brsdincer/alzheimer-features dataset.
    """
    print("Loading original Alzheimer's features dataset...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "brsdincer/alzheimer-features",
        file_path
    )
    print("First 5 records:", df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_new_dataset(df):
    """
    Preprocess the new comprehensive Alzheimer's dataset.
    """
    print(f"Initial dataset shape: {df.shape}")
    
    # The dataset is already clean (no missing values), but let's prepare features
    # Remove non-predictive columns
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1, errors='ignore')
    
    # Select relevant features for prediction
    feature_columns = [
        'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 
        'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
        'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
        'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 
        'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 
        'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 
        'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges', 
        'DifficultyCompletingTasks', 'Forgetfulness'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in df.columns]
    X = df[available_features]
    y = df['Diagnosis']  # 0 = No Alzheimer's, 1 = Alzheimer's
    
    print(f"Using features: {available_features}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

def train_enhanced_model(X, y):
    """
    Train an enhanced machine learning model on the comprehensive dataset.
    """
    print("\n=== Training Enhanced Alzheimer's Model ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier with enhanced parameters
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    print(f"Model trained on {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return clf, X_test, y_test

def evaluate_enhanced_model(clf, X_test, y_test):
    """
    Evaluate the enhanced model with comprehensive metrics.
    """
    y_pred = clf.predict(X_test)
    print("\n=== Enhanced Model Evaluation ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Alzheimer\'s', 'Alzheimer\'s']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance analysis
    if hasattr(clf, 'feature_importances_') and len(clf.feature_importances_) > 0:
        # Get feature names from the last preprocessing step
        feature_names = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 
            'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
            'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 
            'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 
            'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 
            'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 
            'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 
            'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges', 
            'DifficultyCompletingTasks', 'Forgetfulness'
        ][:len(clf.feature_importances_)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(importance_df.head(10))
    
    return y_pred

def save_enhanced_model(clf, model_path="models/enhanced_alzheimer_model.pkl"):
    """
    Save the enhanced trained model to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Enhanced model saved to {model_path}")

def predict_enhanced(clf, new_data):
    """
    Predict Alzheimer's assessment for new patient data using the enhanced model.
    """
    y_pred = clf.predict(new_data)
    probabilities = clf.predict_proba(new_data)
    
    results = []
    for i, (pred, prob) in enumerate(zip(y_pred, probabilities)):
        result = {
            'prediction': 'Alzheimer\'s' if pred == 1 else 'No Alzheimer\'s',
            'confidence': max(prob),
            'probabilities': {
                'No Alzheimer\'s': prob[0],
                'Alzheimer\'s': prob[1] if len(prob) > 1 else 0.0
            }
        }
        results.append(result)
    
    return results

def run_enhanced_training():
    """
    Run the complete enhanced training pipeline.
    """
    print("=== Enhanced Alzheimer's Disease Prediction Training ===")
    
    # Load the new comprehensive dataset (as per problem statement)
    df = load_alzheimer_data_new()
    
    # Preprocess the data
    X, y = preprocess_new_dataset(df)
    
    # Train the model
    clf, X_test, y_test = train_enhanced_model(X, y)
    
    # Evaluate the model
    y_pred = evaluate_enhanced_model(clf, X_test, y_test)
    
    # Save the model
    save_enhanced_model(clf, "models/enhanced_alzheimer_model.pkl")
    
    # Test with example predictions
    print("\n=== Example Enhanced Predictions ===")
    # Create example data with the same structure as the training data
    example_data = pd.DataFrame({
        'Age': [75, 85], 
        'Gender': [0, 1],  # 0=Female, 1=Male
        'Ethnicity': [0, 1],
        'EducationLevel': [2, 3],
        'BMI': [25.5, 28.2],
        'Smoking': [0, 1],
        'AlcoholConsumption': [2.0, 0.0],
        'PhysicalActivity': [3.0, 1.0],
        'DietQuality': [7.0, 5.0],
        'SleepQuality': [6.0, 4.0],
        'FamilyHistoryAlzheimers': [1, 0],
        'CardiovascularDisease': [0, 1],
        'Diabetes': [0, 1],
        'Depression': [0, 1],
        'HeadInjury': [0, 0],
        'Hypertension': [1, 1],
        'SystolicBP': [130, 150],
        'DiastolicBP': [80, 90],
        'CholesterolTotal': [200.0, 250.0],
        'CholesterolLDL': [120.0, 160.0],
        'CholesterolHDL': [50.0, 40.0],
        'CholesterolTriglycerides': [150.0, 200.0],
        'MMSE': [28.0, 20.0],
        'FunctionalAssessment': [8.0, 5.0],
        'MemoryComplaints': [0, 1],
        'BehavioralProblems': [0, 1],
        'ADL': [9.0, 6.0],
        'Confusion': [0, 1],
        'Disorientation': [0, 1],
        'PersonalityChanges': [0, 1],
        'DifficultyCompletingTasks': [0, 1],
        'Forgetfulness': [0, 1]
    })
    
    results = predict_enhanced(clf, example_data)
    print("Enhanced model predictions:")
    for i, result in enumerate(results):
        print(f"Patient {i+1}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        print(f"  Probabilities: No Alzheimer's={result['probabilities']['No Alzheimer\'s']:.3f}, "
              f"Alzheimer's={result['probabilities']['Alzheimer\'s']:.3f}")
    
    print("\n=== Enhanced Training Complete ===")
    print(f"Enhanced model trained on {len(X)} samples with {len(X.columns)} features")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Enhanced model saved and ready for use in advanced medical AI research and applications")
    
    return clf, df

if __name__ == "__main__":
    clf, df = run_enhanced_training()