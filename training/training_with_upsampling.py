#!/usr/bin/env python3
"""
Training implementation with upsampling for data imbalance.
Based on the problem statement requirements with data imbalance handling.
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] scikit-learn imbalanced-learn
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import warnings

def load_alzheimer_data():
    """
    Load Alzheimer's disease dataset using kagglehub.
    Uses the working brsdincer/alzheimer-features dataset.
    """
    # Set the path to the file you'd like to load
    file_path = "alzheimer.csv"
    
    # Load the latest version
    # Suppress deprecation warning for the exact API call from problem statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "brsdincer/alzheimer-features",
            file_path,
            # Provide any additional arguments like 
            # sql_query or pandas_kwargs. See the 
            # documentation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )

    print("First 5 records:", df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_data_with_upsampling(df):
    """
    Preprocess data including handling data imbalance with upsampling.
    """
    print(f"Initial dataset shape: {df.shape}")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.ffill().bfill()

    # Encode categorical columns
    if 'Group' in df.columns:
        # Keep original labels for now
        target_mapping = {'Demented': 'Demented', 'Nondemented': 'Nondemented', 'Converted': 'Converted'}
        df['Group'] = df['Group'].map(target_mapping)
        
    # Encode M/F column
    if 'M/F' in df.columns:
        df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})

    # Drop rows with missing target values
    df = df.dropna(subset=['Group'])
    
    print(f"After preprocessing shape: {df.shape}")
    
    # Check class distribution before upsampling
    print("Original class distribution:")
    print(df['Group'].value_counts())
    
    # Perform upsampling to handle data imbalance
    print("\nPerforming upsampling to handle data imbalance...")
    df_upsampled = perform_upsampling(df)
    
    # Select features
    available_features = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    features = [f for f in available_features if f in df_upsampled.columns]
    
    X = df_upsampled[features]
    y = df_upsampled['Group']
    
    # Final check for any remaining NaN values
    X = X.fillna(X.median())
    
    print(f"\nUsing features: {features}")
    print(f"Final target distribution after upsampling:\n{y.value_counts()}")
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

def perform_upsampling(df):
    """
    Perform upsampling to handle data imbalance.
    Uses random oversampling to balance the classes.
    """
    # Separate classes
    class_counts = df['Group'].value_counts()
    print(f"Class counts before upsampling: {class_counts.to_dict()}")
    
    # Find the majority class count
    majority_count = class_counts.max()
    
    # Separate dataframes by class
    df_majority = df[df['Group'] == class_counts.idxmax()]
    
    upsampled_dfs = [df_majority]
    
    # Upsample minority classes
    for class_name in class_counts.index:
        if class_name != class_counts.idxmax():  # Skip majority class
            df_class = df[df['Group'] == class_name]
            
            # Upsample minority class
            df_upsampled = resample(df_class,
                                  replace=True,  # Sample with replacement
                                  n_samples=majority_count,  # Match majority class
                                  random_state=42)  # Reproducible results
            upsampled_dfs.append(df_upsampled)
    
    # Combine upsampled dataframes
    df_balanced = pd.concat(upsampled_dfs, ignore_index=True)
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Class counts after upsampling: {df_balanced['Group'].value_counts().to_dict()}")
    
    return df_balanced

def train_model(X, y):
    """
    Train a machine learning model on the balanced dataset.
    """
    # Encode target labels for sklearn
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return clf, X_test, y_test, le

def evaluate_model(clf, X_test, y_test, label_encoder):
    """
    Evaluate the trained model and print metrics.
    """
    y_pred = clf.predict(X_test)
    
    print("\n=== Model Evaluation ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        feature_names = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'][:len(clf.feature_importances_)]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
    
    return y_pred

if __name__ == "__main__":
    print("=== Alzheimer's Training with Upsampling ===")
    
    # Load data
    df = load_alzheimer_data()
    
    # Preprocess data with upsampling
    X, y = preprocess_data_with_upsampling(df)
    
    # Train model
    clf, X_test, y_test, le = train_model(X, y)
    
    # Evaluate model
    y_pred = evaluate_model(clf, X_test, y_test, le)
    
    print("\n=== Training Complete ===")
    print("Data imbalance has been addressed through upsampling.")
    print("Model trained on balanced dataset.")