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

def load_alzheimer_data(file_path="alzheimer.csv"):
    """
    Load Alzheimer's disease dataset using kagglehub.
    Uses the working brsdincer/alzheimer-features dataset.
    """
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "brsdincer/alzheimer-features",
        file_path
    )
    print("First 5 records:", df.head())
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess_data(df, apply_upsampling=True):
    """
    Basic preprocessing: handle missing values, encode categorical variables, feature selection, and upsampling.
    
    Args:
        df: Input dataframe
        apply_upsampling: Whether to apply upsampling to handle data imbalance
    """
    print(f"Initial dataset shape: {df.shape}")
    
    # Handle missing values - fill with median for numeric columns first
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill any remaining missing values
    df = df.ffill().bfill()

    # Encode categorical columns
    if 'Group' in df.columns:
        # Check original distribution before encoding
        print("Original class distribution:")
        print(df['Group'].value_counts())
        
        # Handle multiple classes including 'Converted'
        group_mapping = {'Demented': 1, 'Nondemented': 0, 'Converted': 2}
        df['Group'] = df['Group'].map(group_mapping)
        
    # Encode M/F column
    if 'M/F' in df.columns:
        df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})

    # Drop rows with missing target values after encoding
    df = df.dropna(subset=['Group'])
    
    print(f"After preprocessing shape: {df.shape}")
    
    # Apply upsampling if requested
    if apply_upsampling:
        print("\nPerforming upsampling to handle data imbalance...")
        df = perform_upsampling(df)

    # Select features that are available in the dataset
    # Based on the columns: ['Group', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    available_features = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    features = [f for f in available_features if f in df.columns]
    
    X = df[features]
    y = df['Group']
    
    # Final check for any remaining NaN values
    X = X.fillna(X.median())
    
    print(f"Using features: {features}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    print(f"Any NaN in X: {X.isnull().any().any()}")
    print(f"Any NaN in y: {y.isnull().any()}")

    return X, y

def perform_upsampling(df):
    """
    Perform upsampling to handle data imbalance.
    Uses random oversampling to balance the classes.
    """
    from sklearn.utils import resample
    
    # Separate classes
    class_counts = df['Group'].value_counts()
    print(f"Class counts before upsampling: {class_counts.to_dict()}")
    
    # Find the majority class count
    majority_count = class_counts.max()
    
    # Separate dataframes by class
    df_majority = df[df['Group'] == class_counts.idxmax()]
    
    upsampled_dfs = [df_majority]
    
    # Upsample minority classes
    for class_value in class_counts.index:
        if class_value != class_counts.idxmax():  # Skip majority class
            df_class = df[df['Group'] == class_value]
            
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
    Train a machine learning model on the provided features and target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Use a more conservative model to avoid overfitting
    clf = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=5,      # Add max depth limit
        min_samples_split=5,  # Increase minimum samples for split
        min_samples_leaf=2,   # Minimum samples in leaf
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return clf, X_test, y_test

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    """
    y_pred = clf.predict(X_test)
    print("\n=== Model Evaluation ===")
    
    # Create target names based on unique values
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    label_map = {0: 'Nondemented', 1: 'Demented', 2: 'Converted'}
    target_names = [label_map.get(label, f'Class_{label}') for label in unique_labels]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
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
    
    # Optionally create visualization (commented out to avoid display issues in headless environment)
    # plt.figure(figsize=(8,6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
    #             xticklabels=target_names,
    #             yticklabels=target_names)
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()
    
    return y_pred

def save_model(clf, model_path="alzheimer_model.pkl"):
    """
    Save the trained model to disk.
    """
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

def load_model(model_path="alzheimer_model.pkl"):
    """
    Load a trained model from disk.
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return clf
    else:
        print(f"Model file {model_path} not found")
        return None

def predict(clf, new_data):
    """
    Predict Alzheimer's assessment for new patient data.
    'new_data' should be a DataFrame with the same columns as X.
    """
    y_pred = clf.predict(new_data)
    probabilities = clf.predict_proba(new_data)
    
    # Map numeric predictions back to labels
    label_map = {0: 'Nondemented', 1: 'Demented', 2: 'Converted'}
    
    results = []
    for i, (pred, prob) in enumerate(zip(y_pred, probabilities)):
        result = {
            'prediction': label_map.get(pred, f'Class_{pred}'),
            'confidence': max(prob),
            'probabilities': {}
        }
        
        # Map probabilities back to class names
        for j, prob_val in enumerate(prob):
            class_name = label_map.get(j, f'Class_{j}')
            result['probabilities'][class_name] = prob_val
        
        results.append(result)
    
    return results

if __name__ == "__main__":
    print("=== Comprehensive Alzheimer's Training System with Upsampling ===")
    
    import sys
    
    # Check if upsampling should be applied (default: True)
    apply_upsampling = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-upsampling":
        apply_upsampling = False
        print("Note: Upsampling disabled by command line argument")
    else:
        print("Note: Upsampling enabled to handle data imbalance")
    
    # Load and preprocess data
    df = load_alzheimer_data(file_path="alzheimer.csv")
    X, y = preprocess_data(df, apply_upsampling=apply_upsampling)

    # Train the model
    clf, X_test, y_test = train_model(X, y)

    # Evaluate
    y_pred = evaluate_model(clf, X_test, y_test)
    
    # Save the model
    save_model(clf, "models/alzheimer_model_upsampled.pkl")
    
    # Test model loading and prediction
    print("\n=== Testing Model Prediction ===")
    
    # Create example patient data for prediction
    example_data = pd.DataFrame({
        'M/F': [1, 0],  # Male, Female
        'Age': [75, 82], 
        'EDUC': [16, 12], 
        'SES': [2.0, 3.0], 
        'MMSE': [28, 20], 
        'CDR': [0.5, 1.0], 
        'eTIV': [1500, 1400], 
        'nWBV': [0.70, 0.65], 
        'ASF': [1.0, 1.1]
    })
    
    results = predict(clf, example_data)
    print("Example predictions:")
    for i, result in enumerate(results):
        print(f"Patient {i+1}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        if len(result['probabilities']) == 2:
            print(f"  Probabilities: Nondemented={result['probabilities']['Nondemented']:.3f}, "
                  f"Demented={result['probabilities']['Demented']:.3f}")
        else:
            print(f"  Probabilities: {result['probabilities']}")
    
    print("\n=== Training Complete ===")
    print(f"Model trained on {len(X)} samples with {len(X.columns)} features")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    if apply_upsampling:
        print("Data imbalance was addressed through upsampling")
    print("Model saved and ready for use in simulations")
