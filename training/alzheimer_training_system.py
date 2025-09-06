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

def load_alzheimer_data(file_path="Alzheimer_Dataset.csv"):
    """
    Load Alzheimer's disease dataset using kagglehub.
    You may need to adjust 'file_path' to match the correct CSV in the Kaggle dataset.
    """
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ananthu19/alzheimer-disease-and-healthy-aging-data-in-us",
        file_path
    )
    print("First 5 records:", df.head())
    return df

def preprocess_data(df):
    """
    Basic preprocessing: handle missing values, encode categorical variables, feature selection.
    """
    # Drop rows with missing target or features (customize as needed)
    df = df.dropna(subset=['Group'])  # Target column example: 'Group' (Demented/NonDemented)
    df = df.fillna(df.median(numeric_only=True))  # Fill missing values in numeric columns

    # Encode categorical columns
    if 'Group' in df.columns:
        df['Group'] = df['Group'].map({'Demented': 1, 'NonDemented': 0, 'Converted': 2})  # Adjust as needed

    # Select features (customize for your dataset)
    features = ['Age', 'Education', 'MMSE', 'CDR', 'eTIV', 'nWBV']  # Example features
    X = df[features]
    y = df['Group']

    return X, y

def train_model(X, y):
    """
    Train a machine learning model on the provided features and target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def evaluate_model(clf, X_test, y_test):
    """
    Evaluate the trained model and print metrics.
    """
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def predict(clf, new_data):
    """
    Predict Alzheimer's assessment for new patient data.
    'new_data' should be a DataFrame with the same columns as X.
    """
    y_pred = clf.predict(new_data)
    return y_pred

if __name__ == "__main__":
    # Load and preprocess data
    df = load_alzheimer_data(file_path="Alzheimer_Dataset.csv")  # Update CSV file name/path as needed
    X, y = preprocess_data(df)

    # Train the model
    clf, X_test, y_test = train_model(X, y)

    # Evaluate
    evaluate_model(clf, X_test, y_test)

    # Example prediction for new patient(s)
    # new_data = pd.DataFrame({'Age': [75], 'Education': [16], 'MMSE': [28], 'CDR': [0.5], 'eTIV': [1500], 'nWBV': [0.70]})
    # result = predict(clf, new_data)
    # print("Predicted group:", result)
