#!/usr/bin/env python3
"""
Enhanced Medical Data Processing Module
Handles preprocessing of medical data including Alzheimer's datasets with optimization features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Dict, Any, Optional, List
import time
import logging

logger = logging.getLogger(__name__)

def preprocess_alzheimer_data(df: pd.DataFrame, target_col: str = 'diagnosis', 
                            optimization_level: str = 'standard') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Enhanced Alzheimer's dataset preprocessing with optimization for large datasets
    
    Args:
        df: Raw Alzheimer's dataset
        target_col: Name of target column
        optimization_level: 'fast', 'standard', or 'comprehensive'
        
    Returns:
        Tuple of (features, target)
    """
    start_time = time.time()
    logger.info(f"Starting Alzheimer's data preprocessing (level: {optimization_level})")
    
    # Separate features and target
    if target_col in df.columns:
        features = df.drop(target_col, axis=1)
        target = df[target_col]
    else:
        features = df
        target = None
        logger.warning(f"Target column '{target_col}' not found")
    
    # Apply preprocessing based on optimization level
    if optimization_level == 'fast':
        # Minimal preprocessing for speed
        features = features.fillna(features.median(numeric_only=True))
    elif optimization_level == 'comprehensive':
        # Full preprocessing pipeline
        features = comprehensive_preprocessing(features)
    else:
        # Standard preprocessing
        features = standard_preprocessing(features)
    
    processing_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {processing_time:.2f}s")
    
    return features, target

def comprehensive_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive preprocessing with advanced techniques"""
    
    # Handle missing values with KNN imputation for better accuracy
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # KNN imputation for numerical features
    if len(numerical_cols) > 0:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])
    
    # Mode imputation for categorical features
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    # Feature engineering for medical data
    df = add_medical_features(df)
    
    return df

def standard_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Standard preprocessing for balanced speed and accuracy"""
    
    # Simple imputation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Median imputation for numerical, mode for categorical
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def add_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific medical features for Alzheimer's prediction"""
    
    # Age-related features
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 65, 75, 85, 100], 
                                labels=[0, 1, 2, 3])  # Use numeric labels
        df['age_squared'] = df['age'] ** 2
    
    # Cognitive assessment ratios
    if 'mmse_score' in df.columns and 'education_level' in df.columns:
        df['mmse_education_ratio'] = df['mmse_score'] / (df['education_level'] + 1)
    
    if 'cdr_score' in df.columns and 'mmse_score' in df.columns:
        df['cognitive_decline_indicator'] = df['cdr_score'] * (30 - df['mmse_score'])
    
    # APOE risk encoding
    if 'apoe_genotype' in df.columns:
        df['apoe_risk_score'] = df['apoe_genotype'].map({
            'E2/E2': 0, 'E2/E3': 0, 'E2/E4': 1,
            'E3/E3': 1, 'E3/E4': 2, 'E4/E4': 3
        }).fillna(1)  # Default to moderate risk
    
    return df

def clean_medical_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """Enhanced medical data cleaning with outlier detection"""
    
    logger.info("Starting medical data cleaning...")
    initial_rows = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Outlier removal using IQR method
    if remove_outliers:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            outliers_removed = outliers_before - len(df)
            
            if outliers_removed > 0:
                logger.info(f"Removed {outliers_removed} outliers from {col}")
    
    logger.info(f"Data cleaning completed. Final dataset: {len(df)} rows")
    return df

def normalize_features(X: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, object]:
    """Enhanced feature normalization with multiple methods"""
    
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()  # Less sensitive to outliers
    else:
        logger.warning(f"Unknown normalization method {method}, using standard")
        scaler = StandardScaler()
    
    X_scaled = X.copy()
    if len(numerical_columns) > 0:
        X_scaled[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        logger.info(f"Normalized {len(numerical_columns)} numerical features using {method} scaling")
    
    return X_scaled, scaler

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 'all', 
                   method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
    """Enhanced feature selection for medical data"""
    
    if k == 'all' or k >= X.shape[1]:
        return X, list(X.columns)
    
    logger.info(f"Selecting top {k} features using {method}")
    
    # Use only numerical features for statistical tests
    numerical_features = X.select_dtypes(include=[np.number])
    
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(k, numerical_features.shape[1]))
        X_selected = selector.fit_transform(numerical_features, y)
        
        # Get selected feature names
        selected_features = numerical_features.columns[selector.get_support()].tolist()
        
        # Add back categorical features if space allows
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        remaining_slots = k - len(selected_features)
        
        if remaining_slots > 0:
            selected_features.extend(categorical_features[:remaining_slots])
        
        X_final = X[selected_features]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:5]}...")
        return X_final, selected_features
    
    return X, list(X.columns)

def optimize_for_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Memory optimization for large datasets"""
    
    logger.info("Optimizing memory usage...")
    
    # Downcast numerical types
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category if beneficial
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    logger.info("Memory optimization completed")
    return df

__all__ = [
    'preprocess_alzheimer_data', 
    'clean_medical_data', 
    'normalize_features',
    'select_features',
    'optimize_for_memory',
    'add_medical_features'
]