#!/usr/bin/env python3
"""
Enhanced Ensemble Training System with Advanced Model Performance Optimization
Implements comprehensive cross-validation, hyperparameter tuning, and ensemble methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import pickle
import json
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_validate, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline

# Import base training components
from training import AlzheimerTrainer, TrainingConfig

logger = logging.getLogger("EnhancedEnsembleTraining")


class AdvancedFeatureEngineering:
    """Advanced feature engineering for enhanced model performance"""
    
    def __init__(self):
        self.feature_interactions = {}
        self.derived_features = {}
        self.scaler = None
        self.feature_selector = None
        
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        X_enhanced = X.copy()
        
        # Age-related interactions
        if 'Age' in X.columns and 'MMSE' in X.columns:
            X_enhanced['Age_MMSE_ratio'] = X['Age'] / (X['MMSE'] + 1)
            
        if 'Age' in X.columns and 'EDUC' in X.columns:
            X_enhanced['Age_Education_interaction'] = X['Age'] * X['EDUC']
            
        # Cognitive-imaging interactions
        if 'MMSE' in X.columns and 'nWBV' in X.columns:
            X_enhanced['MMSE_BrainVolume_correlation'] = X['MMSE'] * X['nWBV']
            
        if 'CDR' in X.columns and 'nWBV' in X.columns:
            X_enhanced['CDR_BrainVolume_interaction'] = X['CDR'] * (1 - X['nWBV'])
            
        # Brain volume ratios
        if 'eTIV' in X.columns and 'nWBV' in X.columns:
            X_enhanced['Brain_Atrophy_Index'] = (X['eTIV'] * (1 - X['nWBV'])) / 1000
            
        # Socioeconomic interactions
        if 'EDUC' in X.columns and 'SES' in X.columns:
            X_enhanced['Education_SES_combined'] = X['EDUC'] / (X['SES'] + 1)
            
        return X_enhanced
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key continuous variables"""
        X_poly = X.copy()
        
        continuous_cols = ['Age', 'MMSE', 'nWBV', 'eTIV', 'ASF']
        available_cols = [col for col in continuous_cols if col in X.columns]
        
        for col in available_cols:
            if degree >= 2:
                X_poly[f'{col}_squared'] = X[col] ** 2
            if degree >= 3:
                X_poly[f'{col}_cubed'] = X[col] ** 3
                
        return X_poly
    
    def create_binned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create binned categorical features from continuous variables"""
        X_binned = X.copy()
        
        # Age groups
        if 'Age' in X.columns:
            X_binned['Age_Group'] = pd.cut(X['Age'], 
                                         bins=[0, 65, 75, 85, 100], 
                                         labels=['Young', 'Early_Elderly', 'Late_Elderly', 'Very_Old'])
            X_binned = pd.get_dummies(X_binned, columns=['Age_Group'], prefix='Age')
        
        # MMSE categories
        if 'MMSE' in X.columns:
            X_binned['MMSE_Category'] = pd.cut(X['MMSE'],
                                             bins=[0, 17, 23, 30],
                                             labels=['Severe', 'Mild', 'Normal'])
            X_binned = pd.get_dummies(X_binned, columns=['MMSE_Category'], prefix='MMSE')
        
        # Brain volume categories
        if 'nWBV' in X.columns:
            X_binned['BrainVol_Category'] = pd.cut(X['nWBV'],
                                                  bins=[0, 0.7, 0.75, 1.0],
                                                  labels=['Severe_Atrophy', 'Mild_Atrophy', 'Normal'])
            X_binned = pd.get_dummies(X_binned, columns=['BrainVol_Category'], prefix='BrainVol')
            
        return X_binned
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Fit and transform features with comprehensive engineering"""
        
        # Step 1: Create interaction features
        X_enhanced = self.create_interaction_features(X)
        
        # Step 2: Add polynomial features
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        
        # Step 3: Add binned features
        X_enhanced = self.create_binned_features(X_enhanced)
        
        # Step 4: Feature selection (if target is provided)
        if y is not None:
            # Select top K features based on statistical tests
            self.feature_selector = SelectKBest(f_classif, k=min(50, X_enhanced.shape[1]))
            X_enhanced_selected = self.feature_selector.fit_transform(X_enhanced, y)
            
            # Get selected feature names
            selected_features = X_enhanced.columns[self.feature_selector.get_support()].tolist()
            X_enhanced = pd.DataFrame(X_enhanced_selected, columns=selected_features, index=X_enhanced.index)
        
        return X_enhanced
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted engineering pipeline"""
        
        # Apply same transformations
        X_enhanced = self.create_interaction_features(X)
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        X_enhanced = self.create_binned_features(X_enhanced)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            # Ensure all features are present
            missing_features = set(self.feature_selector.get_feature_names_out()) - set(X_enhanced.columns)
            for feature in missing_features:
                X_enhanced[feature] = 0
                
            X_enhanced_selected = self.feature_selector.transform(X_enhanced)
            selected_features = [f for f in X_enhanced.columns if f in self.feature_selector.get_feature_names_out()]
            X_enhanced = pd.DataFrame(X_enhanced_selected, columns=selected_features, index=X_enhanced.index)
        
        return X_enhanced


class EnhancedEnsembleTrainer:
    """Enhanced ensemble trainer with comprehensive ML optimization"""
    
    def __init__(self, trainer: AlzheimerTrainer):
        self.trainer = trainer
        self.feature_engineer = AdvancedFeatureEngineering()
        self.scalers = {}
        self.trained_models = {}
        self.ensemble_models = {}
        self.validation_results = {}
        
    def get_advanced_model_grid(self) -> Dict[str, Dict]:
        """Get comprehensive hyperparameter grids for all models"""
        
        param_grids = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'class_weight': [None, 'balanced']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (100, 50, 25)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive'],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            }
        }
        
        return param_grids
    
    def comprehensive_hyperparameter_tuning(self, X: pd.DataFrame, y: np.ndarray, 
                                          cv_folds: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """Comprehensive hyperparameter tuning with multiple search strategies"""
        
        logger.info("🔍 Starting comprehensive hyperparameter tuning...")
        
        model_grids = self.get_advanced_model_grid()
        tuning_results = {}
        
        # Stratified K-Fold for better cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model_config in model_grids.items():
            logger.info(f"📊 Tuning {model_name}...")
            
            try:
                # Use RandomizedSearchCV for efficiency with large parameter spaces
                if model_name in ['SVM', 'NeuralNetwork']:
                    search = RandomizedSearchCV(
                        model_config['model'],
                        model_config['params'],
                        n_iter=n_iter,
                        cv=skf,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0,
                        random_state=42
                    )
                else:
                    # Use GridSearchCV for smaller parameter spaces
                    search = GridSearchCV(
                        model_config['model'],
                        model_config['params'],
                        cv=skf,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                
                # Fit the search
                search.fit(X, y)
                
                # Store results
                tuning_results[model_name] = {
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'best_model': search.best_estimator_,
                    'cv_results': search.cv_results_
                }
                
                logger.info(f"   {model_name}: Best CV Score = {search.best_score_:.3f}")
                
            except Exception as e:
                logger.warning(f"   {model_name} tuning failed: {e}")
                continue
        
        return tuning_results
    
    def create_advanced_ensembles(self, tuned_models: Dict[str, Any], 
                                X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
        """Create advanced ensemble models with multiple strategies"""
        
        logger.info("🤝 Creating advanced ensemble models...")
        
        # Extract best models
        best_models = {}
        for model_name, results in tuned_models.items():
            if 'best_model' in results:
                best_models[model_name] = results['best_model']
        
        if len(best_models) < 2:
            logger.warning("Not enough models for ensemble creation")
            return {}
        
        ensemble_models = {}
        
        # 1. Voting Classifier (Soft Voting)
        voting_estimators = [(name, model) for name, model in best_models.items()]
        voting_classifier = VotingClassifier(estimators=voting_estimators, voting='soft')
        
        # 2. Stacking Classifier
        # Select top 3 models as base learners
        top_models = sorted(tuned_models.items(), key=lambda x: x[1].get('best_score', 0), reverse=True)[:3]
        stacking_estimators = [(name, results['best_model']) for name, results in top_models]
        
        stacking_classifier = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        # 3. Bagging with best single model
        best_single_model_name = max(tuned_models.keys(), key=lambda k: tuned_models[k].get('best_score', 0))
        best_single_model = tuned_models[best_single_model_name]['best_model']
        
        bagging_classifier = BaggingClassifier(
            base_estimator=best_single_model,
            n_estimators=10,
            random_state=42
        )
        
        # Train ensemble models
        ensemble_models['VotingClassifier'] = voting_classifier
        ensemble_models['StackingClassifier'] = stacking_classifier  
        ensemble_models['BaggingClassifier'] = bagging_classifier
        
        for name, ensemble in ensemble_models.items():
            logger.info(f"Training {name}...")
            ensemble.fit(X_train, y_train)
        
        return ensemble_models
    
    def comprehensive_cross_validation(self, models: Dict[str, Any], 
                                     X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Comprehensive cross-validation with multiple metrics"""
        
        logger.info("📈 Running comprehensive cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name, model in models.items():
            logger.info(f"   Evaluating {model_name}...")
            
            try:
                # Multi-metric cross-validation
                cv_scores = cross_validate(model, X, y, cv=skf, scoring=scoring_metrics, 
                                         return_train_score=True)
                
                # Calculate statistics for each metric
                results = {}
                for metric in scoring_metrics:
                    test_scores = cv_scores[f'test_{metric}']
                    train_scores = cv_scores[f'train_{metric}']
                    
                    results[metric] = {
                        'test_mean': np.mean(test_scores),
                        'test_std': np.std(test_scores),
                        'train_mean': np.mean(train_scores),
                        'train_std': np.std(train_scores),
                        'overfitting': np.mean(train_scores) - np.mean(test_scores)
                    }
                
                cv_results[model_name] = results
                
                logger.info(f"     Accuracy: {results['accuracy']['test_mean']:.3f} ± {results['accuracy']['test_std']:.3f}")
                
            except Exception as e:
                logger.warning(f"   {model_name} evaluation failed: {e}")
                continue
        
        return cv_results
    
    def run_enhanced_ensemble_training(self) -> Dict[str, Any]:
        """Run complete enhanced ensemble training pipeline"""
        
        logger.info("🚀 Starting Enhanced Ensemble Training Pipeline...")
        
        # Load and preprocess data
        df = self.trainer.load_data()
        X, y = self.trainer.preprocess_data(df)
        
        # Convert to DataFrame for feature engineering
        if isinstance(X, np.ndarray):
            feature_names = self.trainer.feature_columns if hasattr(self.trainer, 'feature_columns') else [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Advanced feature engineering
        logger.info("🔧 Applying advanced feature engineering...")
        X_train_enhanced = self.feature_engineer.fit_transform(X_train, y_train)
        X_test_enhanced = self.feature_engineer.transform(X_test)
        
        # Hyperparameter tuning
        tuned_models = self.comprehensive_hyperparameter_tuning(X_train_enhanced, y_train)
        self.trained_models = tuned_models
        
        # Create ensemble models
        ensemble_models = self.create_advanced_ensembles(tuned_models, X_train_enhanced, y_train)
        self.ensemble_models = ensemble_models
        
        # Combine all models for evaluation
        all_models = {}
        for name, results in tuned_models.items():
            if 'best_model' in results:
                all_models[name] = results['best_model']
        all_models.update(ensemble_models)
        
        # Comprehensive cross-validation
        cv_results = self.comprehensive_cross_validation(all_models, X_train_enhanced, y_train)
        self.validation_results = cv_results
        
        # Final evaluation on test set
        logger.info("📊 Final evaluation on test set...")
        test_results = {}
        
        for model_name, model in all_models.items():
            try:
                y_pred = model.predict(X_test_enhanced)
                y_prob = model.predict_proba(X_test_enhanced)[:, 1] if hasattr(model, 'predict_proba') else None
                
                test_results[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                logger.info(f"   {model_name}: Test Accuracy = {test_results[model_name]['accuracy']:.3f}")
                
            except Exception as e:
                logger.warning(f"   {model_name} test evaluation failed: {e}")
                continue
        
        # Select best model
        best_model_name = max(test_results.keys(), 
                            key=lambda k: test_results[k]['accuracy'] if test_results[k]['accuracy'] else 0)
        best_model = all_models[best_model_name]
        
        # Save best model
        self.trainer.model = best_model
        self.trainer.save_model("enhanced_ensemble_best_model.pkl")
        
        # Compile comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'hyperparameter_tuning_results': {k: {
                'best_params': v['best_params'],
                'best_score': v['best_score']
            } for k, v in tuned_models.items()},
            'cross_validation_results': cv_results,
            'test_results': test_results,
            'best_model': best_model_name,
            'best_test_accuracy': test_results[best_model_name]['accuracy'],
            'ensemble_models_created': list(ensemble_models.keys()),
            'feature_engineering_summary': {
                'original_features': X.shape[1],
                'enhanced_features': X_train_enhanced.shape[1],
                'selected_features': X_train_enhanced.shape[1] if self.feature_engineer.feature_selector else None
            }
        }
        
        logger.info(f"✅ Enhanced ensemble training completed!")
        logger.info(f"   Best model: {best_model_name} (Test Accuracy: {results['best_test_accuracy']:.3f})")
        
        return results


def run_enhanced_training_pipeline(data_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the complete enhanced training pipeline"""
    
    # Initialize trainer
    from data_loaders import CSVDataLoader, MockDataLoader
    
    if data_path and Path(data_path).exists():
        data_loader = CSVDataLoader(data_path)
    else:
        data_loader = MockDataLoader()
        
    trainer = AlzheimerTrainer(data_loader=data_loader)
    
    # Run enhanced training
    enhanced_trainer = EnhancedEnsembleTrainer(trainer)
    results = enhanced_trainer.run_enhanced_ensemble_training()
    
    # Save results
    results_path = Path("enhanced_training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run enhanced training
    results = run_enhanced_training_pipeline()
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED ENSEMBLE TRAINING SUMMARY")
    print("="*60)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Test Accuracy: {results['best_test_accuracy']:.3f}")
    print(f"Models Trained: {len(results['hyperparameter_tuning_results'])}")
    print(f"Ensemble Models Created: {len(results['ensemble_models_created'])}")
    print(f"Enhanced Features: {results['feature_engineering_summary']['enhanced_features']}")
    print("="*60)