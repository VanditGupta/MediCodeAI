#!/usr/bin/env python3
"""
Local Model Training Script for ICD-10 Billing Code Prediction
Multi-label Classification using ClinicalBERT + XGBoost Ensemble

This is a local version that doesn't require MLflow server.

PRODUCTION ENHANCEMENTS (COMMENTED):
- Grid Search for hyperparameter optimization
- Cross-validation for robust evaluation
- Bayesian optimization for efficient hyperparameter search
- Stratified sampling for multi-label data
- Early stopping during hyperparameter search
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Progress tracking
from tqdm import tqdm
import threading

# ML and NLP libraries
import torch
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# PRODUCTION: Additional imports for hyperparameter optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate
from scipy.stats import uniform, randint
# from optuna import create_study, Trial
# import optuna

# MLflow tracking
import mlflow
import mlflow.sklearn

class ICD10Predictor:
    """Main class for ICD-10 code prediction model training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model trainer."""
        self.config = config
        # Use M1 GPU (MPS) if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🚀 Using M1 GPU (MPS) for training")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 Using CUDA GPU for training")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU for training")
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5001")
        
        # Model components
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.label_binarizer = None
        
        # Data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        print(f"🚀 Initializing ICD-10 Predictor on {self.device}")
        print(f"📊 Configuration: {json.dumps(config, indent=2)}")
        print(f"📝 MLflow tracking enabled at: http://localhost:5001")
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed data from local path."""
        print(f"📥 Loading data from {data_path}")
        start_time = time.time()
        
        try:
            # Load train/validation/test sets with progress
            print("   Loading training data...")
            train_df = pd.read_parquet(f"{data_path}/train/")
            print(f"   ✅ Training data: {len(train_df)} records")
            
            print("   Loading validation data...")
            val_df = pd.read_parquet(f"{data_path}/validation/")
            print(f"   ✅ Validation data: {len(val_df)} records")
            
            print("   Loading test data...")
            test_df = pd.read_parquet(f"{data_path}/test/")
            print(f"   ✅ Test data: {len(test_df)} records")
            
            load_time = time.time() - start_time
            print(f"✅ Data loading completed in {load_time:.2f} seconds")
            print(f"📊 Total records: {len(train_df) + len(val_df) + len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def prepare_labels(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Prepare multi-label targets for training."""
        print("🏷️ Preparing multi-label targets...")
        start_time = time.time()
        
        # Extract ICD-10 codes from all datasets
        print("   Collecting all ICD-10 codes...")
        all_codes = []
        for df in [train_df, val_df, test_df]:
            codes = df['processed_codes'].tolist()
            all_codes.extend([code for code_list in codes for code in code_list])
        
        # Create label binarizer
        unique_codes = sorted(list(set(all_codes)))
        self.label_binarizer = MultiLabelBinarizer(classes=unique_codes)
        
        # Transform labels with progress
        print("   Transforming training labels...")
        train_labels = self.label_binarizer.fit_transform(train_df['processed_codes'].tolist())
        
        print("   Transforming validation labels...")
        val_labels = self.label_binarizer.transform(val_df['processed_codes'].tolist())
        
        print("   Transforming test labels...")
        test_labels = self.label_binarizer.transform(test_df['processed_codes'].tolist())
        
        prep_time = time.time() - start_time
        print(f"✅ Label preparation completed in {prep_time:.2f} seconds")
        print(f"📋 Created {len(unique_codes)} unique ICD-10 code labels")
        print(f"📊 Label matrix shape: {train_labels.shape}")
        
        return train_labels, val_labels, test_labels
    
    def initialize_bert_model(self):
        """Initialize ClinicalBERT model and tokenizer."""
        print("🤖 Initializing ClinicalBERT model...")
        start_time = time.time()
        
        # Use ClinicalBERT or fallback to BioBERT
        model_name = self.config.get('bert_model', 'emilyalsentzer/Bio_ClinicalBERT')
        cache_dir = "model_cache"  # Use our cached models
        
        try:
            print(f"   Loading tokenizer from cache: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # Allow downloading if not cached
            )
            print("   ✅ Tokenizer loaded successfully")
            
            print(f"   Loading BERT model from cache: {model_name}")
            self.bert_model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False  # Allow downloading if not cached
            )
            self.bert_model.to(self.device)
            print("   ✅ BERT model loaded successfully")
            
            init_time = time.time() - start_time
            print(f"✅ Model initialization completed in {init_time:.2f} seconds")
            print(f"✅ Loaded {model_name} from cache")
            
        except Exception as e:
            print(f"⚠️ Could not load {model_name} from cache, falling back to BERT-base")
            try:
                print("   Loading BERT-base tokenizer from cache...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'bert-base-uncased', 
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                print("   ✅ BERT-base tokenizer loaded")
                
                print("   Loading BERT-base model from cache...")
                self.bert_model = AutoModel.from_pretrained(
                    'bert-base-uncased', 
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                self.bert_model.to(self.device)
                print("   ✅ BERT-base model loaded")
                
                init_time = time.time() - start_time
                print(f"✅ Model initialization completed in {init_time:.2f} seconds")
                print("✅ Loaded BERT-base from cache")
                
            except Exception as e2:
                print(f"❌ Could not load any cached model: {e2}")
                raise
    
    def extract_bert_features(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract BERT embeddings from clinical notes."""
        print(f"🔍 Extracting BERT features from {len(texts)} texts...")
        start_time = time.time()
        
        self.bert_model.eval()
        features = []
        
        # Calculate total batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            # Use tqdm for progress bar
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc="BERT Feature Extraction", 
                         total=total_batches,
                         unit="batch"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('max_length', 512),
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get BERT embeddings
                outputs = self.bert_model(**inputs)
                
                # Use [CLS] token embeddings (first token)
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(batch_features)
        
        # Concatenate all features
        all_features = np.vstack(features)
        
        extract_time = time.time() - start_time
        print(f"✅ BERT feature extraction completed in {extract_time:.2f} seconds")
        print(f"📊 Feature shape: {all_features.shape}")
        
        return all_features
    
    def create_ensemble_features(self, df: pd.DataFrame, bert_features: np.ndarray) -> np.ndarray:
        """Create ensemble features combining BERT embeddings with engineered features."""
        print("🔧 Creating ensemble features...")
        start_time = time.time()
        
        # Extract engineered features
        engineered_features = []
        
        # Progress bar for feature engineering
        for idx in tqdm(range(len(df)), desc="Feature Engineering", unit="record"):
            row = df.iloc[idx]
            
            # Basic features
            features = [
                row['age'],
                row['note_length'],
                row['symptoms_count'],
                row['body_parts_count'],
                row['severity_count'],
                row['medical_terms_count']
            ]
            
            # Gender encoding
            features.append(1 if row['gender'] == 'M' else 0)
            
            engineered_features.append(features)
        
        engineered_features = np.array(engineered_features)
        
        # Combine BERT and engineered features
        ensemble_features = np.hstack([bert_features, engineered_features])
        
        ensemble_time = time.time() - start_time
        print(f"✅ Ensemble feature creation completed in {ensemble_time:.2f} seconds")
        print(f"📊 Final feature shape: {ensemble_features.shape}")
        
        return ensemble_features

    # PRODUCTION: Grid Search Implementation (Commented)
    """
    def perform_grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        \"\"\"
        Perform grid search for hyperparameter optimization.
        This would significantly improve model performance but takes much longer.
        \"\"\"
        print("🔍 Performing Grid Search for Hyperparameter Optimization...")
        
        # Define parameter grids for different classifiers
        if self.config.get('classifier') == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5]
            }
            
            # Create base classifier
            base_classifier = xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
        elif self.config.get('classifier') == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_classifier = RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_classifier,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_micro',  # Multi-label scoring
            n_jobs=-1,
            verbose=2,
            refit=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"✅ Grid Search completed!")
        print(f"📊 Best parameters: {best_params}")
        print(f"📈 Best CV score: {best_score:.4f}")
        
        # Update config with best parameters
        self.config.update(best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    """

    # PRODUCTION: Cross-Validation Implementation (Commented)
    """
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                n_splits: int = 5) -> Dict[str, float]:
        \"\"\"
        Perform k-fold cross-validation for robust model evaluation.
        \"\"\"
        print(f"🔄 Performing {n_splits}-fold Cross-Validation...")
        
        # Create classifier
        classifier_type = self.config.get('classifier', 'xgboost')
        
        if classifier_type == 'xgboost':
            classifier = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 6),
                random_state=42,
                n_jobs=-1
            )
        
        # Define scoring metrics for multi-label classification
        scoring = {
            'accuracy': 'accuracy',
            'f1_micro': 'f1_micro',
            'precision_micro': 'precision_micro',
            'recall_micro': 'recall_micro',
            'hamming_loss': 'hamming_loss'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            classifier,
            X, y,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Calculate mean and std of CV results
        cv_metrics = {}
        for metric in scoring.keys():
            scores = cv_results[f'test_{metric}']
            cv_metrics[f'cv_{metric}_mean'] = np.mean(scores)
            cv_metrics[f'cv_{metric}_std'] = np.std(scores)
        
        print("✅ Cross-Validation Results:")
        for metric, value in cv_metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return cv_metrics
    """

    # PRODUCTION: Bayesian Optimization Implementation (Commented)
    """
    def perform_bayesian_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray,
                                    n_trials: int = 50) -> Dict[str, Any]:
        \"\"\"
        Perform Bayesian optimization for efficient hyperparameter search.
        More efficient than grid search for high-dimensional spaces.
        \"\"\"
        print(f"🎯 Performing Bayesian Optimization ({n_trials} trials)...")
        
        def objective(trial: Trial) -> float:
            # Define hyperparameter space
            if self.config.get('classifier') == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
                }
                
                classifier = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            
            # Train and evaluate
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_val)
            
            # Calculate F1 score (objective to maximize)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, y_pred, average='micro')
            
            return f1
        
        # Create study
        study = create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"✅ Bayesian Optimization completed!")
        print(f"📊 Best parameters: {best_params}")
        print(f"📈 Best F1 score: {best_score:.4f}")
        
        # Update config with best parameters
        self.config.update(best_params)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    """

    # PRODUCTION: Stratified Sampling for Multi-label Data (Commented)
    """
    def create_stratified_splits(self, X: np.ndarray, y: np.ndarray, 
                                n_splits: int = 5) -> List[Tuple]:
        \"\"\"
        Create stratified splits for multi-label classification.
        Ensures each fold has similar label distribution.
        \"\"\"
        from sklearn.model_selection import StratifiedKFold
        
        # For multi-label data, we need to create a stratification target
        # One approach is to use the sum of labels or most frequent label
        if y.ndim > 1:
            # Use sum of labels as stratification target
            stratify_target = np.sum(y, axis=1)
        else:
            stratify_target = y
        
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        
        for train_idx, val_idx in skf.split(X, stratify_target):
            splits.append((train_idx, val_idx))
        
        return splits
    """

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train the classifier with progress tracking."""
        print("🎯 Training classifier...")
        start_time = time.time()
        
        # PRODUCTION: Uncomment to enable hyperparameter optimization
        """
        # Perform grid search for hyperparameter optimization
        grid_search_results = self.perform_grid_search(X_train, y_train, X_val, y_val)
        
        # Or use Bayesian optimization for more efficient search
        # bayesian_results = self.perform_bayesian_optimization(X_train, y_train, X_val, y_val)
        
        # Log optimization results
        self.mlflow_tracker.log_params(grid_search_results['best_params'])
        self.mlflow_tracker.log_metrics({'best_cv_score': grid_search_results['best_score']})
        """
        
        classifier_type = self.config.get('classifier', 'xgboost')
        print(f"   Using classifier: {classifier_type}")
        
        if classifier_type == 'xgboost':
            # XGBoost for multi-label classification
            print("   Initializing XGBoost classifier...")
            self.classifier = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            print("   Training XGBoost with early stopping...")
            # Train with early stopping and progress tracking
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=True  # Enable XGBoost's built-in progress
            )
            
        elif classifier_type == 'random_forest':
            print("   Initializing Random Forest classifier...")
            # Random Forest for multi-label classification
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                verbose=1  # Enable progress output
            )
            print("   Training Random Forest...")
            self.classifier.fit(X_train, y_train)
            
        elif classifier_type == 'logistic_regression':
            print("   Initializing Logistic Regression classifier...")
            # Logistic Regression for multi-label classification
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=1  # Enable progress output
            )
            print("   Training Logistic Regression...")
            self.classifier.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"✅ Classifier training completed in {train_time:.2f} seconds")
        print(f"✅ Trained {classifier_type} classifier")
        return self.classifier
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model."""
        print("📊 Evaluating model performance...")
        start_time = time.time()
        
        # PRODUCTION: Uncomment to enable cross-validation
        """
        # Perform cross-validation for robust evaluation
        cv_metrics = self.perform_cross_validation(X_test, y_test, n_splits=5)
        
        # Log cross-validation results
        self.mlflow_tracker.log_metrics(cv_metrics)
        """
        
        # Predictions with progress
        print("   Making predictions...")
        y_pred_proba = self.classifier.predict_proba(X_test)
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics for each label
        print("   Calculating metrics...")
        metrics = {}
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_test)
        metrics['overall_accuracy'] = accuracy
        
        # Per-label metrics (with progress for large label sets)
        print("   Computing per-label metrics...")
        for i, label in enumerate(tqdm(self.label_binarizer.classes_, 
                                     desc="Per-label Metrics", 
                                     unit="label")):
            if np.sum(y_test[:, i]) > 0:  # Only evaluate labels that appear in test set
                label_accuracy = np.mean(y_pred[:, i] == y_test[:, i])
                metrics[f'accuracy_{label}'] = label_accuracy
        
        # Hamming loss (lower is better)
        hamming_loss = np.mean(y_pred != y_test)
        metrics['hamming_loss'] = hamming_loss
        
        # F1 score (micro-averaged)
        from sklearn.metrics import f1_score
        f1_micro = f1_score(y_test, y_pred, average='micro')
        metrics['f1_micro'] = f1_micro
        
        # Precision and recall (micro-averaged)
        from sklearn.metrics import precision_score, recall_score
        precision_micro = precision_score(y_test, y_pred, average='micro')
        recall_micro = recall_score(y_test, y_pred, average='micro')
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        
        eval_time = time.time() - start_time
        print(f"✅ Model evaluation completed in {eval_time:.2f} seconds")
        print(f"📈 Model Performance:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Hamming Loss: {hamming_loss:.4f}")
        print(f"   F1 Score (Micro): {f1_micro:.4f}")
        print(f"   Precision (Micro): {precision_micro:.4f}")
        print(f"   Recall (Micro): {recall_micro:.4f}")
        
        return metrics
    
    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
        """Perform K-Fold cross-validation on the dataset."""
        print(f"🔄 Performing {n_splits}-fold cross-validation...")
        start_time = time.time()
        
        # Initialize cross-validation (use KFold for multi-label data)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Perform cross-validation for each metric
        cv_scores = {}
        
        # Accuracy
        accuracy_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='accuracy')
        cv_scores['cv_accuracy_mean'] = accuracy_scores.mean()
        cv_scores['cv_accuracy_std'] = accuracy_scores.std()
        
        # F1 Score (micro-averaged)
        f1_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='f1_micro')
        cv_scores['cv_f1_mean'] = f1_scores.mean()
        cv_scores['cv_f1_std'] = f1_scores.std()
        
        # Precision (micro-averaged)
        precision_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='precision_micro')
        cv_scores['cv_precision_mean'] = precision_scores.mean()
        cv_scores['cv_precision_std'] = precision_scores.std()
        
        # Recall (micro-averaged)
        recall_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='recall_micro')
        cv_scores['cv_recall_mean'] = recall_scores.mean()
        cv_scores['cv_recall_std'] = recall_scores.std()
        
        cv_time = time.time() - start_time
        print(f"✅ Cross-validation completed in {cv_time:.2f} seconds")
        print(f"📊 Cross-validation Results:")
        print(f"   Accuracy: {cv_scores['cv_accuracy_mean']:.4f} ± {cv_scores['cv_accuracy_std']:.4f}")
        print(f"   F1 Score: {cv_scores['cv_f1_mean']:.4f} ± {cv_scores['cv_f1_std']:.4f}")
        print(f"   Precision: {cv_scores['cv_precision_mean']:.4f} ± {cv_scores['cv_precision_std']:.4f}")
        print(f"   Recall: {cv_scores['cv_recall_mean']:.4f} ± {cv_scores['cv_recall_std']:.4f}")
        
        return cv_scores
    
    def save_model(self, model_path: str):
        """Save the trained model and components."""
        print(f"💾 Saving model to {model_path}")
        start_time = time.time()
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save model components
        model_components = {
            'tokenizer': self.tokenizer,
            'bert_model': self.bert_model,
            'classifier': self.classifier,
            'label_binarizer': self.label_binarizer,
            'config': self.config
        }
        
        # Save each component with progress
        for name, component in tqdm(model_components.items(), 
                                  desc="Saving Model Components", 
                                  unit="component"):
            if name == 'tokenizer':
                component.save_pretrained(f"{model_path}/{name}")
            elif name == 'bert_model':
                component.save_pretrained(f"{model_path}/{name}")
            elif name == 'label_binarizer':
                with open(f"{model_path}/{name}.pkl", "wb") as f:
                    pickle.dump(component, f)
            elif name == 'classifier':
                with open(f"{model_path}/{name}.pkl", "wb") as f:
                    pickle.dump(component, f)
            elif name == 'config':
                with open(f"{model_path}/{name}.json", "w") as f:
                    json.dump(component, f, indent=2)
        
        save_time = time.time() - start_time
        print(f"✅ Model saving completed in {save_time:.2f} seconds")
        print("✅ Model saved successfully")
    
    def train(self, data_path: str, model_path: str):
        """Main training pipeline with comprehensive progress tracking."""
        print("🏥 Starting ICD-10 Prediction Model Training")
        print("=" * 60)
        total_start_time = time.time()
        
        try:
            # Start MLflow run
            mlflow.set_experiment("MediCodeAI_Training")
            with mlflow.start_run() as run:
                # Log parameters
                print("📊 Logging training parameters...")
                for key, value in self.config.items():
                    mlflow.log_param(key, value)
                
                # Load data
                train_df, val_df, test_df = self.load_data(data_path)
                
                # Prepare labels
                train_labels, val_labels, test_labels = self.prepare_labels(train_df, val_df, test_df)
                
                # Initialize BERT model
                self.initialize_bert_model()
                
                # Extract BERT features with progress
                print("🔍 Extracting BERT features for training set...")
                train_bert_features = self.extract_bert_features(train_df['cleaned_notes'].tolist())
                
                print("🔍 Extracting BERT features for validation set...")
                val_bert_features = self.extract_bert_features(val_df['cleaned_notes'].tolist())
                
                print("🔍 Extracting BERT features for test set...")
                test_bert_features = self.extract_bert_features(test_df['cleaned_notes'].tolist())
                
                # Create ensemble features
                print("🔧 Creating ensemble features for training set...")
                X_train = self.create_ensemble_features(train_df, train_bert_features)
                
                print("🔧 Creating ensemble features for validation set...")
                X_val = self.create_ensemble_features(val_df, val_bert_features)
                
                print("🔧 Creating ensemble features for test set...")
                X_test = self.create_ensemble_features(test_df, test_bert_features)
                
                # Train classifier first
                self.train_classifier(X_train, train_labels, X_val, val_labels)
                
                # Perform cross-validation on a sample for faster execution
                print("🔄 Performing cross-validation on sample dataset...")
                full_X = np.vstack([X_train, X_val])
                full_y = np.vstack([train_labels, val_labels])
                
                # Use smaller sample for faster CV
                sample_size = min(500, len(full_X))
                sample_indices = np.random.choice(len(full_X), sample_size, replace=False)
                sample_X = full_X[sample_indices]
                sample_y = full_y[sample_indices]
                
                print(f"📊 Using {sample_size} samples for CV (out of {len(full_X)} total)")
                cv_metrics = self.perform_cross_validation(sample_X, sample_y, n_splits=2)
                
                # Log cross-validation metrics
                print("📈 Logging cross-validation metrics...")
                for key, value in cv_metrics.items():
                    mlflow.log_metric(key, value)
                
                # Evaluate model
                metrics = self.evaluate_model(X_test, test_labels, test_df)
                
                # Log final metrics
                print("📈 Logging final metrics...")
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                # Save model
                self.save_model(model_path)
                
                # Log model artifacts
                mlflow.log_artifact(model_path, "model")
                
                # Note: Skipping mlflow.sklearn.log_model due to API compatibility
                print("📝 Model artifacts logged to MLflow")
                
                total_time = time.time() - total_start_time
                print("=" * 60)
                print(f"✅ Training completed successfully in {total_time:.2f} seconds!")
                print(f"🎉 Total training time: {total_time/60:.2f} minutes")
                print(f"📊 View results at: http://localhost:5001")
                
                return metrics
                
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training function."""
    
    # Enhanced configuration with MLflow tracking
    config = {
        'bert_model': 'emilyalsentzer/Bio_ClinicalBERT',
        'classifier': 'xgboost',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'cv_folds': 3,
        'dataset_size': '5,000 records',
        'model_type': 'ClinicalBERT + XGBoost Ensemble'
    }
    
    # Paths
    data_path = "data/preprocessed"  # Local path
    model_path = "model/saved_model"
    
    print("🚀 Starting MediCodeAI Training with MLflow Tracking")
    print("=" * 60)
    print(f"📊 Configuration: {config}")
    print(f"📝 MLflow UI: http://localhost:5001")
    print(f"📁 Data path: {data_path}")
    print(f"💾 Model path: {model_path}")
    print("=" * 60)
    
    # Initialize and train
    trainer = ICD10Predictor(config)
    metrics = trainer.train(data_path, model_path)
    
    print("=" * 60)
    print(f"🎉 Training completed successfully!")
    print(f"📊 Final metrics: {metrics}")
    print(f"📈 View detailed results at: http://localhost:5001")
    print("=" * 60)

if __name__ == "__main__":
    main() 