#!/usr/bin/env python3
"""
Model Training Script for ICD-10 Billing Code Prediction
Multi-label Classification using ClinicalBERT + XGBoost Ensemble

This script trains a production-ready model for predicting ICD-10 codes
from clinical notes using state-of-the-art NLP and ML techniques.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML and NLP libraries
import torch
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Custom imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mlflow_tracking import MLflowTracker

class ICD10Predictor:
    """Main class for ICD-10 code prediction model training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model trainer."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlflow_tracker = MLflowTracker()
        
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
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed data from S3/local path."""
        print(f"📥 Loading data from {data_path}")
        
        try:
            # Load train/validation/test sets
            train_df = pd.read_parquet(f"{data_path}/train/")
            val_df = pd.read_parquet(f"{data_path}/validation/")
            test_df = pd.read_parquet(f"{data_path}/test/")
            
            print(f"✅ Loaded {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test records")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def prepare_labels(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Prepare multi-label targets for training."""
        print("🏷️ Preparing multi-label targets...")
        
        # Extract ICD-10 codes from all datasets
        all_codes = []
        for df in [train_df, val_df, test_df]:
            codes = df['processed_codes'].tolist()
            all_codes.extend([code for code_list in codes for code in code_list])
        
        # Create label binarizer
        unique_codes = sorted(list(set(all_codes)))
        self.label_binarizer = MultiLabelBinarizer(classes=unique_codes)
        
        # Transform labels
        train_labels = self.label_binarizer.fit_transform(train_df['processed_codes'].tolist())
        val_labels = self.label_binarizer.transform(val_df['processed_codes'].tolist())
        test_labels = self.label_binarizer.transform(test_df['processed_codes'].tolist())
        
        print(f"📋 Created {len(unique_codes)} unique ICD-10 code labels")
        print(f"📊 Label matrix shape: {train_labels.shape}")
        
        return train_labels, val_labels, test_labels
    
    def initialize_bert_model(self):
        """Initialize ClinicalBERT model and tokenizer."""
        print("🤖 Initializing ClinicalBERT model...")
        
        # Use ClinicalBERT or fallback to BioBERT
        model_name = self.config.get('bert_model', 'emilyalsentzer/Bio_ClinicalBERT')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.to(self.device)
            print(f"✅ Loaded {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load {model_name}, falling back to BERT-base")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_model.to(self.device)
    
    def extract_bert_features(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract BERT embeddings from clinical notes."""
        print(f"🔍 Extracting BERT features from {len(texts)} texts...")
        
        self.bert_model.eval()
        features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get BERT outputs
                outputs = self.bert_model(**inputs)
                
                # Use [CLS] token embeddings (mean pooling for longer sequences)
                if outputs.last_hidden_state.size(1) > 1:
                    # Mean pooling
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                else:
                    embeddings = outputs.last_hidden_state.squeeze(1)
                
                features.append(embeddings.cpu().numpy())
        
        return np.vstack(features)
    
    def create_ensemble_features(self, df: pd.DataFrame, bert_features: np.ndarray) -> np.ndarray:
        """Create ensemble features combining BERT embeddings with engineered features."""
        print("🔧 Creating ensemble features...")
        
        # Extract engineered features
        engineered_features = df[[
            'age', 'symptoms_count', 'body_parts_count', 
            'severity_count', 'medical_terms_count', 'note_length'
        ]].values
        
        # Normalize age and note_length
        engineered_features[:, 0] = (engineered_features[:, 0] - 50) / 20  # Age normalization
        engineered_features[:, 5] = (engineered_features[:, 5] - 200) / 100  # Note length normalization
        
        # Combine features
        ensemble_features = np.hstack([bert_features, engineered_features])
        
        print(f"📊 Ensemble features shape: {ensemble_features.shape}")
        return ensemble_features
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train the multi-label classifier."""
        print("🎯 Training multi-label classifier...")
        
        classifier_type = self.config.get('classifier', 'xgboost')
        
        if classifier_type == 'xgboost':
            # XGBoost for multi-label classification
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            # Train with early stopping
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
        elif classifier_type == 'random_forest':
            # Random Forest for multi-label classification
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.classifier.fit(X_train, y_train)
            
        elif classifier_type == 'logistic_regression':
            # Logistic Regression for multi-label classification
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
            self.classifier.fit(X_train, y_train)
        
        print(f"✅ Trained {classifier_type} classifier")
        return self.classifier
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model."""
        print("📊 Evaluating model performance...")
        
        # Predictions
        y_pred_proba = self.classifier.predict_proba(X_test)
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics for each label
        metrics = {}
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_test)
        metrics['overall_accuracy'] = accuracy
        
        # Per-label metrics
        for i, label in enumerate(self.label_binarizer.classes_):
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
        
        print(f"📈 Model Performance:")
        print(f"   Overall Accuracy: {accuracy:.4f}")
        print(f"   Hamming Loss: {hamming_loss:.4f}")
        print(f"   F1 Score (Micro): {f1_micro:.4f}")
        print(f"   Precision (Micro): {precision_micro:.4f}")
        print(f"   Recall (Micro): {recall_micro:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str):
        """Save the trained model and components."""
        print(f"💾 Saving model to {model_path}")
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save model components
        model_components = {
            'tokenizer': self.tokenizer,
            'bert_model': self.bert_model,
            'classifier': self.classifier,
            'label_binarizer': self.label_binarizer,
            'config': self.config
        }
        
        # Save each component
        for name, component in model_components.items():
            if name == 'tokenizer':
                component.save_pretrained(f"{model_path}/{name}")
            elif name == 'bert_model':
                component.save_pretrained(f"{model_path}/{name}")
            elif name == 'label_binarizer':
                with open(f"{model_path}/{name}.pkl", 'wb') as f:
                    pickle.dump(component, f)
            elif name == 'classifier':
                with open(f"{model_path}/{name}.pkl", 'wb') as f:
                    pickle.dump(component, f)
            elif name == 'config':
                with open(f"{model_path}/{name}.json", 'w') as f:
                    json.dump(component, f, indent=2)
        
        print("✅ Model saved successfully")
    
    def train(self, data_path: str, model_path: str):
        """Main training pipeline."""
        print("🏥 Starting ICD-10 Prediction Model Training")
        print("=" * 60)
        
        try:
            # Start MLflow run
            with self.mlflow_tracker.start_run() as run:
                # Log parameters
                self.mlflow_tracker.log_params(self.config)
                
                # Load data
                train_df, val_df, test_df = self.load_data(data_path)
                
                # Prepare labels
                train_labels, val_labels, test_labels = self.prepare_labels(train_df, val_df, test_df)
                
                # Initialize BERT model
                self.initialize_bert_model()
                
                # Extract BERT features
                print("🔍 Extracting BERT features for training set...")
                train_bert_features = self.extract_bert_features(train_df['cleaned_notes'].tolist())
                
                print("🔍 Extracting BERT features for validation set...")
                val_bert_features = self.extract_bert_features(val_df['cleaned_notes'].tolist())
                
                print("🔍 Extracting BERT features for test set...")
                test_bert_features = self.extract_bert_features(test_df['cleaned_notes'].tolist())
                
                # Create ensemble features
                X_train = self.create_ensemble_features(train_df, train_bert_features)
                X_val = self.create_ensemble_features(val_df, val_bert_features)
                X_test = self.create_ensemble_features(test_df, test_bert_features)
                
                # Train classifier
                self.train_classifier(X_train, train_labels, X_val, val_labels)
                
                # Evaluate model
                metrics = self.evaluate_model(X_test, test_labels, test_df)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(metrics)
                
                # Save model
                self.save_model(model_path)
                
                # Log model artifacts
                self.mlflow_tracker.log_artifact(model_path, "model")
                
                print("✅ Training completed successfully!")
                return metrics
                
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training function."""
    
    # Configuration
    config = {
        'bert_model': 'emilyalsentzer/Bio_ClinicalBERT',
        'classifier': 'xgboost',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # Paths
    data_path = "data/preprocessed"  # Local path or S3 path
    model_path = "model/saved_model"
    
    # Initialize and train
    trainer = ICD10Predictor(config)
    metrics = trainer.train(data_path, model_path)
    
    print(f"🎉 Training completed with metrics: {metrics}")

if __name__ == "__main__":
    main() 