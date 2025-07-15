#!/usr/bin/env python3
"""
AWS Lambda Handler for ICD-10 Prediction API
Serverless inference service for real-time predictions

This Lambda function provides a serverless endpoint for ICD-10 code
prediction that can be integrated with API Gateway.
"""

import json
import os
import sys
import logging
import boto3
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64

# Add model directory to path
sys.path.append('/opt/python/lib/python3.9/site-packages')

import torch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

class LambdaICD10Predictor:
    """Lambda-based ICD-10 prediction model."""
    
    def __init__(self):
        """Initialize the Lambda predictor."""
        self.model_loaded = False
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.label_binarizer = None
        self.config = None
        
        # Model configuration
        self.model_bucket = os.environ.get('MODEL_BUCKET', 'your-hipaa-bucket')
        self.model_key = os.environ.get('MODEL_KEY', 'models/icd10-predictor/latest')
        self.temp_dir = '/tmp'
        
        # Load model on initialization
        self.load_model()
    
    def load_model_from_s3(self):
        """Load model artifacts from S3."""
        try:
            logger.info(f"📥 Loading model from s3://{self.model_bucket}/{self.model_key}")
            
            # Create temp directory
            os.makedirs(f"{self.temp_dir}/model", exist_ok=True)
            
            # Download model artifacts
            artifacts = [
                'config.json',
                'classifier.pkl',
                'label_binarizer.pkl'
            ]
            
            for artifact in artifacts:
                s3_path = f"{self.model_key}/{artifact}"
                local_path = f"{self.temp_dir}/model/{artifact}"
                
                s3_client.download_file(self.model_bucket, s3_path, local_path)
                logger.info(f"✅ Downloaded {artifact}")
            
            # Load configuration
            with open(f"{self.temp_dir}/model/config.json", 'r') as f:
                self.config = json.load(f)
            
            # Load classifier
            with open(f"{self.temp_dir}/model/classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load label binarizer
            with open(f"{self.temp_dir}/model/label_binarizer.pkl", 'rb') as f:
                self.label_binarizer = pickle.load(f)
            
            # Load BERT model and tokenizer
            model_name = self.config.get('bert_model', 'emilyalsentzer/Bio_ClinicalBERT')
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_model = AutoModel.from_pretrained(model_name)
                logger.info(f"✅ Loaded BERT model: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {model_name}, using BERT-base")
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            
            self.model_loaded = True
            logger.info("✅ Model loaded successfully from S3")
            
        except Exception as e:
            logger.error(f"❌ Error loading model from S3: {str(e)}")
            raise
    
    def load_model(self):
        """Load the model (from S3 or local cache)."""
        try:
            # Check if model is already loaded
            if not self.model_loaded:
                self.load_model_from_s3()
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess clinical notes text."""
        import re
        
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_medical_features(self, text: str) -> Dict[str, int]:
        """Extract medical-specific features from text."""
        import re
        
        features = {}
        
        # Medical terminology patterns
        medical_patterns = {
            'symptoms': r'\b(pain|discomfort|pressure|burning|nausea|dizziness|fatigue|weakness)\b',
            'body_parts': r'\b(chest|abdomen|head|back|legs|arms|neck|shoulder|knee|hip|throat|stomach)\b',
            'severity': r'\b(mild|moderate|severe|acute|chronic|intermittent|persistent)\b',
            'medical_terms': r'\b(examination|diagnosis|treatment|medication|symptoms|condition)\b'
        }
        
        for feature_name, pattern in medical_patterns.items():
            matches = re.findall(pattern, text.lower())
            features[f'{feature_name}_count'] = len(matches)
            features[f'{feature_name}_present'] = 1 if matches else 0
        
        return features
    
    def extract_bert_features(self, text: str) -> np.ndarray:
        """Extract BERT embeddings from text."""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
            # Use mean pooling
            if outputs.last_hidden_state.size(1) > 1:
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                embeddings = outputs.last_hidden_state.squeeze(1)
        
        return embeddings.numpy()
    
    def create_features(self, text: str, age: Optional[int] = None) -> np.ndarray:
        """Create ensemble features for prediction."""
        # Extract BERT features
        bert_features = self.extract_bert_features(text)
        
        # Extract medical features
        medical_features = self.extract_medical_features(text)
        
        # Create engineered features array
        engineered_features = np.array([
            age or 50,  # Default age if not provided
            medical_features.get('symptoms_count', 0),
            medical_features.get('body_parts_count', 0),
            medical_features.get('severity_count', 0),
            medical_features.get('medical_terms_count', 0),
            len(text)  # Note length
        ]).reshape(1, -1)
        
        # Normalize engineered features
        engineered_features[:, 0] = (engineered_features[:, 0] - 50) / 20  # Age normalization
        engineered_features[:, 5] = (engineered_features[:, 5] - 200) / 100  # Note length normalization
        
        # Combine features
        ensemble_features = np.hstack([bert_features, engineered_features])
        
        return ensemble_features
    
    def predict(self, text: str, age: Optional[int] = None, 
                confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Make ICD-10 code predictions."""
        start_time = datetime.now()
        
        try:
            # Ensure model is loaded
            if not self.model_loaded:
                self.load_model()
            
            # Create features
            features = self.create_features(text, age)
            
            # Make predictions
            if hasattr(self.classifier, 'predict_proba'):
                # Get probability predictions
                proba = self.classifier.predict_proba(features)
                
                # Handle multi-label case
                if len(proba) > 1:
                    # Multi-label case - each class has its own probability
                    predictions = []
                    confidences = []
                    
                    for i, class_proba in enumerate(proba):
                        if class_proba[1] >= confidence_threshold:  # Positive class probability
                            predictions.append(self.label_binarizer.classes_[i])
                            confidences.append(float(class_proba[1]))
                else:
                    # Single-label case
                    predictions = []
                    confidences = []
                    for i, prob in enumerate(proba[0]):
                        if prob >= confidence_threshold:
                            predictions.append(self.label_binarizer.classes_[i])
                            confidences.append(float(prob))
            else:
                # Fallback for models without predict_proba
                predictions_binary = self.classifier.predict(features)
                predictions = []
                confidences = []
                
                for i, pred in enumerate(predictions_binary[0]):
                    if pred == 1:
                        predictions.append(self.label_binarizer.classes_[i])
                        confidences.append(0.8)  # Default confidence
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'predicted_codes': predictions,
                'confidence_scores': confidences,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            raise

# Global model instance
predictor = None

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: API Gateway event
        context: Lambda context
    
    Returns:
        API Gateway response
    """
    global predictor
    
    try:
        # Initialize predictor if not already done
        if predictor is None:
            predictor = LambdaICD10Predictor()
        
        # Parse request
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        # Extract parameters
        doctor_notes = body.get('doctor_notes', '')
        patient_age = body.get('patient_age')
        patient_gender = body.get('patient_gender')
        confidence_threshold = body.get('confidence_threshold', 0.5)
        
        # Validate input
        if not doctor_notes or len(doctor_notes.strip()) < 10:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST,OPTIONS'
                },
                'body': json.dumps({
                    'error': 'Invalid input: doctor_notes must be at least 10 characters long'
                })
            }
        
        # Make prediction
        result = predictor.predict(
            text=doctor_notes,
            age=patient_age,
            confidence_threshold=confidence_threshold
        )
        
        # Prepare response
        response = {
            'predicted_codes': result['predicted_codes'],
            'confidence_scores': result['confidence_scores'],
            'processing_time': result['processing_time'],
            'model_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'request_id': context.aws_request_id
        }
        
        # Log prediction (for monitoring)
        logger.info(f"📊 Prediction completed: {len(result['predicted_codes'])} codes predicted in {result['processing_time']:.3f}s")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"❌ Lambda execution error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'request_id': context.aws_request_id
            })
        }

def health_check_handler(event, context):
    """Health check handler for Lambda."""
    global predictor
    
    try:
        # Check if model is loaded
        model_loaded = predictor is not None and predictor.model_loaded
        
        response = {
            'status': 'healthy' if model_loaded else 'unhealthy',
            'model_loaded': model_loaded,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'request_id': context.aws_request_id
        }
        
        return {
            'statusCode': 200 if model_loaded else 503,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e),
                'request_id': context.aws_request_id
            })
        }

def model_info_handler(event, context):
    """Model information handler."""
    global predictor
    
    try:
        if predictor is None or not predictor.model_loaded:
            return {
                'statusCode': 503,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET,OPTIONS'
                },
                'body': json.dumps({
                    'error': 'Model not loaded',
                    'request_id': context.aws_request_id
                })
            }
        
        response = {
            'model_type': 'clinical-bert-xgboost',
            'version': '1.0.0',
            'num_classes': len(predictor.label_binarizer.classes_) if predictor.label_binarizer else 0,
            'config': predictor.config,
            'request_id': context.aws_request_id
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"❌ Model info error: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'request_id': context.aws_request_id
            })
        } 