#!/usr/bin/env python3
"""
Docker Entrypoint for ICD-10 Prediction Model
Production-ready inference service

This script loads the trained model and starts a FastAPI service
for real-time ICD-10 code prediction from clinical notes.
"""

import os
import sys
import json
import pickle
import logging
import uvicorn
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add app directory to path
sys.path.append('/app')

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for ICD-10 prediction."""
    doctor_notes: str = Field(..., min_length=10, max_length=5000, description="Clinical notes text")
    patient_age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    patient_gender: Optional[str] = Field(None, regex="^(M|F)$", description="Patient gender (M/F)")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Prediction confidence threshold")

class PredictionResponse(BaseModel):
    """Response model for ICD-10 prediction."""
    predicted_codes: List[str] = Field(..., description="Predicted ICD-10 codes")
    confidence_scores: List[float] = Field(..., description="Confidence scores for each code")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Health check timestamp")

# =============================================================================
# Model Loading and Prediction
# =============================================================================

class ICD10PredictionModel:
    """ICD-10 prediction model wrapper."""
    
    def __init__(self, model_path: str = "/app/model/saved_model"):
        """Initialize the prediction model."""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.label_binarizer = None
        self.config = None
        
        # Load model
        self.load_model()
        
        logger.info(f"✅ ICD-10 prediction model loaded on {self.device}")
    
    def load_model(self):
        """Load the trained model components."""
        try:
            logger.info(f"📥 Loading model from {self.model_path}")
            
            # Load configuration
            with open(f"{self.model_path}/config.json", 'r') as f:
                self.config = json.load(f)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/tokenizer")
            
            # Load BERT model
            self.bert_model = AutoModel.from_pretrained(f"{self.model_path}/bert_model")
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # Load classifier
            with open(f"{self.model_path}/classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load label binarizer
            with open(f"{self.model_path}/label_binarizer.pkl", 'rb') as f:
                self.label_binarizer = pickle.load(f)
            
            logger.info("✅ All model components loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
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
        ).to(self.device)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
            # Use mean pooling
            if outputs.last_hidden_state.size(1) > 1:
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                embeddings = outputs.last_hidden_state.squeeze(1)
        
        return embeddings.cpu().numpy()
    
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
                            confidences.append(class_proba[1])
                else:
                    # Single-label case
                    predictions = []
                    confidences = []
                    for i, prob in enumerate(proba[0]):
                        if prob >= confidence_threshold:
                            predictions.append(self.label_binarizer.classes_[i])
                            confidences.append(prob)
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

# =============================================================================
# FastAPI Application
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="ICD-10 Prediction API",
    description="HIPAA-aware API for predicting ICD-10 billing codes from clinical notes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model
    try:
        model = ICD10PredictionModel()
        logger.info("🚀 ICD-10 prediction service started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {str(e)}")
        sys.exit(1)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_icd10(request: PredictionRequest):
    """Predict ICD-10 codes from clinical notes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        result = model.predict(
            text=request.doctor_notes,
            age=request.patient_age,
            confidence_threshold=request.confidence_threshold
        )
        
        return PredictionResponse(
            predicted_codes=result['predicted_codes'],
            confidence_scores=result['confidence_scores'],
            processing_time=result['processing_time'],
            model_version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "clinical-bert-xgboost",
        "version": "1.0.0",
        "num_classes": len(model.label_binarizer.classes_) if model.label_binarizer else 0,
        "device": str(model.device),
        "config": model.config
    }

@app.get("/available-codes")
async def get_available_codes():
    """Get list of available ICD-10 codes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "available_codes": model.label_binarizer.classes_.tolist() if model.label_binarizer else [],
        "total_codes": len(model.label_binarizer.classes_) if model.label_binarizer else 0
    }

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting ICD-10 prediction service on {host}:{port}")
    
    # Start the server
    uvicorn.run(
        "docker.entrypoint:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    ) 