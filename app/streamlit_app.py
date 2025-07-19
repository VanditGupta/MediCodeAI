#!/usr/bin/env python3
"""
Local Streamlit Frontend for ICD-10 Prediction
Interactive web application for clinical note analysis

This version works directly with the trained model without requiring an API server.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import os
import sys
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

# Page configuration
st.set_page_config(
    page_title="ICD-10 Code Predictor (Local)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

class LocalICD10Predictor:
    """Local predictor that loads the trained model directly."""
    
    def __init__(self, model_path: str = "model/saved_model"):
        """Initialize the local predictor."""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.label_binarizer = None
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model components."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path}/tokenizer")
            
            # Load BERT model
            self.bert_model = AutoModel.from_pretrained(f"{self.model_path}/bert_model")
            self.bert_model.to(self.device)
            
            # Load classifier
            with open(f"{self.model_path}/classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)
            
            # Load label binarizer
            with open(f"{self.model_path}/label_binarizer.pkl", "rb") as f:
                self.label_binarizer = pickle.load(f)
            
            # Load config
            with open(f"{self.model_path}/config.json", "r") as f:
                self.config = json.load(f)
            
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess clinical text."""
        import re
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        return text
    
    def extract_bert_features(self, text: str) -> np.ndarray:
        """Extract BERT features from text."""
        self.bert_model.eval()
        
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get BERT outputs
            outputs = self.bert_model(**inputs)
            
            # Mean pooling
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
            return embeddings.cpu().numpy()
    
    def create_ensemble_features(self, text: str, age: Optional[int] = None, gender: Optional[str] = None) -> np.ndarray:
        """Create ensemble features."""
        # Extract BERT features
        bert_features = self.extract_bert_features(text)
        
        # Create engineered features (7 features to match training)
        engineered_features = np.zeros(7)
        
        # Age (normalized)
        if age is not None:
            engineered_features[0] = (age - 50) / 20
        else:
            engineered_features[0] = 0
        
        # Text length (normalized)
        engineered_features[1] = (len(text) - 200) / 100
        
        # Simple feature extraction
        text_lower = text.lower()
        
        # Symptoms count
        symptoms = ['pain', 'discomfort', 'pressure', 'burning', 'nausea', 'dizziness', 'fatigue', 'weakness']
        engineered_features[2] = sum(1 for symptom in symptoms if symptom in text_lower)
        
        # Body parts count
        body_parts = ['chest', 'abdomen', 'head', 'back', 'legs', 'arms', 'neck', 'shoulder', 'knee', 'hip', 'throat', 'stomach']
        engineered_features[3] = sum(1 for part in body_parts if part in text_lower)
        
        # Severity indicators
        severity_words = ['severe', 'acute', 'chronic', 'mild', 'moderate', 'intermittent', 'persistent']
        engineered_features[4] = sum(1 for word in severity_words if word in text_lower)
        
        # Medical terms count
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'examination', 'test', 'medication', 'condition']
        engineered_features[5] = sum(1 for term in medical_terms if term in text_lower)
        
        # Gender encoding (1 if 'M', 0 otherwise)
        engineered_features[6] = 1 if gender == 'M' else 0
        
        # Combine features - ensure both are 2D
        if bert_features.ndim == 1:
            bert_features = bert_features.reshape(1, -1)
        if engineered_features.ndim == 1:
            engineered_features = engineered_features.reshape(1, -1)
        
        ensemble_features = np.hstack([bert_features, engineered_features])
        
        return ensemble_features
    
    def predict(self, doctor_notes: str, patient_age: Optional[int] = None,
                patient_gender: Optional[str] = None, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Make ICD-10 code prediction."""
        start_time = time.time()
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(doctor_notes)
            
            # Create features
            features = self.create_ensemble_features(cleaned_text, patient_age, patient_gender)
            
            # Ensure features is 2D for prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction for multi-label classification
            predictions = []
            
            try:
                # Get probabilities for all labels
                probabilities = self.classifier.predict_proba(features)
                # probabilities shape is (1, 99) - each column is probability for one label
                
                # Always get top 3 predictions regardless of threshold
                top_indices = np.argsort(probabilities[0])[-3:][::-1]  # Top 3, descending
                
                for idx in top_indices:
                    code = self.label_binarizer.classes_[idx]
                    prob = probabilities[0][idx]
                    
                    # Only add if above threshold, but always show top 3
                    if prob >= confidence_threshold or len(predictions) < 3:
                        predictions.append({
                            'code': code,
                            'confidence': float(prob),
                            'description': f"ICD-10 Code: {code}",
                            'above_threshold': prob >= confidence_threshold
                        })
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return {
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = time.time() - start_time
            
            return {
                'predictions': predictions,
                'processing_time': processing_time,
                'total_codes_checked': len(self.label_binarizer.classes_),
                'codes_above_threshold': len(predictions),
                'model_info': {
                    'model_type': 'ClinicalBERT + XGBoost',
                    'feature_dimensions': features.shape[0],
                    'total_labels': len(self.label_binarizer.classes_)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }

# Initialize predictor
@st.cache_resource
def get_predictor():
    """Get cached predictor instance."""
    return LocalICD10Predictor()

# =============================================================================
# Sidebar Configuration
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model Configuration
    st.sidebar.subheader("🤖 Model Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for code predictions"
    )
    
    # Display Options
    st.sidebar.subheader("📊 Display Options")
    show_confidence_bars = st.sidebar.checkbox(
        "Show Confidence Bars",
        value=True,
        help="Display confidence scores as progress bars"
    )
    
    show_processing_time = st.sidebar.checkbox(
        "Show Processing Time",
        value=True,
        help="Display processing time"
    )
    
    # About Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **ICD-10 Code Predictor (Local)**
    
    This application uses a locally trained ClinicalBERT + XGBoost ensemble model to predict ICD-10 billing codes from clinical notes.
    
    **Features:**
    - ClinicalBERT + XGBoost ensemble
    - Multi-label classification
    - Real-time predictions
    - Local processing (no API required)
    """)
    
    return {
        'confidence_threshold': confidence_threshold,
        'show_confidence_bars': show_confidence_bars,
        'show_processing_time': show_processing_time
    }

# =============================================================================
# Main Application
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🏥 ICD-10 Code Predictor (Local)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered billing code prediction from clinical notes</p>', unsafe_allow_html=True)

def render_input_section():
    """Render the input section for clinical notes."""
    st.subheader("📝 Clinical Notes Input")
    
    # Text area for clinical notes
    doctor_notes = st.text_area(
        "Enter clinical notes:",
        height=200,
        placeholder="Enter the patient's clinical notes here...\n\nExample: Patient presents with chest pain and shortness of breath for the past 2 days. Examination reveals normal heart sounds and clear lung fields. EKG shows normal sinus rhythm.",
        help="Enter detailed clinical notes describing the patient's symptoms, examination findings, and diagnosis."
    )
    
    # Patient demographics
    col1, col2 = st.columns(2)
    
    with col1:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=50,
            help="Patient age (optional, helps improve prediction accuracy)"
        )
    
    with col2:
        patient_gender = st.selectbox(
            "Patient Gender",
            options=["", "Male", "Female"],
            help="Patient gender (optional)"
        )
    
    return {
        'doctor_notes': doctor_notes,
        'patient_age': patient_age if patient_age > 0 else None,
        'patient_gender': patient_gender if patient_gender else None
    }

def render_prediction_results(results: Dict[str, Any], config: Dict[str, Any]):
    """Render prediction results."""
    st.subheader("🎯 Prediction Results")
    
    if 'error' in results:
        st.error(f"❌ Prediction Error: {results['error']}")
        return
    
    predictions = results.get('predictions', [])
    
    if not predictions:
        st.warning("⚠️ No ICD-10 codes found above the confidence threshold. Try lowering the threshold or providing more detailed clinical notes.")
        return
    
    # Display processing time
    if config['show_processing_time']:
        processing_time = results.get('processing_time', 0)
        st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
    
    # Display model info
    model_info = results.get('model_info', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_info.get('model_type', 'Unknown'))
    
    with col2:
        st.metric("Feature Dimensions", model_info.get('feature_dimensions', 0))
    
    with col3:
        st.metric("Total Labels", model_info.get('total_labels', 0))
    
    # Display predictions
    st.markdown("### 📋 Top 3 Predicted ICD-10 Codes")
    
    for i, pred in enumerate(predictions[:3]):  # Show top 3
        confidence = pred['confidence']
        code = pred['code']
        above_threshold = pred.get('above_threshold', True)
        
        # Color coding for threshold status
        status_color = "#28a745" if above_threshold else "#ffc107"
        status_text = "✅ Above Threshold" if above_threshold else "⚠️ Below Threshold"
        
        # Create confidence bar
        if config['show_confidence_bars']:
            st.markdown(f"""
            <div class="prediction-card">
                <h4>#{i+1} - ICD-10: {code}</h4>
                <p>Confidence: {confidence:.1%} <span style="color: {status_color};">({status_text})</span></p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence:.1%}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h4>#{i+1} - ICD-10: {code}</h4>
                <p>Confidence: {confidence:.1%} <span style="color: {status_color};">({status_text})</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show summary
    st.markdown("### 📊 Summary")
    above_threshold_count = sum(1 for pred in predictions[:3] if pred.get('above_threshold', True))
    st.info(f"Showing top 3 predictions: {above_threshold_count} above {config['confidence_threshold']:.1%} threshold, {3-above_threshold_count} below threshold")

def main():
    """Main application function."""
    
    # Get configuration
    config = render_sidebar()
    
    # Render header
    render_header()
    
    # Get predictor
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("💡 Make sure you have run the training script first and the model files are available in `model/saved_model/`")
        return
    
    # Render input section
    input_data = render_input_section()
    
    # Prediction button
    if st.button("🔍 Predict ICD-10 Codes", type="primary"):
        if not input_data['doctor_notes'].strip():
            st.warning("⚠️ Please enter clinical notes to make a prediction.")
        else:
            with st.spinner("🤖 Analyzing clinical notes..."):
                results = predictor.predict(
                    doctor_notes=input_data['doctor_notes'],
                    patient_age=input_data['patient_age'],
                    patient_gender=input_data['patient_gender'],
                    confidence_threshold=config['confidence_threshold']
                )
            
            # Render results
            render_prediction_results(results, config)
    
    # Sample data button
    if st.button("📋 Load Sample Data"):
        sample_text = """Patient presents with chest pain and shortness of breath for the past 2 days. 
        Examination reveals normal heart sounds and clear lung fields. 
        EKG shows normal sinus rhythm. Patient has a history of hypertension and diabetes.
        Blood pressure is elevated at 160/95 mmHg."""
        
        st.session_state.sample_text = sample_text
        st.rerun()
    
    # Load sample text if available
    if 'sample_text' in st.session_state:
        st.text_area("Sample Clinical Notes:", value=st.session_state.sample_text, height=150)

if __name__ == "__main__":
    main() 