#!/usr/bin/env python3
"""
Test script that mimics the exact feature creation from Streamlit app
"""

import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os

def test_streamlit_features():
    """Test the exact feature creation process from Streamlit app."""
    
    print("🧪 Testing Streamlit feature creation process...")
    
    # Load model components
    model_path = "../model/saved_model"
    
    try:
        # Load tokenizer and BERT model
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
        bert_model = AutoModel.from_pretrained(f"{model_path}/bert_model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        
        # Load classifier and label binarizer
        with open(f"{model_path}/classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(f"{model_path}/label_binarizer.pkl", "rb") as f:
            label_binarizer = pickle.load(f)
        
        print("✅ All model components loaded")
        
        # Test text (same as sample in Streamlit)
        test_text = """Patient presents with chest pain and shortness of breath for the past 2 days. 
        Examination reveals normal heart sounds and clear lung fields. 
        EKG shows normal sinus rhythm. Patient has a history of hypertension and diabetes.
        Blood pressure is elevated at 160/95 mmHg."""
        
        print(f"📝 Test text length: {len(test_text)} characters")
        
        # Step 1: Preprocess text
        import re
        cleaned_text = re.sub(r'\s+', ' ', test_text.strip())
        cleaned_text = re.sub(r'[^\w\s\-\.]', ' ', cleaned_text)
        print(f"🧹 Cleaned text length: {len(cleaned_text)} characters")
        
        # Step 2: Extract BERT features
        print("🔍 Extracting BERT features...")
        bert_model.eval()
        
        with torch.no_grad():
            inputs = tokenizer(
                cleaned_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = bert_model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            bert_features = embeddings.cpu().numpy()
        
        print(f"✅ BERT features shape: {bert_features.shape}")
        
        # Step 3: Create engineered features
        print("🔧 Creating engineered features...")
        engineered_features = np.zeros(7)  # 7 features to match training
        
        # Age (normalized) - using default 50
        engineered_features[0] = (50 - 50) / 20  # = 0
        
        # Text length (normalized)
        engineered_features[1] = (len(cleaned_text) - 200) / 100
        
        # Simple feature extraction
        text_lower = cleaned_text.lower()
        
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
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'examination', 'test', 'condition', 'medication']
        engineered_features[5] = sum(1 for term in medical_terms if term in text_lower)
        
        # Gender (default to 0 for female)
        engineered_features[6] = 0
        
        print(f"✅ Engineered features shape: {engineered_features.shape}")
        print(f"📊 Engineered features: {engineered_features}")
        
        # Step 4: Combine features
        print("🔗 Combining features...")
        print(f"BERT features shape: {bert_features.shape}")
        print(f"Engineered features shape: {engineered_features.shape}")
        
        # Ensure both are 2D
        if bert_features.ndim == 1:
            bert_features = bert_features.reshape(1, -1)
        if engineered_features.ndim == 1:
            engineered_features = engineered_features.reshape(1, -1)
        
        ensemble_features = np.hstack([bert_features, engineered_features])
        print(f"✅ Ensemble features shape: {ensemble_features.shape}")
        
        # Step 5: Ensure 2D for prediction
        if ensemble_features.ndim == 1:
            ensemble_features = ensemble_features.reshape(1, -1)
        print(f"✅ Final features shape: {ensemble_features.shape}")
        
        # Step 6: Make prediction
        print("🎯 Making prediction...")
        probabilities = classifier.predict_proba(ensemble_features)
        print(f"✅ Probabilities shape: {probabilities.shape}")
        
        # Step 7: Get predictions above threshold
        confidence_threshold = 0.5
        predictions = []
        
        for i, code in enumerate(label_binarizer.classes_):
            prob = probabilities[0][i]
            if prob >= confidence_threshold:
                predictions.append({
                    'code': code,
                    'confidence': float(prob)
                })
        
        print(f"✅ Found {len(predictions)} predictions above threshold")
        
        # Show top predictions
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        for i, pred in enumerate(predictions[:5]):
            print(f"  {pred['code']}: {pred['confidence']:.4f}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streamlit_features() 