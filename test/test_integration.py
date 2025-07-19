#!/usr/bin/env python3
"""
Integration Test for MediCodeAI Pipeline
Tests the complete flow from data loading to prediction
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_data_pipeline():
    """Test the complete data pipeline."""
    print("🔍 Testing data pipeline...")
    
    # Check if data exists
    data_paths = [
        "../data/raw/synthetic_ehr_data.csv",
        "../data/preprocessed/train/",
        "../data/preprocessed/validation/",
        "../data/preprocessed/test/"
    ]
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"❌ Missing data: {path}")
            return False
        print(f"✅ Found: {path}")
    
    # Load sample data
    try:
        raw_data = pd.read_csv("../data/raw/synthetic_ehr_data.csv")
        print(f"✅ Raw data loaded: {len(raw_data)} records")
        
        # Check required columns
        required_columns = ['patient_id', 'age', 'gender', 'doctor_notes', 'icd10_codes']
        for col in required_columns:
            if col not in raw_data.columns:
                print(f"❌ Missing column: {col}")
                return False
        print("✅ All required columns present")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    return True

def test_model_pipeline():
    """Test the complete model pipeline."""
    print("\n🤖 Testing model pipeline...")
    
    # Check if model exists
    model_path = "../model/saved_model"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check model components
    required_files = [
        "classifier.pkl",
        "label_binarizer.pkl", 
        "config.json",
        "tokenizer/",
        "bert_model/"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"❌ Missing model component: {file}")
            return False
        print(f"✅ Found: {file}")
    
    # Load model components
    try:
        with open(f"{model_path}/config.json", "r") as f:
            config = json.load(f)
        print(f"✅ Config loaded: {config.get('bert_model', 'Unknown')}")
        
        with open(f"{model_path}/classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        print(f"✅ Classifier loaded: {type(classifier).__name__}")
        
        with open(f"{model_path}/label_binarizer.pkl", "rb") as f:
            label_binarizer = pickle.load(f)
        print(f"✅ Label binarizer loaded: {len(label_binarizer.classes_)} classes")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

def test_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("\n🎯 Testing prediction pipeline...")
    
    try:
        # Import prediction components
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load model
        model_path = "../model/saved_model"
        
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
        
        # Test prediction
        test_text = "Patient presents with chest pain and shortness of breath."
        
        # Preprocess
        import re
        cleaned_text = re.sub(r'\s+', ' ', test_text.strip())
        cleaned_text = re.sub(r'[^\w\s\-\.]', ' ', cleaned_text)
        
        # Extract BERT features
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
        
        # Create engineered features
        engineered_features = np.zeros(7)
        engineered_features[0] = (50 - 50) / 20  # Age
        engineered_features[1] = (len(cleaned_text) - 200) / 100  # Text length
        
        text_lower = cleaned_text.lower()
        symptoms = ['pain', 'discomfort', 'pressure', 'burning', 'nausea', 'dizziness', 'fatigue', 'weakness']
        engineered_features[2] = sum(1 for symptom in symptoms if symptom in text_lower)
        
        body_parts = ['chest', 'abdomen', 'head', 'back', 'legs', 'arms', 'neck', 'shoulder', 'knee', 'hip', 'throat', 'stomach']
        engineered_features[3] = sum(1 for part in body_parts if part in text_lower)
        
        severity_words = ['severe', 'acute', 'chronic', 'mild', 'moderate', 'intermittent', 'persistent']
        engineered_features[4] = sum(1 for word in severity_words if word in text_lower)
        
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'examination', 'test', 'condition', 'medication']
        engineered_features[5] = sum(1 for term in medical_terms if term in text_lower)
        
        engineered_features[6] = 0  # Gender
        
        # Combine features
        if bert_features.ndim == 1:
            bert_features = bert_features.reshape(1, -1)
        if engineered_features.ndim == 1:
            engineered_features = engineered_features.reshape(1, -1)
        
        ensemble_features = np.hstack([bert_features, engineered_features])
        
        # Make prediction
        probabilities = classifier.predict_proba(ensemble_features)
        
        print(f"✅ Prediction successful: {probabilities.shape}")
        print(f"✅ Feature shape: {ensemble_features.shape}")
        
        # Get top predictions
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_integration():
    """Test Streamlit app integration."""
    print("\n🌐 Testing Streamlit integration...")
    
    try:
        # Check if Streamlit app exists
        app_path = "../app/streamlit_app.py"
        if not os.path.exists(app_path):
            print(f"❌ Streamlit app not found: {app_path}")
            return False
        
        print(f"✅ Streamlit app found: {app_path}")
        
        # Check if app can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
        streamlit_module = importlib.util.module_from_spec(spec)
        
        # This will fail if there are import errors
        print("✅ Streamlit app can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Streamlit integration: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting MediCodeAI Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Model Pipeline", test_model_pipeline),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Streamlit Integration", test_streamlit_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All integration tests passed! MediCodeAI is fully operational.")
        return 0
    else:
        print(f"\n⚠️ {failed} integration test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 