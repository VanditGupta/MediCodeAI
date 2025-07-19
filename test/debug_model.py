#!/usr/bin/env python3
"""
Debug script to understand the trained model structure
"""

import pickle
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def debug_model():
    """Debug the trained model to understand its structure."""
    
    print("🔍 Debugging trained model structure...")
    
    # Load model components
    model_path = "../model/saved_model"
    
    # Load classifier
    with open(f"{model_path}/classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    # Load label binarizer
    with open(f"{model_path}/label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)
    
    # Load config
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    print(f"📊 Model Configuration: {config}")
    print(f"📋 Classifier Type: {type(classifier)}")
    print(f"📋 Classifier: {classifier}")
    print(f"📋 Label Binarizer Classes: {len(label_binarizer.classes_)}")
    print(f"📋 Sample Classes: {label_binarizer.classes_[:5]}")
    
    # Test with sample features
    print("\n🧪 Testing with sample features...")
    
    # Create sample features (775 dimensions as expected)
    sample_features = np.random.random(775).reshape(1, -1)
    print(f"📊 Sample features shape: {sample_features.shape}")
    
    # Test different prediction methods
    print("\n🔍 Testing prediction methods...")
    
    try:
        # Method 1: predict()
        print("Testing predict()...")
        pred = classifier.predict(sample_features)
        print(f"predict() result shape: {pred.shape}")
        print(f"predict() result: {pred}")
        
        # Method 2: predict_proba()
        print("\nTesting predict_proba()...")
        proba = classifier.predict_proba(sample_features)
        print(f"predict_proba() result type: {type(proba)}")
        print(f"predict_proba() result length: {len(proba)}")
        
        if isinstance(proba, list):
            for i, p in enumerate(proba[:3]):  # Show first 3
                print(f"  Label {i} proba shape: {p.shape}")
                print(f"  Label {i} proba: {p}")
        else:
            print(f"predict_proba() result shape: {proba.shape}")
            print(f"predict_proba() result: {proba}")
        
        # Method 3: decision_function()
        print("\nTesting decision_function()...")
        if hasattr(classifier, 'decision_function'):
            decision = classifier.decision_function(sample_features)
            print(f"decision_function() result shape: {decision.shape}")
            print(f"decision_function() result: {decision}")
        else:
            print("No decision_function() available")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model() 