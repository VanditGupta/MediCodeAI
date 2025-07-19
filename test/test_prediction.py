#!/usr/bin/env python3
"""
Simple test script to isolate the prediction issue
"""

import pickle
import numpy as np
import os

def test_prediction():
    """Test prediction with the saved model."""
    
    print("🧪 Testing prediction with saved model...")
    
    # Load model components
    model_path = "../model/saved_model"
    
    try:
        # Load classifier
        with open(f"{model_path}/classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        
        # Load label binarizer
        with open(f"{model_path}/label_binarizer.pkl", "rb") as f:
            label_binarizer = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"📋 Classifier type: {type(classifier)}")
        print(f"📋 Number of classes: {len(label_binarizer.classes_)}")
        
        # Create test features (775 dimensions: 768 BERT + 7 engineered features)
        test_features = np.random.random(775).reshape(1, -1)
        print(f"📊 Test features shape: {test_features.shape}")
        
        # Test prediction
        print("\n🔍 Testing prediction...")
        
        # Method 1: predict()
        print("Testing predict()...")
        pred = classifier.predict(test_features)
        print(f"✅ predict() successful, shape: {pred.shape}")
        
        # Method 2: predict_proba()
        print("Testing predict_proba()...")
        proba = classifier.predict_proba(test_features)
        print(f"✅ predict_proba() successful, shape: {proba.shape}")
        
        # Test individual label access
        print("\n🔍 Testing individual label access...")
        for i in range(min(5, len(label_binarizer.classes_))):
            code = label_binarizer.classes_[i]
            prob = proba[0][i]
            print(f"  {code}: {prob:.4f}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction() 