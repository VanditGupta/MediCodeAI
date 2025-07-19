#!/usr/bin/env python3
"""
Download BERT Models for ICD-10 Prediction
Pre-downloads required models to avoid network issues during training
"""

import os
import sys
from transformers import AutoTokenizer, AutoModel
import torch

def download_models():
    """Download required BERT models."""
    
    print("🚀 Downloading BERT models for ICD-10 prediction...")
    
    # Create cache directory
    cache_dir = "model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Models to download
    models = [
        "emilyalsentzer/Bio_ClinicalBERT",  # Primary model
        "bert-base-uncased"  # Fallback model
    ]
    
    for model_name in models:
        print(f"\n📥 Downloading {model_name}...")
        try:
            # Download tokenizer
            print(f"   Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            print(f"   ✅ Tokenizer downloaded")
            
            # Download model
            print(f"   Downloading model...")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            print(f"   ✅ Model downloaded")
            
            # Test the model
            print(f"   Testing model...")
            test_input = tokenizer("Test sentence", return_tensors="pt")
            with torch.no_grad():
                output = model(**test_input)
            print(f"   ✅ Model test successful")
            
        except Exception as e:
            print(f"   ❌ Error downloading {model_name}: {str(e)}")
            continue
    
    print(f"\n✅ Model download completed!")
    print(f"📁 Models cached in: {os.path.abspath(cache_dir)}")

if __name__ == "__main__":
    download_models() 