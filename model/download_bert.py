#!/usr/bin/env python3
"""
BERT Model Download Script with Progress Tracking
Downloads and caches BERT models for local development
"""

import os
import sys
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import time

def download_model_with_progress(model_name: str, cache_dir: str = None):
    """
    Download BERT model with progress tracking.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache the model (optional)
    """
    
    print(f"🚀 Starting download of {model_name}")
    print(f"📁 Cache directory: {cache_dir or 'default'}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Download tokenizer with progress
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print("✅ Tokenizer downloaded successfully")
        
        # Download model with progress
        print("📥 Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print("✅ Model downloaded successfully")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate download time
        download_time = time.time() - start_time
        
        print("\n📊 Model Information:")
        print(f"   Model Name: {model_name}")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size: ~{total_params * 4 / (1024**2):.1f} MB")
        print(f"   Download Time: {download_time:.1f} seconds")
        
        # Test model
        print("\n🧪 Testing model...")
        test_input = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            output = model(**test_input)
        print("✅ Model test successful")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise

def main():
    """Main function to download models."""
    
    # List of models to download
    models_to_download = [
        "emilyalsentzer/Bio_ClinicalBERT",  # Primary choice
        "bert-base-uncased",                # Fallback option
    ]
    
    # Create cache directory
    cache_dir = "model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("🏥 ICD-10 Prediction Model Downloader")
    print("=" * 50)
    
    for i, model_name in enumerate(models_to_download, 1):
        print(f"\n📦 Downloading model {i}/{len(models_to_download)}: {model_name}")
        
        try:
            tokenizer, model = download_model_with_progress(model_name, cache_dir)
            
            # Save model info
            model_info = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "download_time": time.time(),
                "total_params": sum(p.numel() for p in model.parameters()),
                "status": "success"
            }
            
            # Save to cache directory
            import json
            with open(os.path.join(cache_dir, f"{model_name.replace('/', '_')}_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            print(f"💾 Model cached in: {cache_dir}")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {str(e)}")
            if i < len(models_to_download):
                print("🔄 Trying next model...")
                continue
            else:
                print("❌ All model downloads failed!")
                return
    
    print("\n🎉 All models downloaded successfully!")
    print(f"📁 Models are cached in: {os.path.abspath(cache_dir)}")
    print("\n💡 You can now run the training script without download delays.")

if __name__ == "__main__":
    main() 