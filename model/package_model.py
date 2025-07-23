#!/usr/bin/env python3
"""
Model Packaging Script for DVC Pipeline
Packages the trained ICD-10 prediction model for deployment
"""

import os
import sys
import json
import pickle
import shutil
import tarfile
from datetime import datetime
from typing import Dict, List
import pandas as pd

def create_model_package(model_dir: str, xgboost_path: str, metadata_path: str, 
                        output_dir: str) -> str:
    """Create a deployable model package."""
    print("📦 Creating model package...")
    
    # Create package directory
    package_name = f"icd10_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_path = os.path.join(output_dir, package_name)
    os.makedirs(package_path, exist_ok=True)
    
    # Copy BERT model
    bert_dest = os.path.join(package_path, "clinical_bert_model")
    shutil.copytree(model_dir, bert_dest)
    print(f"✅ Copied BERT model to {bert_dest}")
    
    # Copy XGBoost model
    xgb_dest = os.path.join(package_path, "xgboost_model.pkl")
    shutil.copy2(xgboost_path, xgb_dest)
    print(f"✅ Copied XGBoost model to {xgb_dest}")
    
    # Copy metadata
    metadata_dest = os.path.join(package_path, "model_metadata.json")
    shutil.copy2(metadata_path, metadata_dest)
    print(f"✅ Copied metadata to {metadata_dest}")
    
    # Create model info file
    model_info = {
        "model_name": "ICD-10 Prediction Model",
        "model_type": "ClinicalBERT + XGBoost Ensemble",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "description": "Multi-label ICD-10 code prediction from clinical notes",
        "architecture": {
            "bert_model": "emilyalsentzer/Bio_ClinicalBERT",
            "classifier": "XGBoost",
            "feature_dimension": 768,
            "num_classes": "variable"
        },
        "performance": {
            "overall_accuracy": "~98%",
            "f1_score": "~21%",
            "training_samples": "50,000"
        },
        "usage": {
            "input": "Clinical notes text",
            "output": "ICD-10 codes (multi-label)",
            "threshold": 0.5
        }
    }
    
    info_path = os.path.join(package_path, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✅ Created model info file: {info_path}")
    
    return package_path

def create_inference_script(package_path: str):
    """Create inference script for the packaged model."""
    print("📝 Creating inference script...")
    
    inference_script = '''#!/usr/bin/env python3
"""
ICD-10 Prediction Model Inference Script
Loads the packaged model and provides prediction functionality
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch

class ICD10Predictor:
    """ICD-10 code prediction model."""
    
    def __init__(self, model_path: str):
        """Initialize the predictor with model path."""
        self.model_path = model_path
        self.bert_model = None
        self.tokenizer = None
        self.xgb_model = None
        self.metadata = None
        self.class_names = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained models."""
        print("🤖 Loading ICD-10 prediction model...")
        
        # Load BERT model
        bert_path = os.path.join(self.model_path, "clinical_bert_model")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModel.from_pretrained(bert_path)
        self.bert_model.eval()
        
        # Load XGBoost model
        xgb_path = os.path.join(self.model_path, "xgboost_model.pkl")
        with open(xgb_path, 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(self.model_path, "model_metadata.json")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Get class names from XGBoost model
        if hasattr(self.xgb_model, 'classes_'):
            self.class_names = self.xgb_model.classes_
        
        print("✅ Model loaded successfully")
    
    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Preprocess text using BERT tokenizer."""
        features = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embeddings
                batch_features = outputs.last_hidden_state[:, 0, :].numpy()
            
            features.append(batch_features)
        
        return np.vstack(features)
    
    def predict(self, texts: List[str], threshold: float = 0.5) -> Dict:
        """Predict ICD-10 codes for given texts."""
        if not texts:
            return {"predictions": [], "probabilities": []}
        
        # Preprocess texts
        features = self.preprocess_text(texts)
        
        # Make predictions
        probabilities = self.xgb_model.predict_proba(features)
        predictions = (probabilities > threshold).astype(int)
        
        # Convert to ICD-10 codes
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            predicted_codes = []
            code_probabilities = {}
            
            for j, (is_predicted, probability) in enumerate(zip(pred, prob)):
                if is_predicted and self.class_names is not None:
                    code = self.class_names[j]
                    predicted_codes.append(code)
                    code_probabilities[code] = float(probability)
            
            results.append({
                "text": texts[i],
                "predicted_codes": predicted_codes,
                "probabilities": code_probabilities,
                "confidence": float(np.mean(prob[pred == 1])) if np.any(pred) else 0.0
            })
        
        return {
            "predictions": results,
            "model_info": {
                "model_type": "ClinicalBERT + XGBoost",
                "threshold": threshold,
                "num_classes": len(self.class_names) if self.class_names else 0
            }
        }
    
    def predict_single(self, text: str, threshold: float = 0.5) -> Dict:
        """Predict ICD-10 codes for a single text."""
        result = self.predict([text], threshold)
        return result["predictions"][0] if result["predictions"] else {}

def main():
    """Example usage of the ICD-10 predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICD-10 Code Prediction")
    parser.add_argument("--model_path", required=True, help="Path to model package")
    parser.add_argument("--text", required=True, help="Clinical note text")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ICD10Predictor(args.model_path)
    
    # Make prediction
    result = predictor.predict_single(args.text, args.threshold)
    
    # Print results
    print("\\n📋 ICD-10 Prediction Results:")
    print(f"Text: {result.get('text', '')}")
    print(f"Predicted Codes: {result.get('predicted_codes', [])}")
    print(f"Confidence: {result.get('confidence', 0):.3f}")
    print(f"Probabilities: {result.get('probabilities', {})}")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(package_path, "predict.py")
    with open(script_path, 'w') as f:
        f.write(inference_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"✅ Created inference script: {script_path}")

def create_requirements_file(package_path: str):
    """Create requirements.txt for the model package."""
    print("📋 Creating requirements file...")
    
    requirements = '''# Core dependencies for ICD-10 prediction model
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
xgboost>=2.0.0,<3.0.0
transformers>=4.30.0,<5.0.0
torch>=2.0.0,<3.0.0
tokenizers>=0.13.0,<1.0.0

# Optional dependencies for advanced features
plotly>=5.0.0,<6.0.0
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<1.0.0

# Utilities
tqdm>=4.65.0,<5.0.0
click>=8.0.0,<9.0.0
'''
    
    req_path = os.path.join(package_path, "requirements.txt")
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print(f"✅ Created requirements file: {req_path}")

def create_readme(package_path: str):
    """Create README for the model package."""
    print("📖 Creating README...")
    
    readme = '''# ICD-10 Prediction Model Package

This package contains a trained machine learning model for predicting ICD-10 billing codes from clinical notes.

## Model Information

- **Model Type**: ClinicalBERT + XGBoost Ensemble
- **Input**: Clinical notes text
- **Output**: ICD-10 codes (multi-label classification)
- **Performance**: ~98% overall accuracy

## Files

- `clinical_bert_model/`: BERT model and tokenizer
- `xgboost_model.pkl`: XGBoost classifier
- `model_metadata.json`: Model training metadata
- `model_info.json`: Model information and usage details
- `predict.py`: Inference script
- `requirements.txt`: Python dependencies

## Usage

### Basic Usage

```python
from predict import ICD10Predictor

# Initialize predictor
predictor = ICD10Predictor("./")

# Make prediction
result = predictor.predict_single(
    "Patient presents with chest pain and shortness of breath.",
    threshold=0.5
)

print(f"Predicted codes: {result['predicted_codes']}")
```

### Command Line Usage

```bash
python predict.py --model_path ./ --text "Patient with diabetes and hypertension" --threshold 0.5
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Use the model:
   ```python
   from predict import ICD10Predictor
   predictor = ICD10Predictor("./")
   ```

## Model Performance

- **Overall Accuracy**: ~98%
- **F1 Score**: ~21%
- **Training Samples**: 50,000 synthetic EHR records
- **Number of ICD-10 Codes**: Variable (based on training data)

## Notes

- This model was trained on synthetic data for demonstration purposes
- For production use, ensure proper validation and testing
- The model requires clinical text input for predictions
- Threshold can be adjusted based on precision/recall requirements
'''
    
    readme_path = os.path.join(package_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✅ Created README: {readme_path}")

def create_tarball(package_path: str, output_dir: str) -> str:
    """Create a compressed tarball of the model package."""
    print("🗜️ Creating model package tarball...")
    
    package_name = os.path.basename(package_path)
    tarball_path = os.path.join(output_dir, f"{package_name}.tar.gz")
    
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(package_path, arcname=package_name)
    
    print(f"✅ Created tarball: {tarball_path}")
    return tarball_path

def save_package_metrics(package_path: str, tarball_path: str, output_dir: str):
    """Save package metrics for DVC."""
    print("📊 Saving package metrics...")
    
    # Calculate package size
    package_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(package_path)
        for filename in filenames
    )
    
    tarball_size = os.path.getsize(tarball_path)
    
    metrics = {
        "package_date": datetime.now().isoformat(),
        "package_name": os.path.basename(package_path),
        "package_size_bytes": package_size,
        "package_size_mb": round(package_size / (1024 * 1024), 2),
        "tarball_size_bytes": tarball_size,
        "tarball_size_mb": round(tarball_size / (1024 * 1024), 2),
        "compression_ratio": round((1 - tarball_size / package_size) * 100, 2),
        "files_included": [
            "clinical_bert_model/",
            "xgboost_model.pkl",
            "model_metadata.json",
            "model_info.json",
            "predict.py",
            "requirements.txt",
            "README.md"
        ]
    }
    
    metrics_path = os.path.join(output_dir, "package_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Package metrics saved: {metrics_path}")
    print(f"📦 Package size: {metrics['package_size_mb']} MB")
    print(f"🗜️ Tarball size: {metrics['tarball_size_mb']} MB")
    print(f"📉 Compression: {metrics['compression_ratio']}%")

def main():
    """Main packaging function."""
    print("📦 ICD-10 Model Packaging")
    print("=" * 40)
    
    # File paths
    model_dir = "models/clinical_bert_model/"
    xgboost_path = "models/xgboost_model.pkl"
    metadata_path = "models/model_metadata.json"
    
    # Output directory
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if models exist
    if not os.path.exists(model_dir):
        print(f"❌ BERT model not found: {model_dir}")
        print("Please run model training first")
        return False
    
    if not os.path.exists(xgboost_path):
        print(f"❌ XGBoost model not found: {xgboost_path}")
        print("Please run model training first")
        return False
    
    # Create model package
    package_path = create_model_package(model_dir, xgboost_path, metadata_path, output_dir)
    
    # Create additional files
    create_inference_script(package_path)
    create_requirements_file(package_path)
    create_readme(package_path)
    
    # Create tarball
    tarball_path = create_tarball(package_path, output_dir)
    
    # Save metrics
    save_package_metrics(package_path, tarball_path, output_dir)
    
    print("\n✅ Model packaging completed successfully!")
    print(f"📁 Package location: {package_path}")
    print(f"🗜️ Tarball location: {tarball_path}")
    print(f"📋 Ready for deployment!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 