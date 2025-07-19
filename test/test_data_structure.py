#!/usr/bin/env python3
"""
Simple Data Structure Test
Validates preprocessed data structure without requiring PySpark
"""

import os
import json
import pandas as pd
from pathlib import Path

def test_data_structure():
    """Test the data structure without PySpark."""
    print("🔍 Testing data structure...")
    
    # Check if data directories exist
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
    
    # Load and validate raw data
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
        
        # Check data quality
        print(f"📊 Data quality check:")
        print(f"   - Age range: {raw_data['age'].min()} - {raw_data['age'].max()}")
        print(f"   - Gender values: {raw_data['gender'].unique()}")
        print(f"   - Notes length range: {raw_data['doctor_notes'].str.len().min()} - {raw_data['doctor_notes'].str.len().max()}")
        
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")
        return False
    
    # Check preprocessed data structure
    try:
        # Check if train/validation/test splits exist
        for split in ['train', 'validation', 'test']:
            split_path = f"../data/preprocessed/{split}/"
            if os.path.exists(split_path):
                files = os.listdir(split_path)
                print(f"✅ {split} split: {len(files)} files")
            else:
                print(f"❌ Missing {split} split")
                return False
        
        # Check if ICD-10 mapping exists
        mapping_path = "../data/preprocessed/icd10_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            print(f"✅ ICD-10 mapping: {len(mapping)} codes")
        else:
            print("❌ Missing ICD-10 mapping")
            return False
            
    except Exception as e:
        print(f"❌ Error checking preprocessed data: {e}")
        return False
    
    print("✅ Data structure validation completed successfully")
    return True

if __name__ == "__main__":
    success = test_data_structure()
    exit(0 if success else 1) 