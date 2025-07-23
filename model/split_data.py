#!/usr/bin/env python3
"""
Data Splitting Script for Model Training
Splits preprocessed data into train/validation/test sets
"""

import os
import json
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

def split_preprocessed_data():
    """Split preprocessed data into train/validation/test sets."""
    
    print("\n🔀 Splitting preprocessed data for model training", flush=True)
    print("=" * 50, flush=True)
    start_time = time.time()
    
    # Initialize Spark
    print("🚀 Initializing Spark session...", flush=True)
    spark = SparkSession.builder.appName("DataSplit").getOrCreate()
    print("✅ Spark session started.", flush=True)
    
    # Load preprocessed features
    print("📥 Loading preprocessed features from data/preprocessed/features/...", flush=True)
    features_df = spark.read.parquet("data/preprocessed/features/")
    total_records = features_df.count()
    print(f"📊 Total records: {total_records}", flush=True)
    
    # Convert to pandas for easier splitting
    print("🔄 Converting Spark DataFrame to pandas DataFrame (this may take a while for large datasets)...", flush=True)
    t0 = time.time()
    df = features_df.toPandas()
    print(f"✅ Conversion complete in {time.time() - t0:.1f} seconds.", flush=True)
    
    # Add note_length feature if not present
    print("📝 Checking/adding engineered features...", flush=True)
    t0 = time.time()
    if 'note_length' not in df.columns:
        df['note_length'] = df['cleaned_notes'].str.len()
    if 'symptoms_count' not in df.columns:
        df['symptoms_count'] = df['medical_features'].apply(lambda x: x.get('symptoms_count', 0) if x else 0)
    if 'body_parts_count' not in df.columns:
        df['body_parts_count'] = df['medical_features'].apply(lambda x: x.get('body_parts_count', 0) if x else 0)
    if 'severity_count' not in df.columns:
        df['severity_count'] = df['medical_features'].apply(lambda x: x.get('severity_count', 0) if x else 0)
    if 'medical_terms_count' not in df.columns:
        df['medical_terms_count'] = df['medical_features'].apply(lambda x: x.get('medical_terms_count', 0) if x else 0)
    print(f"✅ Feature engineering complete in {time.time() - t0:.1f} seconds.", flush=True)
    print(f"📋 Features available: {list(df.columns)}", flush=True)
    
    # Split data: 70% train, 15% validation, 15% test
    print("✂️ Splitting data into train/validation/test sets...", flush=True)
    t0 = time.time()
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=None)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=None)
    print(f"✅ Data split complete in {time.time() - t0:.1f} seconds.", flush=True)
    print(f"📊 Split sizes:")
    print(f"  Train: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create output directories
    print("📁 Creating output directories...", flush=True)
    output_path = "data/preprocessed"
    os.makedirs(f"{output_path}/train", exist_ok=True)
    os.makedirs(f"{output_path}/validation", exist_ok=True)
    os.makedirs(f"{output_path}/test", exist_ok=True)
    
    # Create DVC expected directories
    dvc_output_path = "data"
    os.makedirs(f"{dvc_output_path}/train", exist_ok=True)
    os.makedirs(f"{dvc_output_path}/validation", exist_ok=True)
    os.makedirs(f"{dvc_output_path}/test", exist_ok=True)
    os.makedirs(f"{dvc_output_path}/splits", exist_ok=True)
    print("✅ Output directories ready.", flush=True)
    
    # Convert back to Spark DataFrames and save
    print("💾 Saving splits as Parquet files (train/validation/test)...", flush=True)
    t0 = time.time()
    train_spark = spark.createDataFrame(train_df)
    val_spark = spark.createDataFrame(val_df)
    test_spark = spark.createDataFrame(test_df)
    
    # Save to both locations
    train_spark.write.mode("overwrite").parquet(f"{output_path}/train/")
    val_spark.write.mode("overwrite").parquet(f"{output_path}/validation/")
    test_spark.write.mode("overwrite").parquet(f"{output_path}/test/")
    
    # Save to DVC expected locations
    train_spark.write.mode("overwrite").parquet(f"{dvc_output_path}/train/train_data.parquet")
    val_spark.write.mode("overwrite").parquet(f"{dvc_output_path}/validation/validation_data.parquet")
    test_spark.write.mode("overwrite").parquet(f"{dvc_output_path}/test/test_data.parquet")
    
    print(f"✅ Parquet files saved in {time.time() - t0:.1f} seconds.", flush=True)
    print(f"✅ Data splits saved to:")
    print(f"  Train: {output_path}/train/")
    print(f"  Validation: {output_path}/validation/")
    print(f"  Test: {output_path}/test/")
    
    # Save split metadata and metrics for DVC
    split_metadata = {
        "split_date": datetime.now().isoformat(),
        "total_records": int(len(df)),
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "split_ratios": {
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15
        },
        "random_state": 42
    }
    
    metadata_path = f"{dvc_output_path}/splits/split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=2)
    
    # Save split metrics
    split_metrics = {
        "data_splitting": {
            "total_records": int(len(df)),
            "train_records": int(len(train_df)),
            "validation_records": int(len(val_df)),
            "test_records": int(len(test_df)),
            "train_ratio": float(len(train_df) / len(df)),
            "validation_ratio": float(len(val_df) / len(df)),
            "test_ratio": float(len(test_df) / len(df))
        }
    }
    
    metrics_path = f"{dvc_output_path}/splits/split_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(split_metrics, f, indent=2)
    
    print(f"✅ Split metadata saved to: {metadata_path}")
    print(f"✅ Split metrics saved to: {metrics_path}")
    
    # Show sample statistics
    print("\n📈 Sample Statistics:")
    print(f"  Average age: {df['age'].mean():.1f}")
    print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"  Average note length: {df['note_length'].mean():.1f} characters")
    
    # Show ICD-10 code distribution
    all_codes = []
    for codes in df['processed_codes']:
        if codes:
            all_codes.extend(codes)
    unique_codes = set(all_codes)
    print(f"  Unique ICD-10 codes: {len(unique_codes)}")
    print(f"  Average codes per patient: {len(all_codes)/len(df):.2f}")
    
    spark.stop()
    print(f"\n🎉 Data splitting completed successfully in {time.time() - start_time:.1f} seconds!", flush=True)

if __name__ == "__main__":
    try:
        split_preprocessed_data()
    except Exception as e:
        print(f"❌ Data splitting failed: {str(e)}")
        raise 