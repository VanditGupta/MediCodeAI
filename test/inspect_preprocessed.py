#!/usr/bin/env python3
"""
Inspect Preprocessed Data
Shows what's in the Parquet files and why we use Parquet instead of CSV
"""

import pandas as pd
import json
from pyspark.sql import SparkSession

def inspect_preprocessed_data():
    print("🔍 Inspecting Preprocessed Data")
    print("=" * 50)
    
    # Initialize Spark
    spark = SparkSession.builder.appName("InspectData").getOrCreate()
    
    # Load the preprocessed features
    features_df = spark.read.parquet("../data/preprocessed/features/")
    
    print(f"📊 Total records: {features_df.count()}")
    print(f"📋 Columns: {features_df.columns}")
    print()
    
    # Show schema to understand data types
    print("🏗️  Schema (showing why Parquet is needed):")
    features_df.printSchema()
    print()
    
    # Show sample data
    print("📋 Sample Records:")
    sample_df = features_df.limit(3).toPandas()
    
    # Display key columns
    display_cols = ['patient_id', 'age', 'gender', 'cleaned_notes', 'icd10_codes']
    print(sample_df[display_cols].to_string())
    print()
    
    # Show medical features for first record
    print("🏥 Medical Features (Map type - can't be stored in CSV):")
    first_record = features_df.first()
    if first_record.medical_features:
        for key, value in first_record.medical_features.items():
            print(f"  {key}: {value}")
    print()
    
    # Show processed codes for first record
    print("📋 Processed ICD-10 Codes (Array type - can't be stored in CSV):")
    if first_record.processed_codes:
        print(f"  {list(first_record.processed_codes)}")
    print()
    
    # Show one-hot encoding for first record
    print("🔢 ICD-10 One-Hot Encoding (Array type - can't be stored in CSV):")
    if first_record.icd10_onehot:
        print(f"  Length: {len(first_record.icd10_onehot)}")
        print(f"  First 10 values: {list(first_record.icd10_onehot[:10])}")
    print()
    
    # Show TF-IDF features
    print("📈 TF-IDF Features (Sparse Vector - can't be stored in CSV):")
    if first_record.tfidf_features:
        print(f"  Vector size: {first_record.tfidf_features.size}")
        print(f"  Non-zero indices: {first_record.tfidf_features.indices[:10] if len(first_record.tfidf_features.indices) > 10 else first_record.tfidf_features.indices}")
        print(f"  Non-zero values: {first_record.tfidf_features.values[:10] if len(first_record.tfidf_features.values) > 10 else first_record.tfidf_features.values}")
    print()
    
    # Load ICD-10 mapping
    with open("../data/preprocessed/icd10_mapping.json", "r") as f:
        icd10_mapping = json.load(f)
    
    print(f"📋 ICD-10 Mapping: {len(icd10_mapping)} unique codes")
    print("Sample mappings:")
    for i, (code, idx) in enumerate(list(icd10_mapping.items())[:5]):
        print(f"  {code} -> {idx}")
    print()
    
    # File size comparison
    import os
    parquet_size = os.path.getsize("../data/preprocessed/features/part-00000-02da0bb5-8f44-49cc-85cf-1ba462ffc1ac-c000.snappy.parquet")
    raw_csv_size = os.path.getsize("../data/raw/synthetic_ehr_data.csv")
    
    print("💾 File Size Comparison:")
    print(f"  Raw CSV: {raw_csv_size:,} bytes")
    print(f"  Preprocessed Parquet: {parquet_size:,} bytes")
    print(f"  Compression ratio: {raw_csv_size/parquet_size:.1f}x smaller")
    print()
    
    print("✅ Why Parquet instead of CSV:")
    print("  1. Complex data types (arrays, maps, vectors) can't be stored in CSV")
    print("  2. Much better compression (10-50x smaller)")
    print("  3. Faster read/write performance")
    print("  4. Schema preservation")
    print("  5. Native Spark integration")
    
    spark.stop()

if __name__ == "__main__":
    inspect_preprocessed_data() 