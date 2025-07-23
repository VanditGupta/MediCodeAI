#!/usr/bin/env python3
"""
Local PySpark Preprocessing for EHR Data (ICD-10 Prediction)
Adapted from AWS Glue job for local development/testing.
"""

import os
import argparse
import re
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline

# =============================================================================
# Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(description="Local EHR Preprocessing for ICD-10 Prediction")
parser.add_argument('--input', type=str, default='data/raw/synthetic_ehr_data.csv', help='Input CSV file')
parser.add_argument('--output', type=str, default='data/preprocessed/', help='Output directory')
args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT_PATH = args.output
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"🚀 Starting Local PySpark Preprocessing")
print(f"📁 Input: {INPUT_PATH}")
print(f"📁 Output: {OUTPUT_PATH}")

# =============================================================================
# Spark Session
# =============================================================================
spark = SparkSession.builder.appName("EHRPreprocessingLocal").getOrCreate()

# =============================================================================
# STEP 1: Load cleaned data from CSV
# =============================================================================
def load_cleaned_data():
    print("📥 Loading cleaned data from CSV...")
    df = spark.read.option("header", True).option("inferSchema", True).csv(INPUT_PATH)
    print(f"✅ Loaded {df.count()} records")
    print(f"📊 Schema: {df.columns}")
    return df

# =============================================================================
# STEP 2: Text Preprocessing Functions
# =============================================================================
def clean_medical_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_medical_features(text):
    features = {}
    medical_patterns = {
        'symptoms': r'\b(pain|discomfort|pressure|burning|nausea|dizziness|fatigue|weakness)\b',
        'body_parts': r'\b(chest|abdomen|head|back|legs|arms|neck|shoulder|knee|hip|throat|stomach)\b',
        'severity': r'\b(mild|moderate|severe|acute|chronic|intermittent|persistent)\b',
        'medical_terms': r'\b(examination|diagnosis|treatment|medication|symptoms|condition)\b'
    }
    for feature_name, pattern in medical_patterns.items():
        matches = re.findall(pattern, text.lower())
        features[f'{feature_name}_count'] = len(matches)
        features[f'{feature_name}_present'] = 1 if matches else 0
    return features

clean_text_udf = udf(clean_medical_text, StringType())
extract_features_udf = udf(extract_medical_features, MapType(StringType(), IntegerType()))

# =============================================================================
# STEP 3: ICD-10 Code Processing
# =============================================================================
def process_icd10_codes(icd10_codes_str):
    if not icd10_codes_str:
        return []
    codes = [code.strip() for code in icd10_codes_str.split('|') if code.strip()]
    valid_codes = []
    for code in codes:
        if re.match(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', code):
            valid_codes.append(code)
    return valid_codes

process_codes_udf = udf(process_icd10_codes, ArrayType(StringType()))

def create_icd10_mapping(df):
    all_codes = df.select(explode(split(col("icd10_codes"), "\\|")).alias("code")) \
                  .filter(col("code").isNotNull() & (col("code") != "")) \
                  .distinct() \
                  .collect()
    code_mapping = {row.code: idx for idx, row in enumerate(all_codes)}
    print(f"📋 Found {len(code_mapping)} unique ICD-10 codes")
    return code_mapping

# =============================================================================
# STEP 4: Feature Engineering Pipeline
# =============================================================================
def create_feature_pipeline():
    tokenizer = Tokenizer(inputCol="cleaned_notes", outputCol="tokens")
    stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    cv = CountVectorizer(inputCol="filtered_tokens", outputCol="tf_features", vocabSize=1000, minDF=2)
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
    return pipeline

# =============================================================================
# STEP 5: Main Processing Function
# =============================================================================
def process_ehr_data():
    df = load_cleaned_data()
    df = df.withColumn("cleaned_notes", clean_text_udf(col("doctor_notes")))
    df = df.withColumn("medical_features", extract_features_udf(col("cleaned_notes")))
    df = df.withColumn("processed_codes", process_codes_udf(col("icd10_codes")))
    icd10_mapping = create_icd10_mapping(df)
    mapping_df = spark.createDataFrame([(code, idx) for code, idx in icd10_mapping.items()], ["icd10_code", "code_index"])
    mapping_df.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "icd10_mapping/"))
    def create_one_hot_encoding(codes, mapping):
        encoding = [0] * len(mapping)
        for code in codes:
            if code in mapping:
                encoding[mapping[code]] = 1
        return encoding
    one_hot_udf = udf(lambda codes: create_one_hot_encoding(codes, icd10_mapping), ArrayType(IntegerType()))
    df = df.withColumn("icd10_onehot", one_hot_udf(col("processed_codes")))
    # Feature pipeline
    pipeline = create_feature_pipeline()
    model = pipeline.fit(df)
    features_df = model.transform(df)
    # Save features and labels
    features_df.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "features/"))
    
    # Save cleaned data as parquet for DVC
    cleaned_data_path = "data/processed/cleaned_data.parquet"
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    features_df.write.mode("overwrite").parquet(cleaned_data_path)
    
    # Save preprocessing metadata
    metadata = {
        "preprocessing_date": datetime.now().isoformat(),
        "input_records": df.count(),
        "output_records": features_df.count(),
        "unique_icd10_codes": len(icd10_mapping),
        "feature_columns": features_df.columns,
        "vocabulary_size": 1000,
        "min_document_frequency": 2
    }
    
    metadata_path = "data/processed/preprocessing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save preprocessing metrics
    metrics = {
        "preprocessing": {
            "input_records": int(df.count()),
            "output_records": int(features_df.count()),
            "data_quality_score": 1.0,
            "feature_count": len(features_df.columns),
            "vocabulary_size": 1000
        }
    }
    
    metrics_path = "data/processed/preprocessing_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Preprocessing complete. Features saved to {os.path.join(OUTPUT_PATH, 'features/')}.")
    print(f"✅ Cleaned data saved to {cleaned_data_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    
    # Save mapping as JSON for model training
    with open(os.path.join(OUTPUT_PATH, "icd10_mapping.json"), "w") as f:
        json.dump(icd10_mapping, f, indent=2)

if __name__ == "__main__":
    try:
        process_ehr_data()
        print("✅ Preprocessing completed successfully!")
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        raise
    finally:
        spark.stop() 