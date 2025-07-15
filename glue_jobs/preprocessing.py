#!/usr/bin/env python3
"""
AWS Glue ETL Job for EHR Data Preprocessing
HIPAA-Aware Data Processing for ICD-10 Prediction Model

This Glue job processes cleaned EHR data from Databricks validation
and prepares it for machine learning model training.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
import re
import json
from datetime import datetime

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_BUCKET', 'ENVIRONMENT'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Configuration
S3_BUCKET = args['S3_BUCKET']
ENVIRONMENT = args['ENVIRONMENT']
INPUT_PATH = f"s3://{S3_BUCKET}/data/cleaned/"
OUTPUT_PATH = f"s3://{S3_BUCKET}/data/preprocessed/"
TEMP_PATH = f"s3://{S3_BUCKET}/temp/"

print(f"🚀 Starting Glue ETL Job: {args['JOB_NAME']}")
print(f"📁 Input: {INPUT_PATH}")
print(f"📁 Output: {OUTPUT_PATH}")

# =============================================================================
# STEP 1: Load cleaned data from S3
# =============================================================================

def load_cleaned_data():
    """Load cleaned EHR data from S3 Parquet files."""
    print("📥 Loading cleaned data from S3...")
    
    # Read cleaned data
    cleaned_df = spark.read.parquet(INPUT_PATH)
    
    print(f"✅ Loaded {cleaned_df.count()} records")
    print(f"📊 Schema: {cleaned_df.columns}")
    
    return cleaned_df

# =============================================================================
# STEP 2: Text Preprocessing Functions
# =============================================================================

def clean_medical_text(text):
    """Clean and normalize medical text."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_medical_features(text):
    """Extract medical-specific features from text."""
    features = {}
    
    # Medical terminology patterns
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

# Register UDFs
clean_text_udf = udf(clean_medical_text, StringType())
extract_features_udf = udf(extract_medical_features, MapType(StringType(), IntegerType()))

# =============================================================================
# STEP 3: ICD-10 Code Processing
# =============================================================================

def process_icd10_codes(icd10_codes_str):
    """Process ICD-10 codes string into individual codes."""
    if not icd10_codes_str:
        return []
    
    # Split by pipe and clean
    codes = [code.strip() for code in icd10_codes_str.split('|') if code.strip()]
    
    # Validate format
    valid_codes = []
    for code in codes:
        if re.match(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', code):
            valid_codes.append(code)
    
    return valid_codes

def create_icd10_mapping(df):
    """Create mapping of all unique ICD-10 codes."""
    # Extract all ICD-10 codes
    all_codes = df.select(explode(split(col("icd10_codes"), "\\|")).alias("code")) \
                  .filter(col("code").isNotNull() & (col("code") != "")) \
                  .distinct() \
                  .collect()
    
    # Create mapping
    code_mapping = {row.code: idx for idx, row in enumerate(all_codes)}
    
    print(f"📋 Found {len(code_mapping)} unique ICD-10 codes")
    
    return code_mapping

# Register UDFs
process_codes_udf = udf(process_icd10_codes, ArrayType(StringType()))

# =============================================================================
# STEP 4: Feature Engineering Pipeline
# =============================================================================

def create_feature_pipeline():
    """Create Spark ML pipeline for text feature extraction."""
    
    # Text tokenization
    tokenizer = Tokenizer(inputCol="cleaned_notes", outputCol="tokens")
    
    # Remove stop words
    stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    
    # TF-IDF features
    cv = CountVectorizer(inputCol="filtered_tokens", outputCol="tf_features", vocabSize=1000, minDF=2)
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    
    # Create pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
    
    return pipeline

# =============================================================================
# STEP 5: Main Processing Function
# =============================================================================

def process_ehr_data():
    """Main function to process EHR data for ML training."""
    
    # Load data
    df = load_cleaned_data()
    
    # Clean text
    df = df.withColumn("cleaned_notes", clean_text_udf(col("doctor_notes")))
    
    # Extract medical features
    df = df.withColumn("medical_features", extract_features_udf(col("cleaned_notes")))
    
    # Process ICD-10 codes
    df = df.withColumn("processed_codes", process_codes_udf(col("icd10_codes")))
    
    # Create ICD-10 mapping
    icd10_mapping = create_icd10_mapping(df)
    
    # Save mapping for later use
    mapping_df = spark.createDataFrame([(code, idx) for code, idx in icd10_mapping.items()], 
                                      ["icd10_code", "code_index"])
    mapping_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/icd10_mapping/")
    
    # Create one-hot encoded ICD-10 features
    def create_one_hot_encoding(codes, mapping):
        """Create one-hot encoding for ICD-10 codes."""
        encoding = [0] * len(mapping)
        for code in codes:
            if code in mapping:
                encoding[mapping[code]] = 1
        return encoding
    
    one_hot_udf = udf(lambda codes: create_one_hot_encoding(codes, icd10_mapping), 
                      ArrayType(IntegerType()))
    
    df = df.withColumn("icd10_one_hot", one_hot_udf(col("processed_codes")))
    
    # Apply text feature extraction pipeline
    feature_pipeline = create_feature_pipeline()
    feature_model = feature_pipeline.fit(df)
    df = feature_model.transform(df)
    
    # Extract features from medical_features map
    df = df.withColumn("symptoms_count", col("medical_features.symptoms_count"))
    df = df.withColumn("body_parts_count", col("medical_features.body_parts_count"))
    df = df.withColumn("severity_count", col("medical_features.severity_count"))
    df = df.withColumn("medical_terms_count", col("medical_features.medical_terms_count"))
    
    # Create final features DataFrame
    features_df = df.select(
        "patient_id",
        "age",
        "gender",
        "doctor_notes",
        "cleaned_notes",
        "diagnosis_date",
        "icd10_codes",
        "processed_codes",
        "icd10_one_hot",
        "tfidf_features",
        "symptoms_count",
        "body_parts_count",
        "severity_count",
        "medical_terms_count",
        "note_length",
        "num_icd10_codes",
        "processed_at"
    )
    
    return features_df, feature_model, icd10_mapping

# =============================================================================
# STEP 6: Data Splitting and Export
# =============================================================================

def split_and_export_data(features_df, icd10_mapping):
    """Split data into train/validation/test sets and export."""
    
    print("🔄 Splitting data into train/validation/test sets...")
    
    # Split data (70% train, 15% validation, 15% test)
    train_df, temp_df = features_df.randomSplit([0.7, 0.3], seed=42)
    val_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=42)
    
    print(f"📊 Train set: {train_df.count()} records")
    print(f"📊 Validation set: {val_df.count()} records")
    print(f"📊 Test set: {test_df.count()} records")
    
    # Export datasets
    train_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/train/")
    val_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/validation/")
    test_df.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/test/")
    
    # Export metadata
    metadata = {
        "processing_date": datetime.now().isoformat(),
        "total_records": features_df.count(),
        "train_records": train_df.count(),
        "validation_records": val_df.count(),
        "test_records": test_df.count(),
        "unique_icd10_codes": len(icd10_mapping),
        "feature_columns": features_df.columns,
        "environment": ENVIRONMENT
    }
    
    # Save metadata
    metadata_df = spark.createDataFrame([(json.dumps(metadata),)], ["metadata"])
    metadata_df.write.mode("overwrite").text(f"{OUTPUT_PATH}/metadata/")
    
    return train_df, val_df, test_df

# =============================================================================
# STEP 7: Quality Checks
# =============================================================================

def perform_quality_checks(features_df, icd10_mapping):
    """Perform quality checks on processed data."""
    
    print("🔍 Performing quality checks...")
    
    # Check for null values
    null_counts = {}
    for col in features_df.columns:
        null_count = features_df.filter(col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count
    
    # Check data distribution
    age_stats = features_df.select("age").summary("count", "min", "25%", "50%", "75%", "max").collect()
    note_length_stats = features_df.select("note_length").summary("count", "min", "25%", "50%", "75%", "max").collect()
    
    # Check ICD-10 code distribution
    icd10_distribution = features_df.select(explode(col("processed_codes")).alias("code")) \
                                   .groupBy("code") \
                                   .count() \
                                   .orderBy("count", ascending=False) \
                                   .limit(10) \
                                   .collect()
    
    quality_report = {
        "null_values": null_counts,
        "age_statistics": {row.statistic: row.age for row in age_stats},
        "note_length_statistics": {row.statistic: row.note_length for row in note_length_stats},
        "top_icd10_codes": [(row.code, row.count) for row in icd10_distribution],
        "total_unique_codes": len(icd10_mapping)
    }
    
    print("✅ Quality checks completed")
    print(f"📊 Quality report: {json.dumps(quality_report, indent=2)}")
    
    return quality_report

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    try:
        print("🏥 Starting EHR Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Process data
        features_df, feature_model, icd10_mapping = process_ehr_data()
        
        # Perform quality checks
        quality_report = perform_quality_checks(features_df, icd10_mapping)
        
        # Split and export data
        train_df, val_df, test_df = split_and_export_data(features_df, icd10_mapping)
        
        # Save feature model
        feature_model.write().overwrite().save(f"{OUTPUT_PATH}/feature_model/")
        
        print("✅ Preprocessing pipeline completed successfully!")
        print(f"📁 Output location: {OUTPUT_PATH}")
        print(f"📊 Total records processed: {features_df.count()}")
        print(f"🏷️ Unique ICD-10 codes: {len(icd10_mapping)}")
        
        # Job completion
        job.commit()
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 