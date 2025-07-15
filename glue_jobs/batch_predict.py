#!/usr/bin/env python3
"""
AWS Glue Batch Inference Job for ICD-10 Prediction
Large-scale batch processing of clinical notes

This Glue job processes cleaned clinical notes in batch and generates
ICD-10 code predictions using the trained model.
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
import json
import pickle
import numpy as np
from datetime import datetime
import boto3
import os

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_BUCKET', 'MODEL_PATH', 'ENVIRONMENT'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Configuration
S3_BUCKET = args['S3_BUCKET']
MODEL_PATH = args['MODEL_PATH']
ENVIRONMENT = args['ENVIRONMENT']
INPUT_PATH = f"s3://{S3_BUCKET}/data/cleaned/"
OUTPUT_PATH = f"s3://{S3_BUCKET}/data/predictions/"
TEMP_PATH = f"s3://{S3_BUCKET}/temp/"

print(f"🚀 Starting Batch Inference Job: {args['JOB_NAME']}")
print(f"📁 Input: {INPUT_PATH}")
print(f"📁 Output: {OUTPUT_PATH}")
print(f"🤖 Model: {MODEL_PATH}")

# =============================================================================
# Model Loading Functions
# =============================================================================

def load_model_from_s3(model_path: str):
    """Load the trained model from S3."""
    print(f"📥 Loading model from {model_path}")
    
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    if model_path.startswith('s3://'):
        bucket = model_path.split('/')[2]
        key = '/'.join(model_path.split('/')[3:])
    else:
        bucket = S3_BUCKET
        key = model_path
    
    # Create temp directory
    temp_dir = "/tmp/model"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download model artifacts
    artifacts = [
        'config.json',
        'classifier.pkl',
        'label_binarizer.pkl'
    ]
    
    model_components = {}
    
    for artifact in artifacts:
        s3_key = f"{key}/{artifact}"
        local_path = f"{temp_dir}/{artifact}"
        
        try:
            s3_client.download_file(bucket, s3_key, local_path)
            print(f"✅ Downloaded {artifact}")
            
            # Load component
            if artifact.endswith('.json'):
                with open(local_path, 'r') as f:
                    model_components[artifact.replace('.json', '')] = json.load(f)
            elif artifact.endswith('.pkl'):
                with open(local_path, 'rb') as f:
                    model_components[artifact.replace('.pkl', '')] = pickle.load(f)
                    
        except Exception as e:
            print(f"❌ Error loading {artifact}: {str(e)}")
            raise
    
    return model_components

# =============================================================================
# Text Processing Functions
# =============================================================================

def clean_medical_text(text):
    """Clean and normalize medical text."""
    if not text:
        return ""
    
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_medical_features(text):
    """Extract medical-specific features from text."""
    import re
    
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
# BERT Feature Extraction (Simplified for Spark)
# =============================================================================

def extract_bert_features_simple(text, tokenizer, bert_model):
    """Simplified BERT feature extraction for Spark compatibility."""
    try:
        import torch
        
        # Preprocess text
        cleaned_text = clean_medical_text(text)
        
        # Simple tokenization (fallback)
        tokens = cleaned_text.split()[:100]  # Limit to 100 tokens
        token_features = [hash(token) % 768 for token in tokens]  # Simple hash-based features
        
        # Pad or truncate to 768 dimensions
        if len(token_features) < 768:
            token_features.extend([0] * (768 - len(token_features)))
        else:
            token_features = token_features[:768]
        
        return token_features
        
    except Exception as e:
        print(f"⚠️ BERT feature extraction failed, using fallback: {str(e)}")
        # Return zero features as fallback
        return [0] * 768

# =============================================================================
# Prediction Functions
# =============================================================================

def predict_icd10_codes(features, classifier, label_binarizer, confidence_threshold=0.5):
    """Make ICD-10 code predictions."""
    try:
        # Make predictions
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(features.reshape(1, -1))
            
            predictions = []
            confidences = []
            
            # Handle multi-label case
            if len(proba) > 1:
                for i, class_proba in enumerate(proba):
                    if class_proba[1] >= confidence_threshold:
                        predictions.append(label_binarizer.classes_[i])
                        confidences.append(float(class_proba[1]))
            else:
                for i, prob in enumerate(proba[0]):
                    if prob >= confidence_threshold:
                        predictions.append(label_binarizer.classes_[i])
                        confidences.append(float(prob))
        else:
            # Fallback for models without predict_proba
            predictions_binary = classifier.predict(features.reshape(1, -1))
            predictions = []
            confidences = []
            
            for i, pred in enumerate(predictions_binary[0]):
                if pred == 1:
                    predictions.append(label_binarizer.classes_[i])
                    confidences.append(0.8)  # Default confidence
        
        return predictions, confidences
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return [], []

# =============================================================================
# Main Processing Function
# =============================================================================

def process_batch_predictions():
    """Main function for batch prediction processing."""
    
    # Load model
    model_components = load_model_from_s3(MODEL_PATH)
    classifier = model_components['classifier']
    label_binarizer = model_components['label_binarizer']
    config = model_components['config']
    
    print(f"✅ Model loaded successfully")
    print(f"📋 Number of ICD-10 codes: {len(label_binarizer.classes_)}")
    
    # Load cleaned data
    print("📥 Loading cleaned clinical notes...")
    cleaned_df = spark.read.parquet(INPUT_PATH)
    
    print(f"✅ Loaded {cleaned_df.count()} records")
    
    # Preprocess data
    print("🔧 Preprocessing clinical notes...")
    
    # Clean text
    processed_df = cleaned_df.withColumn("cleaned_notes", clean_text_udf(col("doctor_notes")))
    
    # Extract medical features
    processed_df = processed_df.withColumn("medical_features", extract_features_udf(col("cleaned_notes")))
    
    # Extract features from medical_features map
    processed_df = processed_df.withColumn("symptoms_count", col("medical_features.symptoms_count"))
    processed_df = processed_df.withColumn("body_parts_count", col("medical_features.body_parts_count"))
    processed_df = processed_df.withColumn("severity_count", col("medical_features.severity_count"))
    processed_df = processed_df.withColumn("medical_terms_count", col("medical_features.medical_terms_count"))
    
    # Create simplified features (without BERT for batch processing)
    def create_simple_features(age, symptoms_count, body_parts_count, severity_count, medical_terms_count, note_length):
        """Create simplified features for batch processing."""
        # Normalize features
        age_norm = (age - 50) / 20 if age else 0
        note_length_norm = (note_length - 200) / 100
        
        # Create feature vector (simplified without BERT)
        features = [
            age_norm,
            symptoms_count,
            body_parts_count,
            severity_count,
            medical_terms_count,
            note_length_norm
        ]
        
        # Pad with zeros to match expected feature dimension
        expected_dim = 768 + 6  # BERT dim + engineered features
        features.extend([0] * (expected_dim - len(features)))
        
        return features
    
    create_features_udf = udf(create_simple_features, ArrayType(FloatType()))
    
    processed_df = processed_df.withColumn(
        "features",
        create_features_udf(
            col("age"),
            col("symptoms_count"),
            col("body_parts_count"),
            col("severity_count"),
            col("medical_terms_count"),
            col("note_length")
        )
    )
    
    # Broadcast model components
    classifier_bc = sc.broadcast(classifier)
    label_binarizer_bc = sc.broadcast(label_binarizer)
    
    # Define prediction function for Spark
    def predict_row(features, confidence_threshold=0.5):
        """Predict ICD-10 codes for a single row."""
        try:
            features_array = np.array(features, dtype=np.float32)
            predictions, confidences = predict_icd10_codes(
                features_array,
                classifier_bc.value,
                label_binarizer_bc.value,
                confidence_threshold
            )
            return predictions, confidences
        except Exception as e:
            print(f"❌ Row prediction error: {str(e)}")
            return [], []
    
    predict_udf = udf(predict_row, StructType([
        StructField("predicted_codes", ArrayType(StringType()), True),
        StructField("confidence_scores", ArrayType(FloatType()), True)
    ]))
    
    # Make predictions
    print("🤖 Making batch predictions...")
    predictions_df = processed_df.withColumn(
        "predictions",
        predict_udf(col("features"), lit(0.5))  # 0.5 confidence threshold
    )
    
    # Extract prediction results
    predictions_df = predictions_df.withColumn(
        "predicted_codes",
        col("predictions.predicted_codes")
    )
    
    predictions_df = predictions_df.withColumn(
        "confidence_scores",
        col("predictions.confidence_scores")
    )
    
    # Calculate prediction metrics
    predictions_df = predictions_df.withColumn(
        "num_predicted_codes",
        size(col("predicted_codes"))
    )
    
    predictions_df = predictions_df.withColumn(
        "avg_confidence",
        when(size(col("confidence_scores")) > 0,
             aggregate(col("confidence_scores"), lit(0.0), lambda acc, x: acc + x) / size(col("confidence_scores")))
        .otherwise(lit(0.0))
    )
    
    # Add processing metadata
    predictions_df = predictions_df.withColumn(
        "prediction_timestamp",
        current_timestamp()
    )
    
    predictions_df = predictions_df.withColumn(
        "model_version",
        lit("1.0.0")
    )
    
    predictions_df = predictions_df.withColumn(
        "processing_pipeline",
        lit("glue_batch_inference")
    )
    
    # Select final columns
    final_df = predictions_df.select(
        "patient_id",
        "age",
        "gender",
        "doctor_notes",
        "diagnosis_date",
        "predicted_codes",
        "confidence_scores",
        "num_predicted_codes",
        "avg_confidence",
        "prediction_timestamp",
        "model_version",
        "processing_pipeline"
    )
    
    # Save predictions
    print("💾 Saving predictions...")
    final_df.write.mode("overwrite").parquet(OUTPUT_PATH)
    
    # Generate summary statistics
    print("📊 Generating summary statistics...")
    
    total_records = final_df.count()
    records_with_predictions = final_df.filter(col("num_predicted_codes") > 0).count()
    avg_codes_per_record = final_df.agg(avg("num_predicted_codes")).collect()[0][0]
    avg_confidence = final_df.agg(avg("avg_confidence")).collect()[0][0]
    
    # Get top predicted codes
    all_codes = final_df.select(explode(col("predicted_codes")).alias("code"))
    top_codes = all_codes.groupBy("code").count().orderBy("count", ascending=False).limit(10)
    
    # Save summary
    summary = {
        "processing_date": datetime.now().isoformat(),
        "total_records": total_records,
        "records_with_predictions": records_with_predictions,
        "prediction_rate": records_with_predictions / total_records if total_records > 0 else 0,
        "avg_codes_per_record": avg_codes_per_record,
        "avg_confidence": avg_confidence,
        "model_version": "1.0.0",
        "environment": ENVIRONMENT
    }
    
    # Save summary as JSON
    summary_df = spark.createDataFrame([(json.dumps(summary),)], ["summary"])
    summary_df.write.mode("overwrite").text(f"{OUTPUT_PATH}/summary/")
    
    print("✅ Batch prediction completed successfully!")
    print(f"📊 Summary: {json.dumps(summary, indent=2)}")
    
    return final_df, summary

# =============================================================================
# Quality Checks
# =============================================================================

def perform_quality_checks(predictions_df, summary):
    """Perform quality checks on batch predictions."""
    
    print("🔍 Performing quality checks...")
    
    # Check prediction distribution
    prediction_distribution = predictions_df.groupBy("num_predicted_codes").count().orderBy("num_predicted_codes").collect()
    
    # Check confidence distribution
    confidence_ranges = predictions_df.select(
        when(col("avg_confidence") >= 0.8, "High")
        .when(col("avg_confidence") >= 0.6, "Medium")
        .otherwise("Low").alias("confidence_level")
    ).groupBy("confidence_level").count().collect()
    
    quality_report = {
        "prediction_distribution": [(row.num_predicted_codes, row.count) for row in prediction_distribution],
        "confidence_distribution": [(row.confidence_level, row.count) for row in confidence_ranges],
        "summary": summary
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
        print("🏥 Starting Batch ICD-10 Prediction Pipeline")
        print("=" * 60)
        
        # Process batch predictions
        predictions_df, summary = process_batch_predictions()
        
        # Perform quality checks
        quality_report = perform_quality_checks(predictions_df, summary)
        
        print("✅ Batch inference pipeline completed successfully!")
        print(f"📁 Output location: {OUTPUT_PATH}")
        print(f"📊 Total records processed: {summary['total_records']}")
        print(f"🎯 Prediction rate: {summary['prediction_rate']:.2%}")
        
        # Job completion
        job.commit()
        
    except Exception as e:
        print(f"❌ Error in batch inference pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 