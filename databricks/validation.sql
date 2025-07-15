-- Databricks SQL Script for EHR Data Validation and Cleaning
-- HIPAA-Aware Data Quality Checks for ICD-10 Prediction Pipeline

-- =============================================================================
-- STEP 1: Load raw data and create initial table
-- =============================================================================

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS hipaa_mlops;

-- Use the database
USE hipaa_mlops;

-- Create raw data table from CSV
CREATE TABLE IF NOT EXISTS raw_ehr_data
USING CSV
OPTIONS (
  path "s3://your-hipaa-bucket/data/raw/synthetic_ehr_data.csv",
  header "true",
  inferSchema "true"
);

-- =============================================================================
-- STEP 2: Data Quality Validation
-- =============================================================================

-- Check for null or empty doctor notes
CREATE OR REPLACE TEMPORARY VIEW notes_validation AS
SELECT 
  patient_id,
  doctor_notes,
  CASE 
    WHEN doctor_notes IS NULL THEN 'NULL_NOTES'
    WHEN TRIM(doctor_notes) = '' THEN 'EMPTY_NOTES'
    WHEN LENGTH(TRIM(doctor_notes)) < 10 THEN 'TOO_SHORT_NOTES'
    ELSE 'VALID_NOTES'
  END as notes_validation_status
FROM raw_ehr_data;

-- Validate ICD-10 code format using regex
CREATE OR REPLACE TEMPORARY VIEW icd10_validation AS
SELECT 
  patient_id,
  icd10_codes,
  -- Split ICD-10 codes and validate each
  ARRAY(
    SELECT code 
    FROM (
      SELECT EXPLODE(SPLIT(icd10_codes, '\\|')) as code
    ) 
    WHERE code RLIKE '^[A-Z][0-9]{2}(\\.[0-9A-Z]{1,4})?$'
  ) as valid_codes,
  ARRAY(
    SELECT code 
    FROM (
      SELECT EXPLODE(SPLIT(icd10_codes, '\\|')) as code
    ) 
    WHERE NOT (code RLIKE '^[A-Z][0-9]{2}(\\.[0-9A-Z]{1,4})?$')
  ) as invalid_codes
FROM raw_ehr_data;

-- =============================================================================
-- STEP 3: Data Cleaning and Deduplication
-- =============================================================================

-- Create cleaned data with quality filters
CREATE OR REPLACE TEMPORARY VIEW cleaned_ehr_data AS
SELECT 
  r.patient_id,
  r.age,
  r.gender,
  r.doctor_notes,
  r.diagnosis_date,
  r.icd10_codes,
  -- Add validation flags
  n.notes_validation_status,
  i.valid_codes,
  i.invalid_codes,
  -- Add data quality metrics
  LENGTH(r.doctor_notes) as note_length,
  SIZE(SPLIT(r.icd10_codes, '\\|')) as num_icd10_codes,
  -- Add hash for deduplication
  HASH(r.patient_id, r.doctor_notes) as content_hash
FROM raw_ehr_data r
LEFT JOIN notes_validation n ON r.patient_id = n.patient_id
LEFT JOIN icd10_validation i ON r.patient_id = i.patient_id
WHERE 
  -- Filter out invalid records
  n.notes_validation_status = 'VALID_NOTES'
  AND SIZE(i.valid_codes) > 0  -- At least one valid ICD-10 code
  AND r.age BETWEEN 0 AND 120  -- Reasonable age range
  AND r.gender IN ('M', 'F')   -- Valid gender values
  AND r.diagnosis_date IS NOT NULL;

-- Remove duplicates based on patient_id and note content
CREATE OR REPLACE TEMPORARY VIEW deduplicated_data AS
SELECT 
  *,
  ROW_NUMBER() OVER (
    PARTITION BY patient_id, content_hash 
    ORDER BY diagnosis_date DESC
  ) as row_num
FROM cleaned_ehr_data;

-- =============================================================================
-- STEP 4: Final Cleaned Dataset
-- =============================================================================

-- Create final cleaned table
CREATE OR REPLACE TABLE cleaned_ehr_data_final AS
SELECT 
  patient_id,
  age,
  gender,
  doctor_notes,
  diagnosis_date,
  icd10_codes,
  valid_codes,
  note_length,
  num_icd10_codes,
  content_hash,
  -- Add processing metadata
  CURRENT_TIMESTAMP() as processed_at,
  'databricks_validation' as processing_pipeline
FROM deduplicated_data
WHERE row_num = 1;

-- =============================================================================
-- STEP 5: Data Quality Reports
-- =============================================================================

-- Generate quality metrics
CREATE OR REPLACE TEMPORARY VIEW quality_metrics AS
SELECT 
  'TOTAL_RECORDS' as metric,
  COUNT(*) as value
FROM raw_ehr_data

UNION ALL

SELECT 
  'CLEANED_RECORDS' as metric,
  COUNT(*) as value
FROM cleaned_ehr_data_final

UNION ALL

SELECT 
  'DUPLICATE_RECORDS_REMOVED' as metric,
  (SELECT COUNT(*) FROM raw_ehr_data) - (SELECT COUNT(*) FROM cleaned_ehr_data_final) as value

UNION ALL

SELECT 
  'AVG_NOTE_LENGTH' as metric,
  ROUND(AVG(note_length), 2) as value
FROM cleaned_ehr_data_final

UNION ALL

SELECT 
  'AVG_ICD10_CODES_PER_PATIENT' as metric,
  ROUND(AVG(num_icd10_codes), 2) as value
FROM cleaned_ehr_data_final;

-- =============================================================================
-- STEP 6: Export Cleaned Data
-- =============================================================================

-- Write cleaned data to S3 as Parquet (encrypted)
INSERT OVERWRITE DIRECTORY 's3://your-hipaa-bucket/data/cleaned/'
USING PARQUET
OPTIONS (
  compression 'snappy',
  encryption 'AES_GCM_V1'
)
SELECT 
  patient_id,
  age,
  gender,
  doctor_notes,
  diagnosis_date,
  icd10_codes,
  valid_codes,
  note_length,
  num_icd10_codes,
  processed_at,
  processing_pipeline
FROM cleaned_ehr_data_final;

-- =============================================================================
-- STEP 7: Create Summary Views for Monitoring
-- =============================================================================

-- Create summary view for monitoring
CREATE OR REPLACE VIEW ehr_data_summary AS
SELECT 
  COUNT(*) as total_records,
  COUNT(DISTINCT patient_id) as unique_patients,
  ROUND(AVG(age), 2) as avg_age,
  COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_count,
  COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_count,
  ROUND(AVG(note_length), 2) as avg_note_length,
  ROUND(AVG(num_icd10_codes), 2) as avg_icd10_codes,
  MIN(diagnosis_date) as earliest_diagnosis,
  MAX(diagnosis_date) as latest_diagnosis,
  COUNT(DISTINCT valid_codes) as unique_valid_codes
FROM cleaned_ehr_data_final;

-- Create ICD-10 code distribution view
CREATE OR REPLACE VIEW icd10_distribution AS
SELECT 
  code,
  COUNT(*) as frequency,
  ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM cleaned_ehr_data_final), 2) as percentage
FROM (
  SELECT EXPLODE(valid_codes) as code
  FROM cleaned_ehr_data_final
)
GROUP BY code
ORDER BY frequency DESC;

-- =============================================================================
-- STEP 8: Data Quality Alerts
-- =============================================================================

-- Create alerts for data quality issues
CREATE OR REPLACE TEMPORARY VIEW data_quality_alerts AS
SELECT 
  'HIGH_DUPLICATE_RATE' as alert_type,
  'Duplicate rate exceeds 10%' as alert_message,
  COUNT(*) as affected_records
FROM (
  SELECT patient_id, COUNT(*) as dup_count
  FROM raw_ehr_data
  GROUP BY patient_id
  HAVING COUNT(*) > 1
)
WHERE dup_count > 1

UNION ALL

SELECT 
  'INVALID_ICD10_CODES' as alert_type,
  'Records with invalid ICD-10 codes detected' as alert_message,
  COUNT(*) as affected_records
FROM icd10_validation
WHERE SIZE(invalid_codes) > 0

UNION ALL

SELECT 
  'SHORT_NOTES' as alert_type,
  'Records with very short notes detected' as alert_message,
  COUNT(*) as affected_records
FROM notes_validation
WHERE notes_validation_status = 'TOO_SHORT_NOTES';

-- =============================================================================
-- STEP 9: Final Validation Summary
-- =============================================================================

-- Display final summary
SELECT 
  'DATA_VALIDATION_COMPLETE' as status,
  CURRENT_TIMESTAMP() as completed_at,
  (SELECT value FROM quality_metrics WHERE metric = 'TOTAL_RECORDS') as total_input_records,
  (SELECT value FROM quality_metrics WHERE metric = 'CLEANED_RECORDS') as cleaned_output_records,
  (SELECT value FROM quality_metrics WHERE metric = 'DUPLICATE_RECORDS_REMOVED') as duplicates_removed,
  ROUND(
    (SELECT value FROM quality_metrics WHERE metric = 'CLEANED_RECORDS') * 100.0 / 
    (SELECT value FROM quality_metrics WHERE metric = 'TOTAL_RECORDS'), 
    2
  ) as data_quality_percentage;

-- Display quality metrics
SELECT * FROM quality_metrics ORDER BY metric;

-- Display top ICD-10 codes
SELECT * FROM icd10_distribution LIMIT 10;

-- Display any data quality alerts
SELECT * FROM data_quality_alerts WHERE affected_records > 0; 