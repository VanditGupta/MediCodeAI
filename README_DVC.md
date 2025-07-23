# DVC Integration for MediCodeAI

This document explains how to use DVC (Data Version Control) in the MediCodeAI project for managing data, models, and artifacts.

## 🚀 Quick Start

### 1. Setup DVC

```bash
# Run the setup script
./dvc_setup.sh

# Or manually initialize DVC
dvc init
dvc remote add -d myremote s3://medicodeai-ehr-dvc-data/medicodeai
```

### 2. Run the Complete Pipeline

```bash
# Execute the entire MLOps pipeline
dvc repro

# Or run specific stages
dvc repro generate_data
dvc repro preprocess_data
dvc repro train_model
dvc repro evaluate_model
```

### 3. View Results

```bash
# View metrics
dvc metrics show

# View plots
dvc plots show

# Check pipeline status
dvc status
```

## 📁 Project Structure with DVC

```
MediCodeAI/
├── .dvc/                    # DVC configuration and cache
├── data/                    # Data files (tracked by DVC)
│   ├── raw/                 # Raw synthetic EHR data
│   ├── processed/           # Preprocessed data
│   ├── train/              # Training data
│   ├── validation/         # Validation data
│   └── test/               # Test data
├── models/                  # Trained models (tracked by DVC)
├── artifacts/              # Model artifacts (tracked by DVC)
├── dvc.yaml               # Pipeline definition
├── .dvcignore             # DVC ignore patterns
└── dvc_setup.sh           # Setup script
```

## 🔄 Pipeline Stages

The DVC pipeline consists of 6 stages:

### 1. **generate_data**
- **Command**: `python data_gen.py`
- **Inputs**: `data_gen.py`, `requirements.txt`
- **Outputs**: `data/raw/synthetic_ehr_data.csv`, metadata, metrics, plots
- **Purpose**: Generate synthetic EHR data with ICD-10 codes

### 2. **preprocess_data**
- **Command**: `spark-submit glue_jobs/preprocessing_local.py`
- **Inputs**: `glue_jobs/preprocessing_local.py`, raw data
- **Outputs**: `data/processed/cleaned_data.parquet`, preprocessing metadata
- **Purpose**: Clean and preprocess the raw data

### 3. **split_data**
- **Command**: `spark-submit model/split_data.py`
- **Inputs**: `model/split_data.py`, processed data
- **Outputs**: Train/validation/test splits
- **Purpose**: Split data for model training

### 4. **train_model**
- **Command**: `python model/train_model.py`
- **Inputs**: Training data, validation data, model cache
- **Outputs**: BERT model, XGBoost model, training metrics
- **Purpose**: Train the ICD-10 prediction model

### 5. **evaluate_model**
- **Command**: `python model/evaluate_model.py`
- **Inputs**: Test data, trained models
- **Outputs**: Evaluation results, confusion matrices, ROC curves
- **Purpose**: Evaluate model performance

### 6. **package_model**
- **Command**: `python model/package_model.py`
- **Inputs**: Trained models, metadata
- **Outputs**: Deployable model package, tarball
- **Purpose**: Package model for deployment

## 📊 Metrics and Plots

### Metrics Tracked
- **Data Generation**: Record count, code distribution, validation status
- **Preprocessing**: Data quality metrics, processing statistics
- **Model Training**: Accuracy, F1 score, training time
- **Model Evaluation**: Overall accuracy, per-label metrics, ROC AUC
- **Model Packaging**: Package size, compression ratio

### Plots Generated
- **Data Distribution**: ICD-10 code frequency, age/gender distribution
- **Model Performance**: Confusion matrices, ROC curves, accuracy plots
- **Evaluation Results**: Interactive HTML plots with metrics

## 🔧 Configuration

### DVC Configuration (`.dvc/config`)
```ini
[core]
    remote = myremote
    analytics = false

['remote "myremote"']
    url = s3://medicodeai-ehr-dvc-data/medicodeai
    endpointurl = https://s3.amazonaws.com
    region = us-east-1
```

### Pipeline Configuration (`dvc.yaml`)
- Defines all pipeline stages
- Specifies dependencies and outputs
- Configures metrics and plots
- Sets up data persistence

## 🛠️ Common Commands

### Pipeline Management
```bash
# Run the entire pipeline
dvc repro

# Run specific stage
dvc repro train_model

# Check pipeline status
dvc status

# Show pipeline graph
dvc dag
```

### Data Management
```bash
# Add new data to tracking
dvc add data/new_dataset.csv

# Commit changes
dvc commit

# Push to remote
dvc push

# Pull from remote
dvc pull
```

### Metrics and Plots
```bash
# Show all metrics
dvc metrics show

# Show specific metrics
dvc metrics show models/training_metrics.json

# Show plots
dvc plots show

# Show specific plot
dvc plots show data/raw/data_distribution.html
```

### Version Control
```bash
# Checkout specific version
dvc checkout

# List all versions
dvc list myremote

# Compare versions
dvc diff HEAD~1
```

## 🔐 Security and HIPAA Compliance

### Data Protection
- All data is synthetic (no real PHI)
- DVC cache is encrypted
- S3 bucket uses SSE-KMS encryption
- Access controlled via IAM roles

### Best Practices
- Never commit real patient data
- Use synthetic data for development
- Encrypt sensitive configuration
- Audit data access logs

## 🚀 Deployment Integration

### CI/CD Pipeline
The DVC pipeline integrates with the existing CI/CD:

1. **Data Generation**: Automated synthetic data creation
2. **Model Training**: Automated model training with MLflow tracking
3. **Model Evaluation**: Automated performance assessment
4. **Model Packaging**: Automated deployment package creation

### AWS Integration
- S3 for data storage
- ECR for model containers
- Lambda for real-time inference
- CloudWatch for monitoring

## 📈 Monitoring and Observability

### DVC Metrics Dashboard
```bash
# Generate metrics report
dvc metrics show --show-md > metrics_report.md

# Generate plots report
dvc plots show --show-md > plots_report.md
```

### Integration with MLflow
- DVC tracks data and model versions
- MLflow tracks experiments and runs
- Combined for complete MLOps observability

## 🔧 Troubleshooting

### Common Issues

#### 1. DVC not installed
```bash
pip install dvc[s3]
```

#### 2. AWS credentials not configured
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

#### 3. Pipeline stage fails
```bash
# Check stage dependencies
dvc status

# Run stage with verbose output
dvc repro train_model -v
```

#### 4. Remote storage issues
```bash
# Check remote configuration
dvc remote list

# Test remote connection
dvc push --dry-run
```

### Debug Commands
```bash
# Show DVC configuration
dvc config --list

# Show cache information
dvc cache dir

# Show remote information
dvc remote list
```

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Pipeline Tutorial](https://dvc.org/doc/start/data-pipelines)
- [DVC with S3](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3)
- [DVC Metrics and Plots](https://dvc.org/doc/user-guide/experiment-management/metrics-plots)

## 🤝 Contributing

When contributing to the DVC pipeline:

1. **Update `dvc.yaml`** when adding new stages
2. **Add metrics** for new stages
3. **Update `.dvcignore`** for new file patterns
4. **Test the pipeline** with `dvc repro`
5. **Document changes** in this README

## 📝 License

This DVC integration is part of the MediCodeAI project and follows the same MIT license. 