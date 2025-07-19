# 🐳 Docker Setup for ICD-10 Prediction System

## Overview
Single comprehensive Docker container that includes:
- **Streamlit Frontend** (Port 8502)
- **MLflow Server** (Port 5001) 
- **ML Model Inference** (pre-trained models only)
- **HIPAA-Compliant Security**

## Workflow

### 1. Local Training (Before Docker)
```bash
# Generate synthetic data
python data_gen.py

# Train the model locally
python model/train_model.py

# This creates model artifacts in model/saved_model/
```

### 2. Docker Deployment (After Training)
```bash
# Build the image (includes only trained models)
./docker/build.sh

# Run the container
docker-compose up
```

## Quick Start

### 1. Build the Image
```bash
# Option 1: Use the build script
./docker/build.sh

# Option 2: Manual build
docker build -f docker/Dockerfile -t icd10-predictor:latest .
```

### 2. Run the Container
```bash
# Option 1: Using docker-compose (recommended)
docker-compose up

# Option 2: Manual run
docker run -p 8502:8501 -p 5001:5000 icd10-predictor:latest
```

### 3. Access the Application
- **Streamlit UI**: http://localhost:8502
- **MLflow UI**: http://localhost:5001

## Features

### ✅ Production Optimized
- Multi-stage build for smaller images
- Non-root user for security
- Health checks and monitoring
- Optimized dependencies

### ✅ HIPAA Compliant
- Data encryption at rest
- Secure user permissions
- Audit logging capabilities
- Privacy protection measures

### ✅ Inference-Only Container
- Pre-trained models only (no training code)
- Streamlit frontend
- MLflow experiment tracking
- Model inference API

## Container Structure
```
/app/
├── app/                 # Streamlit application
├── model/
│   ├── saved_model/    # Pre-trained model artifacts
│   └── mlflow_tracking.py  # MLflow utilities
├── docker/             # Docker configuration
├── data/               # Data directories (mounted)
├── logs/               # Application logs
└── mlruns/             # MLflow artifacts (mounted)
```

## What's Excluded from Docker
- Training scripts (`train_model.py`, `split_data.py`)
- Data generation (`data_gen.py`, `run_pipeline.py`)
- Development files and logs
- Local MLflow data (mounted as volume)

## Environment Variables
- `PYTHONUNBUFFERED=1` - Real-time logging
- `MLFLOW_TRACKING_URI=http://localhost:5000` - MLflow connection

## Port Mapping
- **8502** → Streamlit frontend (mapped from container port 8501)
- **5001** → MLflow server (mapped from container port 5000)
- **8000** → API endpoint (if needed)

## Health Checks
- Streamlit health endpoint: `/health`
- MLflow server status
- Automatic restart on failure

## Security Features
- Non-root user execution
- Minimal attack surface
- Secure file permissions
- Vulnerability scanning ready 