#!/bin/bash

# Build and run script for ICD-10 Prediction System
# HIPAA-Aware Production Container

echo "🐳 Building ICD-10 Prediction Docker Image..."

# Build the Docker image
docker build -f docker/Dockerfile -t icd10-predictor:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "🚀 To run the container:"
    echo "   docker run -p 8501:8501 -p 5000:5000 icd10-predictor:latest"
    echo ""
    echo "📊 Or use docker-compose:"
    echo "   docker-compose up"
    echo ""
    echo "🌐 Access points:"
    echo "   - Streamlit UI: http://localhost:8501"
    echo "   - MLflow UI: http://localhost:5000"
else
    echo "❌ Docker build failed!"
    exit 1
fi 