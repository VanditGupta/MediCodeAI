# Dockerfile for ICD-10 Prediction Model
# HIPAA-Aware Container for Production Deployment

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/model/saved_model \
    /app/data/raw \
    /app/data/cleaned \
    /app/data/preprocessed \
    /app/logs

# Set permissions for security
RUN chmod -R 755 /app && \
    chown -R nobody:nogroup /app

# Switch to non-root user for security
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "docker/entrypoint.py"] 