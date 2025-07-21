# Comprehensive Dockerfile for ICD-10 Prediction System
# HIPAA-Aware Production Container with Streamlit + ML Model

# Multi-stage build for optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code (excluding training scripts)
COPY app/ ./app/
COPY model/saved_model/ ./model/saved_model/
COPY model/mlflow_tracking.py ./model/
COPY docker/ ./docker/
COPY requirements.txt .

# Create necessary directories with proper permissions
RUN mkdir -p /app/model/saved_model \
    /app/data/raw \
    /app/data/cleaned \
    /app/data/preprocessed \
    /app/logs \
    /app/mlruns \
    && chown -R appuser:appuser /app

# Add this to fix matplotlib config permissions
RUN mkdir -p /app/.config/matplotlib && chown -R appuser:appuser /app/.config/matplotlib

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Set entrypoint to run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"] 