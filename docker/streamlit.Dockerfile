# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./
COPY requirements_local.txt ./

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements_local.txt

# Copy app code
COPY app/ ./app/
COPY model/saved_model/ ./model/saved_model/

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit entrypoint
CMD ["streamlit", "run", "app/streamlit_app_local.py", "--server.port=8501", "--server.address=0.0.0.0"] 