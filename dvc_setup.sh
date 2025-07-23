#!/bin/bash
# DVC Setup Script for MediCodeAI Project
# Initializes DVC and configures remote storage

set -e

echo "🚀 Setting up DVC for MediCodeAI Project"
echo "========================================"

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC is not installed. Please install it first:"
    echo "   pip install dvc[s3]"
    exit 1
fi

echo "✅ DVC is installed"

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "📁 Initializing DVC..."
    dvc init
    echo "✅ DVC initialized"
else
    echo "✅ DVC already initialized"
fi

# Create necessary directories
echo "📂 Creating project directories..."
mkdir -p data/{raw,processed,train,validation,test,splits}
mkdir -p models
mkdir -p artifacts
mkdir -p .dvc/cache

echo "✅ Directories created"

# Configure DVC remote (S3)
echo "☁️ Configuring DVC remote storage..."

# Check if AWS credentials are available
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️ AWS credentials not found in environment variables"
    echo "   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "   Or configure AWS CLI: aws configure"
fi

# Add S3 remote (or update if exists)
echo "🔗 Configuring S3 remote..."
if dvc remote list | grep -q "myremote"; then
    echo "   Updating existing remote..."
    dvc remote modify myremote url s3://medicodeai-ehr-dvc-data/medicodeai
    dvc remote modify myremote endpointurl https://s3.amazonaws.com
    dvc remote modify myremote region us-east-1
    dvc remote default myremote
else
    echo "   Adding new remote..."
    dvc remote add -d myremote s3://medicodeai-ehr-dvc-data/medicodeai
    dvc remote modify myremote endpointurl https://s3.amazonaws.com
    dvc remote modify myremote region us-east-1
fi

echo "✅ S3 remote configured"

# Add local remote as backup (or update if exists)
echo "💾 Configuring local remote..."
if dvc remote list | grep -q "local"; then
    echo "   Updating existing local remote..."
    dvc remote modify local url ./dvc_cache
else
    echo "   Adding new local remote..."
    dvc remote add local ./dvc_cache
fi
echo "✅ Local remote configured"

# Configure DVC settings
echo "⚙️ Configuring DVC settings..."
dvc config core.analytics false

echo "✅ DVC settings configured"

# Create placeholder files for DVC tracking
echo "📝 Creating placeholder files for DVC tracking..."

# Create placeholder files in data directories
echo "# Placeholder file for DVC tracking" > data/raw/.gitkeep
echo "# Placeholder file for DVC tracking" > data/processed/.gitkeep
echo "# Placeholder file for DVC tracking" > data/train/.gitkeep
echo "# Placeholder file for DVC tracking" > data/validation/.gitkeep
echo "# Placeholder file for DVC tracking" > data/test/.gitkeep
echo "# Placeholder file for DVC tracking" > models/.gitkeep
echo "# Placeholder file for DVC tracking" > artifacts/.gitkeep

echo "✅ Placeholder files created"
echo "📝 Note: .dvc files will be created when you run the pipeline"

echo "✅ DVC setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure AWS credentials (aws configure or environment variables)"
echo "2. Run 'dvc repro' to execute the pipeline"
echo "3. Use 'dvc metrics show' to view metrics"
echo "4. Use 'dvc plots show' to view plots"

# Show DVC status
echo ""
echo "📊 DVC Status:"
echo "=============="
dvc status

echo ""
echo "🎉 DVC setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run 'dvc repro' to execute the pipeline"
echo "2. Use 'dvc metrics show' to view metrics"
echo "3. Use 'dvc plots show' to view plots"
echo ""
echo "📚 Useful DVC commands:"
echo "   dvc repro                    # Run the pipeline"
echo "   dvc status                   # Check pipeline status"
echo "   dvc metrics show             # Show metrics"
echo "   dvc plots show               # Show plots"
echo "   dvc push                     # Push to remote"
echo "   dvc pull                     # Pull from remote"
echo "   dvc checkout                 # Checkout specific version" 