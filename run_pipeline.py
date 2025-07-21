#!/usr/bin/env python3
"""
Complete MediCodeAI Pipeline Runner
Runs the entire pipeline from data generation to model training with MLflow tracking
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def run_command(command, description, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"📝 Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        if capture_output:
            # Capture output for quick commands
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"⚠️ Warnings/Info: {result.stderr}")
        else:
            # Show live output for long-running commands
            result = subprocess.run(command, shell=True, check=check)
        
        elapsed_time = time.time() - start_time
        print(f"✅ {description} completed in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {description} failed after {elapsed_time:.2f} seconds")
        if capture_output:
            print(f"Error: {e}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all required tools are available."""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    if not run_command("python --version", "Checking Python", check=False):
        print("❌ Python not found")
        return False
    
    # Check if virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️ Virtual environment not detected. Make sure to activate codametrix_env")
        print("💡 Run: source codametrix_env/bin/activate")
        return False
    
    # Check Spark
    if not run_command("spark-submit --version", "Checking Spark", check=False):
        print("❌ Spark not found. Please install Apache Spark")
        return False
    
    print("✅ All prerequisites checked")
    return True

def main():
    """Run the complete MediCodeAI pipeline."""
    print("🏥 MediCodeAI Complete Pipeline Runner")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues above.")
        return False
    
    # Step 1: Generate synthetic data
    print("\n📊 STEP 1: Generating Synthetic EHR Data")
    if not run_command("python data_gen.py", "Data Generation", capture_output=True):
        print("❌ Data generation failed")
        return False
    
    # Step 2: Preprocess data with Spark
    print("\n🔧 STEP 2: Preprocessing Data with PySpark")
    spark_command = "spark-submit --master local[*] --driver-memory 4g --executor-memory 2g glue_jobs/preprocessing_local.py"
    if not run_command(spark_command, "Data Preprocessing", capture_output=True):
        print("❌ Data preprocessing failed")
        return False
    
    # Step 3: Split data with Spark
    print("\n✂️ STEP 3: Splitting Data for Training")
    split_command = "spark-submit --master local[*] --driver-memory 4g --executor-memory 2g model/split_data.py"
    if not run_command(split_command, "Data Splitting", capture_output=True):
        print("❌ Data splitting failed")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Generated: 5,000 synthetic EHR records")
    print(f"🔧 Preprocessed: Data with PySpark")
    print(f"✂️ Split: Train/Validation/Test sets")
    print(f"📝 You can now run model training separately if desired.")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 