#!/usr/bin/env python3
"""
Master Test Runner for MediCodeAI
Runs all tests in the correct order to validate the entire pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test(test_name, test_file, description=""):
    """Run a single test and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_name}")
    if description:
        print(f"📝 Description: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_name} FAILED ({duration:.2f}s)")
            if result.stderr:
                print("📤 Error:")
                print(result.stderr)
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {test_name} ERROR: {str(e)}")
        return False

def main():
    """Run all tests in the correct order."""
    print("🚀 Starting MediCodeAI Test Suite")
    print("=" * 60)
    
    # Define tests in order of execution
    tests = [
        {
            "name": "Data Structure Test", 
            "file": "test_data_structure.py",
            "description": "Validates preprocessed data structure"
        },
        {
            "name": "Model Prediction Test",
            "file": "test_prediction.py", 
            "description": "Tests basic model prediction functionality"
        },
        {
            "name": "Streamlit Features Test",
            "file": "test_streamlit_features.py",
            "description": "Tests feature creation for Streamlit app"
        },
        {
            "name": "Model Debug Test",
            "file": "debug_model.py",
            "description": "Advanced model debugging and validation"
        }
    ]
    
    # Track results
    passed = 0
    failed = 0
    
    # Run each test
    for test in tests:
        success = run_test(test["name"], test["file"], test["description"])
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! MediCodeAI is ready for use.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 