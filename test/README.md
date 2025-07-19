# MediCodeAI Test Suite

This directory contains all testing utilities and scripts for the MediCodeAI project.

## 📁 Test Structure

```
test/
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── run_all_tests.py              # Master test runner
├── test_integration.py           # Integration tests
├── test_prediction.py            # Basic model prediction tests
├── test_streamlit_features.py    # Streamlit feature creation tests
├── test_data_structure.py        # Data structure validation
├── debug_model.py                # Model debugging utilities
└── inspect_preprocessed.py       # Data inspection utilities (PySpark)
```

## 🚀 Quick Start

### Run All Tests
```bash
python test/run_all_tests.py
```

### Run Individual Tests
```bash
# Basic prediction test
python test/test_prediction.py

# Streamlit features test
python test/test_streamlit_features.py

# Integration test
python test/test_integration.py

# Model debugging
python test/debug_model.py

# Data structure validation
python test/test_data_structure.py

# Data inspection (requires PySpark)
python test/inspect_preprocessed.py

# Download models (one-time setup)
python download_models.py
```

## 🧪 Test Categories

### 1. **Unit Tests**
- `test_prediction.py`: Tests basic model prediction functionality
- `test_streamlit_features.py`: Tests feature creation for Streamlit app

### 2. **Integration Tests**
- `test_integration.py`: Tests complete pipeline from data to prediction
- `run_all_tests.py`: Master test runner that executes all tests

### 3. **Utility Tests**
- `debug_model.py`: Advanced model debugging and validation
- `test_data_structure.py`: Data structure validation (no PySpark required)
- `inspect_preprocessed.py`: Advanced data inspection (requires PySpark)

## 📊 Test Coverage

The test suite covers:

- ✅ **Data Pipeline**: Raw data loading, preprocessing validation
- ✅ **Model Pipeline**: Model loading, component validation
- ✅ **Prediction Pipeline**: End-to-end prediction testing
- ✅ **Feature Engineering**: BERT + engineered features
- ✅ **Streamlit Integration**: App import and functionality
- ✅ **Model Performance**: Accuracy and prediction validation

## 🔧 Test Configuration

### Environment Variables
```bash
# Enable internet access for model downloads
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# Set model paths
export MODEL_PATH=model/saved_model
export DATA_PATH=data/preprocessed
```

### Test Dependencies
All tests require the main project dependencies from `requirements.txt`.

## 📈 Test Results

### Expected Output
```
🚀 Starting MediCodeAI Test Suite
============================================================

🧪 Running: Model Download Test
📝 Description: Downloads and validates BERT models
============================================================
✅ Model Download Test PASSED (45.23s)

🧪 Running: Data Inspection Test
📝 Description: Validates preprocessed data structure
============================================================
✅ Data Inspection Test PASSED (0.12s)

...

📊 TEST SUMMARY
============================================================
✅ Passed: 5
❌ Failed: 0
📈 Success Rate: 100.0%

🎉 All tests passed! MediCodeAI is ready for use.
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model Download Failures**
   ```bash
   # Set environment variables
   export HF_HUB_OFFLINE=0
   export TRANSFORMERS_OFFLINE=0
   
   # Run download utility (one-time setup)
   python download_models.py
   ```

2. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/MediCodeAI
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size in model configuration
   # Edit model/train_model.py: batch_size = 8
   ```

### Debug Mode
Run tests with verbose output:
```bash
python -v test/run_all_tests.py
```

## 📝 Adding New Tests

1. Create test file in `test/` directory
2. Follow naming convention: `test_*.py`
3. Add to `run_all_tests.py` if it should be part of the main suite
4. Update this README with test description

### Test Template
```python
#!/usr/bin/env python3
"""
Test Description
"""

def test_function():
    """Test description."""
    try:
        # Test logic here
        print("✅ Test passed")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_function()
    exit(0 if success else 1)
```

## 🤝 Contributing

When adding new features to MediCodeAI:

1. **Write tests first** (TDD approach)
2. **Update test suite** to include new functionality
3. **Ensure all tests pass** before committing
4. **Document new tests** in this README

## 📞 Support

For test-related issues:
1. Check the troubleshooting section above
2. Review test output for specific error messages
3. Ensure all dependencies are installed
4. Verify data and model files exist 