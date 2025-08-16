#!/usr/bin/env python3
""" 
Utility Functions for ICD-10 Prediction App
API communication and data processing utilities

This module provides utility functions for the Streamlit frontend
to communicate with the ICD-10 prediction API.
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICD10PredictorAPI:
    """API client for ICD-10 prediction service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ICD10-Predictor-App/1.0.0'
        })
        
        # Timeout settings
        self.timeout = 30
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise Exception("Request timeout - the API is taking too long to respond")
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for {url}")
            raise Exception("Connection error - cannot reach the API server")
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}")
            try:
                error_data = e.response.json()
                error_message = error_data.get('detail', error_data.get('error', str(e)))
            except:
                error_message = str(e)
            raise Exception(f"API error: {error_message}")
        
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from {url}")
            raise Exception("Invalid response from API server")
        
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request('GET', '/health')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self._make_request('GET', '/model-info')
    
    def get_available_codes(self) -> Dict[str, Any]:
        """Get list of available ICD-10 codes."""
        return self._make_request('GET', '/available-codes')
    
    def predict(self, doctor_notes: str, patient_age: Optional[int] = None,
                patient_gender: Optional[str] = None, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Make ICD-10 code prediction."""
        data = {
            'doctor_notes': doctor_notes,
            'confidence_threshold': confidence_threshold
        }
        
        if patient_age is not None:
            data['patient_age'] = patient_age
        
        if patient_gender is not None:
            data['patient_gender'] = patient_gender
        
        return self._make_request('POST', '/predict', data)

class DataProcessor:
    """Utility class for data processing and validation."""
    
    @staticmethod
    def validate_icd10_code(code: str) -> bool:
        """Validate ICD-10 code format."""
        import re
        pattern = r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$'
        return bool(re.match(pattern, code))
    
    @staticmethod
    def clean_clinical_text(text: str) -> str:
        """Clean and normalize clinical text."""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        return text
    
    @staticmethod
    def extract_medical_entities(text: str) -> Dict[str, List[str]]:
        """Extract medical entities from clinical text."""
        import re
        
        entities = {
            'symptoms': [],
            'body_parts': [],
            'medications': [],
            'procedures': []
        }
        
        # Medical entity patterns
        patterns = {
            'symptoms': r'\b(pain|discomfort|pressure|burning|nausea|dizziness|fatigue|weakness|shortness of breath|chest pain)\b',
            'body_parts': r'\b(chest|abdomen|head|back|legs|arms|neck|shoulder|knee|hip|throat|stomach|heart|lungs)\b',
            'medications': r'\b(aspirin|ibuprofen|acetaminophen|antibiotics|insulin|metformin)\b',
            'procedures': r'\b(ekg|ecg|x-ray|mri|ct scan|blood test|surgery|examination)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            entities[entity_type] = list(set(matches))
        
        return entities
    
    @staticmethod
    def calculate_text_metrics(text: str) -> Dict[str, Any]:
        """Calculate text metrics for analysis."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0,
            'unique_words': len(set(words)),
            'text_length': len(text)
        }

class VisualizationHelper:
    """Helper class for creating visualizations."""
    
    @staticmethod
    def create_confidence_chart(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create confidence chart data."""
        codes = [pred['code'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        
        return {
            'x': codes,
            'y': confidences,
            'type': 'bar',
            'marker': {
                'color': confidences,
                'colorscale': 'RdYlGn'
            }
        }
    
    @staticmethod
    def create_category_pie_chart(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create category pie chart data."""
        categories = {}
        for pred in predictions:
            category = pred['code'][0]  # First letter of ICD-10 code
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return {
            'labels': list(categories.keys()),
            'values': list(categories.values()),
            'type': 'pie'
        }
    
    @staticmethod
    def create_timeline_chart(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create timeline chart for prediction history."""
        timestamps = [entry['timestamp'] for entry in history]
        avg_confidences = [entry['avg_confidence'] for entry in history]
        
        return {
            'x': timestamps,
            'y': avg_confidences,
            'type': 'scatter',
            'mode': 'lines+markers'
        }

class ErrorHandler:
    """Error handling utilities."""
    
    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format error message for user display."""
        error_type = type(error).__name__
        
        if "timeout" in str(error).lower():
            return "The request timed out. Please try again or check your connection."
        elif "connection" in str(error).lower():
            return "Cannot connect to the API server. Please check if the service is running."
        elif "api error" in str(error).lower():
            return str(error)
        else:
            return f"An unexpected error occurred: {str(error)}"
    
    @staticmethod
    def log_error(error: Exception, context: str = ""):
        """Log error with context."""
        logger.error(f"Error in {context}: {str(error)}")
        logger.error(f"Error type: {type(error).__name__}")

class CacheManager:
    """Simple cache manager for API responses."""
    
    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache with TTL."""
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()

# Global cache instance
api_cache = CacheManager()

def get_cached_api_response(cache_key: str, api_call_func, *args, **kwargs):
    """Get API response with caching."""
    cached_result = api_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    result = api_call_func(*args, **kwargs)
    api_cache.set(cache_key, result)
    return result

def format_prediction_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction results for display."""
    if not results or 'error' in results:
        return results
    
    # Add formatted confidence scores
    if 'confidence_scores' in results:
        results['confidence_percentages'] = [
            f"{score:.1%}" for score in results['confidence_scores']
        ]
    
    # Add confidence levels
    if 'confidence_scores' in results:
        results['confidence_levels'] = []
        for score in results['confidence_scores']:
            if score >= 0.8:
                results['confidence_levels'].append('High')
            elif score >= 0.6:
                results['confidence_levels'].append('Medium')
            else:
                results['confidence_levels'].append('Low')
    
    return results

def validate_input_data(doctor_notes: str, patient_age: Optional[int] = None) -> List[str]:
    """Validate input data and return list of errors."""
    errors = []
    
    if not doctor_notes or len(doctor_notes.strip()) < 10:
        errors.append("Clinical notes must be at least 10 characters long")
    
    if patient_age is not None and (patient_age < 0 or patient_age > 120):
        errors.append("Patient age must be between 0 and 120")
    
    return errors

def create_sample_data() -> Dict[str, Any]:
    """Create sample data for testing."""
    return {
        'doctor_notes': """Patient presents with chest pain and shortness of breath for the past 2 days. 
        Examination reveals normal heart sounds and clear lung fields. EKG shows normal sinus rhythm. 
        Patient reports mild fatigue and occasional dizziness. Blood pressure is 140/90 mmHg.""",
        'patient_age': 65,
        'patient_gender': 'M',
        'confidence_threshold': 0.5
    } 