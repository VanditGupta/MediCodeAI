#!/usr/bin/env python3
"""
Streamlit Frontend for ICD-10 Prediction
Interactive web application for clinical note analysis

This Streamlit app provides a user-friendly interface for predicting
ICD-10 billing codes from clinical notes with real-time feedback.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Import utility functions
from app.utils import ICD10PredictorAPI

# Page configuration
st.set_page_config(
    page_title="ICD-10 Code Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get cached API client instance."""
    return ICD10PredictorAPI()

# =============================================================================
# Sidebar Configuration
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # API Configuration
    st.sidebar.subheader("🔗 API Settings")
    api_url = st.sidebar.text_input(
        "API Endpoint",
        value=st.secrets.get("API_URL", "http://localhost:8000"),
        help="URL of the ICD-10 prediction API"
    )
    
    # Model Configuration
    st.sidebar.subheader("🤖 Model Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for code predictions"
    )
    
    # Display Options
    st.sidebar.subheader("📊 Display Options")
    show_confidence_bars = st.sidebar.checkbox(
        "Show Confidence Bars",
        value=True,
        help="Display confidence scores as progress bars"
    )
    
    show_processing_time = st.sidebar.checkbox(
        "Show Processing Time",
        value=True,
        help="Display API processing time"
    )
    
    # About Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.markdown("""
    **ICD-10 Code Predictor**
    
    This application uses advanced NLP and machine learning to predict ICD-10 billing codes from clinical notes.
    
    **Features:**
    - ClinicalBERT + XGBoost ensemble
    - Multi-label classification
    - Real-time predictions
    - HIPAA-compliant processing
    """)
    
    return {
        'api_url': api_url,
        'confidence_threshold': confidence_threshold,
        'show_confidence_bars': show_confidence_bars,
        'show_processing_time': show_processing_time
    }

# =============================================================================
# Main Application
# =============================================================================

def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🏥 ICD-10 Code Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered billing code prediction from clinical notes</p>', unsafe_allow_html=True)

def render_input_section():
    """Render the input section for clinical notes."""
    st.subheader("📝 Clinical Notes Input")
    
    # Text area for clinical notes
    doctor_notes = st.text_area(
        "Enter clinical notes:",
        height=200,
        placeholder="Enter the patient's clinical notes here...\n\nExample: Patient presents with chest pain and shortness of breath for the past 2 days. Examination reveals normal heart sounds and clear lung fields. EKG shows normal sinus rhythm.",
        help="Enter detailed clinical notes describing the patient's symptoms, examination findings, and diagnosis."
    )
    
    # Patient demographics
    col1, col2 = st.columns(2)
    
    with col1:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=50,
            help="Patient age (optional, helps improve prediction accuracy)"
        )
    
    with col2:
        patient_gender = st.selectbox(
            "Patient Gender",
            options=["", "M", "F"],
            help="Patient gender (optional)"
        )
    
    return {
        'doctor_notes': doctor_notes,
        'patient_age': patient_age if patient_age > 0 else None,
        'patient_gender': patient_gender if patient_gender else None
    }

def render_prediction_results(results: Dict[str, Any], config: Dict[str, Any]):
    """Render the prediction results."""
    st.subheader("🎯 Prediction Results")
    
    if not results or 'error' in results:
        st.error(f"❌ Prediction failed: {results.get('error', 'Unknown error')}")
        return
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Predicted Codes",
            len(results.get('predicted_codes', []))
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_confidence = sum(results.get('confidence_scores', [])) / len(results.get('confidence_scores', [])) if results.get('confidence_scores') else 0
        st.metric(
            "Avg Confidence",
            f"{avg_confidence:.2%}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if config['show_processing_time']:
            st.metric(
                "Processing Time",
                f"{results.get('processing_time', 0):.3f}s"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Model Version",
            results.get('model_version', 'Unknown')
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display predicted codes
    if results.get('predicted_codes'):
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.write("**Predicted ICD-10 Codes:**")
        
        # Create DataFrame for better display
        df = pd.DataFrame({
            'ICD-10 Code': results['predicted_codes'],
            'Confidence': [f"{score:.2%}" for score in results['confidence_scores']],
            'Confidence_Value': results['confidence_scores']
        })
        
        # Sort by confidence
        df = df.sort_values('Confidence_Value', ascending=False)
        
        # Display codes with confidence bars
        for _, row in df.iterrows():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.write(f"**{row['ICD-10 Code']}**")
            
            with col2:
                if config['show_confidence_bars']:
                    confidence_pct = row['Confidence_Value'] * 100
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write(row['Confidence'])
            
            with col3:
                # Color code based on confidence
                if row['Confidence_Value'] >= 0.8:
                    st.success("High")
                elif row['Confidence_Value'] >= 0.6:
                    st.warning("Medium")
                else:
                    st.info("Low")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        render_visualizations(df)
    else:
        st.info("ℹ️ No ICD-10 codes predicted above the confidence threshold. Try lowering the threshold or providing more detailed clinical notes.")

def render_visualizations(df: pd.DataFrame):
    """Render visualizations of the prediction results."""
    st.subheader("📊 Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Confidence Distribution", "Code Categories"])
    
    with tab1:
        # Confidence distribution chart
        fig = px.bar(
            df,
            x='ICD-10 Code',
            y='Confidence_Value',
            title="Prediction Confidence by ICD-10 Code",
            color='Confidence_Value',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Code category analysis
        df['Category'] = df['ICD-10 Code'].str[0]
        category_counts = df['Category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="ICD-10 Code Categories Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_history():
    """Render prediction history."""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.prediction_history:
        st.subheader("📚 Prediction History")
        
        # Create DataFrame from history
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display recent predictions
        for i, record in enumerate(history_df.tail(5).iterrows()):
            with st.expander(f"Prediction {len(history_df) - i} - {record[1]['timestamp']}"):
                st.write(f"**Notes:** {record[1]['notes'][:100]}...")
                st.write(f"**Codes:** {', '.join(record[1]['codes'])}")
                st.write(f"**Avg Confidence:** {record[1]['avg_confidence']:.2%}")

def render_api_status(api_client: ICD10PredictorAPI):
    """Render API status information."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 API Status")
    
    try:
        health = api_client.health_check()
        if health.get('status') == 'healthy':
            st.sidebar.success("✅ API Healthy")
        else:
            st.sidebar.error("❌ API Unhealthy")
        
        # Model info
        model_info = api_client.get_model_info()
        if model_info:
            st.sidebar.write(f"**Model:** {model_info.get('model_type', 'Unknown')}")
            st.sidebar.write(f"**Classes:** {model_info.get('num_classes', 0)}")
            st.sidebar.write(f"**Version:** {model_info.get('version', 'Unknown')}")
    
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {str(e)}")

# =============================================================================
# Main Application Flow
# =============================================================================

def main():
    """Main application function."""
    # Render header
    render_header()
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Initialize API client
    api_client = get_api_client()
    api_client.base_url = config['api_url']
    
    # Render API status
    render_api_status(api_client)
    
    # Render input section
    input_data = render_input_section()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 Predict ICD-10 Codes",
            type="primary",
            use_container_width=True
        )
    
    # Handle prediction
    if predict_button and input_data['doctor_notes']:
        with st.spinner("🤖 Analyzing clinical notes..."):
            try:
                # Make prediction
                results = api_client.predict(
                    doctor_notes=input_data['doctor_notes'],
                    patient_age=input_data['patient_age'],
                    patient_gender=input_data['patient_gender'],
                    confidence_threshold=config['confidence_threshold']
                )
                
                # Store in history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'notes': input_data['doctor_notes'],
                    'codes': results.get('predicted_codes', []),
                    'avg_confidence': sum(results.get('confidence_scores', [])) / len(results.get('confidence_scores', [])) if results.get('confidence_scores') else 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Render results
                render_prediction_results(results, config)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    elif predict_button and not input_data['doctor_notes']:
        st.warning("⚠️ Please enter clinical notes before making a prediction.")
    
    # Render history
    render_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>🏥 ICD-10 Code Predictor | HIPAA-Compliant AI for Medical Coding</p>
        <p>Built with ClinicalBERT + XGBoost | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 