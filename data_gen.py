#!/usr/bin/env python3
"""
Synthetic EHR Data Generator for ICD-10 Billing Code Prediction

This script generates realistic synthetic EHR data for training and testing
the ICD-10 prediction model. All data is synthetic and contains no real PHI.
"""

import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import os

# Realistic medical conditions mapped to actual ICD-10 codes
MEDICAL_CONDITIONS = {
    # Cardiovascular Conditions
    "hypertension": {
        "codes": ["I10", "I11.9", "I12.9", "I13.9"],
        "symptoms": ["elevated blood pressure", "hypertension", "high blood pressure"],
        "notes": [
            "Blood pressure elevated at {bp}/95 mmHg. Patient reports no symptoms.",
            "Hypertension diagnosed. BP reading {bp}/95 mmHg on multiple visits.",
            "Essential hypertension noted. Patient asymptomatic."
        ]
    },
    "heart_failure": {
        "codes": ["I50.9", "I50.22", "I50.32", "I50.42"],
        "symptoms": ["shortness of breath", "fatigue", "swelling in legs", "chest discomfort"],
        "notes": [
            "Patient presents with shortness of breath and fatigue. Signs of heart failure.",
            "Congestive heart failure symptoms. Edema in lower extremities noted.",
            "Heart failure with reduced ejection fraction. Patient reports dyspnea on exertion."
        ]
    },
    "atrial_fibrillation": {
        "codes": ["I48.91", "I48.0", "I48.1", "I48.2"],
        "symptoms": ["irregular heartbeat", "palpitations", "dizziness", "fatigue"],
        "notes": [
            "Atrial fibrillation detected on EKG. Patient reports palpitations.",
            "Irregular heart rhythm noted. EKG shows atrial fibrillation.",
            "Patient with history of atrial fibrillation presents for follow-up."
        ]
    },
    "chest_pain": {
        "codes": ["R07.9", "I20.9", "I21.9", "I25.10"],
        "symptoms": ["chest pain", "chest pressure", "chest discomfort", "angina"],
        "notes": [
            "Patient complains of chest pain. EKG shows normal sinus rhythm.",
            "Chest pressure reported. Cardiac enzymes within normal limits.",
            "Angina symptoms. Stress test recommended for further evaluation."
        ]
    },
    
    # Respiratory Conditions
    "asthma": {
        "codes": ["J45.909", "J45.901", "J45.902", "J45.990"],
        "symptoms": ["wheezing", "shortness of breath", "cough", "chest tightness"],
        "notes": [
            "Asthma exacerbation. Patient reports increased wheezing and shortness of breath.",
            "Asthma symptoms worsening. Peak flow measurements decreased.",
            "Patient with asthma presents with respiratory distress."
        ]
    },
    "copd": {
        "codes": ["J44.9", "J44.0", "J44.1", "J44.8"],
        "symptoms": ["chronic cough", "shortness of breath", "sputum production", "fatigue"],
        "notes": [
            "COPD exacerbation. Increased sputum production and dyspnea.",
            "Chronic obstructive pulmonary disease. Patient reports worsening symptoms.",
            "COPD patient with acute exacerbation. Oxygen saturation decreased."
        ]
    },
    "pneumonia": {
        "codes": ["J18.9", "J15.9", "J44.0", "J90"],
        "symptoms": ["fever", "cough", "shortness of breath", "chest pain"],
        "notes": [
            "Community-acquired pneumonia. Chest X-ray shows infiltrate.",
            "Pneumonia diagnosed. Patient febrile with productive cough.",
            "Lower respiratory infection. Pneumonia confirmed on imaging."
        ]
    },
    
    # Endocrine Conditions
    "diabetes_type2": {
        "codes": ["E11.9", "E11.65", "E11.22", "E11.21"],
        "symptoms": ["increased thirst", "frequent urination", "fatigue", "blurred vision"],
        "notes": [
            "Type 2 diabetes mellitus. Blood glucose elevated at {bg} mg/dL.",
            "Diabetes management visit. HbA1c level {a1c}%.",
            "Patient with diabetes presents for routine follow-up."
        ]
    },
    "hyperthyroidism": {
        "codes": ["E05.90", "E05.00", "E05.01", "E05.02"],
        "symptoms": ["weight loss", "nervousness", "rapid heartbeat", "sweating"],
        "notes": [
            "Hyperthyroidism symptoms. TSH levels decreased, T4 elevated.",
            "Patient reports weight loss and nervousness. Thyroid function tests ordered.",
            "Graves' disease suspected. Patient with hyperthyroid symptoms."
        ]
    },
    "hypothyroidism": {
        "codes": ["E03.9", "E03.1", "E03.2", "E03.8"],
        "symptoms": ["fatigue", "weight gain", "cold intolerance", "depression"],
        "notes": [
            "Hypothyroidism diagnosed. TSH elevated, T4 decreased.",
            "Patient reports fatigue and weight gain. Thyroid function tests show hypothyroidism.",
            "Hashimoto's thyroiditis. Patient on thyroid replacement therapy."
        ]
    },
    
    # Gastrointestinal Conditions
    "gastritis": {
        "codes": ["K29.70", "K29.60", "K29.50", "K29.40"],
        "symptoms": ["abdominal pain", "nausea", "vomiting", "loss of appetite"],
        "notes": [
            "Gastritis symptoms. Patient reports epigastric pain and nausea.",
            "Acute gastritis diagnosed. Endoscopy shows gastric inflammation.",
            "Patient with gastritis presents with abdominal discomfort."
        ]
    },
    "gastroesophageal_reflux": {
        "codes": ["K21.9", "K21.0", "K21.00", "K21.01"],
        "symptoms": ["heartburn", "acid reflux", "chest pain", "regurgitation"],
        "notes": [
            "GERD symptoms. Patient reports frequent heartburn and acid reflux.",
            "Gastroesophageal reflux disease. Symptoms worse after meals.",
            "Patient with GERD presents for medication adjustment."
        ]
    },
    "irritable_bowel_syndrome": {
        "codes": ["K58.9", "K58.0", "K58.1", "K58.2"],
        "symptoms": ["abdominal pain", "bloating", "diarrhea", "constipation"],
        "notes": [
            "Irritable bowel syndrome. Patient reports alternating diarrhea and constipation.",
            "IBS symptoms. Abdominal pain and bloating after meals.",
            "Patient with IBS presents for symptom management."
        ]
    },
    
    # Neurological Conditions
    "migraine": {
        "codes": ["G43.909", "G43.109", "G43.009", "G43.809"],
        "symptoms": ["severe headache", "nausea", "light sensitivity", "vomiting"],
        "notes": [
            "Migraine headache. Patient reports severe unilateral headache with nausea.",
            "Classic migraine symptoms. Aura followed by severe headache.",
            "Patient with migraine presents for acute treatment."
        ]
    },
    "epilepsy": {
        "codes": ["G40.909", "G40.301", "G40.109", "G40.209"],
        "symptoms": ["seizures", "loss of consciousness", "confusion", "memory problems"],
        "notes": [
            "Epileptic seizure. Patient reports loss of consciousness and confusion.",
            "Seizure disorder. EEG shows epileptiform activity.",
            "Patient with epilepsy presents for medication adjustment."
        ]
    },
    "parkinsons_disease": {
        "codes": ["G20", "G21.9", "G21.0", "G21.1"],
        "symptoms": ["tremor", "rigidity", "bradykinesia", "postural instability"],
        "notes": [
            "Parkinson's disease. Patient shows resting tremor and bradykinesia.",
            "Progressive Parkinson's symptoms. Rigidity and postural instability noted.",
            "Patient with Parkinson's disease presents for routine follow-up."
        ]
    },
    
    # Mental Health Conditions
    "depression": {
        "codes": ["F32.9", "F32.1", "F32.2", "F33.2"],
        "symptoms": ["sadness", "loss of interest", "fatigue", "sleep problems"],
        "notes": [
            "Major depressive disorder. Patient reports persistent sadness and fatigue.",
            "Depression symptoms. Loss of interest in usual activities.",
            "Patient with depression presents for medication management."
        ]
    },
    "anxiety": {
        "codes": ["F41.1", "F41.0", "F41.9", "F41.8"],
        "symptoms": ["excessive worry", "nervousness", "panic attacks", "sleep problems"],
        "notes": [
            "Generalized anxiety disorder. Patient reports excessive worry and nervousness.",
            "Anxiety symptoms. Panic attacks occurring frequently.",
            "Patient with anxiety presents for therapy and medication."
        ]
    },
    "bipolar_disorder": {
        "codes": ["F31.9", "F31.1", "F31.2", "F31.6"],
        "symptoms": ["mood swings", "mania", "depression", "irritability"],
        "notes": [
            "Bipolar disorder. Patient reports alternating manic and depressive episodes.",
            "Bipolar symptoms. Recent manic episode with decreased need for sleep.",
            "Patient with bipolar disorder presents for mood stabilization."
        ]
    },
    
    # Musculoskeletal Conditions
    "osteoarthritis": {
        "codes": ["M16.9", "M17.9", "M15.9", "M19.90"],
        "symptoms": ["joint pain", "stiffness", "decreased range of motion", "swelling"],
        "notes": [
            "Osteoarthritis of knee. Patient reports pain and stiffness with activity.",
            "Degenerative joint disease. X-rays show joint space narrowing.",
            "Patient with osteoarthritis presents for pain management."
        ]
    },
    "back_pain": {
        "codes": ["M54.5", "M54.9", "M51.9", "M48.06"],
        "symptoms": ["lower back pain", "radiating pain", "stiffness", "muscle spasms"],
        "notes": [
            "Chronic low back pain. Patient reports pain radiating to left leg.",
            "Lumbar spine pain. MRI shows disc herniation at L4-L5.",
            "Patient with back pain presents for physical therapy evaluation."
        ]
    },
    "fibromyalgia": {
        "codes": ["M79.7", "M79.3", "M79.4", "M79.5"],
        "symptoms": ["widespread pain", "fatigue", "sleep problems", "cognitive issues"],
        "notes": [
            "Fibromyalgia symptoms. Patient reports widespread pain and fatigue.",
            "Chronic pain syndrome. Tender points noted on examination.",
            "Patient with fibromyalgia presents for pain management."
        ]
    },
    
    # Infectious Diseases
    "urinary_tract_infection": {
        "codes": ["N39.0", "N30.90", "N30.00", "N30.01"],
        "symptoms": ["frequent urination", "burning sensation", "urgency", "fever"],
        "notes": [
            "Urinary tract infection. Patient reports dysuria and frequency.",
            "UTI symptoms. Urinalysis shows bacteria and white blood cells.",
            "Patient with UTI presents for antibiotic treatment."
        ]
    },
    "upper_respiratory_infection": {
        "codes": ["J06.9", "J00", "J02.9", "J03.90"],
        "symptoms": ["sore throat", "cough", "congestion", "fever"],
        "notes": [
            "Upper respiratory infection. Patient reports sore throat and cough.",
            "Viral URI symptoms. No signs of bacterial infection.",
            "Patient with cold symptoms presents for symptomatic treatment."
        ]
    },
    "cellulitis": {
        "codes": ["L08.9", "L03.90", "L03.91", "L03.92"],
        "symptoms": ["redness", "swelling", "pain", "warmth"],
        "notes": [
            "Cellulitis of lower extremity. Patient reports redness and swelling.",
            "Skin infection. Erythema and warmth noted on examination.",
            "Patient with cellulitis presents for antibiotic treatment."
        ]
    }
}

def generate_realistic_icd10_codes() -> Tuple[List[str], str]:
    """Generate realistic ICD-10 codes based on actual medical conditions."""
    # Select 1-3 medical conditions
    num_conditions = random.randint(1, 3)
    selected_conditions = random.sample(list(MEDICAL_CONDITIONS.keys()), num_conditions)
    
    # Get codes for selected conditions
    all_codes = []
    for condition in selected_conditions:
        all_codes.extend(MEDICAL_CONDITIONS[condition]["codes"])
    
    # Select 1-3 codes from the available codes
    num_codes = random.randint(1, min(3, len(all_codes)))
    selected_codes = random.sample(all_codes, num_codes)
    
    return selected_codes, selected_conditions

def generate_realistic_medical_note(icd10_codes: List[str], conditions: List[str]) -> str:
    """Generate a realistic medical note based on actual medical conditions."""
    note_parts = []
    
    # Generate note for each condition
    for condition in conditions:
        condition_data = MEDICAL_CONDITIONS[condition]
        
        # Select a random note template
        note_template = random.choice(condition_data["notes"])
        
        # Fill in placeholders
        if "{bp}" in note_template:
            bp = random.randint(140, 180)
            note_template = note_template.replace("{bp}", str(bp))
        elif "{bg}" in note_template:
            bg = random.randint(150, 300)
            note_template = note_template.replace("{bg}", str(bg))
        elif "{a1c}" in note_template:
            a1c = round(random.uniform(7.0, 12.0), 1)
            note_template = note_template.replace("{a1c}", str(a1c))
        
        note_parts.append(note_template)
    
    # Add patient demographics and context
    age = random.randint(18, 95)
    gender = random.choice(["male", "female"])
    
    # Add examination findings
    exam_findings = [
        "Physical examination reveals no acute distress.",
        "Vital signs are stable.",
        "Patient appears alert and oriented.",
        "No significant findings on physical examination."
    ]
    
    # Add treatment plan
    treatment_plans = [
        "Plan: Continue current medications and follow up as scheduled.",
        "Plan: Prescribe appropriate medications and schedule follow-up.",
        "Plan: Recommend lifestyle modifications and monitor symptoms.",
        "Plan: Refer to specialist for further evaluation."
    ]
    
    # Combine all parts
    full_note = f"Patient is a {age}-year-old {gender}. "
    full_note += " ".join(note_parts) + " "
    full_note += random.choice(exam_findings) + " "
    full_note += random.choice(treatment_plans)
    
    return full_note

def generate_patient_data(num_records: int = 5000) -> pd.DataFrame:
    """Generate synthetic patient data with realistic medical notes and ICD-10 codes."""
    
    data = []
    
    for i in range(num_records):
        # Generate patient demographics
        patient_id = f"P{str(i+1).zfill(6)}"
        age = random.randint(18, 95)
        gender = random.choice(["M", "F"])
        
        # Generate diagnosis date (within last 2 years)
        days_ago = random.randint(1, 730)
        diagnosis_date = datetime.now() - timedelta(days=days_ago)
        
        # Generate realistic ICD-10 codes and conditions
        icd10_codes, conditions = generate_realistic_icd10_codes()
        
        # Generate realistic medical note based on conditions
        doctor_notes = generate_realistic_medical_note(icd10_codes, conditions)
        
        # Create record
        record = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "doctor_notes": doctor_notes,
            "diagnosis_date": diagnosis_date.strftime("%Y-%m-%d"),
            "icd10_codes": "|".join(icd10_codes),  # Pipe-separated for CSV
            "icd10_codes_list": icd10_codes,  # List for processing
            "medical_conditions": conditions  # For reference
        }
        
        data.append(record)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1:,} records...")
    
    return pd.DataFrame(data)

def validate_icd10_format(codes: List[str]) -> bool:
    """Validate ICD-10 code format using regex."""
    pattern = r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$'
    return all(re.match(pattern, code) for code in codes)

def save_data_with_metadata(df: pd.DataFrame, output_path: str):
    """Save data with metadata for reproducibility."""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save main data
    df.to_csv(output_path, index=False)
    
    # Calculate code distribution
    all_codes = []
    for codes in df['icd10_codes_list']:
        all_codes.extend(codes)
    
    code_counts = pd.Series(all_codes).value_counts()
    
    # Save metadata
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "num_records": int(len(df)),
        "total_unique_codes": int(len(set(all_codes))),
        "age_range": f"{int(df['age'].min())}-{int(df['age'].max())}",
        "gender_distribution": df['gender'].value_counts().to_dict(),
        "code_distribution": code_counts.head(20).to_dict(),  # Top 20 codes
        "validation": {
            "all_icd10_valid": bool(all(validate_icd10_format(codes) for codes in df['icd10_codes_list'])),
            "no_null_notes": bool(df['doctor_notes'].notna().all()),
            "no_empty_notes": bool((df['doctor_notes'].str.len() > 0).all())
        }
    }
    
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create metrics file for DVC
    metrics = {
        "data_generation": {
            "total_records": int(len(df)),
            "unique_icd10_codes": int(len(set(all_codes))),
            "avg_codes_per_record": round(len(all_codes) / len(df), 2),
            "data_quality_score": 1.0 if metadata['validation']['all_icd10_valid'] else 0.0,
            "completeness_score": 1.0 if metadata['validation']['no_null_notes'] else 0.0
        },
        "demographics": {
            "age_mean": float(df['age'].mean()),
            "age_std": float(df['age'].std()),
            "gender_balance": float(df['gender'].value_counts().min() / df['gender'].value_counts().max())
        },
        "icd10_distribution": {
            "top_5_codes": code_counts.head(5).to_dict(),
            "code_diversity": float(len(set(all_codes)) / len(all_codes))
        }
    }
    
    metrics_path = "data/raw/data_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create plots file for DVC
    create_data_distribution_plots(df, code_counts)
    
    print(f"✅ Generated {len(df):,} synthetic EHR records")
    print(f"📁 Data saved to: {output_path}")
    print(f"📊 Metadata saved to: {metadata_path}")
    print(f"📈 Metrics saved to: {metrics_path}")
    print(f"📊 Plots saved to: data/raw/data_distribution.html")
    print(f"🔍 Validation: {metadata['validation']}")
    print(f"📈 Top 5 ICD-10 codes: {list(code_counts.head(5).items())}")

def create_data_distribution_plots(df: pd.DataFrame, code_counts: pd.Series):
    """Create interactive plots for data distribution."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'Gender Distribution', 
                           'Top 10 ICD-10 Codes', 'Codes per Record'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=df['age'], name='Age Distribution', nbinsx=20),
            row=1, col=1
        )
        
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        fig.add_trace(
            go.Pie(labels=gender_counts.index, values=gender_counts.values, name='Gender'),
            row=1, col=2
        )
        
        # Top 10 ICD-10 codes
        top_10_codes = code_counts.head(10)
        fig.add_trace(
            go.Bar(x=top_10_codes.index, y=top_10_codes.values, name='Top 10 Codes'),
            row=2, col=1
        )
        
        # Codes per record distribution
        codes_per_record = [len(codes) for codes in df['icd10_codes_list']]
        fig.add_trace(
            go.Histogram(x=codes_per_record, name='Codes per Record', nbinsx=10),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Synthetic EHR Data Distribution",
            showlegend=False
        )
        
        # Save as HTML
        plots_path = "data/raw/data_distribution.html"
        fig.write_html(plots_path)
        
    except ImportError:
        # Fallback if plotly is not available
        print("⚠️ Plotly not available, skipping plots generation")
        plots_path = "data/raw/data_distribution.html"
        with open(plots_path, 'w') as f:
            f.write("<html><body><h1>Data Distribution Plots</h1><p>Plots not generated - plotly not available</p></body></html>")

def main():
    """Main function to generate synthetic EHR data."""
    print("🏥 Generating Realistic Synthetic EHR Data for ICD-10 Prediction")
    print("=" * 70)
    print("📊 Target: 100,000 records with realistic medical conditions")
    print("🎯 Using actual ICD-10 codes mapped to real medical problems")
    print("=" * 70)
    
    # Generate data
    df = generate_patient_data(num_records=5000)
    
    # Save to data/raw directory
    output_path = "data/raw/synthetic_ehr_data.csv"
    save_data_with_metadata(df, output_path)

if __name__ == "__main__":
    main() 