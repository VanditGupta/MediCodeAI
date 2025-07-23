#!/usr/bin/env python3
"""
Model Evaluation Script for DVC Pipeline
Evaluates the trained ICD-10 prediction model and generates metrics and plots
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    hamming_loss, multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.train_model import load_model, preprocess_text

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from parquet file."""
    print(f"📊 Loading test data from {test_path}")
    df = pd.read_parquet(test_path)
    print(f"✅ Loaded {len(df)} test records")
    return df

def load_trained_model(model_dir: str, xgboost_path: str):
    """Load the trained BERT and XGBoost models."""
    print(f"🤖 Loading trained models from {model_dir}")
    
    # Load BERT model and tokenizer
    bert_model, tokenizer = load_model(model_dir)
    
    # Load XGBoost model
    with open(xgboost_path, 'rb') as f:
        xgb_model = pickle.load(f)
    
    print("✅ Models loaded successfully")
    return bert_model, tokenizer, xgb_model

def extract_bert_features(texts: List[str], bert_model, tokenizer) -> np.ndarray:
    """Extract BERT features from text."""
    print("🔍 Extracting BERT features...")
    
    features = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_features = preprocess_text(batch_texts, bert_model, tokenizer)
        features.append(batch_features)
        
        if (i + batch_size) % 1000 == 0:
            print(f"   Processed {i + batch_size}/{len(texts)} texts")
    
    return np.vstack(features)

def predict_icd10_codes(texts: List[str], bert_model, tokenizer, xgb_model, 
                       mlb: MultiLabelBinarizer, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Predict ICD-10 codes for given texts."""
    print("🎯 Making predictions...")
    
    # Extract BERT features
    bert_features = extract_bert_features(texts, bert_model, tokenizer)
    
    # Make predictions with XGBoost
    predictions_proba = xgb_model.predict_proba(bert_features)
    
    # Convert to binary predictions using threshold
    predictions_binary = (predictions_proba > threshold).astype(int)
    
    return predictions_binary, predictions_proba

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    print("📈 Calculating metrics...")
    
    metrics = {}
    
    # Overall metrics
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # Per-label metrics
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-label accuracy
    per_label_accuracy = np.mean(y_true == y_pred, axis=0)
    metrics['per_label_accuracy_mean'] = np.mean(per_label_accuracy)
    metrics['per_label_accuracy_std'] = np.std(per_label_accuracy)
    metrics['per_label_accuracy_min'] = np.min(per_label_accuracy)
    metrics['per_label_accuracy_max'] = np.max(per_label_accuracy)
    
    # ROC AUC (if possible)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='micro')
        metrics['roc_auc_micro'] = roc_auc
    except:
        metrics['roc_auc_micro'] = None
    
    return metrics

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: List[str], output_path: str):
    """Create and save confusion matrix plot."""
    print("📊 Creating confusion matrix...")
    
    # Calculate confusion matrix for each label
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    # Create subplot for confusion matrices
    n_labels = min(9, len(class_names))  # Show first 9 labels
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i in range(n_labels):
        sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'],
                   ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {class_names[i]}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Hide empty subplots
    for i in range(n_labels, 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {output_path}")

def create_roc_curves_plot(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          class_names: List[str], output_path: str):
    """Create and save ROC curves plot."""
    print("📈 Creating ROC curves...")
    
    from sklearn.metrics import roc_curve
    
    # Create subplot for ROC curves
    n_labels = min(9, len(class_names))  # Show first 9 labels
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i in range(n_labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        
        axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'ROC Curve - {class_names[i]}')
        axes[i].legend(loc="lower right")
        axes[i].grid(True)
    
    # Hide empty subplots
    for i in range(n_labels, 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved to {output_path}")

def create_evaluation_plots_html(metrics: Dict, output_path: str):
    """Create interactive HTML plots for evaluation results."""
    print("📊 Creating interactive evaluation plots...")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Metrics', 'Per-Label Accuracy Distribution', 
                       'Precision-Recall-F1 Scores', 'Model Performance Summary'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # Overall metrics bar chart
    overall_metrics = ['overall_accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
    overall_values = [metrics.get(m, 0) for m in overall_metrics]
    
    fig.add_trace(
        go.Bar(x=overall_metrics, y=overall_values, name='Overall Metrics'),
        row=1, col=1
    )
    
    # Per-label accuracy histogram
    fig.add_trace(
        go.Histogram(x=[metrics['per_label_accuracy_mean']], name='Accuracy Distribution'),
        row=1, col=2
    )
    
    # Precision-Recall-F1 comparison
    comparison_metrics = ['precision_micro', 'recall_micro', 'f1_micro']
    comparison_values = [metrics.get(m, 0) for m in comparison_metrics]
    
    fig.add_trace(
        go.Bar(x=comparison_metrics, y=comparison_values, name='Micro Scores'),
        row=2, col=1
    )
    
    # Performance summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Overall Accuracy', f"{metrics.get('overall_accuracy', 0):.4f}"],
        ['Hamming Loss', f"{metrics.get('hamming_loss', 0):.4f}"],
        ['F1 Score (Micro)', f"{metrics.get('f1_micro', 0):.4f}"],
        ['ROC AUC (Micro)', f"{metrics.get('roc_auc_micro', 'N/A')}"],
        ['Per-Label Accuracy (Mean)', f"{metrics.get('per_label_accuracy_mean', 0):.4f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=summary_data[0], fill_color='paleturquoise', align='left'),
            cells=dict(values=list(zip(*summary_data[1:])), fill_color='lavender', align='left')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="ICD-10 Prediction Model Evaluation Results",
        showlegend=False
    )
    
    # Save as HTML
    fig.write_html(output_path)
    print(f"✅ Interactive plots saved to {output_path}")

def save_evaluation_results(metrics: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    print(f"💾 Saving evaluation results to {output_path}")
    
    # Add metadata
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'metrics': metrics,
        'model_info': {
            'model_type': 'ClinicalBERT + XGBoost Ensemble',
            'evaluation_set': 'Test Set'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Evaluation results saved")

def main():
    """Main evaluation function."""
    print("🔍 ICD-10 Prediction Model Evaluation")
    print("=" * 50)
    
    # File paths
    test_data_path = "data/test/test_data.parquet"
    model_dir = "models/clinical_bert_model/"
    xgboost_path = "models/xgboost_model.pkl"
    
    # Output paths
    os.makedirs("artifacts", exist_ok=True)
    results_path = "artifacts/evaluation_results.json"
    metrics_path = "artifacts/evaluation_metrics.json"
    confusion_matrix_path = "artifacts/confusion_matrix.png"
    roc_curves_path = "artifacts/roc_curves.png"
    plots_path = "artifacts/evaluation_plots.html"
    
    # Load test data
    test_df = load_test_data(test_data_path)
    
    # Load models
    bert_model, tokenizer, xgb_model = load_trained_model(model_dir, xgboost_path)
    
    # Prepare data
    texts = test_df['doctor_notes'].tolist()
    
    # Convert ICD-10 codes to binary format
    icd10_codes = [codes.split('|') if isinstance(codes, str) else [] 
                   for codes in test_df['icd10_codes']]
    
    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(icd10_codes)
    class_names = mlb.classes_
    
    print(f"📊 Test set: {len(test_df)} samples, {len(class_names)} ICD-10 codes")
    
    # Make predictions
    y_pred, y_pred_proba = predict_icd10_codes(texts, bert_model, tokenizer, xgb_model, mlb)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print summary
    print("\n📈 Evaluation Summary:")
    print(f"   Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"   Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"   F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"   Precision (Micro): {metrics['precision_micro']:.4f}")
    print(f"   Recall (Micro): {metrics['recall_micro']:.4f}")
    
    # Create visualizations
    create_confusion_matrix_plot(y_true, y_pred, class_names, confusion_matrix_path)
    create_roc_curves_plot(y_true, y_pred_proba, class_names, roc_curves_path)
    create_evaluation_plots_html(metrics, plots_path)
    
    # Save results
    save_evaluation_results(metrics, results_path)
    
    # Save metrics for DVC
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved to artifacts/ directory")

if __name__ == "__main__":
    main() 