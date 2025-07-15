#!/usr/bin/env python3
"""
MLflow Tracking Module for ICD-10 Prediction Model
Experiment Tracking and Model Management

This module provides MLflow integration for tracking experiments,
logging metrics, parameters, and artifacts in the MLOps pipeline.
"""

import os
import json
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow tracking wrapper for experiment management."""
    
    def __init__(self, 
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "icd10-prediction",
                 artifact_location: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            artifact_location: S3 location for artifacts
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location
                )
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"✅ MLflow experiment '{self.experiment_name}' configured")
        except Exception as e:
            logger.warning(f"⚠️ Could not configure MLflow experiment: {e}")
            logger.info("📝 Using default experiment")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        run_name = run_name or f"icd10-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Default tags
        default_tags = {
            "project": "icd10-prediction",
            "model_type": "clinical-bert-xgboost",
            "environment": os.getenv('ENVIRONMENT', 'development'),
            "version": "1.0.0"
        }
        
        if tags:
            default_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=default_tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
            logger.info(f"📊 Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"❌ Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics)
            logger.info(f"📈 Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"❌ Error logging metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to MLflow."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"📦 Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"❌ Error logging artifact: {e}")
    
    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None):
        """Log model to MLflow model registry."""
        try:
            if hasattr(model, 'predict_proba'):
                # Scikit-learn compatible model
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path, 
                    registered_model_name=registered_model_name
                )
            elif hasattr(model, 'state_dict'):
                # PyTorch model
                mlflow.pytorch.log_model(
                    model, 
                    artifact_path, 
                    registered_model_name=registered_model_name
                )
            else:
                # Generic model logging
                mlflow.log_artifact(model, artifact_path)
            
            logger.info(f"🤖 Logged model: {artifact_path}")
        except Exception as e:
            logger.error(f"❌ Error logging model: {e}")
    
    def log_data_quality_metrics(self, quality_report: Dict[str, Any]):
        """Log data quality metrics."""
        try:
            # Extract numeric metrics
            numeric_metrics = {}
            for key, value in quality_report.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[f"data_quality_{key}"] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            numeric_metrics[f"data_quality_{key}_{sub_key}"] = sub_value
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
                logger.info(f"📊 Logged {len(numeric_metrics)} data quality metrics")
            
            # Log full report as artifact
            with open("data_quality_report.json", "w") as f:
                json.dump(quality_report, f, indent=2)
            mlflow.log_artifact("data_quality_report.json", "data_quality")
            
        except Exception as e:
            logger.error(f"❌ Error logging data quality metrics: {e}")
    
    def log_model_performance(self, performance_metrics: Dict[str, Any]):
        """Log comprehensive model performance metrics."""
        try:
            # Log overall metrics
            overall_metrics = {
                'overall_accuracy': performance_metrics.get('overall_accuracy', 0),
                'hamming_loss': performance_metrics.get('hamming_loss', 0),
                'f1_micro': performance_metrics.get('f1_micro', 0),
                'precision_micro': performance_metrics.get('precision_micro', 0),
                'recall_micro': performance_metrics.get('recall_micro', 0)
            }
            mlflow.log_metrics(overall_metrics)
            
            # Log per-label metrics
            per_label_metrics = {}
            for key, value in performance_metrics.items():
                if key.startswith('accuracy_') and isinstance(value, (int, float)):
                    per_label_metrics[key] = value
            
            if per_label_metrics:
                mlflow.log_metrics(per_label_metrics)
            
            # Log detailed performance report
            with open("model_performance_report.json", "w") as f:
                json.dump(performance_metrics, f, indent=2)
            mlflow.log_artifact("model_performance_report.json", "performance")
            
            logger.info(f"📈 Logged model performance metrics")
            
        except Exception as e:
            logger.error(f"❌ Error logging model performance: {e}")
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        try:
            # Log config as parameters
            mlflow.log_params(config)
            
            # Log full config as artifact
            with open("training_config.json", "w") as f:
                json.dump(config, f, indent=2)
            mlflow.log_artifact("training_config.json", "config")
            
            logger.info(f"⚙️ Logged training configuration")
            
        except Exception as e:
            logger.error(f"❌ Error logging training config: {e}")
    
    def log_feature_importance(self, feature_importance: Dict[str, float], model_name: str = "xgboost"):
        """Log feature importance metrics."""
        try:
            # Log top features as metrics
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:20]  # Top 20 features
            
            for feature, importance in top_features:
                mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Log full feature importance as artifact
            with open("feature_importance.json", "w") as f:
                json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact("feature_importance.json", "feature_importance")
            
            logger.info(f"🔍 Logged feature importance for {len(feature_importance)} features")
            
        except Exception as e:
            logger.error(f"❌ Error logging feature importance: {e}")
    
    def log_hyperparameter_tuning(self, trial_results: Dict[str, Any]):
        """Log hyperparameter tuning results."""
        try:
            # Log best parameters
            if 'best_params' in trial_results:
                mlflow.log_params(trial_results['best_params'])
            
            # Log optimization metrics
            if 'best_score' in trial_results:
                mlflow.log_metric('best_cv_score', trial_results['best_score'])
            
            # Log full tuning results
            with open("hyperparameter_tuning.json", "w") as f:
                json.dump(trial_results, f, indent=2)
            mlflow.log_artifact("hyperparameter_tuning.json", "hyperparameter_tuning")
            
            logger.info(f"🎯 Logged hyperparameter tuning results")
            
        except Exception as e:
            logger.error(f"❌ Error logging hyperparameter tuning: {e}")
    
    def register_model(self, model_path: str, model_name: str, version: str = "latest"):
        """Register model in MLflow model registry."""
        try:
            # Register the model
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_path}",
                name=model_name
            )
            
            logger.info(f"📝 Registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error registering model: {e}")
    
    def get_best_model(self, metric_name: str = "f1_micro", max_results: int = 5):
        """Get the best model based on a metric."""
        try:
            # Search for runs
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=max_results
            )
            
            if not runs.empty:
                best_run = runs.iloc[0]
                logger.info(f"🏆 Best model found: {best_run['run_id']} with {metric_name}={best_run[f'metrics.{metric_name}']}")
                return best_run
            else:
                logger.warning("⚠️ No runs found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting best model: {e}")
            return None
    
    def compare_models(self, run_ids: list):
        """Compare multiple model runs."""
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                metrics = run.data.metrics
                params = run.data.params
                
                comparison_data.append({
                    'run_id': run_id,
                    'metrics': metrics,
                    'params': params
                })
            
            # Log comparison as artifact
            with open("model_comparison.json", "w") as f:
                json.dump(comparison_data, f, indent=2)
            mlflow.log_artifact("model_comparison.json", "model_comparison")
            
            logger.info(f"📊 Compared {len(run_ids)} model runs")
            
        except Exception as e:
            logger.error(f"❌ Error comparing models: {e}")

# Utility functions for common MLflow operations
def setup_mlflow_experiment(experiment_name: str = "icd10-prediction"):
    """Setup MLflow experiment with proper configuration."""
    tracker = MLflowTracker(experiment_name=experiment_name)
    return tracker

def log_training_run(tracker: MLflowTracker, 
                    config: Dict[str, Any],
                    metrics: Dict[str, float],
                    model_path: str,
                    run_name: Optional[str] = None):
    """Log a complete training run."""
    with tracker.start_run(run_name=run_name) as run:
        # Log configuration
        tracker.log_training_config(config)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_artifact(model_path, "model")
        
        # Log run info
        logger.info(f"📝 Training run logged: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    # Example usage
    tracker = MLflowTracker()
    
    # Example metrics
    example_metrics = {
        'accuracy': 0.85,
        'f1_score': 0.82,
        'precision': 0.84,
        'recall': 0.80
    }
    
    # Example parameters
    example_params = {
        'model_type': 'clinical-bert-xgboost',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    }
    
    with tracker.start_run("example-run") as run:
        tracker.log_params(example_params)
        tracker.log_metrics(example_metrics)
        print(f"✅ Example run logged: {run.info.run_id}") 