# Django AI App - Complete Structure
# This includes all necessary files for the AI Django application

# ============================================================================
# AI/models.py
# ============================================================================

from django.db import models
from django.contrib.auth.models import User
import uuid

class PDFUpload(models.Model):
    """Model to store PDF upload information"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file_name = models.CharField(max_length=255)
    file_hash = models.CharField(max_length=64, unique=True)
    file_size = models.BigIntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    processing_completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.file_name} ({self.uploaded_at})"

class PredictionResult(models.Model):
    """Model to store prediction results"""
    pdf_upload = models.OneToOneField(PDFUpload, on_delete=models.CASCADE, related_name='prediction')
    
    # XGBoost Results
    xgboost_prediction = models.IntegerField()  # 0 or 1
    xgboost_probability = models.FloatField()
    xgboost_confidence = models.FloatField()
    
    # Wide & Deep Results
    wide_deep_prediction = models.IntegerField()  # 0 or 1
    wide_deep_probability = models.FloatField()
    wide_deep_confidence = models.FloatField()
    
    # Ensemble Results
    ensemble_prediction = models.IntegerField()  # 0 or 1
    ensemble_confidence = models.FloatField()
    
    # Feature extraction info
    extraction_success = models.BooleanField(default=False)
    extraction_time_ms = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        status = "Steganography Detected" if self.ensemble_prediction == 1 else "Clean PDF"
        return f"{self.pdf_upload.file_name}: {status}"
    
    @property
    def is_suspicious(self):
        return self.ensemble_prediction == 1
    
    @property
    def risk_level(self):
        """Return risk level based on confidence"""
        if self.ensemble_confidence >= 0.8:
            return "High"
        elif self.ensemble_confidence >= 0.6:
            return "Medium"
        else:
            return "Low"

class FeatureVector(models.Model):
    """Model to store extracted features"""
    pdf_upload = models.OneToOneField(PDFUpload, on_delete=models.CASCADE, related_name='features')
    
    # Basic Features
    file_size_bytes = models.BigIntegerField(default=0)
    pdf_version = models.FloatField(default=1.4)
    num_pages = models.IntegerField(default=0)
    num_objects = models.IntegerField(default=0)
    num_stream_objects = models.IntegerField(default=0)
    num_embedded_files = models.IntegerField(default=0)
    num_annotation_objects = models.IntegerField(default=0)
    num_form_fields = models.IntegerField(default=0)
    
    # Temporal Features
    creation_date_ts = models.BigIntegerField(default=0)
    mod_date_ts = models.BigIntegerField(default=0)
    creation_mod_date_diff = models.BigIntegerField(default=0)
    
    # Entropy Features
    avg_entropy_per_stream = models.FloatField(default=0.0)
    max_entropy_per_stream = models.FloatField(default=0.0)
    min_entropy_per_stream = models.FloatField(default=0.0)
    std_entropy_per_stream = models.FloatField(default=0.0)
    num_streams_entropy_gt_threshold = models.IntegerField(default=0)
    
    # Security Features
    num_encrypted_streams = models.IntegerField(default=0)
    num_corrupted_objects = models.IntegerField(default=0)
    num_objects_with_random_markers = models.IntegerField(default=0)
    has_broken_name_trees = models.BooleanField(default=False)
    num_suspicious_filters = models.IntegerField(default=0)
    has_javascript = models.BooleanField(default=False)
    has_launch_actions = models.BooleanField(default=False)
    
    # Derived Features
    avg_file_size_per_page = models.FloatField(default=0.0)
    compression_ratio = models.FloatField(default=1.0)
    num_eof_markers = models.IntegerField(default=1)
    extraction_success = models.BooleanField(default=False)
    extraction_time_ms = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    
    def to_dataframe_row(self):
        """Convert to pandas DataFrame row format"""
        import pandas as pd
        
        feature_dict = {
            'file_size_bytes': self.file_size_bytes,
            'pdf_version': self.pdf_version,
            'num_pages': self.num_pages,
            'num_objects': self.num_objects,
            'num_stream_objects': self.num_stream_objects,
            'num_embedded_files': self.num_embedded_files,
            'num_annotation_objects': self.num_annotation_objects,
            'num_form_fields': self.num_form_fields,
            'creation_date_ts': self.creation_date_ts,
            'mod_date_ts': self.mod_date_ts,
            'creation_mod_date_diff': self.creation_mod_date_diff,
            'avg_entropy_per_stream': self.avg_entropy_per_stream,
            'max_entropy_per_stream': self.max_entropy_per_stream,
            'min_entropy_per_stream': self.min_entropy_per_stream,
            'std_entropy_per_stream': self.std_entropy_per_stream,
            'num_streams_entropy_gt_threshold': self.num_streams_entropy_gt_threshold,
            'num_encrypted_streams': self.num_encrypted_streams,
            'num_corrupted_objects': self.num_corrupted_objects,
            'num_objects_with_random_markers': self.num_objects_with_random_markers,
            'has_broken_name_trees': self.has_broken_name_trees,
            'num_suspicious_filters': self.num_suspicious_filters,
            'has_javascript': self.has_javascript,
            'has_launch_actions': self.has_launch_actions,
            'avg_file_size_per_page': self.avg_file_size_per_page,
            'compression_ratio': self.compression_ratio,
            'num_eof_markers': self.num_eof_markers,
            'extraction_success': self.extraction_success,
            'extraction_time_ms': self.extraction_time_ms,
            'error_count': self.error_count,
        }
        
        return pd.DataFrame([feature_dict])

# models.py
from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
import json
import os


class MLModel(models.Model):
    """Base model for storing ML model information"""
    MODEL_TYPES = [
        ('xgboost', 'XGBoost'),
        ('wide_deep', 'Wide & Deep Neural Network'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Model files
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    preprocessor_file = models.FileField(upload_to='models/', null=True, blank=True)
    
    # Model parameters (stored as JSON)
    hyperparameters = models.JSONField(default=dict, blank=True)
    
    # Training metadata
    training_duration = models.FloatField(null=True, blank=True, help_text="Training time in seconds")
    feature_columns = models.JSONField(default=list, blank=True)
    
    # Model description
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"


class TrainingSession(models.Model):
    """Store training session information"""
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Training data
    training_data_file = models.FileField(upload_to='models/training_data/')
    data_shape = models.JSONField(default=dict, blank=True)  # Store shape info
    class_distribution = models.JSONField(default=dict, blank=True)
    
    # Training configuration
    test_size = models.FloatField(default=0.2)
    random_state = models.IntegerField(default=42)
    stratify = models.BooleanField(default=True)
    
    # Status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Training Session: {self.name}"


class ModelEvaluation(models.Model):
    """Store model evaluation metrics"""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='evaluations')
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    
    # Evaluation metrics
    train_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField(null=True, blank=True)
    train_auc = models.FloatField(null=True, blank=True)
    test_auc = models.FloatField(null=True, blank=True)
    
    # Classification report (stored as JSON)
    classification_report = models.JSONField(default=dict, blank=True)
    confusion_matrix = models.JSONField(default=list, blank=True)
    
    # Feature importance (for tree-based models)
    feature_importance = models.JSONField(default=dict, blank=True)
    
    # Cross-validation scores
    cv_scores = models.JSONField(default=list, blank=True)
    cv_mean = models.FloatField(null=True, blank=True)
    cv_std = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Evaluation for {self.model.name}"


class TrainingVisualization(models.Model):
    """Store training visualizations and plots"""
    CHART_TYPES = [
        ('training_history', 'Training History'),
        ('confusion_matrix', 'Confusion Matrix'),
        ('roc_curve', 'ROC Curve'),
        ('feature_importance', 'Feature Importance'),
        ('learning_curve', 'Learning Curve'),
        ('validation_curve', 'Validation Curve'),
    ]
    
    training_session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='visualizations')
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='visualizations', null=True, blank=True)
    
    chart_type = models.CharField(max_length=30, choices=CHART_TYPES)
    title = models.CharField(max_length=200)
    
    # Store chart data as JSON (for recreating charts)
    chart_data = models.JSONField(default=dict, blank=True)
    
    # Store chart image
    chart_image = models.ImageField(upload_to='models/visualizations/', null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['chart_type', '-created_at']
    
    def __str__(self):
        return f"{self.title} ({self.chart_type})"


class PredictionLog(models.Model):
    """Log predictions made by the models"""
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='predictions')
    
    # Input data (file or features)
    input_file = models.FileField(upload_to='prediction_inputs/', null=True, blank=True)
    input_features = models.JSONField(default=dict, blank=True)
    
    # Predictions
    xgboost_prediction = models.IntegerField(null=True, blank=True)
    xgboost_probability = models.FloatField(null=True, blank=True)
    wide_deep_prediction = models.IntegerField(null=True, blank=True)
    wide_deep_probability = models.FloatField(null=True, blank=True)
    
    # Metadata
    prediction_time = models.FloatField(help_text="Time taken for prediction in seconds")
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Additional notes
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction on {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class DatasetInfo(models.Model):
    """Store information about datasets used for training"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='datasets/')
    
    # Dataset statistics
    total_samples = models.IntegerField(null=True, blank=True)
    positive_samples = models.IntegerField(null=True, blank=True)  # Steganographic
    negative_samples = models.IntegerField(null=True, blank=True)  # Clean
    feature_count = models.IntegerField(null=True, blank=True)
    
    # Data quality metrics
    missing_values_count = models.IntegerField(null=True, blank=True)
    duplicate_rows = models.IntegerField(null=True, blank=True)
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # File information
    file_size = models.BigIntegerField(null=True, blank=True)
    file_format = models.CharField(max_length=10, default='csv')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if self.file:
            self.file_size = self.file.size
        super().save(*args, **kwargs)