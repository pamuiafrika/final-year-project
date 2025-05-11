from django.db import models
from django.contrib.auth.models import User
import os
import uuid

def upload_pdf_path(instance, filename):
    """Generate file path for uploaded PDFs"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', filename)

def model_file_path(instance, filename):
    """Generate file path for ML model files"""
    return os.path.join('ml_models', filename)

class Dataset(models.Model):
    """Dataset model for steganography detection training"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    # Dataset statistics
    clean_pdf_count = models.IntegerField(default=0)
    stego_pdf_count = models.IntegerField(default=0)
    
    # Status
    STATUS_CHOICES = (
        ('created', 'Created'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='created')
    status_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.name} ({self.clean_pdf_count}/{self.stego_pdf_count})"

class MLModel(models.Model):
    """ML model for steganography detection"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to=model_file_path)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    # Model type
    MODEL_TYPE_CHOICES = (
        ('cnn', 'CNN'),
        ('lstm', 'LSTM'),
        ('xgboost', 'XGBoost'),
        ('ensemble', 'Ensemble'),
    )
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    
    # Model metadata
    accuracy = models.FloatField(default=0.0)
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    # Status
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('training', 'Training'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='inactive')
    
    def __str__(self):
        return f"{self.name} ({self.model_type}, {self.accuracy:.2f})"

class PDFUpload(models.Model):
    """Uploaded PDF file for detection"""
    file = models.FileField(upload_to=upload_pdf_path)
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    filesize = models.IntegerField(default=0)
    
    # Detection results
    is_stego = models.BooleanField(null=True, blank=True)  # None means not processed
    confidence = models.FloatField(null=True, blank=True)
    ml_model = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Status
    STATUS_CHOICES = (
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    details = models.JSONField(default=dict, blank=True)
    
    def __str__(self):
        return self.filename

class BulkDetectionJob(models.Model):
    """Bulk detection job for multiple PDFs"""
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    # Job configuration
    ml_model = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True)
    
    # Job statistics
    total_files = models.IntegerField(default=0)
    processed_files = models.IntegerField(default=0)
    stego_detected = models.IntegerField(default=0)
    
    # Status
    STATUS_CHOICES = (
        ('created', 'Created'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='created')
    status_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.name} ({self.processed_files}/{self.total_files})"
    
    @property
    def progress_percentage(self):
        if self.total_files == 0:
            return 0
        return int((self.processed_files / self.total_files) * 100)

class BulkDetectionFile(models.Model):
    """Individual file within a bulk detection job"""
    job = models.ForeignKey(BulkDetectionJob, on_delete=models.CASCADE, related_name='files')
    pdf_upload = models.ForeignKey(PDFUpload, on_delete=models.CASCADE)
    
    # Status
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('processed', 'Processed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    def __str__(self):
        return f"{self.pdf_upload.filename} ({self.status})"