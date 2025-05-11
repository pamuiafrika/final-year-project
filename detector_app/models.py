from django.db import models
from django.utils import timezone

class Dataset(models.Model):
    """Model for storing dataset information"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    clean_pdf_count = models.IntegerField(default=0)
    stego_pdf_count = models.IntegerField(default=0)
    
    def __str__(self):
        return self.name

class TrainedModel(models.Model):
    """Model for storing information about trained ML models"""
    MODEL_TYPES = (
        ('cnn', 'Convolutional Neural Network'),
        ('xgboost', 'XGBoost'),
        ('lstm', 'LSTM'),
        ('ensemble', 'Ensemble Model'),
    )
    
    name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    accuracy = models.FloatField(default=0.0)
    file_path = models.CharField(max_length=512)
    is_active = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class PDFScan(models.Model):
    """Model for storing PDF scan requests and results"""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    RESULT_CHOICES = (
        ('clean', 'Clean'),
        ('stego', 'Steganography Detected'),
        ('unknown', 'Unknown'),
    )
    
    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    result = models.CharField(max_length=20, choices=RESULT_CHOICES, default='unknown')
    confidence = models.FloatField(default=0.0)
    model_used = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, blank=True)
    task_id = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return f"{self.filename} - {self.status}"

class BulkScan(models.Model):
    """Model for storing bulk scan jobs"""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_files = models.IntegerField(default=0)
    processed_files = models.IntegerField(default=0)
    clean_count = models.IntegerField(default=0)
    stego_count = models.IntegerField(default=0)
    task_id = models.CharField(max_length=255, blank=True, null=True)
    
    def __str__(self):
        return f"{self.name} - {self.status}"
    
    @property
    def progress(self):
        if self.total_files == 0:
            return 0
        return int((self.processed_files / self.total_files) * 100)