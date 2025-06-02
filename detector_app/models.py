# ==============================================
# 1. models.py - Database Models
# ==============================================

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.files.storage import default_storage
import uuid
import os
import json

def upload_to_pdfs(instance, filename):
    """Generate upload path for PDF files"""
    return f'pdfs/{instance.user.username}/{uuid.uuid4()}/{filename}'

class PDFScanResult(models.Model):
    RISK_LEVELS = [
        ('MINIMAL', 'Minimal'),
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    # Basic info
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='pdf_scans')
    file = models.FileField(upload_to='pdf_uploads/%Y/%m/%d/', max_length=500)
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField()
    
    # Scan results
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    is_malicious = models.BooleanField(null=True, blank=True)
    ensemble_probability = models.FloatField(null=True, blank=True)
    confidence_percentage = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS, null=True, blank=True)
    
    # Individual model predictions
    attention_probability = models.FloatField(null=True, blank=True)
    deep_ff_probability = models.FloatField(null=True, blank=True)
    wide_deep_probability = models.FloatField(null=True, blank=True)
    
    # Extracted features (key ones)
    pdf_pages = models.IntegerField(null=True, blank=True)
    metadata_size = models.IntegerField(null=True, blank=True)
    suspicious_count = models.IntegerField(null=True, blank=True)
    javascript_elements = models.IntegerField(null=True, blank=True)
    auto_actions = models.IntegerField(null=True, blank=True)
    embedded_files = models.IntegerField(null=True, blank=True)
    
    # Complete features as JSON
    extracted_features = models.JSONField(null=True, blank=True)
    individual_predictions = models.JSONField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'PDF Scan Result'
        verbose_name_plural = 'PDF Scan Results'
    
    def __str__(self):
        return f"{self.original_filename} - {self.get_status_display()}"
    
    @property
    def scan_duration(self):
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None


class PDFAnalysis(models.Model):
    RISK_LEVELS = [
        ('CLEAN', 'Clean - No significant indicators'),
        ('LOW_RISK', 'Low Risk - Minor anomalies'),
        ('MEDIUM_RISK', 'Medium Risk - Suspicious patterns'),
        ('HIGH_RISK', 'High Risk - Strong evidence of steganography'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='pdf_analyses')
    pdf_file = models.FileField(upload_to='pdf_uploads/%Y/%m/%d/')
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField()
    upload_date = models.DateTimeField(auto_now_add=True)
    analysis_date = models.DateTimeField(null=True, blank=True)
    
    # Analysis Results
    assessment = models.CharField(max_length=20, choices=RISK_LEVELS, null=True, blank=True)
    risk_score = models.FloatField(null=True, blank=True)
    total_indicators = models.IntegerField(default=0)
    ml_anomaly_score = models.FloatField(null=True, blank=True)
    is_anomaly = models.BooleanField(default=False)
    
    # JSON fields for detailed results
    indicators_data = models.JSONField(default=dict, blank=True)
    features_data = models.JSONField(default=dict, blank=True)
    recommendations = models.JSONField(default=list, blank=True)
    
    # Analysis metadata
    analysis_duration = models.FloatField(null=True, blank=True)  # seconds
    technique_used = models.CharField(max_length=50, default='auto')
    model_version = models.CharField(max_length=50, null=True, blank=True)
    
    class Meta:
        ordering = ['-analysis_date', '-upload_date']
        verbose_name = 'PDF Analysis'
        verbose_name_plural = 'PDF Analyses'
    
    def __str__(self):
        return f"{self.original_filename} - {self.assessment or 'Pending'}"
    
    @property
    def is_analyzed(self):
        return self.analysis_date is not None
    
    @property
    def risk_level_display(self):
        if self.risk_score is None:
            return "Not Analyzed"
        elif self.risk_score >= 20:
            return "CRITICAL"
        elif self.risk_score >= 10:
            return "HIGH"
        elif self.risk_score >= 5:
            return "MEDIUM"
        else:
            return "LOW"

class AnalysisIndicator(models.Model):
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'), 
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    ]
    
    analysis = models.ForeignKey(PDFAnalysis, on_delete=models.CASCADE, related_name='indicators')
    category = models.CharField(max_length=50)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    description = models.TextField()
    confidence = models.FloatField()
    technical_details = models.JSONField(default=dict, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    
    class Meta:
        ordering = ['-severity', '-confidence']
    
    def __str__(self):
        return f"{self.category} - {self.severity}"