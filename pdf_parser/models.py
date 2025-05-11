from django.db import models
import uuid
import os

def upload_to_pdfs(instance, filename):
    """Generate a unique file path for uploaded PDFs"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads/pdfs', filename)

class PDFDocument(models.Model):
    """Model for storing PDF documents and analysis results"""
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to=upload_to_pdfs)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    # Analysis status
    is_analyzed = models.BooleanField(default=False)
    has_anomalies = models.BooleanField(default=False)
    
    # Metadata extracted from PDF
    author = models.CharField(max_length=255, blank=True, null=True)
    creation_date = models.DateTimeField(null=True, blank=True)
    modification_date = models.DateTimeField(null=True, blank=True)
    
    # Analysis results
    num_pages = models.IntegerField(default=0)
    num_images = models.IntegerField(default=0)
    suspicious_areas = models.IntegerField(default=0)
    
    def __str__(self):
        return self.title or os.path.basename(self.file.name)

class PDFImage(models.Model):
    """Model for storing images extracted from PDFs"""
    pdf = models.ForeignKey(PDFDocument, related_name='images', on_delete=models.CASCADE)
    image_data = models.BinaryField()
    page_number = models.IntegerField()
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    size_bytes = models.IntegerField(default=0)
    image_type = models.CharField(max_length=20, default='unknown')
    is_suspicious = models.BooleanField(default=False)
    entropy_score = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Image from {self.pdf} (Page {self.page_number})"

class AnomalyReport(models.Model):
    """Model for storing detailed anomaly reports"""
    pdf = models.ForeignKey(PDFDocument, related_name='anomalies', on_delete=models.CASCADE)
    detection_date = models.DateTimeField(auto_now_add=True)
    anomaly_type = models.CharField(max_length=100)
    description = models.TextField()
    confidence_score = models.FloatField(default=0.0)
    page_number = models.IntegerField(null=True, blank=True)
    location_data = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Anomaly in {self.pdf}: {self.anomaly_type}"