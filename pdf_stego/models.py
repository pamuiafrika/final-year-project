import os
import uuid
from django.db import models
from django.utils import timezone

class StegoOperation(models.Model):
    """Model to track steganography operations"""
    OPERATION_CHOICES = [
        ('hide', 'Hide Image in PDF'),
        ('extract', 'Extract Image from PDF'),
    ]
    
    METHOD_CHOICES = [
        ('stream', 'Object Stream Manipulation'),
        ('metadata', 'Metadata Embedding'),
        ('whitespace', 'White Space and Comment Exploitation'),
        ('components', 'Document Components Manipulation'),
        ('javascript', 'JavaScript Integration'),
        ('random', 'Random Selection'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    operation_type = models.CharField(max_length=10, choices=OPERATION_CHOICES)
    method = models.CharField(max_length=20, choices=METHOD_CHOICES)
    actual_method = models.CharField(max_length=20, blank=True, null=True)  # For when random is selected
    
    # For hide operation
    input_pdf = models.FileField(upload_to='stego/input_pdfs/', blank=True, null=True)
    input_png = models.ImageField(upload_to='stego/input_pngs/', blank=True, null=True)
    output_pdf = models.FileField(upload_to='stego/output_pdfs/', blank=True, null=True)
    
    # For extract operation
    stego_pdf = models.FileField(upload_to='stego/extracted/stego_pdfs/', blank=True, null=True)
    extracted_png = models.ImageField(upload_to='stego/extracted/extracted_pngs/', blank=True, null=True)
    
    # Operation status and metadata
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.operation_type} - {self.method} - {self.status}"
    
    def get_input_pdf_filename(self):
        return os.path.basename(self.input_pdf.name) if self.input_pdf else None
    
    def get_input_png_filename(self):
        return os.path.basename(self.input_png.name) if self.input_png else None
    
    def get_stego_pdf_filename(self):
        return os.path.basename(self.stego_pdf.name) if self.stego_pdf else None
    
    def get_output_pdf_filename(self):
        return os.path.basename(self.output_pdf.name) if self.output_pdf else None
    
    def get_extracted_png_filename(self):
        return os.path.basename(self.extracted_png.name) if self.extracted_png else None
