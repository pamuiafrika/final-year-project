
from django import forms
from django.core.validators import FileExtensionValidator

class PDFUploadForm(forms.Form):
    TECHNIQUE_CHOICES = [
        ('auto', 'Automatic (All Techniques)'),
        ('object_stream', 'Object Stream Analysis'),
        ('metadata', 'Metadata Analysis'),
        ('font_glyph', 'Font & Glyph Analysis'),
        ('entropy', 'Entropy Pattern Analysis'),
        ('embedded', 'Embedded Files Scan'),
        ('layers', 'Invisible Layers Detection'),
    ]
    
    pdf_file = forms.FileField(
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf'
        }),
        help_text='Select a PDF file to analyze for steganography (max 50MB)'
    )
    
    technique = forms.ChoiceField(
        choices=TECHNIQUE_CHOICES,
        initial='auto',
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Choose analysis technique or use automatic detection'
    )
    
    def clean_pdf_file(self):
        pdf_file = self.cleaned_data.get('pdf_file')
        
        if pdf_file:
            # Check file size (50MB limit)
            if pdf_file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size cannot exceed 50MB.')
            
            # Check if it's actually a PDF
            if not pdf_file.name.lower().endswith('.pdf'):
                raise forms.ValidationError('Please upload a valid PDF file.')
        
        return pdf_file


'''
# ==============================================
# 3. forms.py - Django Forms
# ==============================================

from django import forms
from django.core.validators import FileExtensionValidator

class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(
        label='Select PDF File',
        validators=[FileExtensionValidator(allowed_extensions=['pdf'])],
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf',
            'id': 'pdf-file-input'
        }),
        help_text='Only PDF files are allowed (max 50MB)'
    )
    
    def clean_pdf_file(self):
        file = self.cleaned_data.get('pdf_file')
        if file:
            # Check file size (50MB limit)
            if file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size must be less than 50MB')
            
            # Additional PDF validation can be added here
            
        return file

'''