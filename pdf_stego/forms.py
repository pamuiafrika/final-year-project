"""
Form definitions for PDF steganography application.
"""

from django import forms
from .models import StegoOperation

class HideForm(forms.ModelForm):
    """Form for hiding PNG in PDF"""
    class Meta:
        model = StegoOperation
        fields = ['method', 'input_pdf', 'input_png']
        widgets = {
            'method': forms.Select(attrs={'class': 'form-control'}),
            'input_pdf': forms.FileInput(attrs={'class': 'form-control'}),
            'input_png': forms.FileInput(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['input_pdf'].help_text = 'Upload PDF file (Max size: 10MB)'
        self.fields['input_png'].help_text = 'Upload PNG image (Max size: 5MB)'
        self.fields['method'].help_text = 'Select steganography technique or choose random'
    
    def clean_input_pdf(self):
        pdf = self.cleaned_data.get('input_pdf')
        if pdf:
            # Check file extension
            if not pdf.name.endswith('.pdf'):
                raise forms.ValidationError('File must be a PDF document')
            # Check file size (10MB limit)
            if pdf.size > 10 * 1024 * 1024:
                raise forms.ValidationError('PDF file size must be under 10MB')
        return pdf
    
    def clean_input_png(self):
        img = self.cleaned_data.get('input_png')
        if img:
            # Check file extension
            if not img.name.lower().endswith('.png'):
                raise forms.ValidationError('Image must be in PNG format')
            # Check file size (5MB limit)
            if img.size > 5 * 1024 * 1024:
                raise forms.ValidationError('PNG file size must be under 5MB')
        return img

class ExtractForm(forms.ModelForm):
    """Form for extracting PNG from PDF"""
    class Meta:
        model = StegoOperation
        fields = ['method', 'stego_pdf']
        widgets = {
            'method': forms.Select(attrs={'class': 'form-control'}),
            'stego_pdf': forms.FileInput(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['stego_pdf'].help_text = 'Upload PDF file with hidden PNG (Max size: 10MB)'
        self.fields['method'].help_text = 'Select extraction technique or choose random'
    
    def clean_stego_pdf(self):
        pdf = self.cleaned_data.get('stego_pdf')
        if pdf:
            # Check file extension
            if not pdf.name.endswith('.pdf'):
                raise forms.ValidationError('File must be a PDF document')
            # Check file size (10MB limit)
            if pdf.size > 10 * 1024 * 1024:
                raise forms.ValidationError('PDF file size must be under 10MB')
        return pdf