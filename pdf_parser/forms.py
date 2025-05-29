from django import forms
from .models import PDFDocument

class PDFUploadForm(forms.ModelForm):
    """Form for uploading PDF files"""
    
    class Meta:
        model = PDFDocument
        fields = ['file']
        
    def clean_file(self):
        """Validate that the uploaded file is a PDF"""
        file = self.cleaned_data.get('file', False)
        if file:
            if not file.name.endswith('.pdf'):
                raise forms.ValidationError("File must be a PDF document.")
            if file.size > 25 * 1024 * 1024:  # 25 MB limit
                raise forms.ValidationError("File size must be under 10 MB.")
            return file
        else:
            raise forms.ValidationError("Could not read uploaded file.")