# forms.py
from django import forms
from .models import TrainingSession, DatasetInfo, MLModel


class TrainingSessionForm(forms.ModelForm):
    """Form for creating training sessions"""
    
    class Meta:
        model = TrainingSession
        fields = ['name', 'training_data_file', 'test_size', 'random_state', 'stratify']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter training session name'
            }),
            'training_data_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            }),
            'test_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '0.5',
                'step': '0.05'
            }),
            'random_state': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '1'
            }),
            'stratify': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
        
    def clean_training_data_file(self):
        file = self.cleaned_data['training_data_file']
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError('Please upload a CSV file.')
            if file.size > 100 * 1024 * 1024:  # 100MB limit
                raise forms.ValidationError('File size must be less than 100MB.')
        return file


class DatasetUploadForm(forms.ModelForm):
    """Form for uploading datasets"""
    
    class Meta:
        model = DatasetInfo
        fields = ['name', 'description', 'file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe your dataset...'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls'
            })
        }
    
    def clean_file(self):
        file = self.cleaned_data['file']
        if file:
            allowed_extensions = ['.csv', '.xlsx', '.xls']
            if not any(file.name.lower().endswith(ext) for ext in allowed_extensions):
                raise forms.ValidationError('Please upload a CSV or Excel file.')
            if file.size > 200 * 1024 * 1024:  # 200MB limit
                raise forms.ValidationError('File size must be less than 200MB.')
        return file


class PredictionForm(forms.Form):
    """Form for making predictions"""
    
    PREDICTION_METHODS = [
        ('file', 'Upload File'),
        ('manual', 'Manual Input'),
    ]
    
    model = forms.ModelChoiceField(
        queryset=MLModel.objects.filter(status='completed'),
        widget=forms.Select(attrs={'class': 'form-control'}),
        empty_label="Select a trained model"
    )
    
    prediction_method = forms.ChoiceField(
        choices=PREDICTION_METHODS,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        initial='file'
    )
    
    prediction_file = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        help_text="Upload a CSV file with PDF features"
    )
    
    # Manual input fields for key features
    file_size_bytes = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'File size in bytes'
        })
    )
    
    pdf_version = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'PDF version (e.g., 1.4)',
            'step': '0.1'
        })
    )
    
    num_pages = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of pages'
        })
    )
    
    num_objects = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of PDF objects'
        })
    )
    
    num_stream_objects = forms.IntegerField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of stream objects'
        })
    )
    
    avg_entropy_per_stream = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Average entropy per stream',
            'step': '0.01'
        })
    )
    
    max_entropy_per_stream = forms.FloatField(
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Maximum entropy per stream',
            'step': '0.01'
        })
    )
    
    has_javascript = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
    
    has_launch_actions = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        prediction_method = cleaned_data.get('prediction_method')
        prediction_file = cleaned_data.get('prediction_file')
        
        if prediction_method == 'file' and not prediction_file:
            raise forms.ValidationError('Please upload a file for file-based prediction.')
        
        if prediction_method == 'manual':
            # Check if at least some manual fields are filled
            manual_fields = ['file_size_bytes', 'pdf_version', 'num_pages', 'num_objects']
            if not any(cleaned_data.get(field) for field in manual_fields):
                raise forms.ValidationError('Please fill in at least some feature values for manual prediction.')
        
        return cleaned_data


class ModelComparisonForm(forms.Form):
    """Form for comparing models"""
    
    models = forms.ModelMultipleChoiceField(
        queryset=MLModel.objects.filter(status='completed'),
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        }),
        help_text="Select 2-5 models to compare"
    )
    
    comparison_metrics = forms.MultipleChoiceField(
        choices=[
            ('test_auc', 'Test AUC'),
            ('test_accuracy', 'Test Accuracy'),
            ('train_auc', 'Train AUC'),
            ('train_accuracy', 'Train Accuracy'),
        ],
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        }),
        initial=['test_auc', 'test_accuracy']
    )
    
    def clean_models(self):
        models = self.cleaned_data['models']
        if len(models) < 2:
            raise forms.ValidationError('Please select at least 2 models to compare.')
        if len(models) > 5:
            raise forms.ValidationError('Please select no more than 5 models to compare.')
        return models