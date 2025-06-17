
# ============================================================================
# AI/views.py
# ============================================================================
from django.urls import reverse
import uuid
import hashlib
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
from django.db import transaction
import json
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from django.db import models

from .models import PDFUpload, PredictionResult, FeatureVector
from .ml_predictor import MLPredictor
from .pdf_feature_extractor import PDFFeatureExtractor

# AI/views.py
from django.db.models import Q, Avg
from django.utils import timezone
from datetime import datetime, timedelta
from django.views.generic import ListView

class PDFUploadListView(ListView):
    model = PDFUpload
    template_name = 'ai/upload_list.html'  # Or 'ai/upload_history.html' if you rename it
    context_object_name = 'uploads'
    paginate_by = 20

    def get_queryset(self):
        queryset = PDFUpload.objects.filter(user=self.request.user).select_related('prediction').order_by('-uploaded_at')
        
        # Get filter parameters from request
        status_filter = self.request.GET.get('status', '')
        date_filter = self.request.GET.get('date', '')
        search_query = self.request.GET.get('search', '')
        
        # Apply status filter
        if status_filter == 'clean':
            queryset = queryset.filter(processed=True, prediction__ensemble_prediction=0)
        elif status_filter == 'suspicious':
            queryset = queryset.filter(processed=True, prediction__ensemble_prediction=1)
        elif status_filter == 'processing':
            queryset = queryset.filter(processed=False)
        
        # Apply date filter
        if date_filter:
            today = timezone.now().date()
            if date_filter == 'today':
                queryset = queryset.filter(uploaded_at__date=today)
            elif date_filter == 'week':
                start_of_week = today - timedelta(days=today.weekday())
                queryset = queryset.filter(uploaded_at__date__gte=start_of_week)
            elif date_filter == 'month':
                start_of_month = today.replace(day=1)
                queryset = queryset.filter(uploaded_at__date__gte=start_of_month)
        
        # Apply search filter
        if search_query:
            queryset = queryset.filter(
                Q(file_name__icontains=search_query) |
                Q(file_hash__icontains=search_query)
            )
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get all uploads (unfiltered) for statistics
        all_uploads = PDFUpload.objects.filter(user=self.request.user).select_related('prediction')

        # Calculate statistics
        total_uploads = all_uploads.count()
        processed_uploads = all_uploads.filter(processed=True).count()
        suspicious_files = all_uploads.filter(
            processed=True, 
            prediction__ensemble_prediction=1
        ).count()
        
        # Calculate average processing time for processed files
        processed_predictions = PredictionResult.objects.filter(
            pdf_upload__in=all_uploads, 
            extraction_success=True
        )
        average_processing_time = processed_predictions.aggregate(
            Avg('extraction_time_ms')
        )['extraction_time_ms__avg']
        
        if average_processing_time is None:
            average_processing_time = 0
        
        context.update({
            'total_uploads': total_uploads,
            'processed_uploads': processed_uploads,
            'suspicious_files': suspicious_files,
            'average_processing_time': average_processing_time,
            # Pass current filter values back to template
            'current_status_filter': self.request.GET.get('status', ''),
            'current_date_filter': self.request.GET.get('date', ''),
            'current_search_query': self.request.GET.get('search', ''),
        })
        
        return context
    
    
class PDFUploadDetailView(DetailView):
    """Detail view for PDF upload"""
    model = PDFUpload
    template_name = 'ai/upload_detail.html'
    context_object_name = 'upload'
    
    def get_object(self):
        return get_object_or_404(PDFUpload, id=self.kwargs['pk'])
    
@login_required
def upload_dashboard(request):
    """Main dashboard view"""
    recent_uploads = PDFUpload.objects.filter(user=request.user)[:10]
    total_uploads = PDFUpload.objects.filter(user=request.user).count()
    processed_uploads = PDFUpload.objects.filter(user=request.user, processed=True).count()
    suspicious_files = PredictionResult.objects.filter(pdf_upload__user=request.user, ensemble_prediction=1).count()
    
    context = {
        'recent_uploads': recent_uploads,
        'total_uploads': total_uploads,
        'processed_uploads': processed_uploads,
        'suspicious_files': suspicious_files,
    }
    return render(request, 'ai/dashboard.html', context)

@csrf_exempt
def upload_pdf(request):
    """Handle PDF file upload"""
    if request.method == 'POST':
        if 'pdf_file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        uploaded_file = request.FILES['pdf_file']
        
        # Validate file type
        if not uploaded_file.name.lower().endswith('.pdf'):
            return JsonResponse({'error': 'Only PDF files are allowed'}, status=400)
        
        # Validate file size (max 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            return JsonResponse({'error': 'File size must be less than 50MB'}, status=400)
        
        try:
            # Calculate file hash
            import hashlib
            uploaded_file.seek(0)
            file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
            uploaded_file.seek(0)
            
            # Check if file already exists
            existing_upload = PDFUpload.objects.filter(file_hash=file_hash).first()
            if existing_upload:
                return JsonResponse({
                    'message': 'File already uploaded',
                    'upload_id': str(existing_upload.id),
                    'existing': True
                })
            
            # Create PDFUpload record
            pdf_upload = PDFUpload.objects.create(
                user=request.user,
                file_name=uploaded_file.name,
                file_hash=file_hash,
                file_size=uploaded_file.size
            )
            
            # Save file to storage
            file_path = f"uploads/{pdf_upload.id}/{uploaded_file.name}"
            saved_path = default_storage.save(file_path, ContentFile(uploaded_file.read()))
            
            # Start processing in background
            threading.Thread(
                target=process_pdf_async,
                args=(pdf_upload.id, os.path.join(settings.MEDIA_ROOT, saved_path))
            ).start()
            
            return JsonResponse({
                'message': 'File uploaded successfully',
                'upload_id': str(pdf_upload.id),
                'processing': True
            })
            
        except Exception as e:
            return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)
    
    return render(request, 'ai/upload.html')

def process_pdf_async(upload_id, file_path):
    """Process PDF file asynchronously"""
    try:
        pdf_upload = PDFUpload.objects.get(id=upload_id)
        pdf_upload.processing_started_at = timezone.now()
        pdf_upload.save()
        
        # Initialize components
        feature_extractor = PDFFeatureExtractor()
        ml_predictor = MLPredictor()
        
        # Extract features
        features_df = feature_extractor.extract_single_file(file_path)
        features_dict = features_df.iloc[0].to_dict()
        
        # Save features to database
        with transaction.atomic():
            feature_vector = FeatureVector.objects.create(
                pdf_upload=pdf_upload,
                **{k: v for k, v in features_dict.items() 
                   if k not in ['pdf_name', 'file_hash', 'label']}
            )
            
            # Make predictions
            predictions = ml_predictor.predict(features_df)
            
            # Calculate ensemble prediction
            ensemble_prob = (predictions['xgboost_probability'][0] + 
                           predictions['wide_deep_probability'][0]) / 2
            ensemble_pred = int(ensemble_prob > 0.5)
            
            # Save prediction results
            PredictionResult.objects.create(
                pdf_upload=pdf_upload,
                xgboost_prediction=predictions['xgboost_prediction'][0],
                xgboost_probability=predictions['xgboost_probability'][0],
                xgboost_confidence=abs(predictions['xgboost_probability'][0] - 0.5) * 2,
                wide_deep_prediction=predictions['wide_deep_prediction'][0],
                wide_deep_probability=predictions['wide_deep_probability'][0],
                wide_deep_confidence=abs(predictions['wide_deep_probability'][0] - 0.5) * 2,
                ensemble_prediction=ensemble_pred,
                ensemble_confidence=abs(ensemble_prob - 0.5) * 2,
                extraction_success=features_dict.get('extraction_success', False),
                extraction_time_ms=features_dict.get('extraction_time_ms', 0),
                error_count=features_dict.get('error_count', 0)
            )
            
            # Update upload status
            pdf_upload.processed = True
            pdf_upload.processing_completed_at = timezone.now()
            pdf_upload.save()
            
    except Exception as e:
        print(f"Error processing PDF {upload_id}: {e}")
        # Mark as processed with error
        try:
            pdf_upload = PDFUpload.objects.get(id=upload_id)
            pdf_upload.processed = True
            pdf_upload.processing_completed_at = timezone.now()
            pdf_upload.save()
        except:
            pass

def process_single_upload(uploaded_file, user):
    """
    Processes a single uploaded file: calculates hash, checks for duplicates,
    creates PDFUpload record, saves the file, and triggers asynchronous processing.
    """
    # Calculate hash
    uploaded_file.seek(0)
    file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)

    # Check if file already exists
    existing_upload = PDFUpload.objects.filter(file_hash=file_hash).first()
    if existing_upload:
        return {
            'status': 'duplicate',
            'message': 'File already uploaded',
            'upload_id': str(existing_upload.id),
            'file_name': existing_upload.file_name # Return original filename for display
        }

    # Create PDFUpload record
    pdf_upload = PDFUpload.objects.create(
        user=user if user.is_authenticated else None,
        file_name=uploaded_file.name,
        file_hash=file_hash,
        file_size=uploaded_file.size
    )

    # Save file to storage
    # IMPORTANT: Ensure settings.MEDIA_ROOT and settings.MEDIA_URL are configured in your Django project
    file_path = f"uploads/{pdf_upload.id}/{uploaded_file.name}"
    saved_path = default_storage.save(file_path, ContentFile(uploaded_file.read()))

    # Trigger asynchronous processing in a separate thread (DO NOT BLOCK)
    full_path = os.path.join(settings.MEDIA_ROOT, saved_path)
    threading.Thread(
        target=process_pdf_async,
        args=(pdf_upload.id, full_path)
    ).start()

    return {
        'status': 'success',
        'message': 'Uploaded and queued for processing',
        'upload_id': str(pdf_upload.id),
        'file_name': uploaded_file.name # Return original filename for display
    }

# --- New/Modified Views for Batch Upload Feature ---

# Removed @csrf_exempt here. Ensure {% csrf_token %} is in your form and
# the X-CSRFToken header is sent by your JavaScript.
# If you still get CSRF issues, ensure your base.html includes csrf_token in all forms
# and that your fetch requests include the X-CSRFToken header.
def batch_upload(request):
    """
    Handle batch PDF upload.
    Redirects to batch_results page after queuing files for processing.
    """
    if request.method == 'POST':
        files = request.FILES.getlist('pdf_files')

        if not files:
            messages.error(request, 'No files were selected for upload.')
            return render(request, 'ai/batch_upload.html')

        queued_upload_ids = []
        initial_error_results = [] # To store errors for files that couldn't even be queued

        # Use ThreadPoolExecutor for parallel processing of initial file handling (hash, save, trigger async)
        # Max workers set to 10 for illustration; adjust based on server resources.
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for uploaded_file in files:
                # Client-side validation is preferred, but server-side is a must for security
                if not uploaded_file.name.lower().endswith('.pdf'):
                    initial_error_results.append({
                        'filename': uploaded_file.name,
                        'status': 'error',
                        'message': 'Not a PDF file'
                    })
                    continue

                if uploaded_file.size > 50 * 1024 * 1024: # 50 MB limit
                    initial_error_results.append({
                        'filename': uploaded_file.name,
                        'status': 'error',
                        'message': 'File too large (max 50MB)'
                    })
                    continue

                # Submit the initial file handling to the thread pool
                futures.append(executor.submit(process_single_upload, uploaded_file, request.user))

            # Collect results of initial file handling
            for future in futures:
                try:
                    result = future.result(timeout=60) # 60 seconds for initial file save/hash
                    if result['status'] in ['success', 'duplicate']:
                        queued_upload_ids.append(str(result['upload_id']))
                        messages.info(request, f"'{result['file_name']}' - {result['message']}")
                    else:
                        initial_error_results.append(result)
                        messages.error(request, f"'{result['filename']}' - {result['message']}")
                except Exception as e:
                    # This catches errors from process_single_upload itself
                    initial_error_results.append({
                        'filename': 'Unknown File', # Fallback
                        'status': 'error',
                        'message': f"Initial processing failed: {str(e)}"
                    })
                    messages.error(request, f"An error occurred during initial upload processing: {str(e)}")


        if queued_upload_ids:
            # Redirect to the batch results page with the IDs
            upload_ids_str = ','.join(queued_upload_ids)
            return redirect(f"{reverse('ai:batch_results')}?ids={upload_ids_str}")
        else:
            # If no files were successfully queued, display errors on the same page
            messages.error(request, 'No files were successfully queued for processing. Please review errors.')
            return render(request, 'ai/batch_upload.html', {'initial_error_results': initial_error_results})

    return render(request, 'ai/batch_upload.html')


def batch_results(request):
    """
    Displays the results of a batch upload, polling for file processing status.
    Accepts 'ids' as a comma-separated query parameter.
    """
    import logging
    logger = logging.getLogger('django')
    
    upload_ids_str = request.GET.get('ids')
    logger.info(f"Batch results requested for IDs: {upload_ids_str}")
    initial_results = []

    if upload_ids_str:
        # Convert string IDs from URL to UUID objects
        upload_id_list = []
        for uid_str in upload_ids_str.split(','):
            try:
                upload_id_list.append(uuid.UUID(uid_str))
            except ValueError:
                # Handle invalid UUIDs in the URL if necessary, e.g., skip them
                pass

        # Fetch initial info for files to pre-populate the display
        pdf_uploads = PDFUpload.objects.filter(id__in=upload_id_list)

        # Create a dictionary for faster lookup if you have many IDs
        pdf_upload_map = {str(upload.id): upload for upload in pdf_uploads}

        # Ensure results are in the same order as the incoming IDs string
        for uid in upload_id_list:
            pdf_upload = pdf_upload_map.get(str(uid))
            if pdf_upload:
                status = 'queued'
                message = 'Queued for analysis'
                prediction_data = None

                if pdf_upload.processed:
                    try:
                        # If already processed, fetch result for immediate display
                        prediction = pdf_upload.prediction
                        status = 'completed'
                        message = 'Analysis complete'
                        prediction_data = {
                            'ensemble_prediction': prediction.ensemble_prediction,
                            'ensemble_confidence': prediction.ensemble_confidence,
                            'risk_level': prediction.risk_level,
                            'is_suspicious': prediction.is_suspicious,
                            'processing_time_ms': prediction.extraction_time_ms
                        }
                    except PredictionResult.DoesNotExist:
                        status = 'error'
                        message = 'Processing finished, but results not found.'

                initial_results.append({
                    'upload_id': str(pdf_upload.id),
                    'filename': pdf_upload.file_name,
                    'status': status,
                    'message': message,
                    'prediction': prediction_data
                })
            else:
                # If an ID in the URL was not found in the database
                initial_results.append({
                    'upload_id': str(uid),
                    'filename': f"Unknown file ({str(uid)[:8]}...)",
                    'status': 'error',
                    'message': 'File not found in system or invalid ID.'
                })

    return render(request, 'ai/batch_results.html', {
        'initial_results_json': json.dumps(initial_results)
    })



def get_prediction_status(request, upload_id):
    """Get prediction status for an upload"""
    import logging
    logger = logging.getLogger('django')
    logger.info(f"Received status request for upload ID: {upload_id}")
    
    try:
        pdf_upload = get_object_or_404(PDFUpload, id=upload_id)
        
        if not pdf_upload.processed:
            logger.info(f"Upload {upload_id} still processing")
            return JsonResponse({
                'status': 'processing',
                'message': 'File is being processed...'
            })
        
        try:
            prediction = pdf_upload.prediction
            return JsonResponse({
                'status': 'completed',
                'prediction': {
                    'ensemble_prediction': prediction.ensemble_prediction,
                    'ensemble_confidence': prediction.ensemble_confidence,
                    'risk_level': prediction.risk_level,
                    'is_suspicious': prediction.is_suspicious,
                    'xgboost_probability': prediction.xgboost_probability,
                    'wide_deep_probability': prediction.wide_deep_probability,
                    'extraction_success': prediction.extraction_success,
                    'processing_time_ms': prediction.extraction_time_ms
                }
            })
        except PredictionResult.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Processing completed but no results found'
            })
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

def api_predict(request):
    """API endpoint for prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            upload_id = data.get('upload_id')
            
            if not upload_id:
                return JsonResponse({'error': 'upload_id required'}, status=400)
            
            pdf_upload = get_object_or_404(PDFUpload, id=upload_id)
            
            if not pdf_upload.processed:
                return JsonResponse({
                    'status': 'processing',
                    'message': 'File still being processed'
                })
            
            prediction = pdf_upload.prediction
            
            return JsonResponse({
                'upload_id': str(pdf_upload.id),
                'filename': pdf_upload.file_name,
                'prediction': {
                    'ensemble_prediction': prediction.ensemble_prediction,
                    'ensemble_confidence': prediction.ensemble_confidence,
                    'risk_level': prediction.risk_level,
                    'models': {
                        'xgboost': {
                            'prediction': prediction.xgboost_prediction,
                            'probability': prediction.xgboost_probability,
                            'confidence': prediction.xgboost_confidence
                        },
                        'wide_deep': {
                            'prediction': prediction.wide_deep_prediction,
                            'probability': prediction.wide_deep_probability,
                            'confidence': prediction.wide_deep_confidence
                        }
                    }
                },
                'metadata': {
                    'extraction_success': prediction.extraction_success,
                    'extraction_time_ms': prediction.extraction_time_ms,
                    'error_count': prediction.error_count,
                    'processed_at': prediction.created_at.isoformat()
                }
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)




def model_analysis(request):
    """View for ML model analysis dashboard"""
    return render(request, 'ai/model_analysis.html')

def system_performance(request):
    """View for system performance dashboard"""
    return render(request, 'ai/system_performance.html')




# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
import json
import pandas as pd
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

from .models import (
    MLModel, TrainingSession, ModelEvaluation, 
    TrainingVisualization, PredictionLog, DatasetInfo
)
from .forms import TrainingSessionForm, DatasetUploadForm
from .model_training import PDFSteganographyDetector
from .tasks import train_models_task, analyze_dataset_task

@login_required
def ml_dashboard(request):
    """Main dashboard showing overview of models and training sessions"""
    context = {
        'total_models': MLModel.objects.count(),
        'active_trainings': TrainingSession.objects.filter(status='running').count(),
        'completed_trainings': TrainingSession.objects.filter(status='completed').count(),
        'recent_models': MLModel.objects.select_related('created_by')[:5],
        'recent_sessions': TrainingSession.objects.select_related('created_by')[:5],
    }
    return render(request, 'ml_pipeline/dashboard.html', context)


@login_required
def model_list(request):
    """List all trained models"""
    models = MLModel.objects.select_related('created_by').prefetch_related('evaluations')
    return render(request, 'ml_pipeline/model_list.html', {'models': models})


@login_required
def model_detail(request, model_id):
    """Show detailed information about a specific model"""
    model = get_object_or_404(MLModel, id=model_id)
    evaluations = model.evaluations.all()
    visualizations = model.visualizations.all()
    recent_predictions = model.predictions.all()[:10]
    
    context = {
        'model': model,
        'evaluations': evaluations,
        'visualizations': visualizations,
        'recent_predictions': recent_predictions,
    }
    return render(request, 'ml_pipeline/model_detail.html', context)


@login_required
def training_session_list(request):
    """List all training sessions"""
    sessions = TrainingSession.objects.select_related('created_by')
    return render(request, 'ml_pipeline/training_session_list.html', {'sessions': sessions})


@login_required
def create_training_session(request):
    """Create a new training session"""
    if request.method == 'POST':
        form = TrainingSessionForm(request.POST, request.FILES)
        if form.is_valid():
            session = form.save(commit=False)
            session.created_by = request.user
            session.save()
            
            # Analyze uploaded dataset
            analyze_dataset_task.delay(session.id)
            
            messages.success(request, 'Training session created successfully!')
            return redirect('ai:training_session_detail', session_id=session.id)
    else:
        form = TrainingSessionForm()
    
    return render(request, 'ml_pipeline/create_training_session.html', {'form': form})


@login_required
def training_session_detail(request, session_id):
    """Show detailed information about a training session"""
    session = get_object_or_404(TrainingSession, id=session_id)
    models = MLModel.objects.filter(
        evaluations__training_session=session
    ).distinct()
    visualizations = session.visualizations.all()
    
    context = {
        'session': session,
        'models': models,
        'visualizations': visualizations,
    }
    return render(request, 'ml_pipeline/training_session_detail.html', context)


@login_required
def start_training(request, session_id):
    """Start training process for a session"""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    if session.status != 'pending':
        messages.error(request, 'Training session is not in pending state.')
        return redirect('ai:training_session_detail', session_id=session_id)
    
    # Update status
    session.status = 'running'
    session.save()
    
    # Start training task (asynchronous)
    train_models_task.delay(session.id)
    
    messages.success(request, 'Training started successfully!')
    return redirect('ai:training_session_detail', session_id=session_id)


@login_required
def dataset_list(request):
    """List all uploaded datasets"""
    datasets = DatasetInfo.objects.select_related('created_by')
    return render(request, 'ml_pipeline/dataset_list.html', {'datasets': datasets})


@login_required
def upload_dataset(request):
    """Upload a new dataset"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.created_by = request.user
            dataset.save()
            
            # Analyze dataset
            analyze_dataset_info(dataset)
            
            messages.success(request, 'Dataset uploaded successfully!')
            return redirect('ai:dataset_list')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'ml_pipeline/upload_dataset.html', {'form': form})


@login_required
def predict_view(request):
    """Make predictions using trained models"""
    models = MLModel.objects.filter(status='completed')
    
    if request.method == 'POST':
        model_id = request.POST.get('model_id')
        model = get_object_or_404(MLModel, id=model_id)
        
        # Handle file upload or manual feature input
        if 'prediction_file' in request.FILES:
            # File-based prediction
            prediction_file = request.FILES['prediction_file']
            results = make_file_prediction(model, prediction_file, request.user)
        else:
            # Manual feature input
            features = extract_features_from_form(request.POST)
            results = make_feature_prediction(model, features, request.user) # type: ignore
        
        return JsonResponse(results)
    
    return render(request, 'ml_pipeline/predict.html', {'models': models})


def analyze_dataset_info(dataset):
    """Analyze dataset and update statistics"""
    try:
        # Read the dataset
        if dataset.file_format.lower() == 'csv':
            df = pd.read_csv(dataset.file.path)
        else:
            return  # Unsupported format
        
        # Update statistics
        dataset.total_samples = len(df)
        dataset.feature_count = len(df.columns)
        dataset.missing_values_count = df.isnull().sum().sum()
        dataset.duplicate_rows = df.duplicated().sum()
        
        # Check for label column
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            dataset.positive_samples = label_counts.get(1, 0)
            dataset.negative_samples = label_counts.get(0, 0)
        
        dataset.save()
        
    except Exception as e:
        print(f"Error analyzing dataset {dataset.id}: {str(e)}")


def create_visualization(session, model, chart_type, data, title):
    """Create and save a visualization"""
    plt.figure(figsize=(10, 8))
    
    if chart_type == 'confusion_matrix':
        cm = np.array(data['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
    elif chart_type == 'roc_curve':
        plt.plot(data['fpr'], data['tpr'], label=f'ROC Curve (AUC = {data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend()
        
    elif chart_type == 'feature_importance':
        features = list(data['features'])
        importance = list(data['importance'])
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx[-20:]]  # Top 20
        importance = [importance[i] for i in sorted_idx[-20:]]
        
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {title}')
        plt.tight_layout()
        
    elif chart_type == 'training_history':
        epochs = range(1, len(data['loss']) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(epochs, data['loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, data['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, data['accuracy'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, data['val_accuracy'], 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
    
    # Save plot to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Create visualization record
    visualization = TrainingVisualization.objects.create(
        training_session=session,
        model=model,
        chart_type=chart_type,
        title=title,
        chart_data=data
    )
    
    # Save image
    image_file = ContentFile(image_png)
    visualization.chart_image.save(
        f'{chart_type}_{model.id}_{session.id}.png',
        image_file,
        save=True
    )
    
    return visualization


def make_file_prediction(model, prediction_file, user):
    """Make prediction from uploaded file"""
    try:
        # Save file temporarily
        file_path = default_storage.save(
            f'temp/{prediction_file.name}',
            ContentFile(prediction_file.read())
        )
        
        # Load the ML pipeline
        detector = PDFSteganographyDetector()
        detector.load_models(os.path.dirname(model.model_file.path))
        
        # Read and prepare data
        df = pd.read_csv(default_storage.path(file_path))
        
        # Make predictions
        predictions = detector.predict(df)
        
        # Log prediction
        prediction_log = PredictionLog.objects.create(
            model=model,
            input_file=prediction_file,
            xgboost_prediction=int(predictions['xgboost_prediction'][0]),
            xgboost_probability=float(predictions['xgboost_probability'][0]),
            wide_deep_prediction=int(predictions['wide_deep_prediction'][0]),
            wide_deep_probability=float(predictions['wide_deep_probability'][0]),
            prediction_time=0.1,  # Placeholder
            created_by=user
        )
        
        # Clean up temp file
        default_storage.delete(file_path)
        
        return {
            'success': True,
            'predictions': {
                'xgboost': {
                    'prediction': int(predictions['xgboost_prediction'][0]),
                    'probability': float(predictions['xgboost_probability'][0])
                },
                'wide_deep': {
                    'prediction': int(predictions['wide_deep_prediction'][0]),
                    'probability': float(predictions['wide_deep_probability'][0])
                }
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_features_from_form(post_data):
    """Extract features from form data"""
    features = {}
    feature_fields = [
        'file_size_bytes', 'pdf_version', 'num_pages', 'num_objects',
        'num_stream_objects', 'num_embedded_files', 'num_annotation_objects',
        'num_form_fields', 'creation_date_ts', 'mod_date_ts',
        'avg_entropy_per_stream', 'max_entropy_per_stream',
        # Add all other features...
    ]
    
    for field in feature_fields:
        if field in post_data:
            try:
                features[field] = float(post_data[field])
            except (ValueError, TypeError):
                features[field] = 0.0
    
    return features


@login_required
def training_status_api(request, session_id):
    """API endpoint to check training status"""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    return JsonResponse({
        'status': session.status,
        'completed_at': session.completed_at.isoformat() if session.completed_at else None,
        'error_message': session.error_message,
    })


@login_required
def model_comparison(request):
    """Compare multiple models"""
    models = MLModel.objects.filter(status='completed').prefetch_related('evaluations')
    
    comparison_data = []
    for model in models:
        latest_eval = model.evaluations.first()
        if latest_eval:
            comparison_data.append({
                'model': model,
                'evaluation': latest_eval
            })
    
    return render(request, 'ml_pipeline/model_comparison.html', {
        'comparison_data': comparison_data
    })

'''
# Celery tasks (these would normally be in a separate tasks.py file)
def analyze_dataset_task(session_id):
    """Background task to analyze dataset"""
    try:
        session = TrainingSession.objects.get(id=session_id)
        df = pd.read_csv(session.training_data_file.path)
        
        # Update session with data info
        session.data_shape = {'rows': len(df), 'columns': len(df.columns)}
        
        if 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            session.class_distribution = label_counts
        
        session.save()
        
    except Exception as e:
        session = TrainingSession.objects.get(id=session_id)
        session.status = 'failed'
        session.error_message = str(e)
        session.save()


def train_models_task(session_id):
    """Background task to train models"""
    try:
        session = TrainingSession.objects.get(id=session_id)
        
        # Initialize detector
        detector = PDFSteganographyDetector()
        
        # Load and prepare data
        df = pd.read_csv(session.training_data_file.path)
        X, y = detector.prepare_data(df)
        
        # Create models
        xgb_model = MLModel.objects.create(
            name=f"XGBoost - {session.name}",
            model_type='xgboost',
            created_by=session.created_by,
            status='training'
        )
        
        wide_deep_model = MLModel.objects.create(
            name=f"Wide & Deep - {session.name}",
            model_type='wide_deep',
            created_by=session.created_by,
            status='training'
        )
        
        # Train and evaluate (simplified)
        # Train XGBoost model
        xgb_results = detector.train_xgboost(X, y)
        xgb_model_path = os.path.join(settings.MEDIA_ROOT, f'models/xgb_{session.id}.pkl')
        joblib.dump(xgb_results['model'], xgb_model_path)
        xgb_model.model_file.name = f'models/xgb_{session.id}.pkl'
        xgb_model.save()

        # Create XGBoost evaluation
        ModelEvaluation.objects.create(
            model=xgb_model,
            training_session=session,
            accuracy=xgb_results['metrics']['accuracy'],
            precision=xgb_results['metrics']['precision'],
            recall=xgb_results['metrics']['recall'],
            f1_score=xgb_results['metrics']['f1'],
            auc_roc=xgb_results['metrics']['auc_roc'],
            confusion_matrix=xgb_results['metrics']['confusion_matrix'].tolist()
        )

        # Create XGBoost visualizations
        create_visualization(
            session=session,
            model=xgb_model,
            chart_type='confusion_matrix',
            data={'confusion_matrix': xgb_results['metrics']['confusion_matrix'].tolist()},
            title='XGBoost Confusion Matrix'
        )

        create_visualization(
            session=session,
            model=xgb_model,
            chart_type='roc_curve',
            data={
                'fpr': xgb_results['metrics']['fpr'].tolist(),
                'tpr': xgb_results['metrics']['tpr'].tolist(),
                'auc': xgb_results['metrics']['auc_roc']
            },
            title='XGBoost ROC Curve'
        )

        # Train Wide & Deep model
        wide_deep_results = detector.train_wide_deep(X, y)
        wide_deep_model_path = os.path.join(settings.MEDIA_ROOT, f'models/wide_deep_{session.id}')
        wide_deep_results['model'].save(wide_deep_model_path)
        wide_deep_model.model_file.name = f'models/wide_deep_{session.id}'
        wide_deep_model.save()

        # Create Wide & Deep evaluation
        ModelEvaluation.objects.create(
            model=wide_deep_model,
            training_session=session,
            accuracy=wide_deep_results['metrics']['accuracy'],
            precision=wide_deep_results['metrics']['precision'],
            recall=wide_deep_results['metrics']['recall'],
            f1_score=wide_deep_results['metrics']['f1'],
            auc_roc=wide_deep_results['metrics']['auc_roc'],
            confusion_matrix=wide_deep_results['metrics']['confusion_matrix'].tolist()
        )

        # Create Wide & Deep visualizations
        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='confusion_matrix',
            data={'confusion_matrix': wide_deep_results['metrics']['confusion_matrix'].tolist()},
            title='Wide & Deep Confusion Matrix'
        )

        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='roc_curve',
            data={
                'fpr': wide_deep_results['metrics']['fpr'].tolist(),
                'tpr': wide_deep_results['metrics']['tpr'].tolist(),
                'auc': wide_deep_results['metrics']['auc_roc']
            },
            title='Wide & Deep ROC Curve'
        )

        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='training_history',
            data={
                'loss': wide_deep_results['history']['loss'],
                'val_loss': wide_deep_results['history']['val_loss'],
                'accuracy': wide_deep_results['history']['accuracy'],
                'val_accuracy': wide_deep_results['history']['val_accuracy']
            },
            title='Wide & Deep Training History'
        )
        # Update session status
        session.status = 'completed'
        session.completed_at = timezone.now()
        session.save()
        
        # Update model status
        xgb_model.status = 'completed'
        wide_deep_model.status = 'completed'
        xgb_model.save()
        wide_deep_model.save()
        
    except Exception as e:
        session = TrainingSession.objects.get(id=session_id)
        session.status = 'failed'
        session.error_message = str(e)
        session.save()

'''