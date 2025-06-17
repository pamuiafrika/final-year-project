# ============================================================================
# AI/urls.py - URL Configuration for PDF Steganography Detection
# ============================================================================

from django.urls import path
from . import views

app_name = 'ai'

urlpatterns = [
    # Dashboard
    path('dashboard', views.upload_dashboard, name='dashboard'),
    
    # Upload functionality
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('batch-upload/', views.batch_upload, name='batch_upload'),
    
    # List views
    path('analyses/', views.PDFUploadListView.as_view(), name='analysis_list'),
    path('upload/<uuid:pk>/', views.PDFUploadDetailView.as_view(), name='upload_detail'),
    
    # API endpoints
    path('api/status/<uuid:upload_id>/', views.get_prediction_status, name='prediction_status'),
    path('api/predict/', views.api_predict, name='api_predict'),


    path('batch-upload/', views.batch_upload, name='batch_upload'),
    path('batch-results/', views.batch_results, name='batch_results'),
    # Additional views for complete functionality
    # path('model-analysis/', views.model_analysis_view, name='model_analysis'),
    # path('system-performance/', views.system_performance_view, name='system_performance'),
    # path('bulk-download/', views.bulk_download_results, name='bulk_download'),
    # path('delete-upload/<uuid:upload_id>/', views.delete_upload, name='delete_upload'),
    
    # # AJAX endpoints
    # path('ajax/upload-progress/<uuid:upload_id>/', views.upload_progress, name='upload_progress'),
    # path('ajax/recent-uploads/', views.recent_uploads_ajax, name='recent_uploads_ajax'),

    # Dashboard
    path('ml_pipeline', views.ml_dashboard, name='ml_dashboard'),
    
    # Models
    path('models/', views.model_list, name='model_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/compare/', views.model_comparison, name='model_comparison'),
    
    # Training Sessions
    path('training/', views.training_session_list, name='training_session_list'),
    path('training/create/', views.create_training_session, name='create_training_session'),
    path('training/<int:session_id>/', views.training_session_detail, name='training_session_detail'),
    path('training/<int:session_id>/start/', views.start_training, name='start_training'),
    
    # Datasets
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.upload_dataset, name='upload_dataset'),
    
    # Predictions
    path('predict/', views.predict_view, name='predict'),
    
    # API endpoints
    path('api/training/<int:session_id>/status/', views.training_status_api, name='training_status_api'),
]