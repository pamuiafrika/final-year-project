
# ==============================================
# 6. admin.py - Django Admin
# ==============================================

from django.contrib import admin
from .models import PDFScanResult,PDFAnalysis, AnalysisIndicator

@admin.register(PDFAnalysis)
class PDFAnalysisAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'user', 'assessment', 'risk_score', 
                   'total_indicators', 'upload_date', 'analysis_date']
    list_filter = ['assessment', 'upload_date', 'analysis_date', 'is_anomaly']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['upload_date', 'analysis_date', 'file_size']
    
    fieldsets = (
        ('File Information', {
            'fields': ('user', 'pdf_file', 'original_filename', 'file_size', 'upload_date')
        }),
        ('Analysis Results', {
            'fields': ('assessment', 'risk_score', 'total_indicators', 'analysis_date', 
                      'analysis_duration', 'technique_used')
        }),
        ('Machine Learning', {
            'fields': ('ml_anomaly_score', 'is_anomaly', 'model_version')
        }),
        ('Detailed Data', {
            'fields': ('indicators_data', 'features_data', 'recommendations'),
            'classes': ('collapse',)
        }),
    )

@admin.register(AnalysisIndicator)
class AnalysisIndicatorAdmin(admin.ModelAdmin):
    list_display = ['analysis', 'category', 'severity', 'confidence', 'description']
    list_filter = ['severity', 'category']
    search_fields = ['description', 'analysis__original_filename']


@admin.register(PDFScanResult)
class PDFScanResultAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'user', 'status', 'is_malicious', 'risk_level', 'created_at']
    list_filter = ['status', 'is_malicious', 'risk_level', 'created_at']
    search_fields = ['original_filename', 'user__username']
    readonly_fields = ['id', 'created_at', 'completed_at', 'scan_duration']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'original_filename', 'file', 'file_size', 'status')
        }),
        ('Scan Results', {
            'fields': ('is_malicious', 'ensemble_probability', 'confidence_percentage', 'risk_level')
        }),
        ('Model Predictions', {
            'fields': ('attention_probability', 'deep_ff_probability', 'wide_deep_probability'),
            'classes': ('collapse',)
        }),
        ('Extracted Features', {
            'fields': ('pdf_pages', 'metadata_size', 'suspicious_count', 'javascript_elements', 'auto_actions', 'embedded_files'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )