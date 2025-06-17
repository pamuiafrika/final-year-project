# admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.db.models import Count, Q
import json
from .models import (
    MLModel, TrainingSession, ModelEvaluation, 
    TrainingVisualization, PredictionLog, DatasetInfo,PDFUpload
)


class ModelEvaluationInline(admin.TabularInline):
    """Inline display for model evaluations"""
    model = ModelEvaluation
    fields = ('train_accuracy', 'test_accuracy', 'train_auc', 'test_auc', 'cv_mean', 'cv_std')
    readonly_fields = ('train_accuracy', 'test_accuracy', 'train_auc', 'test_auc', 'cv_mean', 'cv_std')
    extra = 0
    can_delete = False

admin.site.register(PDFUpload)


class TrainingVisualizationInline(admin.TabularInline):
    """Inline display for training visualizations"""
    model = TrainingVisualization
    fields = ('chart_type', 'title', 'chart_image_preview')
    readonly_fields = ('chart_image_preview',)
    extra = 0
    can_delete = False
    
    def chart_image_preview(self, obj):
        if obj.chart_image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 100px;" />',
                obj.chart_image.url
            )
        return "No image"
    chart_image_preview.short_description = "Preview"


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'model_type', 'status_badge', 'is_active', 'created_by', 
        'training_duration_display', 'created_at', 'evaluation_count'
    )
    list_filter = ('model_type', 'status', 'created_at', 'created_by')
    search_fields = ('name', 'description', 'created_by__username')
    readonly_fields = ('created_at', 'updated_at', 'hyperparameters_display', 'feature_columns_display')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'model_type', 'status', 'description', 'created_by', 'is_active')
        }),
        ('Model Files', {
            'fields': ('model_file', 'preprocessor_file'),
            'classes': ('collapse',)
        }),
        ('Training Configuration', {
            'fields': ('hyperparameters_display', 'feature_columns_display', 'training_duration'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [ModelEvaluationInline, TrainingVisualizationInline]
    
    def status_badge(self, obj):
        colors = {
            'training': 'orange',
            'completed': 'green',
            'failed': 'red',
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            colors.get(obj.status, 'black'),
            obj.get_status_display()
        )
    status_badge.short_description = "Status"
    
    def training_duration_display(self, obj):
        if obj.training_duration:
            return f"{obj.training_duration:.2f}s"
        return "N/A"
    training_duration_display.short_description = "Training Time"
    
    def hyperparameters_display(self, obj):
        if obj.hyperparameters:
            return format_html('<pre>{}</pre>', json.dumps(obj.hyperparameters, indent=2))
        return "No hyperparameters"
    hyperparameters_display.short_description = "Hyperparameters"
    
    def feature_columns_display(self, obj):
        if obj.feature_columns:
            return format_html('<pre>{}</pre>', json.dumps(obj.feature_columns, indent=2))
        return "No feature columns"
    feature_columns_display.short_description = "Feature Columns"
    
    def evaluation_count(self, obj):
        count = obj.evaluations.count()
        if count > 0:
            url = reverse('admin:ai_modelevaluation_changelist') + f'?model__id__exact={obj.id}'
            return format_html('<a href="{}">{} evaluations</a>', url, count)
        return "0 evaluations"
    evaluation_count.short_description = "Evaluations"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('created_by')


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'status_badge', 'created_by', 'duration_display', 
        'data_samples_display', 'created_at', 'models_count'
    )
    list_filter = ('status', 'created_at', 'created_by', 'stratify')
    search_fields = ('name', 'created_by__username')
    readonly_fields = ('created_at', 'duration_calc', 'data_shape_display', 'class_distribution_display')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'status', 'created_by')
        }),
        ('Training Data', {
            'fields': ('training_data_file', 'data_shape_display', 'class_distribution_display')
        }),
        ('Configuration', {
            'fields': ('test_size', 'random_state', 'stratify'),
            'classes': ('collapse',)
        }),
        ('Execution', {
            'fields': ('created_at', 'completed_at', 'duration_calc', 'error_message'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [TrainingVisualizationInline]
    
    def status_badge(self, obj):
        colors = {
            'pending': 'gray',
            'running': 'blue',
            'completed': 'green',
            'failed': 'red',
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            colors.get(obj.status, 'black'),
            obj.get_status_display()
        )
    status_badge.short_description = "Status"
    
    def duration_display(self, obj):
        if obj.completed_at and obj.created_at:
            duration = obj.completed_at - obj.created_at
            return str(duration).split('.')[0]  # Remove microseconds
        return "In progress" if obj.status == 'running' else "N/A"
    duration_display.short_description = "Duration"
    
    def duration_calc(self, obj):
        return self.duration_display(obj)
    duration_calc.short_description = "Training Duration"
    
    def data_samples_display(self, obj):
        if obj.data_shape:
            rows = obj.data_shape.get('rows', 'N/A')
            cols = obj.data_shape.get('columns', 'N/A')
            return f"{rows} × {cols}"
        return "N/A"
    data_samples_display.short_description = "Data Shape"
    
    def data_shape_display(self, obj):
        if obj.data_shape:
            return format_html('<pre>{}</pre>', json.dumps(obj.data_shape, indent=2))
        return "No data shape info"
    data_shape_display.short_description = "Data Shape Details"
    
    def class_distribution_display(self, obj):
        if obj.class_distribution:
            return format_html('<pre>{}</pre>', json.dumps(obj.class_distribution, indent=2))
        return "No class distribution info"
    class_distribution_display.short_description = "Class Distribution"
    
    def models_count(self, obj):
        # Count models associated with this training session through evaluations
        count = MLModel.objects.filter(evaluations__training_session=obj).distinct().count()
        return f"{count} models"
    models_count.short_description = "Associated Models"


@admin.register(ModelEvaluation)
class ModelEvaluationAdmin(admin.ModelAdmin):
    list_display = (
        'model_link', 'training_session_link', 'test_accuracy_display', 
        'test_auc_display', 'cv_mean_display', 'created_at'
    )
    list_filter = ('created_at', 'model__model_type')
    search_fields = ('model__name', 'training_session__name')
    readonly_fields = (
        'created_at', 'classification_report_display', 'confusion_matrix_display',
        'feature_importance_display', 'cv_scores_display'
    )
    
    fieldsets = (
        ('Associated Records', {
            'fields': ('model', 'training_session')
        }),
        ('Accuracy Metrics', {
            'fields': ('train_accuracy', 'test_accuracy', 'train_auc', 'test_auc')
        }),
        ('Cross Validation', {
            'fields': ('cv_mean', 'cv_std', 'cv_scores_display'),
            'classes': ('collapse',)
        }),
        ('Detailed Analysis', {
            'fields': ('classification_report_display', 'confusion_matrix_display', 'feature_importance_display'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def model_link(self, obj):
        url = reverse('admin:ai_mlmodel_change', args=[obj.model.pk])
        return format_html('<a href="{}">{}</a>', url, obj.model.name)
    model_link.short_description = "Model"
    
    def training_session_link(self, obj):
        url = reverse('admin:ai_trainingsession_change', args=[obj.training_session.pk])
        return format_html('<a href="{}">{}</a>', url, obj.training_session.name)
    training_session_link.short_description = "Training Session"
    
    def test_accuracy_display(self, obj):
        if obj.test_accuracy:
            return f"{obj.test_accuracy:.4f}"
        return "N/A"
    test_accuracy_display.short_description = "Test Accuracy"
    
    def test_auc_display(self, obj):
        if obj.test_auc:
            return f"{obj.test_auc:.4f}"
        return "N/A"
    test_auc_display.short_description = "Test AUC"
    
    def cv_mean_display(self, obj):
        if obj.cv_mean:
            return f"{obj.cv_mean:.4f} ± {obj.cv_std:.4f}" if obj.cv_std else f"{obj.cv_mean:.4f}"
        return "N/A"
    cv_mean_display.short_description = "CV Score"
    
    def classification_report_display(self, obj):
        if obj.classification_report:
            return format_html('<pre style="font-size: 12px;">{}</pre>', 
                             json.dumps(obj.classification_report, indent=2))
        return "No classification report"
    classification_report_display.short_description = "Classification Report"
    
    def confusion_matrix_display(self, obj):
        if obj.confusion_matrix:
            return format_html('<pre>{}</pre>', json.dumps(obj.confusion_matrix, indent=2))
        return "No confusion matrix"
    confusion_matrix_display.short_description = "Confusion Matrix"
    
    def feature_importance_display(self, obj):
        if obj.feature_importance:
            return format_html('<pre style="font-size: 12px;">{}</pre>', 
                             json.dumps(obj.feature_importance, indent=2))
        return "No feature importance data"
    feature_importance_display.short_description = "Feature Importance"
    
    def cv_scores_display(self, obj):
        if obj.cv_scores:
            return format_html('<pre>{}</pre>', json.dumps(obj.cv_scores, indent=2))
        return "No CV scores"
    cv_scores_display.short_description = "CV Scores Details"


@admin.register(TrainingVisualization)
class TrainingVisualizationAdmin(admin.ModelAdmin):
    list_display = ('title', 'chart_type', 'model_link', 'training_session_link', 'image_preview', 'created_at')
    list_filter = ('chart_type', 'created_at')
    search_fields = ('title', 'model__name', 'training_session__name')
    readonly_fields = ('created_at', 'chart_data_display', 'image_preview_large')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'chart_type', 'training_session', 'model')
        }),
        ('Visualization', {
            'fields': ('chart_image', 'image_preview_large')
        }),
        ('Data', {
            'fields': ('chart_data_display',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def model_link(self, obj):
        if obj.model:
            url = reverse('admin:ai_mlmodel_change', args=[obj.model.pk])
            return format_html('<a href="{}">{}</a>', url, obj.model.name)
        return "N/A"
    model_link.short_description = "Model"
    
    def training_session_link(self, obj):
        url = reverse('admin:ai_trainingsession_change', args=[obj.training_session.pk])
        return format_html('<a href="{}">{}</a>', url, obj.training_session.name)
    training_session_link.short_description = "Training Session"
    
    def image_preview(self, obj):
        if obj.chart_image:
            return format_html(
                '<img src="{}" style="max-height: 50px; max-width: 100px;" />',
                obj.chart_image.url
            )
        return "No image"
    image_preview.short_description = "Preview"
    
    def image_preview_large(self, obj):
        if obj.chart_image:
            return format_html(
                '<img src="{}" style="max-height: 300px; max-width: 500px;" />',
                obj.chart_image.url
            )
        return "No image available"
    image_preview_large.short_description = "Chart Preview"
    
    def chart_data_display(self, obj):
        if obj.chart_data:
            return format_html('<pre style="font-size: 12px;">{}</pre>', 
                             json.dumps(obj.chart_data, indent=2))
        return "No chart data"
    chart_data_display.short_description = "Chart Data"


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        'created_at', 'model_link', 'xgboost_result', 'wide_deep_result', 
        'prediction_time_display', 'created_by'
    )
    list_filter = ('created_at', 'model__model_type', 'created_by')
    search_fields = ('model__name', 'created_by__username', 'notes')
    readonly_fields = ('created_at', 'input_features_display')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('model', 'created_by', 'created_at')
        }),
        ('Input Data', {
            'fields': ('input_file', 'input_features_display')
        }),
        ('Predictions', {
            'fields': (
                ('xgboost_prediction', 'xgboost_probability'),
                ('wide_deep_prediction', 'wide_deep_probability')
            )
        }),
        ('Metadata', {
            'fields': ('prediction_time', 'notes'),
            'classes': ('collapse',)
        }),
    )
    
    def model_link(self, obj):
        url = reverse('admin:ai_mlmodel_change', args=[obj.model.pk])
        return format_html('<a href="{}">{}</a>', url, obj.model.name)
    model_link.short_description = "Model"
    
    def xgboost_result(self, obj):
        if obj.xgboost_prediction is not None:
            result = "Steganographic" if obj.xgboost_prediction == 1 else "Clean"
            prob = f" ({obj.xgboost_probability:.4f})" if obj.xgboost_probability else ""
            color = "red" if obj.xgboost_prediction == 1 else "green"
            return format_html('<span style="color: {};">{}{}</span>', color, result, prob)
        return "N/A"
    xgboost_result.short_description = "XGBoost"
    
    def wide_deep_result(self, obj):
        if obj.wide_deep_prediction is not None:
            result = "Steganographic" if obj.wide_deep_prediction == 1 else "Clean"
            prob = f" ({obj.wide_deep_probability:.4f})" if obj.wide_deep_probability else ""
            color = "red" if obj.wide_deep_prediction == 1 else "green"
            return format_html('<span style="color: {};">{}{}</span>', color, result, prob)
        return "N/A"
    wide_deep_result.short_description = "Wide & Deep"
    
    def prediction_time_display(self, obj):
        return f"{obj.prediction_time:.4f}s"
    prediction_time_display.short_description = "Time"
    
    def input_features_display(self, obj):
        if obj.input_features:
            return format_html('<pre style="font-size: 12px;">{}</pre>', 
                             json.dumps(obj.input_features, indent=2))
        return "No input features"
    input_features_display.short_description = "Input Features"


@admin.register(DatasetInfo)
class DatasetInfoAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'samples_summary', 'feature_count', 'quality_summary', 
        'file_size_display', 'created_by', 'created_at'
    )
    list_filter = ('file_format', 'created_at', 'created_by')
    search_fields = ('name', 'description', 'created_by__username')
    readonly_fields = ('created_at', 'updated_at', 'file_size_display', 'balance_ratio')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'created_by')
        }),
        ('Dataset File', {
            'fields': ('file', 'file_format', 'file_size_display')
        }),
        ('Dataset Statistics', {
            'fields': (
                ('total_samples', 'feature_count'),
                ('positive_samples', 'negative_samples', 'balance_ratio')
            )
        }),
        ('Data Quality', {
            'fields': ('missing_values_count', 'duplicate_rows'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def samples_summary(self, obj):
        if obj.total_samples:
            pos = obj.positive_samples or 0
            neg = obj.negative_samples or 0
            return f"{obj.total_samples} ({pos} pos, {neg} neg)"
        return "N/A"
    samples_summary.short_description = "Samples"
    
    def quality_summary(self, obj):
        issues = []
        if obj.missing_values_count:
            issues.append(f"{obj.missing_values_count} missing")
        if obj.duplicate_rows:
            issues.append(f"{obj.duplicate_rows} duplicates")
        
        if issues:
            return format_html('<span style="color: orange;">{}</span>', ", ".join(issues))
        return format_html('<span style="color: green;">Clean</span>')
    quality_summary.short_description = "Quality"
    
    def file_size_display(self, obj):
        if obj.file_size:
            size_mb = obj.file_size / (1024 * 1024)
            if size_mb >= 1:
                return f"{size_mb:.2f} MB"
            else:
                return f"{obj.file_size / 1024:.2f} KB"
        return "N/A"
    file_size_display.short_description = "File Size"
    
    def balance_ratio(self, obj):
        if obj.positive_samples and obj.negative_samples:
            ratio = obj.positive_samples / obj.negative_samples
            return f"1:{ratio:.2f}"
        return "N/A"
    balance_ratio.short_description = "Class Balance"


# Admin site configuration
admin.site.site_header = "ML Model Management System"
admin.site.site_title = "ML Admin"
admin.site.index_title = "Machine Learning Model Administration"