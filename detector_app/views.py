import os
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from django.urls import reverse
from django.views.generic import ListView, DetailView, CreateView
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Dataset, TrainedModel, PDFScan, BulkScan
from .serializers import DatasetSerializer, TrainedModelSerializer, PDFScanSerializer, BulkScanSerializer
from .tasks import process_single_pdf, process_bulk_pdfs, train_model_task
from .ml.detector import StegoPDFDetector

class HomeView(ListView):
    """Home page view showing recent scans"""
    model = PDFScan
    template_name = 'detector_app/index.html'
    context_object_name = 'recent_scans'
    ordering = ['-uploaded_at']
    paginate_by = 10

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['active_models'] = TrainedModel.objects.filter(is_active=True)
        return context

class DatasetViewSet(viewsets.ModelViewSet):
    """API endpoint for datasets"""
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    
    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        """Process a dataset for training"""
        dataset = self.get_object()
        task = train_model_task.delay(dataset.id)
        return Response({'task_id': task.id}, status=status.HTTP_202_ACCEPTED)

class TrainedModelViewSet(viewsets.ModelViewSet):
    """API endpoint for trained models"""
    queryset = TrainedModel.objects.all()
    serializer_class = TrainedModelSerializer
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Set a model as active"""
        model = self.get_object()
        # Deactivate all other models of the same type
        TrainedModel.objects.filter(model_type=model.model_type).update(is_active=False)
        model.is_active = True
        model.save()
        return Response({'status': 'success'})

class PDFScanViewSet(viewsets.ModelViewSet):
    """API endpoint for PDF scans"""
    queryset = PDFScan.objects.all()
    serializer_class = PDFScanSerializer
    
    def perform_create(self, serializer):
        pdf_scan = serializer.save()
        task = process_single_pdf.delay(pdf_scan.id)
        pdf_scan.task_id = task.id
        pdf_scan.save()
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get the status of a PDF scan"""
        scan = self.get_object()
        return Response({
            'id': scan.id,
            'status': scan.status,
            'result': scan.result,
            'confidence': scan.confidence
        })

class BulkScanViewSet(viewsets.ModelViewSet):
    """API endpoint for bulk scans"""
    queryset = BulkScan.objects.all()
    serializer_class = BulkScanSerializer
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Handle bulk upload of PDF files"""
        files = request.FILES.getlist('files')
        
        if not files:
            return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        bulk_scan = BulkScan.objects.create(
            name=request.data.get('name', f"Bulk Scan {BulkScan.objects.count() + 1}"),
            total_files=len(files)
        )
        
        # Save all files
        pdf_ids = []
        for file in files:
            pdf_scan = PDFScan.objects.create(
                file=file,
                filename=file.name,
                status='pending'
            )
            pdf_ids.append(pdf_scan.id)
        
        # Start bulk processing task
        task = process_bulk_pdfs.delay(bulk_scan.id, pdf_ids)
        bulk_scan.task_id = task.id
        bulk_scan.save()
        
        return Response({
            'id': bulk_scan.id,
            'task_id': task.id,
            'total_files': bulk_scan.total_files
        }, status=status.HTTP_202_ACCEPTED)
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get the status of a bulk scan"""
        scan = self.get_object()
        return Response({
            'id': scan.id,
            'status': scan.status,
            'total_files': scan.total_files,
            'processed_files': scan.processed_files,
            'clean_count': scan.clean_count,
            'stego_count': scan.stego_count,
            'progress': scan.progress
        })

def upload_view(request):
    """View for uploading files"""
    if request.method == 'POST':
        file = request.FILES.get('pdf_file')
        
        if not file:
            return render(request, 'detector_app/upload.html', {'error': 'No file selected'})
        
        if not file.name.lower().endswith('.pdf'):
            return render(request, 'detector_app/upload.html', {'error': 'File must be a PDF'})
        
        pdf_scan = PDFScan.objects.create(
            file=file,
            filename=file.name,
            status='pending'
        )
        
        task = process_single_pdf.delay(pdf_scan.id)
        pdf_scan.task_id = task.id
        pdf_scan.save()
        
        return redirect('scan_results', pk=pdf_scan.id)
    
    return render(request, 'detector_app/upload.html')

def scan_results(request, pk):
    """View for displaying scan results"""
    scan = get_object_or_404(PDFScan, pk=pk)
    return render(request, 'detector_app/results.html', {'scan': scan})

def check_scan_status(request, pk):
    """AJAX endpoint for checking scan status"""
    scan = get_object_or_404(PDFScan, pk=pk)
    return JsonResponse({
        'status': scan.status,
        'result': scan.result,
        'confidence': scan.confidence
    })

def check_bulk_scan_status(request, pk):
    """AJAX endpoint for checking bulk scan status"""
    scan = get_object_or_404(BulkScan, pk=pk)
    return JsonResponse({
        'status': scan.status,
        'total_files': scan.total_files,
        'processed_files': scan.processed_files,
        'clean_count': scan.clean_count,
        'stego_count': scan.stego_count,
        'progress': scan.progress
    })
    
    
    
class DatasetListView(ListView):
    model = Dataset
    template_name = 'detector_app/datasets.html'
    context_object_name = 'datasets'
    
from django.views import View
from django.views.generic import TemplateView
from celery.result import AsyncResult

class TrainModelView(View):
    def get(self, request, dataset_id):
        task = train_model_task.delay(dataset_id)
        return redirect('training_status', task_id=task.id)

class TrainingStatusView(TemplateView):
    template_name = 'detector_app/training_status.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        task_id = self.kwargs['task_id']
        task = AsyncResult(task_id)
        context['task'] = task
        context['task_id'] = task_id
        context['status'] = task.status
        if task.status == 'SUCCESS':
            context['result'] = task.result
        return context
    
    

class TrainedModelListView(ListView):
    model = TrainedModel
    template_name = 'detector_app/models.html'
    context_object_name = 'models'
    
    
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .models import TrainedModel
from .ml.detector import StegoPDFDetector
import os
from django.conf import settings

class DetectionAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        pdf_file = request.FILES.get('pdf')
        if not pdf_file:
            return Response({'error': 'No PDF file provided'}, status=400)
        
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp.pdf')
        with open(temp_path, 'wb') as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)
        
        try:
            model = TrainedModel.objects.get(is_active=True)
        except TrainedModel.DoesNotExist:
            return Response({'error': 'No active model available'}, status=500)
        
        detector = StegoPDFDetector(model.file_path)
        result, confidence = detector.detect(temp_path)
        
        os.remove(temp_path)
        
        return Response({
            'result': 'stego' if result else 'clean',
            'confidence': confidence
        })