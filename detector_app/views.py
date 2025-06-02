
# ==============================================
# 4. views.py - Django Views
# ==============================================


from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.utils import timezone
from .models import PDFScanResult
from .forms import PDFUploadForm
from .inference import DjangoPDFMalwareDetector
import json
import logging

logger = logging.getLogger(__name__)

# Initialize detector (singleton pattern)
_detector_instance = None

def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DjangoPDFMalwareDetector()
    return _detector_instance
    
'''
@login_required
def dashboard(request):
    """Main dashboard view"""
    recent_scans = PDFScanResult.objects.filter(user=request.user)[:5]
    
    # Statistics
    total_scans = PDFScanResult.objects.filter(user=request.user).count()
    malicious_count = PDFScanResult.objects.filter(
        user=request.user, is_malicious=True
    ).count()
    
    context = {
        'recent_scans': recent_scans,
        'total_scans': total_scans,
        'malicious_count': malicious_count,
        'benign_count': total_scans - malicious_count,
        'upload_form': PDFUploadForm()
    }
    
    return render(request, 'pdf_detector/dashboard.html', context)

@login_required
@require_http_methods(['POST'])
def upload_pdf(request):
    """Handle PDF upload and analysis"""
    form = PDFUploadForm(request.POST, request.FILES)
    
    if form.is_valid():
        pdf_file = form.cleaned_data['pdf_file']
        
        # Create scan record
        scan_result = PDFScanResult.objects.create(
            user=request.user,
            file=pdf_file,
            original_filename=pdf_file.name,
            file_size=pdf_file.size,
            status='PROCESSING'
        )
        
        try:
            # Get detector and analyze
            detector = get_detector()
            result = detector.predict(scan_result.file)
            
            # Update scan result
            scan_result.status = 'COMPLETED'
            scan_result.is_malicious = result['is_malicious']
            scan_result.ensemble_probability = result['ensemble_probability']
            scan_result.confidence_percentage = result['confidence']
            scan_result.risk_level = result['risk_level']
            scan_result.completed_at = timezone.now()
            
            # Individual model predictions
            individual = result['individual_predictions']
            scan_result.attention_probability = individual.get('attention', {}).get('probability')
            scan_result.deep_ff_probability = individual.get('deep_ff', {}).get('probability')
            scan_result.wide_deep_probability = individual.get('wide_deep', {}).get('probability')
            
            # Feature extraction
            features = result['extracted_features']
            scan_result.pdf_pages = features.get('pages', 0)
            scan_result.metadata_size = features.get('metadata_size', 0)
            scan_result.suspicious_count = features.get('suspicious_count', 0)
            scan_result.javascript_elements = features.get('JS', 0) + features.get('Javascript', 0)
            scan_result.auto_actions = features.get('AA', 0) + features.get('OpenAction', 0)
            scan_result.embedded_files = features.get('EmbeddedFile', 0)
            
            # Store complete data
            scan_result.extracted_features = features
            scan_result.individual_predictions = individual
            
            scan_result.save()
            
            messages.success(request, f'PDF analysis completed! File is classified as {"MALICIOUS" if result["is_malicious"] else "BENIGN"}')
            return redirect('pdf_detector:scan_detail', scan_id=scan_result.id)
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            scan_result.status = 'FAILED'
            scan_result.error_message = str(e)
            scan_result.completed_at = timezone.now()
            scan_result.save()
            
            messages.error(request, f'Analysis failed: {str(e)}')
            return redirect('pdf_detector:dashboard')
    
    else:
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(request, f'{field}: {error}')
        return redirect('pdf_detector:dashboard')

@login_required
def scan_detail(request, scan_id):
    """View detailed scan results"""
    scan = get_object_or_404(PDFScanResult, id=scan_id, user=request.user)
    
    context = {
        'scan': scan,
    }
    
    return render(request, 'pdf_detector/scan_detail.html', context)

@login_required
def scan_history(request):
    """View scan history with pagination"""
    scans = PDFScanResult.objects.filter(user=request.user)
    
    # Filter by status if requested
    status_filter = request.GET.get('status')
    if status_filter:
        scans = scans.filter(status=status_filter)
    
    # Filter by risk level
    risk_filter = request.GET.get('risk')
    if risk_filter:
        scans = scans.filter(risk_level=risk_filter)
    
    paginator = Paginator(scans, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'risk_filter': risk_filter,
    }
    
    return render(request, 'pdf_detector/scan_history.html', context)

@login_required
def delete_scan(request, scan_id):
    """Delete a scan result"""
    scan = get_object_or_404(PDFScanResult, id=scan_id, user=request.user)
    
    if request.method == 'POST':
        # Delete file
        if scan.file:
            try:
                scan.file.delete()
            except:
                pass
        
        scan.delete()
        messages.success(request, 'Scan result deleted successfully')
        return redirect('pdf_detector:scan_history')
    
    return render(request, 'pdf_detector/confirm_delete.html', {'scan': scan})




'''






from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
import json
from django.conf import settings

from .models import PDFAnalysis
from .forms import PDFUploadForm
from .services import DjangoPDFAnalysisService
#from .tasks import analyze_pdf_async  # For async processing (optional)

@login_required
def dashboard(request):
    """Main dashboard showing user's PDF analyses."""
    analyses = PDFAnalysis.objects.filter(user=request.user)
    
    # Statistics
    total_analyses = analyses.count()
    completed_analyses = analyses.filter(analysis_date__isnull=False).count()
    high_risk_count = analyses.filter(assessment__in=['HIGH_RISK', 'CRITICAL']).count()
    
    # Recent analyses
    recent_analyses = analyses[:10]
    
    context = {
        'total_analyses': total_analyses,
        'completed_analyses': completed_analyses,
        'high_risk_count': high_risk_count,
        'recent_analyses': recent_analyses,
    }
    
    return render(request, 'detector_app/dashboard.html', context)

@login_required
def upload_pdf(request):
    """Handle PDF file upload."""
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = request.FILES['pdf_file']
            
            # Create PDFAnalysis record
            analysis = PDFAnalysis.objects.create(
                user=request.user,
                pdf_file=pdf_file,
                original_filename=pdf_file.name,
                file_size=pdf_file.size,
            )
            
            # Start analysis (sync or async)
            if getattr(settings, 'USE_ASYNC_ANALYSIS', False):
                # Queue for background processing
                analyze_pdf_async.delay(analysis.id, form.cleaned_data.get('technique', 'auto'))
                messages.success(request, f'PDF uploaded successfully. Analysis queued.')
            else:
                # Synchronous analysis
                try:
                    service = DjangoPDFAnalysisService()
                    service.analyze_pdf(analysis, form.cleaned_data.get('technique', 'auto'))
                    messages.success(request, f'PDF analyzed successfully. Risk Level: {analysis.risk_level_display}')
                except Exception as e:
                    messages.error(request, f'Analysis failed: {str(e)}')
            
            return redirect('pdf_detector:analysis_detail', pk=analysis.id)
    else:
        form = PDFUploadForm()
    
    return render(request, 'detector_app/upload.html', {'form': form})

@login_required
def analysis_list(request):
    """List all user's PDF analyses with filtering and pagination."""
    analyses = PDFAnalysis.objects.filter(user=request.user)
    
    # Filtering
    risk_filter = request.GET.get('risk')
    if risk_filter:
        analyses = analyses.filter(assessment=risk_filter)
    
    search = request.GET.get('search')
    if search:
        analyses = analyses.filter(
            Q(original_filename__icontains=search) |
            Q(assessment__icontains=search)
        )
    
    # Pagination
    paginator = Paginator(analyses, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'risk_filter': risk_filter,
        'search': search,
        'risk_choices': PDFAnalysis.RISK_LEVELS,
    }
    
    return render(request, 'detector_app/analysis_list.html', context)

@login_required
def analysis_detail(request, pk):
    """Detailed view of a specific PDF analysis."""
    analysis = get_object_or_404(PDFAnalysis, pk=pk, user=request.user)
    
    # Get raw indicator data for debugging
    raw_indicators = list(analysis.indicators.all().values())

    # Get distinct indicators ordered by severity
    indicators = analysis.indicators.all().distinct().order_by('-severity', '-confidence')
    
    # grouped indicators
    indicators_by_category = {}
    for indicator in indicators:
        if indicator.category not in indicators_by_category:
            indicators_by_category[indicator.category] = []
        indicators_by_category[indicator.category].append(indicator)
    
    
    context = {
        'analysis': analysis,
        'indicators_by_category': indicators_by_category,
        'features_data': json.loads(analysis.features_data) if analysis.features_data else {}
    }
    
    return render(request, 'detector_app/analysis_detail.html', context)

@login_required
def download_report(request, pk):
    """Download detailed analysis report as text file."""
    analysis = get_object_or_404(PDFAnalysis, pk=pk, user=request.user)
    
    if not analysis.is_analyzed:
        messages.error(request, 'Analysis not completed yet.')
        return redirect('detector_app:analysis_detail', pk=pk)
    
    service = DjangoPDFAnalysisService()
    report_content = service.generate_detailed_report(analysis)
    
    response = HttpResponse(report_content, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{analysis.original_filename}_analysis_report.txt"'
    
    return response

@login_required
@require_http_methods(["POST"])
def reanalyze_pdf(request, pk):
    """Re-analyze a PDF with different technique."""
    analysis = get_object_or_404(PDFAnalysis, pk=pk, user=request.user)
    technique = request.POST.get('technique', 'auto')
    
    try:
        service = DjangoPDFAnalysisService()
        service.analyze_pdf(analysis, technique)
        messages.success(request, 'PDF re-analyzed successfully.')
    except Exception as e:
        messages.error(request, f'Re-analysis failed: {str(e)}')
    
    return redirect('pdf_detector:analysis_detail', pk=pk)

@login_required
def api_analysis_status(request, pk):
    """API endpoint to check analysis status."""
    analysis = get_object_or_404(PDFAnalysis, pk=pk, user=request.user)
    
    data = {
        'id': analysis.id,
        'filename': analysis.original_filename,
        'is_analyzed': analysis.is_analyzed,
        'assessment': analysis.get_assessment_display() if analysis.assessment else None,
        'risk_score': analysis.risk_score,
        'total_indicators': analysis.total_indicators,
        'analysis_date': analysis.analysis_date.isoformat() if analysis.analysis_date else None,
    }
    
    return JsonResponse(data)
