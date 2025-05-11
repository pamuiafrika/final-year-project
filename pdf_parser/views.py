import os
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from django.db import transaction
from .models import PDFDocument, AnomalyReport
from .modules import PDFStegAnalyzer
from .forms import PDFUploadForm
import datetime

def index(request):
    """
    View for the home page
    """
    form = PDFUploadForm()
    recent_pdfs = PDFDocument.objects.all().order_by('-upload_date')[:5]
    analysis_no = PDFDocument.objects.count()
    positive_analysis = PDFDocument.objects.filter(has_anomalies=True).count()
    false_positives = PDFDocument.objects.filter(has_anomalies=True, is_analyzed=False).count()
    pending_analysis = PDFDocument.objects.filter(is_analyzed=False).count()
    
    context = {
        'form': form,
        'recent_pdfs': recent_pdfs,
        'analysis_no': analysis_no,
        'positive_analysis': positive_analysis,
        'false_positives': false_positives,
        'pending_analysis': pending_analysis,
    }
    
    return render(request, 'pdf_parser/index.html', context)

def analysis_list(request):
    """
    View for listing all analyzed PDFs
    """
    pdf_documents = PDFDocument.objects.all().order_by('-upload_date')
    
    context = {
        'pdf_documents': pdf_documents,
    }
    
    return render(request, 'pdf_parser/analysis_list.html', context)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        return super().default(obj)

@csrf_exempt
@transaction.atomic
def upload_pdf(request):
    """
    View for handling PDF uploads
    """
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded PDF
            pdf_document = form.save()
            
            try:
                # Analyze the PDF for steganographic content
                pdf_path = pdf_document.file.path
                analyzer = PDFStegAnalyzer(pdf_path)
                report = analyzer.analyze_pdf()
                
                # Update the PDF document with analysis results
                pdf_document.title = os.path.basename(pdf_path)
                pdf_document.is_analyzed = True
                
                # Extract basic metadata
                if 'metadata' in report['results'] and 'raw_metadata' in report['results']['metadata']:
                    metadata = report['results']['metadata']['raw_metadata']
                    if 'Author' in metadata:
                        pdf_document.author = metadata['Author']
                    if 'CreationDate' in metadata:
                        pdf_document.creation_date = metadata['CreationDate']
                    if 'ModDate' in metadata:
                        pdf_document.modification_date = metadata['ModDate']
                
                # Extract structure info
                if 'structure' in report['results'] and 'basic_info' in report['results']['structure']:
                    pdf_document.num_pages = report['results']['structure']['basic_info'].get('num_pages', 0)
                
                # Extract image info
                if 'images' in report['results'] and 'summary' in report['results']['images']:
                    pdf_document.num_images = report['results']['images']['summary'].get('total_embedded_images', 0)
                
                # Set anomalies flag
                pdf_document.has_anomalies = report['summary']['suspicious_score'] >= 2.0
                pdf_document.suspicious_areas = report['summary']['anomalies_detected']
                
                pdf_document.save()
                
                # Create anomaly reports
                if 'metadata' in report['results'] and 'anomalies' in report['results']['metadata']:
                    for anomaly in report['results']['metadata']['anomalies']:
                        AnomalyReport.objects.create(
                            pdf=pdf_document,
                            anomaly_type=anomaly['type'],
                            description=anomaly['description'],
                            confidence_score=0.8 if anomaly.get('severity') == 'high' else 
                                         (0.6 if anomaly.get('severity') == 'medium' else 0.4)
                        )
                
                if 'images' in report['results'] and 'potential_hidden_pngs' in report['results']['images']:
                    for png in report['results']['images']['potential_hidden_pngs']:
                        if png.get('suspicious', False):
                            AnomalyReport.objects.create(
                                pdf=pdf_document,
                                anomaly_type='hidden_png',
                                description=f"Potential hidden PNG found at position {png['position']}",
                                confidence_score=0.7,
                                location_data={'position': png['position']}
                            )
                
                # Save the full report as a file
                report_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(report_dir, f"{pdf_document.id}_report.json")
                
                # Use the custom encoder to handle datetime objects
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, cls=DateTimeEncoder)
                
                messages.success(request, f"PDF analyzed successfully. Suspicious score: {report['summary']['suspicious_score']}")
                return redirect('pdf_parser:pdf_detail', pk=pdf_document.id)
                
            except Exception as e:
                messages.error(request, f"Error analyzing PDF: {str(e)}")
                return redirect('pdf_parser:index')
        else:
            messages.error(request, "Invalid form submission.")
            return redirect('pdf_parser:index')
    
    return render(request, 'pdf_parser/upload.html')


def pdf_detail(request, pk):
    try:
        pdf_document = PDFDocument.objects.get(pk=pk)
        anomalies = pdf_document.anomalies.all()
        
        # Load the full report if it exists
        report_data = {}
        report_path = os.path.join(settings.MEDIA_ROOT, 'reports', f"{pdf_document.id}_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
            except json.JSONDecodeError as e:
                # Log the error for debugging
                print(f"JSON decode error in {report_path}: {str(e)}")
                messages.warning(request, "Could not load the full report due to a formatting issue.")
                # Continue with empty report data
        
        context = {
            'pdf': pdf_document,
            'anomalies': anomalies,
            'report': report_data
        }
        
        return render(request, 'pdf_parser/pdf_detail.html', context)
        
    except PDFDocument.DoesNotExist:
        messages.error(request, "PDF not found.")
        return redirect('pdf_parser:index')

def download_report(request, pk):
    """
    View for downloading the JSON report
    """
    try:
        pdf_document = PDFDocument.objects.get(pk=pk)
        report_path = os.path.join(settings.MEDIA_ROOT, 'reports', f"{pdf_document.id}_report.json")
        
        if os.path.exists(report_path):
            with open(report_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='application/json')
                response['Content-Disposition'] = f'attachment; filename="{pdf_document.title}_report.json"'
                return response
        else:
            messages.error(request, "Report not found.")
            return redirect('pdf_parser:pdf_detail', pk=pk)
            
    except PDFDocument.DoesNotExist:
        messages.error(request, "PDF not found.")
        return redirect('pdf_parser:index')

def api_analyze_pdf(request, pk):
    """
    API endpoint for getting analysis results as JSON
    """
    try:
        pdf_document = PDFDocument.objects.get(pk=pk)
        report_path = os.path.join(settings.MEDIA_ROOT, 'reports', f"{pdf_document.id}_report.json")
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_data = json.load(f)
                return JsonResponse(report_data)
        else:
            return JsonResponse({'error': 'Report not found'}, status=404)
            
    except PDFDocument.DoesNotExist:
        return JsonResponse({'error': 'PDF not found'}, status=404)




import os
import tempfile
from django.shortcuts import render
from django.http import HttpResponse
import pikepdf

def embed_png_in_pdf_view(request):
    if request.method == 'POST':
        uploaded_pdf = request.FILES.get('pdf')
        uploaded_png = request.FILES.get('png')

        if not uploaded_pdf or not uploaded_png:
            return HttpResponse("Both PDF and PNG are required.", status=400)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, uploaded_pdf.name)
            png_path = os.path.join(tmpdir, uploaded_png.name)
            output_path = os.path.join(tmpdir, 'stegano_output.pdf')

            # Save uploaded files to temporary directory
            with open(pdf_path, 'wb+') as f:
                for chunk in uploaded_pdf.chunks():
                    f.write(chunk)

            with open(png_path, 'wb+') as f:
                for chunk in uploaded_png.chunks():
                    f.write(chunk)

            # Embed PNG in PDF as an attachment
            with pikepdf.open(pdf_path) as pdf:
                with open(png_path, 'rb') as png_file:
                    # Add PNG as an attachment to the PDF
                    pdf.attachments[os.path.basename(png_path)] = png_file.read()

                # Save the modified PDF
                pdf.save(output_path)

            # Serve the modified PDF to the user
            with open(output_path, 'rb') as final_pdf:
                response = HttpResponse(final_pdf.read(), content_type='application/pdf')
                response['Content-Disposition'] = 'attachment; filename="stegano_result.pdf"'
                return response

    return render(request, 'embed_stegano_pdf.html')
