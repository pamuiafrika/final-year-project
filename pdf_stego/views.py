"""
View functions for PDF steganography application.
"""

import os
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, Http404
from django.utils import timezone
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import StegoOperation
from .forms import HideForm, ExtractForm
from .steganography import PDFSteganography

def home(request):
    """Home page view"""
    return render(request, 'pdf_stego/home.html')

def hide_form(request):
    """View for the hide form"""
    if request.method == 'POST':
        form = HideForm(request.POST, request.FILES)
        if form.is_valid():
            # Create operation object but don't save yet
            operation = form.save(commit=False)
            operation.operation_type = 'hide'
            operation.status = 'processing'
            operation.save()
            
            # Process the files
            try:
                # Save paths
                input_pdf_path = os.path.join(settings.MEDIA_ROOT, operation.input_pdf.name)
                input_png_path = os.path.join(settings.MEDIA_ROOT, operation.input_png.name)
                
                # Create output path
                output_filename = f"output_{uuid.uuid4().hex}.pdf"
                output_pdf_path = os.path.join(settings.MEDIA_ROOT, 'output_pdfs', output_filename)
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
            
                # Process with steganography engine
                stego = PDFSteganography()
                result, actual_method = stego.hide(operation.method, input_pdf_path, output_pdf_path, input_png_path)
                
                if result:
                    # Update operation with result
                    operation.status = 'completed'
                    operation.actual_method = actual_method if operation.method == 'random' else operation.method
                    
                    # Save output file path
                    relative_path = os.path.join('output_pdfs', output_filename)
                    operation.output_pdf.name = relative_path
                    
                    # Mark completion time
                    operation.completed_at = timezone.now()
                    operation.save()
                    
                    messages.success(request, "Image hidden successfully in PDF!")
                    return redirect('pdf_stego:operation_detail', operation_id=operation.id)
                else:
                    operation.status = 'failed'
                    operation.error_message = actual_method  # In case of failure, actual_method contains error message
                    operation.completed_at = timezone.now()
                    operation.save()
                    messages.error(request, f"Failed to hide image: {actual_method}")
            
            except Exception as e:
                operation.status = 'failed'
                operation.error_message = str(e)
                operation.completed_at = timezone.now()
                operation.save()
                messages.error(request, f"Error processing files: {str(e)}")
    else:
        form = HideForm()

    return render(request, 'pdf_stego/hide_form.html', {'form': form})


def extract_form(request):
    """View for the extract form"""
    if request.method == 'POST':
        form = ExtractForm(request.POST, request.FILES)

        if form.is_valid():
            # Create operation object but don't save yet
            operation = form.save(commit=False)
            operation.operation_type = 'extract'
            operation.status = 'processing'
            operation.save()
            # Process the files
            try:
                # Save paths
                stego_pdf_path = os.path.join(settings.MEDIA_ROOT, operation.stego_pdf.name)
                
                # Create output path
                output_filename = f"extracted_{uuid.uuid4().hex}.png"
                output_png_path = os.path.join(settings.MEDIA_ROOT, 'extracted_pngs', output_filename)
                os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
                
                # Process with steganography engine
                stego = PDFSteganography()
                result, actual_method = stego.extract(operation.method, stego_pdf_path, output_png_path)
                
                if result:
                    # Update operation with result
                    operation.status = 'completed'
                    operation.actual_method = actual_method if operation.method == 'random' else operation.method
                    
                    # Save output file path
                    relative_path = os.path.join('extracted_pngs', output_filename)
                    operation.extracted_png.name = relative_path
                    
                    # Mark completion time
                    operation.completed_at = timezone.now()
                    operation.save()
                    
                    messages.success(request, "Image extracted successfully from PDF!")
                    return redirect('pdf_stego:operation_detail', operation_id=operation.id)
                else:
                    operation.status = 'failed'
                    operation.error_message = actual_method  # In case of failure, actual_method contains error message
                    operation.completed_at = timezone.now()
                    operation.save()
                    messages.error(request, f"Failed to extract image: {actual_method}")
            
            except Exception as e:
                operation.status = 'failed'
                operation.error_message = str(e)
                operation.completed_at = timezone.now()
                operation.save()
                messages.error(request, f"Error processing files: {str(e)}")
    else:
        form = ExtractForm()

    return render(request, 'pdf_stego/extract_form.html', {'form': form})


def operations_list(request):
    """List all operations"""
    operations = StegoOperation.objects.all().order_by('-created_at')
    return render(request, 'pdf_stego/operations_list.html', {'operations': operations})

def operation_detail(request, operation_id):
    """Show operation details"""
    operation = get_object_or_404(StegoOperation, id=operation_id)
    return render(request, 'pdf_stego/operation_detail.html', {'operation': operation})

def download_file(request, operation_id, file_type):
    """Download files associated with an operation"""

    operation = get_object_or_404(StegoOperation, id=operation_id)

    if file_type == 'input_pdf':
        if operation.input_pdf:
            file_path = operation.input_pdf.path
            filename = operation.get_input_pdf_filename()
            content_type = 'application/pdf'
        else:
            raise Http404("File not found")

    elif file_type == 'input_png':
        if operation.input_png:
            file_path = operation.input_png.path
            filename = operation.get_input_png_filename()
            content_type = 'image/png'
        else:
            raise Http404("File not found")

    elif file_type == 'output_pdf':
        if operation.output_pdf:
            file_path = operation.output_pdf.path
            filename = operation.get_output_pdf_filename()
            content_type = 'application/pdf'
        else:
            raise Http404("File not found")

    elif file_type == 'stego_pdf':
        if operation.stego_pdf:
            file_path = operation.stego_pdf.path
            filename = operation.get_stego_pdf_filename()
            content_type = 'application/pdf'
        else:
            raise Http404("File not found")

    elif file_type == 'extracted_png':
        if operation.extracted_png:
            file_path = operation.extracted_png.path
            filename = operation.get_extracted_png_filename()
            content_type = 'image/png'
        else:
            raise Http404("File not found")

    else:
        raise Http404("Invalid file type")

    # Serve the file
    with open(file_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response