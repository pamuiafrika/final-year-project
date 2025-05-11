"""
URL patterns for PDF steganography application.
"""

from django.urls import path
from . import views

app_name = 'pdf_stego'

urlpatterns = [
    path('', views.home, name='home'),
    path('hide/', views.hide_form, name='hide_form'),
    path('extract/', views.extract_form, name='extract_form'),
    path('operations/', views.operations_list, name='operations_list'),
    path('operation/<uuid:operation_id>/', views.operation_detail, name='operation_detail'),
    path('download/<uuid:operation_id>/<str:file_type>/', views.download_file, name='download_file'),
]