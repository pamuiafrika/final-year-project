from django.urls import path
from . import views

app_name = 'pdf_parser'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_pdf, name='upload'),
    path('pdf/<int:pk>/', views.pdf_detail, name='pdf_detail'),
    path('pdf/<int:pk>/download-report/', views.download_report, name='download_report'),
    path('api/pdf/<int:pk>/', views.api_analyze_pdf, name='api_analyze_pdf'),
    path('embed/', views.embed_png_in_pdf_view, name='hide_png_in_pdf'),
    path('analyses/', views.analysis_list, name='analysis_list'),
]