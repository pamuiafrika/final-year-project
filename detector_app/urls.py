from django.urls import path
from . import views

app_name = 'pdf_detector'

urlpatterns = [

    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('analyses/', views.analysis_list, name='analysis_list'),
    path('analysis/<int:pk>/', views.analysis_detail, name='analysis_detail'),
    path('analysis/<int:pk>/report/', views.download_report, name='download_report'),
    path('analysis/<int:pk>/reanalyze/', views.reanalyze_pdf, name='reanalyze_pdf'),
    path('api/analysis/<int:pk>/status/', views.api_analysis_status, name='api_analysis_status'),
]


