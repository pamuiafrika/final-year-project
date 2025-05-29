from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'datasets', views.DatasetViewSet)
router.register(r'models', views.TrainedModelViewSet)
router.register(r'scans', views.PDFScanViewSet)
router.register(r'bulk-scans', views.BulkScanViewSet)

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.upload_view, name='upload'),
    path('results/<int:pk>/', views.scan_results, name='scan_results'),
    path('api/', include(router.urls)),
    path('api/check-scan/<int:pk>/', views.check_scan_status, name='check_scan_status'),
    path('api/check-bulk-scan/<int:pk>/', views.check_bulk_scan_status, name='check_bulk_scan_status'),
    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('train/<int:dataset_id>/', views.TrainModelView.as_view(), name='train_model'),
    path('training/<str:task_id>/', views.TrainingStatusView.as_view(), name='training_status'),
    path('models/', views.TrainedModelListView.as_view(), name='model_list'),
    path('api/detect/', views.DetectionAPIView.as_view(), name='detect_api'),
]
