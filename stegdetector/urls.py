from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('pdf_parser.urls')),
    path('tools/', include('pdf_stego.urls')),
    path('ai/', include('detector_app.urls')),
    # path('administrator', include('detector_app.urls')),
]

# Add URL patterns for serving media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)