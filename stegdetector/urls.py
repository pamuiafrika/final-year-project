from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('', lambda request: redirect('accounts:login')),
    path('app', include('pdf_parser.urls')),
    path('tools/', include('pdf_stego.urls')),
    path('', include('ai.urls')),
    path('ai/', include('ai.urls')),
    path('pdf-detector/', include('detector_app.urls')),
    # path('administrator', include('detector_app.urls')),
]

# Add URL patterns for serving media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)