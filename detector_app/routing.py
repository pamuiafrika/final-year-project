# detector_app/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/scan/(?P<scan_id>\w+)/$', consumers.ScanStatusConsumer.as_asgi()),
]
