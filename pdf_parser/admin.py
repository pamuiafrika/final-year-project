from django.contrib import admin
from .models import PDFDocument, AnomalyReport, PDFImage

# Register your models here.
admin.site.register(PDFDocument)
admin.site.register(AnomalyReport)
admin.site.register(PDFImage)
admin.site.site_header = "StegDetector Admin"
admin.site.site_title = "StegDetector Admin Portal"
admin.site.index_title = "Welcome to the StegDetector Admin Portal"
