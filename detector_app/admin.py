from django.contrib import admin
from . models import Dataset, TrainedModel, PDFScan, BulkScan

# Register your models here.
admin.site.register(Dataset)
admin.site.register(TrainedModel)
admin.site.register(PDFScan)
admin.site.register(BulkScan)