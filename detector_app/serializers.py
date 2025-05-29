from rest_framework import serializers
from .models import Dataset, TrainedModel, PDFScan, BulkScan

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'

class TrainedModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainedModel
        fields = '__all__'

class PDFScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFScan
        fields = '__all__'
        read_only_fields = ('status', 'result', 'confidence', 'model_used', 'task_id')

class BulkScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = BulkScan
        fields = '__all__'
        read_only_fields = ('status', 'total_files', 'processed_files', 'clean_count', 'stego_count', 'task_id')