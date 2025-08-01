# Generated by Django 5.2.1 on 2025-06-16 11:52

import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PDFUpload',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('file_name', models.CharField(max_length=255)),
                ('file_hash', models.CharField(max_length=64, unique=True)),
                ('file_size', models.BigIntegerField()),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('processed', models.BooleanField(default=False)),
                ('processing_started_at', models.DateTimeField(blank=True, null=True)),
                ('processing_completed_at', models.DateTimeField(blank=True, null=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-uploaded_at'],
            },
        ),
        migrations.CreateModel(
            name='FeatureVector',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_size_bytes', models.BigIntegerField(default=0)),
                ('pdf_version', models.FloatField(default=1.4)),
                ('num_pages', models.IntegerField(default=0)),
                ('num_objects', models.IntegerField(default=0)),
                ('num_stream_objects', models.IntegerField(default=0)),
                ('num_embedded_files', models.IntegerField(default=0)),
                ('num_annotation_objects', models.IntegerField(default=0)),
                ('num_form_fields', models.IntegerField(default=0)),
                ('creation_date_ts', models.BigIntegerField(default=0)),
                ('mod_date_ts', models.BigIntegerField(default=0)),
                ('creation_mod_date_diff', models.BigIntegerField(default=0)),
                ('avg_entropy_per_stream', models.FloatField(default=0.0)),
                ('max_entropy_per_stream', models.FloatField(default=0.0)),
                ('min_entropy_per_stream', models.FloatField(default=0.0)),
                ('std_entropy_per_stream', models.FloatField(default=0.0)),
                ('num_streams_entropy_gt_threshold', models.IntegerField(default=0)),
                ('num_encrypted_streams', models.IntegerField(default=0)),
                ('num_corrupted_objects', models.IntegerField(default=0)),
                ('num_objects_with_random_markers', models.IntegerField(default=0)),
                ('has_broken_name_trees', models.BooleanField(default=False)),
                ('num_suspicious_filters', models.IntegerField(default=0)),
                ('has_javascript', models.BooleanField(default=False)),
                ('has_launch_actions', models.BooleanField(default=False)),
                ('avg_file_size_per_page', models.FloatField(default=0.0)),
                ('compression_ratio', models.FloatField(default=1.0)),
                ('num_eof_markers', models.IntegerField(default=1)),
                ('extraction_success', models.BooleanField(default=False)),
                ('extraction_time_ms', models.IntegerField(default=0)),
                ('error_count', models.IntegerField(default=0)),
                ('pdf_upload', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='features', to='ai.pdfupload')),
            ],
        ),
        migrations.CreateModel(
            name='PredictionResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('xgboost_prediction', models.IntegerField()),
                ('xgboost_probability', models.FloatField()),
                ('xgboost_confidence', models.FloatField()),
                ('wide_deep_prediction', models.IntegerField()),
                ('wide_deep_probability', models.FloatField()),
                ('wide_deep_confidence', models.FloatField()),
                ('ensemble_prediction', models.IntegerField()),
                ('ensemble_confidence', models.FloatField()),
                ('extraction_success', models.BooleanField(default=False)),
                ('extraction_time_ms', models.IntegerField(default=0)),
                ('error_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('pdf_upload', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='prediction', to='ai.pdfupload')),
            ],
        ),
    ]
