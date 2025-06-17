import os
import time
import tempfile
import json
from typing import Dict, Any, Optional
from django.conf import settings
from django.utils import timezone
from django.core.files.base import ContentFile
from .models import PDFAnalysis, AnalysisIndicator
from .pdf_detector import PDFSteganoDetector  # Your original detector class

class DjangoPDFAnalysisService:
    """Django service wrapper for PDF steganography detection."""
    
    def __init__(self):
        # Initialize detector with model path from settings
        model_path = getattr(settings, 'PDF_STEGANO_MODEL_PATH', None)
        cache_dir = getattr(settings, 'PDF_STEGANO_CACHE_DIR', 
                          os.path.join(settings.MEDIA_ROOT, 'analysis_cache'))
        
        self.detector =PDFSteganoDetector(
            ml_model_path='/home/d3bugger/Projects/FINAL YEAR PROJECT/src/ml_models/stegov2/v02_model.pkl',
            cache_dir=cache_dir
        )
    
    def analyze_pdf(self, pdf_analysis: PDFAnalysis, 
                   focus_technique: str = 'auto') -> PDFAnalysis:
        """
        Analyze a PDF file and update the PDFAnalysis model.
        
        Args:
            pdf_analysis: PDFAnalysis model instance
            focus_technique: Analysis technique to focus on
            
        Returns:
            Updated PDFAnalysis instance
        """
        start_time = time.time()
        
        try:
            # Create temporary file from uploaded PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                # Read file content
                pdf_analysis.pdf_file.seek(0)
                temp_file.write(pdf_analysis.pdf_file.read())
                temp_file.flush()
                temp_path = temp_file.name
            
            try:
                # Run the analysis
                result = self.detector.analyze_pdf(
                    temp_path, 
                    focus_technique=focus_technique
                )
                
                # Update the PDFAnalysis model
                pdf_analysis.analysis_date = timezone.now()
                pdf_analysis.assessment = self._map_assessment(str(result['assessment']))
                pdf_analysis.risk_score = float(result['risk_score'])
                pdf_analysis.total_indicators = int(result['total_indicators'])
                pdf_analysis.technique_used = focus_technique
                pdf_analysis.analysis_duration = time.time() - start_time
                
                # ML analysis results
                if result.get('ml_analysis'):
                    ml_data = result['ml_analysis']
                    pdf_analysis.ml_anomaly_score = ml_data.get('anomaly_score')
                    is_anomaly = ml_data.get('is_anomaly', False)
                    pdf_analysis.is_anomaly = bool(is_anomaly)  # Ensure boolean type
                
                # Store detailed results with proper serialization
                pdf_analysis.indicators_data = json.dumps(result.get('indicators_by_category', {}), default=str)
                pdf_analysis.features_data = json.dumps(result.get('features_extracted', {}), default=str)
                pdf_analysis.recommendations = json.dumps(result.get('recommendations', []), default=str)
                
                pdf_analysis.save()
                
                # Create individual indicator records
                self._create_indicators(pdf_analysis, result.get('indicators_by_category', {}))
                
                return pdf_analysis
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            # Store error information
            pdf_analysis.assessment = 'ERROR'
            pdf_analysis.analysis_date = timezone.now()
            pdf_analysis.analysis_duration = time.time() - start_time
            pdf_analysis.recommendations = [f"Analysis failed: {str(e)}"]
            pdf_analysis.save()
            raise
    
    def _map_assessment(self, assessment: str) -> str:
        """Map detector assessment to model choices."""
        assessment_mapping = {
            'CLEAN - No significant suspicious indicators': 'CLEAN',
            'LOW RISK - Minor anomalies found': 'LOW_RISK',
            'MEDIUM RISK - Suspicious patterns detected': 'MEDIUM_RISK',
            'HIGH RISK - Strong evidence of steganography': 'HIGH_RISK',
        }
        return assessment_mapping.get(assessment, 'LOW_RISK')
    
    def _create_indicators(self, pdf_analysis: PDFAnalysis, 
                          indicators_by_category: Dict[str, Any]) -> None:
        """Create AnalysisIndicator records from detection results."""
        for category, indicators in indicators_by_category.items():
            for indicator_data in indicators:
                AnalysisIndicator.objects.create(
                    analysis=pdf_analysis,
                    category=category,
                    severity=indicator_data.get('severity', 'LOW'),
                    description=indicator_data.get('description', ''),
                    confidence=indicator_data.get('confidence', 0.0),
                    technical_details=indicator_data.get('technical_details', {}),
                    location=indicator_data.get('location')
                )
    
    def generate_detailed_report(self, pdf_analysis: PDFAnalysis) -> str:
        """Generate detailed report for a PDF analysis."""
        if not pdf_analysis.is_analyzed:
            return "Analysis not completed yet."
        
        # Reconstruct result dictionary for report generation
        result = {
            'assessment': dict(pdf_analysis.RISK_LEVELS).get(pdf_analysis.assessment, 'Unknown'),
            'risk_score': float(pdf_analysis.risk_score or 0),
            'total_indicators': int(pdf_analysis.total_indicators),
            'indicators_by_category': json.loads(pdf_analysis.indicators_data or '{}'),
            'ml_analysis': {
                'anomaly_score': float(pdf_analysis.ml_anomaly_score or 0),
                'is_anomaly': bool(pdf_analysis.is_anomaly),
            },
            'recommendations': json.loads(pdf_analysis.recommendations or '[]'),
        }
        
        return self.detector.generate_detailed_report(
            result, 
            pdf_analysis.original_filename
        )




