import os
import logging
from .pdf_extractor import PDFExtractor
from .metadata_analyzer import MetadataAnalyzer
from .image_detector import ImageDetector
from .compression_analyzer import CompressionAnalyzer
from ..utils.helper_functions import create_analysis_report, save_report_to_file, get_file_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStegAnalyzer:
    """Main controller class for analyzing PDFs for steganographic content"""
    
    def __init__(self, pdf_path):
        """
        Initialize with path to PDF file
        
        Args:
            pdf_path (str): Path to the PDF file to analyze
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        self.pdf_path = pdf_path
        self.results = {}
        self.report = {}
        logger.info(f"Initialized PDF Steganography Analyzer for {os.path.basename(pdf_path)}")
    
    def analyze_pdf(self):
        """
        Perform comprehensive analysis of the PDF file using all modules
        
        Returns:
            dict: Analysis results
        """
        logger.info(f"Starting analysis of {self.pdf_path}")
        
        # Calculate file hash for identification
        file_hash = get_file_hash(self.pdf_path)
        
        # Initialize results dictionary
        self.results = {
            'file_info': {
                'path': self.pdf_path,
                'size': os.path.getsize(self.pdf_path),
                'sha256': file_hash
            }
        }
        
        # 1. Extract basic PDF structure
        try:
            pdf_extractor = PDFExtractor(self.pdf_path)
            self.results['structure'] = pdf_extractor.analyze_pdf_structure()
            logger.info("PDF structure analysis complete")
        except Exception as e:
            logger.error(f"Error during PDF structure analysis: {e}")
            self.results['structure'] = {'error': str(e)}
        
        # 2. Analyze metadata
        try:
            metadata_analyzer = MetadataAnalyzer(self.pdf_path)
            self.results['metadata'] = metadata_analyzer.analyze_metadata()
            logger.info("Metadata analysis complete")
        except Exception as e:
            logger.error(f"Error during metadata analysis: {e}")
            self.results['metadata'] = {'error': str(e)}
        
        # 3. Detect images
        try:
            image_detector = ImageDetector(self.pdf_path)
            self.results['images'] = image_detector.analyze_images()
            logger.info("Image analysis complete")
        except Exception as e:
            logger.error(f"Error during image detection: {e}")
            self.results['images'] = {'error': str(e)}
        
        # 4. Analyze compression patterns
        try:
            compression_analyzer = CompressionAnalyzer(self.pdf_path)
            self.results['compression'] = compression_analyzer.analyze_compression_patterns()
            logger.info("Compression pattern analysis complete")
        except Exception as e:
            logger.error(f"Error during compression analysis: {e}")
            self.results['compression'] = {'error': str(e)}
        
        # Create final report
        self.report = create_analysis_report(self.pdf_path, self.results)
        logger.info("Analysis complete")
        
        return self.report
    
    def save_report(self, output_path=None):
        """
        Save analysis report to file
        
        Args:
            output_path (str, optional): Path to save the report to. If None, a default path is used.
            
        Returns:
            str: Path to the saved report
        """
        if not self.report:
            logger.warning("No analysis has been performed yet")
            return None
            
        if output_path is None:
            # Generate default output path
            base_name = os.path.basename(self.pdf_path)
            file_name = os.path.splitext(base_name)[0]
            output_path = os.path.join('reports', f"{file_name}_analysis.json")
        
        success = save_report_to_file(self.report, output_path)
        if success:
            return output_path
        return None
    
    def get_summary(self):
        """
        Get a summary of the analysis results
        
        Returns:
            dict: Summary information
        """
        if not self.report:
            logger.warning("No analysis has been performed yet")
            return {}
            
        return self.report.get('summary', {})