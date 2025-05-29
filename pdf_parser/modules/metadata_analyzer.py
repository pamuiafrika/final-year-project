import PyPDF2
import datetime
import logging
from dateutil.parser import parse as date_parse
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataAnalyzer:
    """Class for analyzing metadata in PDF files to detect anomalies"""
    
    def __init__(self, pdf_path):
        """
        Initialize with path to PDF file
        
        Args:
            pdf_path (str): Path to the PDF file to analyze
        """
        self.pdf_path = pdf_path
        self.metadata = {}
        self.anomalies = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Extract metadata from the PDF's document information dictionary
                if pdf_reader.metadata:
                    self.metadata = {k.replace('/', ''): v for k, v in pdf_reader.metadata.items()}
                    logger.info(f"Successfully extracted metadata from {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
    
    def extract_metadata(self):
        """
        Extract and format all metadata from the PDF
        
        Returns:
            dict: Formatted metadata from the PDF
        """
        formatted_metadata = {}
        
        # Process common metadata fields
        for key, value in self.metadata.items():
            # Convert date strings to datetime objects when possible
            if key in ['CreationDate', 'ModDate'] and value:
                try:
                    # Handle PDF date format (D:YYYYMMDDHHmmSSOHH'mm')
                    if value.startswith('D:'):
                        # Extract the date portion without the 'D:' prefix
                        date_str = value[2:]
                        # Basic parsing for PDF date format
                        year = date_str[0:4]
                        month = date_str[4:6]
                        day = date_str[6:8]
                        hour = date_str[8:10] if len(date_str) > 8 else "00"
                        minute = date_str[10:12] if len(date_str) > 10 else "00"
                        
                        formatted_date = f"{year}-{month}-{day} {hour}:{minute}"
                        formatted_metadata[key] = date_parse(formatted_date)
                    else:
                        formatted_metadata[key] = date_parse(value)
                except Exception as e:
                    logger.warning(f"Could not parse date {value}: {e}")
                    formatted_metadata[key] = value
            else:
                formatted_metadata[key] = value
        
        return formatted_metadata
    
    def check_date_consistency(self):
        """
        Check for inconsistencies in date metadata
        
        Returns:
            list: List of anomalies found in date metadata
        """
        anomalies = []
        metadata = self.extract_metadata()
        
        # Check if creation date is after modification date
        if 'CreationDate' in metadata and 'ModDate' in metadata:
            if isinstance(metadata['CreationDate'], datetime.datetime) and isinstance(metadata['ModDate'], datetime.datetime):
                if metadata['CreationDate'] > metadata['ModDate']:
                    anomaly = {
                        'type': 'date_inconsistency',
                        'description': 'Creation date is after modification date',
                        'creation_date': metadata['CreationDate'],
                        'modification_date': metadata['ModDate'],
                        'severity': 'medium'
                    }
                    anomalies.append(anomaly)
                    logger.info(f"Detected date inconsistency: {anomaly['description']}")
        
        # Check for future dates
        current_date = datetime.datetime.now()
        for date_field in ['CreationDate', 'ModDate']:
            if date_field in metadata and isinstance(metadata[date_field], datetime.datetime):
                if metadata[date_field] > current_date:
                    anomaly = {
                        'type': 'future_date',
                        'description': f'{date_field} is set to a future date',
                        'date': metadata[date_field],
                        'current_date': current_date,
                        'severity': 'high'
                    }
                    anomalies.append(anomaly)
                    logger.info(f"Detected future date: {anomaly['description']}")
        
        return anomalies
    
    def check_for_suspicious_fields(self):
        """
        Check for suspicious or unusual metadata fields
        
        Returns:
            list: List of suspicious metadata fields found
        """
        anomalies = []
        metadata = self.extract_metadata()
        
        # Define a list of known PDF metadata fields
        known_fields = [
            'Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Producer',
            'CreationDate', 'ModDate', 'Trapped', 'PTEX.Fullbanner'
        ]
        
        # Check for unknown fields that might indicate tampering
        for field in metadata:
            if field not in known_fields:
                anomaly = {
                    'type': 'unknown_field',
                    'description': f'Unknown metadata field: {field}',
                    'field': field,
                    'value': metadata[field],
                    'severity': 'low'
                }
                anomalies.append(anomaly)
                logger.info(f"Detected unknown field: {anomaly['description']}")
        
        # Check for suspicious content in known fields (e.g., script or code)
        code_patterns = [
            r'<script', r'function\s*\(', r'eval\(', r'exec\(', 
            r'document\.write', r'\\x[0-9a-fA-F]{2}'
        ]
        
        for field, value in metadata.items():
            if isinstance(value, str):
                for pattern in code_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        anomaly = {
                            'type': 'suspicious_content',
                            'description': f'Possible code or script in {field}',
                            'field': field,
                            'value': value,
                            'pattern': pattern,
                            'severity': 'high'
                        }
                        anomalies.append(anomaly)
                        logger.info(f"Detected suspicious content: {anomaly['description']}")
        
        return anomalies
    
    def analyze_metadata(self):
        """
        Perform comprehensive analysis of PDF metadata
        
        Returns:
            dict: Complete metadata analysis results
        """
        analysis_results = {
            'raw_metadata': self.extract_metadata(),
            'anomalies': []
        }
        
        # Run all checks and collect anomalies
        date_anomalies = self.check_date_consistency()
        field_anomalies = self.check_for_suspicious_fields()
        
        analysis_results['anomalies'] = date_anomalies + field_anomalies
        analysis_results['has_anomalies'] = len(analysis_results['anomalies']) > 0
        
        logger.info(f"Metadata analysis complete. Found {len(analysis_results['anomalies'])} anomalies.")
        return analysis_results