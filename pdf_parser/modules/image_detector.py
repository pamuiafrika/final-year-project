import io
import struct
import zlib
import binascii
import logging
from PIL import Image
from PyPDF2 import PdfReader
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDetector:
    """Class for detecting and analyzing images in PDF files"""
    
    # PNG signature bytes
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    def __init__(self, pdf_path):
        """
        Initialize with path to PDF file
        
        Args:
            pdf_path (str): Path to the PDF file to analyze
        """
        self.pdf_path = pdf_path
        self.pdf_data = None
        self.images = []
        
        try:
            with open(pdf_path, 'rb') as file:
                self.pdf_data = file.read()
                logger.info(f"Successfully read {len(self.pdf_data)} bytes from {pdf_path}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def detect_png_signature(self, data):
        """
        Detect PNG file signature in binary data
        
        Args:
            data (bytes): Binary data to search for PNG signatures
            
        Returns:
            list: List of positions where PNG signatures were found
        """
        positions = []
        
        # Find all occurrences of PNG signature in data
        pos = 0
        while True:
            pos = data.find(self.PNG_SIGNATURE, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += 1
            
        logger.info(f"Found {len(positions)} potential PNG signatures")
        return positions
    
    def extract_embedded_images(self):
        """
        Extract images embedded in PDF objects
        
        Returns:
            list: List of extracted image data and metadata
        """
        extracted_images = []
        
        try:
            # Open the PDF with PyPDF2
            reader = PdfReader(io.BytesIO(self.pdf_data))
            
            # Iterate through pages
            for page_num, page in enumerate(reader.pages, 1):
                
                # Check if the page has resources
                if '/Resources' in page:
                    resources = page['/Resources']
                    
                    # Check if resources has XObjects (which often contain images)
                    if '/XObject' in resources:
                        x_objects = resources['/XObject']
                        
                        # Iterate through XObjects
                        for obj_name, obj in x_objects.items():
                            if obj['/Subtype'] == '/Image':
                                # Extract basic image metadata
                                image_data = {
                                    'page': page_num,
                                    'name': obj_name,
                                    'width': obj['/Width'],
                                    'height': obj['/Height'],
                                    'bits_per_component': obj.get('/BitsPerComponent', None),
                                    'color_space': obj.get('/ColorSpace', None),
                                    'filter': obj.get('/Filter', None),
                                }
                                
                                # Try to get the raw image data
                                try:
                                    image_data['data'] = obj.get_data()
                                    image_data['data_sample'] = binascii.hexlify(image_data['data'][:20])
                                    image_data['data_length'] = len(image_data['data'])
                                    
                                    # Check if this data contains a PNG signature
                                    png_positions = self.detect_png_signature(image_data['data'])
                                    image_data['contains_png_signature'] = len(png_positions) > 0
                                    if png_positions:
                                        image_data['png_positions'] = png_positions
                                except Exception as e:
                                    logger.warning(f"Could not extract image data: {e}")
                                    image_data['data'] = None
                                
                                extracted_images.append(image_data)
            
            logger.info(f"Extracted {len(extracted_images)} images from PDF")
            return extracted_images
            
        except Exception as e:
            logger.error(f"Error extracting embedded images: {e}")
            return []
    
    def scan_for_hidden_pngs(self):
        """
        Scan raw PDF data for PNG signatures that might indicate hidden PNGs
        
        Returns:
            list: List of potential hidden PNG locations
        """
        hidden_pngs = []
        
        # Find all PNG signatures in the raw PDF data
        png_positions = self.detect_png_signature(self.pdf_data)
        
        for pos in png_positions:
            # Try to read the PNG header to get width and height
            try:
                # IHDR chunk should start 8 bytes after PNG signature
                ihdr_data_pos = pos + 8 + 4  # signature + chunk length
                
                # Extract width and height from IHDR chunk (4 bytes each)
                width_bytes = self.pdf_data[ihdr_data_pos:ihdr_data_pos + 4]
                height_bytes = self.pdf_data[ihdr_data_pos + 4:ihdr_data_pos + 8]
                
                width = struct.unpack('>I', width_bytes)[0]
                height = struct.unpack('>I', height_bytes)[0]
                
                # Create a record for this potential hidden PNG
                hidden_png = {
                    'position': pos,
                    'width': width,
                    'height': height,
                    'data_sample': binascii.hexlify(self.pdf_data[pos:pos + 50])
                }
                
                # Check if this PNG is within a valid PDF object or between objects
                # This is a simplified heuristic - real implementation would need more complex PDF parsing
                start_obj_pattern = rb'\d+ \d+ obj'
                end_obj_pattern = rb'endobj'
                
                # Find the nearest start and end object markers before this position
                prev_start = max([m.end() for m in re.finditer(start_obj_pattern, self.pdf_data[:pos])] or [0])
                prev_end = max([m.end() for m in re.finditer(end_obj_pattern, self.pdf_data[:pos])] or [0])
                
                # If the previous marker was 'endobj', this PNG might be between objects (suspicious)
                hidden_png['between_objects'] = prev_end > prev_start
                hidden_png['suspicious'] = hidden_png['between_objects']
                
                hidden_pngs.append(hidden_png)
                
            except Exception as e:
                logger.warning(f"Could not parse PNG header at position {pos}: {e}")
                # Still record the position as potentially suspicious
                hidden_pngs.append({
                    'position': pos,
                    'error': str(e),
                    'data_sample': binascii.hexlify(self.pdf_data[pos:pos + 50]),
                    'suspicious': True
                })
        
        logger.info(f"Found {len(hidden_pngs)} potential hidden PNGs in raw PDF data")
        return hidden_pngs
    
    def analyze_image_properties(self, image_data):
        """
        Analyze properties of an image to detect anomalies
        
        Args:
            image_data (bytes): Raw image data to analyze
            
        Returns:
            dict: Analysis results
        """
        analysis = {
            'size': len(image_data),
            'anomalies': []
        }
        
        try:
            # Try to load the image data with PIL
            img = Image.open(io.BytesIO(image_data))
            analysis['format'] = img.format
            analysis['mode'] = img.mode
            analysis['width'] = img.width
            analysis['height'] = img.height
            
            # Check if the image dimensions are unusually small
            if img.width < 4 or img.height < 4:
                analysis['anomalies'].append({
                    'type': 'small_dimensions',
                    'description': f'Unusually small image dimensions: {img.width}x{img.height}',
                    'severity': 'medium'
                })
            
            # Check if the image has unusual proportions
            aspect_ratio = img.width / max(img.height, 1)  # Avoid division by zero
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                analysis['anomalies'].append({
                    'type': 'unusual_proportions',
                    'description': f'Unusual aspect ratio: {aspect_ratio:.2f}',
                    'severity': 'low'
                })
            
        except Exception as e:
            logger.warning(f"Could not analyze image with PIL: {e}")
            analysis['error'] = str(e)
            
            # Check if this might actually be encrypted or corrupted data
            if len(image_data) > 100:
                # Calculate entropy as a measure of randomness
                byte_counts = {}
                for byte in image_data:
                    byte_counts[byte] = byte_counts.get(byte, 0) + 1
                
                # High entropy can indicate encryption or compression
                entropy = sum((-count/len(image_data) * (count/len(image_data)).bit_length() 
                             for count in byte_counts.values()))
                analysis['entropy'] = entropy
                
                if entropy > 7.5:  # Very high entropy
                    analysis['anomalies'].append({
                        'type': 'high_entropy',
                        'description': f'Data has very high entropy ({entropy:.2f}), may be encrypted',
                        'severity': 'high'
                    })
        
        return analysis
    
    def analyze_images(self):
        """
        Perform comprehensive analysis of images in the PDF
        
        Returns:
            dict: Complete image analysis results
        """
        results = {
            'embedded_images': self.extract_embedded_images(),
            'potential_hidden_pngs': self.scan_for_hidden_pngs(),
            'summary': {}
        }
        
        # Add summary statistics
        results['summary'] = {
            'total_embedded_images': len(results['embedded_images']),
            'suspicious_embedded_images': sum(1 for img in results['embedded_images'] 
                                           if img.get('contains_png_signature', False)),
            'potential_hidden_pngs': len(results['potential_hidden_pngs']),
            'suspicious_hidden_pngs': sum(1 for png in results['potential_hidden_pngs'] 
                                        if png.get('suspicious', False))
        }
        
        # Add overall suspicion score
        suspicion_score = (
            results['summary']['suspicious_embedded_images'] * 2 + 
            results['summary']['suspicious_hidden_pngs'] * 3
        ) / max(results['summary']['total_embedded_images'], 1)
        
        results['summary']['overall_suspicion_score'] = min(suspicion_score, 10)  # Cap at 10
        
        logger.info(f"Image analysis complete. Suspicion score: {results['summary']['overall_suspicion_score']:.2f}")
        return results