import zlib
import binascii
import io
import math
import logging
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionAnalyzer:
    """Class for analyzing compression patterns in PDF files to detect hidden data"""
    
    def __init__(self, pdf_path):
        """
        Initialize with path to PDF file
        
        Args:
            pdf_path (str): Path to the PDF file to analyze
        """
        self.pdf_path = pdf_path
        self.pdf_data = None
        
        try:
            with open(pdf_path, 'rb') as file:
                self.pdf_data = file.read()
                logger.info(f"Successfully read {len(self.pdf_data)} bytes from {pdf_path}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def calculate_entropy(self, data):
        """
        Calculate Shannon entropy of data
        
        Args:
            data (bytes): Binary data to analyze
            
        Returns:
            float: Entropy value (between 0 and 8 for byte data)
        """
        if not data:
            return 0
            
        # Count occurrences of each byte value
        counter = Counter(data)
        
        # Calculate probability of each byte value
        length = len(data)
        probabilities = [count / length for count in counter.values()]
        
        # Calculate entropy using Shannon's formula
        entropy = -sum(p * math.log2(p) for p in probabilities)
        
        return entropy
    
    def sliding_window_entropy(self, window_size=1024, step=512):
        """
        Calculate entropy in sliding windows across the PDF data
        
        Args:
            window_size (int): Size of the window in bytes
            step (int): Number of bytes to slide the window each time
            
        Returns:
            dict: Entropy analysis results
        """
        results = {
            'window_size': window_size,
            'step': step,
            'windows': []
        }
        
        # Calculate entropy for each window
        pos = 0
        while pos + window_size <= len(self.pdf_data):
            window_data = self.pdf_data[pos:pos + window_size]
            entropy = self.calculate_entropy(window_data)
            
            window_info = {
                'start': pos,
                'end': pos + window_size,
                'entropy': entropy,
                'data_sample': binascii.hexlify(window_data[:20]).decode('ascii')
            }
            
            results['windows'].append(window_info)
            pos += step
        
        # Calculate statistics
        entropies = [w['entropy'] for w in results['windows']]
        results['statistics'] = {
            'min_entropy': min(entropies) if entropies else 0,
            'max_entropy': max(entropies) if entropies else 0,
            'avg_entropy': sum(entropies) / len(entropies) if entropies else 0,
            'std_entropy': np.std(entropies) if entropies else 0
        }
        
        logger.info(f"Sliding window entropy analysis complete: {len(results['windows'])} windows analyzed")
        return results
    
    def detect_anomalous_entropy_regions(self, threshold=1.5):
        """
        Detect regions with anomalous entropy levels
        
        Args:
            threshold (float): Number of standard deviations from mean to consider anomalous
            
        Returns:
            list: Regions with anomalous entropy levels
        """
        # First get sliding window entropy analysis
        entropy_analysis = self.sliding_window_entropy()
        
        # Calculate mean and standard deviation
        entropies = [w['entropy'] for w in entropy_analysis['windows']]
        mean_entropy = entropy_analysis['statistics']['avg_entropy']
        std_entropy = entropy_analysis['statistics']['std_entropy']
        
        # Detect anomalous regions
        anomalous_regions = []
        
        for window in entropy_analysis['windows']:
            # Calculate z-score
            z_score = abs(window['entropy'] - mean_entropy) / max(std_entropy, 0.001)  # Avoid division by zero
            
            if z_score > threshold:
                # This region has anomalous entropy
                region = {
                    'start': window['start'],
                    'end': window['end'],
                    'entropy': window['entropy'],
                    'z_score': z_score,
                    'is_high_entropy': window['entropy'] > mean_entropy,
                    'data_sample': window['data_sample']
                }
                anomalous_regions.append(region)
        
        logger.info(f"Detected {len(anomalous_regions)} regions with anomalous entropy")
        return anomalous_regions
    
    def detect_compression_artifacts(self):
        """
        Detect potential compression artifacts that might indicate hidden data
        
        Returns:
            list: Detected compression artifacts
        """
        artifacts = []
        
        # Look for zlib stream markers
        zlib_header = b'\x78\x9c'  # Common zlib header
        zlib_positions = []
        
        pos = 0
        while True:
            pos = self.pdf_data.find(zlib_header, pos)
            if pos == -1:
                break
            zlib_positions.append(pos)
            pos += 1
        
        # Analyze each potential zlib stream
        for pos in zlib_positions:
            try:
                # Try to decompress starting from this position
                # We don't know the length, so try increasingly larger chunks
                for chunk_size in [1024, 2048, 4096, 8192]:
                    try:
                        data_chunk = self.pdf_data[pos:pos + chunk_size]
                        decompressed = zlib.decompress(data_chunk)
                        
                        # Check if the decompressed data contains PNG signatures
                        png_signature = b'\x89PNG\r\n\x1a\n'
                        contains_png = png_signature in decompressed
                        
                        artifact = {
                            'type': 'zlib_stream',
                            'position': pos,
                            'compressed_size': len(data_chunk),
                            'decompressed_size': len(decompressed),
                            'compression_ratio': len(decompressed) / len(data_chunk),
                            'contains_png': contains_png,
                            'suspicious': contains_png,
                            'data_sample': binascii.hexlify(decompressed[:50]).decode('ascii')
                        }
                        
                        artifacts.append(artifact)
                        break  # Successfully decompressed, no need to try larger chunks
                        
                    except zlib.error:
                        # Try a larger chunk or move to the next position
                        continue
            
            except Exception as e:
                logger.warning(f"Error analyzing potential zlib stream at position {pos}: {e}")
        
        logger.info(f"Detected {len(artifacts)} potential compression artifacts")
        return artifacts
    
    def analyze_compression_patterns(self):
        """
        Perform comprehensive analysis of compression patterns in the PDF
        
        Returns:
            dict: Complete compression analysis results
        """
        results = {
            'entropy_analysis': self.sliding_window_entropy(),
            'anomalous_regions': self.detect_anomalous_entropy_regions(),
            'compression_artifacts': self.detect_compression_artifacts(),
            'summary': {}
        }
        
        # Add summary statistics
        results['summary'] = {
            'overall_entropy': self.calculate_entropy(self.pdf_data),
            'num_anomalous_regions': len(results['anomalous_regions']),
            'high_entropy_regions': sum(1 for r in results['anomalous_regions'] if r.get('is_high_entropy', False)),
            'compression_artifacts': len(results['compression_artifacts']),
            'suspicious_artifacts': sum(1 for a in results['compression_artifacts'] if a.get('suspicious', False))
        }
        
        # Calculate suspicious score based on findings
        suspicion_score = (
            results['summary']['high_entropy_regions'] * 1.5 + 
            results['summary']['suspicious_artifacts'] * 3
        ) / max(10, len(self.pdf_data) / 10000)  # Normalize by file size (10KB chunks)
        
        results['summary']['compression_suspicion_score'] = min(suspicion_score, 10)  # Cap at 10
        
        logger.info(f"Compression analysis complete. Suspicion score: {results['summary']['compression_suspicion_score']:.2f}")
        return results