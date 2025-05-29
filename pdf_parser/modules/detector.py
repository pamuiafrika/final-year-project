import os
import io
import re
import sys
import zlib
import time
import base64
import struct
import hashlib
import binascii
import warnings
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Union, Set, BinaryIO
from dataclasses import dataclass, field
import logging
from pathlib import Path
from collections import Counter
import math
import json
from enum import Enum, auto

# Third-party imports
import numpy as np
import PyPDF2
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import (
    LTImage, LTFigure, LTTextBox, LTCurve,
    LTLine, LTRect, LTChar, LAParams
)
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.psparser import PSLiteral
from PIL import Image

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging with formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SteganoMethod(Enum):
    """Enumeration of steganography methods"""
    OBJECT_STREAM = auto()
    METADATA = auto()
    WHITESPACE_COMMENT = auto()
    DOCUMENT_COMPONENTS = auto()
    JAVASCRIPT = auto()
    IMAGE_DATA = auto()
    XREF_TABLE = auto()
    UNKNOWN = auto()


@dataclass
class SteganoDetection:
    """Data class for storing information about detected steganography"""
    method: SteganoMethod
    location: str
    confidence: float  # 0.0-1.0
    size_bytes: int = 0
    page_number: Optional[int] = None
    object_id: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Optional[bytes] = None
    sample: Optional[bytes] = None


@dataclass
class PNGDetails:
    """Data class for storing information about PNG images"""
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int
    chunks: List[Dict[str, Any]]
    has_unusual_chunks: bool = False
    has_suspicious_data: bool = False
    entropy: float = 0.0


class PDFSteganographyDetector:
    """Class for detecting various steganography techniques in PDF documents"""

    def __init__(self, pdf_path: Union[str, Path], 
                 enable_threading: bool = True, 
                 max_workers: int = 4,
                 deep_scan: bool = False,
                 entropy_threshold: float = 7.5,
                 verbose: bool = False):
        """
        Initialize with path to PDF file
        
        Args:
            pdf_path (str or Path): Path to the PDF file to analyze
            enable_threading (bool): Whether to use multithreading for extraction tasks
            max_workers (int): Maximum number of worker threads
            deep_scan (bool): Whether to perform deeper and more CPU-intensive analysis
            entropy_threshold (float): Threshold for flagging high entropy data (0-8)
            verbose (bool): Whether to output detailed information during scanning
        """
        self.pdf_path = Path(pdf_path)
        self.raw_pdf_data = None
        self.pdf_reader = None
        self.enable_threading = enable_threading
        self.max_workers = max_workers
        self.deep_scan = deep_scan
        self.entropy_threshold = entropy_threshold
        self.verbose = verbose
        self._lock = threading.RLock()
        
        # Known PNG signatures and chunk types
        self.png_signature = b'\x89PNG\r\n\x1a\n'
        self.standard_png_chunks = {
            b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tRNS', b'cHRM', b'gAMA', 
            b'iCCP', b'sBIT', b'sRGB', b'iTXt', b'tEXt', b'zTXt', b'bKGD', 
            b'hIST', b'pHYs', b'sPLT', b'tIME'
        }
        
        # Signatures for hidden data detection
        self.signatures = {
            'zip': b'PK\x03\x04',
            'rar': b'Rar!\x1a\x07',
            'jpg': b'\xff\xd8\xff',
            'gif': b'GIF8',
            'png': self.png_signature,
            'pdf': b'%PDF',
            'bmp': b'BM',
            'wav': b'RIFF',
            'mp3': b'ID3',
        }
        
        if not self.pdf_path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        try:
            start_time = time.time()
            with open(self.pdf_path, 'rb') as file:
                self.raw_pdf_data = file.read()
                self.pdf_reader = PyPDF2.PdfReader(io.BytesIO(self.raw_pdf_data))
                
                # Initialize pdfminer objects for more detailed analysis
                self.pdf_parser = PDFParser(io.BytesIO(self.raw_pdf_data))
                self.pdf_document = PDFDocument(self.pdf_parser)
                self.rsrcmgr = PDFResourceManager()
                self.laparams = LAParams()
                self.device = PDFPageAggregator(self.rsrcmgr, laparams=self.laparams)
                self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)
                
            logger.info(f"Successfully loaded PDF: {self.pdf_path.name} in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading PDF: {e}", exc_info=True)
            raise
    
    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of the data
        
        Args:
            data (bytes): The data to calculate entropy for
            
        Returns:
            float: The calculated entropy value (0-8)
        """
        if not data:
            return 0.0
            
        # Count frequency of each byte value
        counter = Counter(data)
        length = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _is_high_entropy(self, data: bytes) -> bool:
        """
        Check if data has high entropy (potentially encrypted/compressed)
        
        Args:
            data (bytes): Data to check
            
        Returns:
            bool: True if data has high entropy
        """
        if not data:
            return False
        
        # Calculate entropy and compare to threshold
        entropy = self._calculate_entropy(data)
        return entropy > self.entropy_threshold
    
    def _extract_potential_png(self, data: bytes) -> List[Tuple[int, bytes]]:
        """
        Extract potential PNG images from binary data
        
        Args:
            data (bytes): Binary data to search for PNG signatures
            
        Returns:
            List[Tuple[int, bytes]]: List of offset and PNG data pairs
        """
        results = []
        
        # Find PNG signatures
        offset = 0
        while True:
            offset = data.find(self.png_signature, offset)
            if offset == -1:
                break
                
            # Try to determine PNG boundaries
            try:
                # Start from the signature
                current_pos = offset + len(self.png_signature)
                
                # Track PNG chunks to find IEND
                while current_pos < len(data) - 12:  # Need at least 12 bytes for chunk header and CRC
                    # Chunk structure: Length (4 bytes) + Type (4 bytes) + Data + CRC (4 bytes)
                    chunk_length = struct.unpack('>I', data[current_pos:current_pos+4])[0]
                    chunk_type = data[current_pos+4:current_pos+8]
                    
                    # Check if we've reached the IEND chunk
                    if chunk_type == b'IEND':
                        # Include chunk header, data, and CRC
                        end_pos = current_pos + 12 + chunk_length
                        png_data = data[offset:end_pos]
                        results.append((offset, png_data))
                        break
                        
                    # Move to next chunk
                    current_pos += 12 + chunk_length
                    
                    # Safety check
                    if current_pos > len(data) or chunk_length > 100000000:  # Arbitrary large value
                        break
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error extracting PNG at offset {offset}: {e}")
            
            # Move past this signature for next iteration
            offset += 1
            
        return results
    
    def _analyze_png_data(self, png_data: bytes) -> Optional[PNGDetails]:
        """
        Analyze PNG data for anomalies
        
        Args:
            png_data (bytes): PNG image data
            
        Returns:
            Optional[PNGDetails]: Details about the PNG or None if invalid
        """
        if not png_data.startswith(self.png_signature):
            return None
            
        try:
            # Check if it's a valid PNG by opening with PIL
            img = Image.open(io.BytesIO(png_data))
            img.verify()  # Verify PNG header
            
            # Reset the buffer
            img_data = io.BytesIO(png_data)
            img = Image.open(img_data)
            
            # Get basic info
            width, height = img.size
            if hasattr(img, 'bit_depth'):
                bit_depth = img.bit_depth
            else:
                bit_depth = 8  # assume default
                
            color_type = 0  # Default placeholder
            compression = 0  # Default placeholder
            filter_method = 0  # Default placeholder
            interlace = 0  # Default placeholder
            
            # Parse chunks
            chunks = []
            unusual_chunks = False
            suspicious_data = False
            
            # Parse IHDR for detailed info
            try:
                # Skip PNG signature
                img_data.seek(8)
                
                # Read IHDR
                length = struct.unpack('>I', img_data.read(4))[0]
                chunk_type = img_data.read(4)
                
                if chunk_type == b'IHDR' and length == 13:
                    width, height = struct.unpack('>II', img_data.read(8))
                    bit_depth = ord(img_data.read(1))
                    color_type = ord(img_data.read(1))
                    compression = ord(img_data.read(1))
                    filter_method = ord(img_data.read(1))
                    interlace = ord(img_data.read(1))
                    
                    # Skip CRC
                    img_data.read(4)
                else:
                    # Reset if IHDR not found as expected
                    img_data.seek(8)
                
                # Process all chunks
                while True:
                    try:
                        chunk_data = img_data.read(8)
                        if len(chunk_data) < 8:  # End of file
                            break
                            
                        length = struct.unpack('>I', chunk_data[:4])[0]
                        chunk_type = chunk_data[4:8]
                        
                        # Save chunk info
                        chunk_info = {
                            'type': chunk_type,
                            'length': length,
                        }
                        
                        # Special handling for interesting chunks
                        if chunk_type == b'tEXt' or chunk_type == b'zTXt' or chunk_type == b'iTXt':
                            chunk_data = img_data.read(length)
                            if len(chunk_data) >= length:
                                # For text chunks, extract the keyword
                                null_pos = chunk_data.find(b'\0')
                                if null_pos != -1:
                                    keyword = chunk_data[:null_pos].decode('latin-1', errors='replace')
                                    chunk_info['keyword'] = keyword
                                    
                                    # Check for unusual keywords that might indicate steganography
                                    unusual_keywords = ['steg', 'hidden', 'secret', 'data', 'payload']
                                    if any(kw in keyword.lower() for kw in unusual_keywords):
                                        chunk_info['suspicious'] = True
                                        suspicious_data = True
                        else:
                            # Skip data
                            img_data.seek(length, io.SEEK_CUR)
                            
                        # Skip CRC
                        img_data.read(4)
                        
                        # Check if this is an unusual chunk type
                        if chunk_type not in self.standard_png_chunks:
                            chunk_info['unusual'] = True
                            unusual_chunks = True
                            
                        chunks.append(chunk_info)
                        
                        # End of PNG
                        if chunk_type == b'IEND':
                            break
                    except Exception as e:
                        if self.verbose:
                            logger.debug(f"Error parsing PNG chunk: {e}")
                        break
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error parsing PNG structure: {e}")
            
            # Calculate entropy of the PNG data
            entropy = self._calculate_entropy(png_data)
            
            # Check for unusual color depth combinations
            if (bit_depth == 16 and color_type in [4, 6]) or (bit_depth > 8 and color_type == 3):
                suspicious_data = True
            
            # Create PNG details object
            return PNGDetails(
                width=width,
                height=height,
                bit_depth=bit_depth,
                color_type=color_type,
                compression=compression,
                filter_method=filter_method,
                interlace=interlace,
                chunks=chunks,
                has_unusual_chunks=unusual_chunks,
                has_suspicious_data=suspicious_data,
                entropy=entropy
            )
        except Exception as e:
            if self.verbose:
                logger.debug(f"Error analyzing PNG: {e}")
            return None
    
    def _check_lsb_steganography(self, png_data: bytes) -> bool:
        """
        Check for potential LSB steganography in PNG data
        
        Args:
            png_data (bytes): PNG image data
            
        Returns:
            bool: True if LSB steganography is suspected
        """
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(png_data))
            img_array = np.array(img)
            
            # Check transparency if alpha channel exists
            if img.mode == 'RGBA':
                alpha = img_array[:, :, 3]
                # Check for unusual alpha values
                unique_alpha = np.unique(alpha)
                if len(unique_alpha) > 2 and len(unique_alpha) < 20:
                    return True  # Suspicious alpha channel
                    
            # Check for LSB patterns in each channel
            for channel in range(min(3, img_array.shape[2])):  # RGB channels
                # Get the least significant bits
                lsb = img_array[:, :, channel] & 1
                
                # Calculate the percentage of 1s in LSB
                ones_percentage = np.mean(lsb)
                
                # If significantly deviates from 0.5 (random distribution)
                if abs(ones_percentage - 0.5) < 0.05:
                    # Very close to 0.5 might indicate artificially balanced LSBs
                    return True
                    
                # Simple pattern check: look for regular patterns in LSBs
                if img_array.shape[0] > 10 and img_array.shape[1] > 10:
                    # Sample a small portion for pattern analysis
                    sample = lsb[:min(100, img_array.shape[0]), :min(100, img_array.shape[1])]
                    
                    # Check for repeating patterns
                    # This is a simplified approach and could be expanded
                    row_patterns = []
                    for row in range(1, min(20, sample.shape[0])):
                        match_ratio = np.mean(sample[0:sample.shape[0]-row, :] == sample[row:, :])
                        row_patterns.append(match_ratio)
                    
                    # If any strong pattern is found
                    if any(p > 0.9 for p in row_patterns):
                        return True
            
            return False
        except Exception as e:
            if self.verbose:
                logger.debug(f"Error in LSB check: {e}")
            return False
    
    def _detect_object_stream_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in PDF object streams
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        # Process all objects in the PDF
        try:
            for obj_id in self.pdf_reader.xref:
                try:
                    obj = self.pdf_reader.get_object(obj_id)
                    
                    # Focus on stream objects
                    if isinstance(obj, PyPDF2.generic.StreamObject):
                        stream_data = None
                        try:
                            stream_data = obj.get_data()
                        except Exception as e:
                            if self.verbose:
                                logger.debug(f"Error extracting stream data from object {obj_id}: {e}")
                            continue
                        
                        if not stream_data:
                            continue
                        
                        # Calculate entropy to detect potential encrypted/compressed data
                        entropy = self._calculate_entropy(stream_data)
                        
                        # Look for PNG signatures in stream data
                        png_matches = self._extract_potential_png(stream_data)
                        
                        for offset, png_data in png_matches:
                            # Analyze the PNG
                            png_details = self._analyze_png_data(png_data)
                            
                            if png_details:
                                # Check for LSB steganography in standard image
                                has_lsb = self._check_lsb_steganography(png_data) if self.deep_scan else False
                                
                                confidence = 0.5  # Base confidence
                                
                                # Increase confidence based on various factors
                                if png_details.has_unusual_chunks:
                                    confidence += 0.15
                                if png_details.has_suspicious_data:
                                    confidence += 0.15
                                if has_lsb:
                                    confidence += 0.2
                                if png_details.entropy > 7.0:
                                    confidence += 0.1
                                
                                # Check if image is referenced properly or "hidden"
                                is_referenced = False
                                for i, page in enumerate(self.pdf_reader.pages):
                                    if '/Resources' in page and '/XObject' in page['/Resources']:
                                        for xobj_name, xobj_ref in page['/Resources']['/XObject'].items():
                                            if isinstance(xobj_ref, PyPDF2.generic.IndirectObject) and xobj_ref.idnum == obj_id:
                                                is_referenced = True
                                                break
                                
                                # Hidden images are more suspicious
                                if not is_referenced:
                                    confidence += 0.1
                                
                                # Create a detection entry
                                details = {
                                    'width': png_details.width,
                                    'height': png_details.height,
                                    'bit_depth': png_details.bit_depth,
                                    'color_type': png_details.color_type,
                                    'unusual_chunks': png_details.has_unusual_chunks,
                                    'suspicious_data': png_details.has_suspicious_data,
                                    'entropy': png_details.entropy,
                                    'has_lsb_steganography': has_lsb,
                                    'stream_offset': offset,
                                    'is_properly_referenced': is_referenced
                                }
                                
                                results.append(SteganoDetection(
                                    method=SteganoMethod.OBJECT_STREAM,
                                    location=f"Object {obj_id} stream",
                                    confidence=min(confidence, 0.99),  # Cap at 0.99
                                    size_bytes=len(png_data),
                                    object_id=obj_id,
                                    details=details,
                                    sample=png_data[:100] if len(png_data) > 100 else png_data
                                ))
                                
                        # Check for other suspicious features in streams
                        if len(stream_data) > 20 and entropy > 7.2:
                            # Look for suspicious patterns that might indicate steganography
                            if b'FlateDecode' in stream_data and self._is_high_entropy(stream_data):
                                # Check for unusually sized data or unusual compression ratios
                                try:
                                    decompressed = zlib.decompress(stream_data)
                                    compression_ratio = len(decompressed) / len(stream_data)
                                    
                                    # Unusual compression ratio can indicate hidden data
                                    if compression_ratio < 0.5 or compression_ratio > 10:
                                        results.append(SteganoDetection(
                                            method=SteganoMethod.OBJECT_STREAM,
                                            location=f"Object {obj_id} stream (unusual compression)",
                                            confidence=0.6,
                                            size_bytes=len(stream_data),
                                            object_id=obj_id,
                                            details={
                                                'entropy': entropy,
                                                'compression_ratio': compression_ratio,
                                                'original_size': len(stream_data),
                                                'decompressed_size': len(decompressed)
                                            },
                                            sample=stream_data[:100] if len(stream_data) > 100 else stream_data
                                        ))
                                except Exception:
                                    # Failed decompression might indicate manipulated data
                                    pass
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Error examining object {obj_id}: {e}")
                        
            logger.info(f"Detected {len(results)} potential instances of object stream steganography")
            return results
        except Exception as e:
            logger.error(f"Error in object stream steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_metadata_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in PDF metadata
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Extract metadata
            metadata = self.pdf_reader.metadata
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, str) and len(value) > 20:
                        # Check for base64-encoded data
                        if re.search(r'^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$', value):
                            try:
                                # Try to decode base64
                                decoded = base64.b64decode(value)
                                
                                # Check if decoded data contains any known file signatures
                                for sig_name, signature in self.signatures.items():
                                    if signature in decoded:
                                        confidence = 0.85
                                        
                                        if sig_name == 'png':
                                            # Extract and analyze the PNG
                                            png_matches = self._extract_potential_png(decoded)
                                            for _, png_data in png_matches:
                                                png_details = self._analyze_png_data(png_data)
                                                if png_details:
                                                    results.append(SteganoDetection(
                                                        method=SteganoMethod.METADATA,
                                                        location=f"Metadata field: {key}",
                                                        confidence=confidence,
                                                        size_bytes=len(png_data),
                                                        details={
                                                            'metadata_field': key,
                                                            'encoded_type': 'base64',
                                                            'contains': sig_name,
                                                            'png_details': {
                                                                'width': png_details.width,
                                                                'height': png_details.height,
                                                                'bit_depth': png_details.bit_depth,
                                                                'unusual_chunks': png_details.has_unusual_chunks
                                                            }
                                                        },
                                                        extracted_data=png_data,
                                                        sample=decoded[:100] if len(decoded) > 100 else decoded
                                                    ))
                                        else:
                                            results.append(SteganoDetection(
                                                method=SteganoMethod.METADATA,
                                                location=f"Metadata field: {key}",
                                                confidence=confidence,
                                                size_bytes=len(decoded),
                                                details={
                                                    'metadata_field': key,
                                                    'encoded_type': 'base64',
                                                    'contains': sig_name
                                                },
                                                sample=decoded[:100] if len(decoded) > 100 else decoded
                                            ))
                            except Exception:
                                # Not valid base64
                                pass
                                
                        # Check for hex-encoded data
                        elif re.search(r'^[0-9a-fA-F]+$', value) and len(value) % 2 == 0:
                            try:
                                # Try to decode hex
                                decoded = binascii.unhexlify(value)
                                
                                # Check if decoded data contains any known file signatures
                                for sig_name, signature in self.signatures.items():
                                    if signature in decoded:
                                        confidence = 0.8
                                        
                                        if sig_name == 'png':
                                            # Extract and analyze the PNG
                                            png_matches = self._extract_potential_png(decoded)
                                            for _, png_data in png_matches:
                                                png_details = self._analyze_png_data(png_data)
                                                if png_details:
                                                    results.append(SteganoDetection(
                                                        method=SteganoMethod.METADATA,
                                                        location=f"Metadata field: {key}",
                                                        confidence=confidence,
                                                        size_bytes=len(png_data),
                                                        details={
                                                            'metadata_field': key,
                                                            'encoded_type': 'hex',
                                                            'contains': sig_name,
                                                            'png_details': {
                                                                'width': png_details.width,
                                                                'height': png_details.height,
                                                                'bit_depth': png_details.bit_depth,
                                                                'unusual_chunks': png_details.has_unusual_chunks
                                                            }
                                                        },
                                                        extracted_data=png_data,
                                                        sample=decoded[:100] if len(decoded) > 100 else decoded
                                                    ))
                                        else:
                                            results.append(SteganoDetection(
                                                method=SteganoMethod.METADATA,
                                                location=f"Metadata field: {key}",
                                                confidence=confidence,
                                                size_bytes=len(decoded),
                                                details={
                                                    'metadata_field': key,
                                                    'encoded_type': 'hex',
                                                    'contains': sig_name
                                                },
                                                sample=decoded[:100] if len(decoded) > 100 else decoded
                                            ))
                            except Exception:
                                # Not valid hex
                                pass
                                
                        # Check for unusual high entropy in metadata value
                        elif len(value) > 100:
                            entropy = self._calculate_entropy(value.encode('utf-8', errors='replace'))
                            if entropy > 6.5:  # High entropy for text data
                                results.append(SteganoDetection(
                                    method=SteganoMethod.METADATA,
                                    location=f"Metadata field: {key}",
                                    confidence=0.6,
                                    size_bytes=len(value),
                                    details={
                                        'metadata_field': key,
                                        'entropy': entropy,
                                        'length': len(value)
                                    },
                                    sample=value[:100].encode('utf-8', errors='replace')
                                ))
                    
            # Check document information dictionary
            if hasattr(self.pdf_document, 'info'):
                for info_dict in self.pdf_document.info:
                    for key, value in info_dict.items():
                        if isinstance(value, bytes) and len(value) > 20:
                            # Basic entropy check
                            entropy = self._calculate_entropy(value)
                            if entropy > 6.8:
                                # Look for PNG signatures
                                png_matches = self._extract_potential_png(value)
                                for offset, png_data in png_matches:
                                    png_details = self._analyze_png_data(png_data)
                                    if png_details:
                                        results.append(SteganoDetection(
                                            method=SteganoMethod.METADATA,
                                            location=f"Info dictionary field: {key.decode('utf-8', errors='replace')}",
                                            confidence=0.85,
                                            size_bytes=len(png_data),
                                            details={
                                                'info_field': key.decode('utf-8', errors='replace'),
                                                'png_details': {
                                                    'width': png_details.width,
                                                    'height': png_details.height,
                                                    'bit_depth': png_details.bit_depth,
                                                    'unusual_chunks': png_details.has_unusual_chunks
                                                },
                                                'offset': offset
                                            },
                                            extracted_data=png_data,
                                            sample=value[:100]
                                        ))
                                        
# If no PNG but high entropy
                                if not png_matches and entropy > 7.0:
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.METADATA,
                                        location=f"Info dictionary field: {key.decode('utf-8', errors='replace')}",
                                        confidence=0.7,
                                        size_bytes=len(value),
                                        details={
                                            'info_field': key.decode('utf-8', errors='replace'),
                                            'entropy': entropy
                                        },
                                        sample=value[:100]
                                    ))
                        elif isinstance(value, PSLiteral):
                            # Handle PSLiteral values which can sometimes contain hidden data
                            if isinstance(value.name, bytes) and len(value.name) > 50:
                                entropy = self._calculate_entropy(value.name)
                                if entropy > 6.5:
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.METADATA,
                                        location=f"Info dictionary PSLiteral: {key.decode('utf-8', errors='replace')}",
                                        confidence=0.6,
                                        size_bytes=len(value.name),
                                        details={
                                            'info_field': key.decode('utf-8', errors='replace'),
                                            'entropy': entropy,
                                            'type': 'PSLiteral'
                                        },
                                        sample=value.name[:100]
                                    ))
                                    
            logger.info(f"Detected {len(results)} potential instances of metadata steganography")
            return results
        except Exception as e:
            logger.error(f"Error in metadata steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_whitespace_comment_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in PDF whitespace and comments
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Look for patterns of unusual whitespace or comments in the raw PDF data
            pdf_text = self.raw_pdf_data.decode('latin-1', errors='replace')
            
            # Check for comments that might contain hidden data
            comment_pattern = re.compile(r'%[^\n]*\n')
            comments = comment_pattern.findall(pdf_text)
            
            suspicious_comments = []
            for comment in comments:
                # Skip standard PDF header comments
                if comment.startswith('%PDF-') or comment.startswith('%EOF'):
                    continue
                    
                # Check for long comments
                if len(comment) > 50:
                    # Check for base64-like content
                    if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', comment):
                        suspicious_comments.append(comment)
                    
                    # Check for high entropy
                    entropy = self._calculate_entropy(comment.encode('latin-1', errors='replace'))
                    if entropy > 6.0:  # Lower threshold since comments are usually readable
                        suspicious_comments.append(comment)
            
            # Group suspicious comments that appear in sequence (might be split data)
            if suspicious_comments:
                grouped_comments = []
                current_group = []
                last_index = -1
                
                for comment in suspicious_comments:
                    current_index = pdf_text.find(comment)
                    if last_index == -1 or current_index - last_index < 100:  # Group nearby comments
                        current_group.append(comment)
                    else:
                        if current_group:
                            grouped_comments.append(current_group)
                        current_group = [comment]
                    last_index = current_index
                
                if current_group:
                    grouped_comments.append(current_group)
                
                # Analyze each group
                for group in grouped_comments:
                    combined = ''.join(group)
                    # Strip common comment characters
                    data = re.sub(r'%|\n|\r', '', combined)
                    
                    # Check for possible encoding patterns
                    confidence = 0.0
                    
                    # Base64 check
                    if re.match(r'^[A-Za-z0-9+/]+={0,2}$', data):
                        try:
                            decoded = base64.b64decode(data)
                            # Check if decoded data contains known file signatures
                            for sig_name, signature in self.signatures.items():
                                if signature in decoded:
                                    confidence = 0.85
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.WHITESPACE_COMMENT,
                                        location="PDF comments",
                                        confidence=confidence,
                                        size_bytes=len(decoded),
                                        details={
                                            'encoding': 'base64',
                                            'contains': sig_name,
                                            'num_comments': len(group)
                                        },
                                        extracted_data=decoded,
                                        sample=data[:100].encode('latin-1', errors='replace')
                                    ))
                                    break
                            
                            # If no known signature but still high entropy
                            if confidence == 0.0 and len(decoded) > 20:
                                entropy = self._calculate_entropy(decoded)
                                if entropy > 7.0:
                                    confidence = 0.6
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.WHITESPACE_COMMENT,
                                        location="PDF comments",
                                        confidence=confidence,
                                        size_bytes=len(decoded),
                                        details={
                                            'encoding': 'base64',
                                            'entropy': entropy,
                                            'num_comments': len(group)
                                        },
                                        sample=data[:100].encode('latin-1', errors='replace')
                                    ))
                        except Exception:
                            # Not valid base64
                            pass
                    
                    # Hex check
                    elif re.match(r'^[0-9A-Fa-f]+$', data) and len(data) % 2 == 0:
                        try:
                            decoded = binascii.unhexlify(data)
                            # Check if decoded data contains known file signatures
                            for sig_name, signature in self.signatures.items():
                                if signature in decoded:
                                    confidence = 0.85
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.WHITESPACE_COMMENT,
                                        location="PDF comments",
                                        confidence=confidence,
                                        size_bytes=len(decoded),
                                        details={
                                            'encoding': 'hex',
                                            'contains': sig_name,
                                            'num_comments': len(group)
                                        },
                                        extracted_data=decoded,
                                        sample=data[:100].encode('latin-1', errors='replace')
                                    ))
                                    break
                            
                            # If no known signature but still high entropy
                            if confidence == 0.0 and len(decoded) > 20:
                                entropy = self._calculate_entropy(decoded)
                                if entropy > 7.0:
                                    confidence = 0.6
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.WHITESPACE_COMMENT,
                                        location="PDF comments",
                                        confidence=confidence,
                                        size_bytes=len(decoded),
                                        details={
                                            'encoding': 'hex',
                                            'entropy': entropy,
                                            'num_comments': len(group)
                                        },
                                        sample=data[:100].encode('latin-1', errors='replace')
                                    ))
                        except Exception:
                            # Not valid hex
                            pass
            
            # Check for unusual whitespace patterns
            # PDF spec allows arbitrary whitespace in many places
            whitespace_pattern = re.compile(r'(\s{10,})')  # Look for long stretches of whitespace
            whitespaces = whitespace_pattern.findall(pdf_text)
            
            for whitespace in whitespaces:
                if len(whitespace) > 50:  # Very long whitespace is suspicious
                    # Check for binary data encoded in whitespace patterns
                    # e.g., spaces and tabs can encode binary data
                    space_tab_pattern = re.compile(r'[ \t]+')
                    if space_tab_pattern.match(whitespace):
                        # Convert spaces to 0 and tabs to 1 (or vice versa)
                        binary_data = whitespace.replace(' ', '0').replace('\t', '1')
                        
                        # Try to convert to bytes
                        try:
                            # Check if we have complete bytes (multiples of 8 bits)
                            if len(binary_data) % 8 == 0:
                                # Convert binary string to bytes
                                byte_data = bytearray()
                                for i in range(0, len(binary_data), 8):
                                    byte = int(binary_data[i:i+8], 2)
                                    byte_data.append(byte)
                                
                                # Check entropy of the data
                                entropy = self._calculate_entropy(byte_data)
                                
                                if entropy > 6.5:  # Moderate entropy threshold
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.WHITESPACE_COMMENT,
                                        location="PDF whitespace",
                                        confidence=0.7,
                                        size_bytes=len(byte_data),
                                        details={
                                            'encoding': 'whitespace_binary',
                                            'entropy': entropy,
                                            'pattern': 'space_tab'
                                        },
                                        extracted_data=bytes(byte_data),
                                        sample=whitespace[:100].encode('latin-1', errors='replace')
                                    ))
                        except Exception as e:
                            if self.verbose:
                                logger.debug(f"Error analyzing whitespace pattern: {e}")
                    
                    # Check for other patterns of whitespace that might encode data
                    # e.g., varying amounts of spaces between words
                    # This is more complex and would require custom pattern recognition
            
            logger.info(f"Detected {len(results)} potential instances of whitespace/comment steganography")
            return results
        except Exception as e:
            logger.error(f"Error in whitespace/comment steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_document_component_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in document components (e.g., fonts, images)
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Process each page in the PDF
            for page_idx, page in enumerate(self.pdf_reader.pages):
                # Check for hidden images
                if '/Resources' in page and '/XObject' in page['/Resources']:
                    for xobj_name, xobj in page['/Resources']['/XObject'].items():
                        if isinstance(xobj, PyPDF2.generic.IndirectObject):
                            # Get the actual object
                            obj = self.pdf_reader.get_object(xobj.idnum)
                            
                            # Check if it's an image
                            if '/Subtype' in obj and obj['/Subtype'] == '/Image':
                                # Examine image properties
                                suspicious = False
                                confidence = 0.0
                                details = {
                                    'xobject_name': xobj_name,
                                    'page': page_idx + 1
                                }
                                
                                # Check for unusual filter combinations
                                if '/Filter' in obj:
                                    filters = obj['/Filter']
                                    if isinstance(filters, list) and len(filters) > 1:
                                        # Multiple filters are sometimes used to hide data
                                        suspicious = True
                                        confidence += 0.2
                                        details['filters'] = [str(f) for f in filters]
                                
                                # Check for unusual color spaces
                                if '/ColorSpace' in obj:
                                    colorspace = obj['/ColorSpace']
                                    if isinstance(colorspace, PyPDF2.generic.ArrayObject) and len(colorspace) > 3:
                                        # Complex color spaces can hide data
                                        suspicious = True
                                        confidence += 0.2
                                        details['colorspace'] = str(colorspace)
                                
                                # Check image dimensions
                                if '/Width' in obj and '/Height' in obj:
                                    width = obj['/Width']
                                    height = obj['/Height']
                                    details['width'] = width
                                    details['height'] = height
                                    
                                    # Very small or oddly sized images might hide data
                                    if (width < 4 or height < 4) and not (width == 1 and height == 1):
                                        suspicious = True
                                        confidence += 0.3
                                        details['unusual_size'] = True
                                
                                # Check for image data
                                try:
                                    stream_data = obj.get_data()
                                    if stream_data:
                                        entropy = self._calculate_entropy(stream_data)
                                        details['entropy'] = entropy
                                        
                                        # High entropy indicates potential hidden data
                                        if entropy > 7.2:
                                            suspicious = True
                                            confidence += 0.2
                                        
                                        # Check for embedded PNG signatures
                                        png_matches = self._extract_potential_png(stream_data)
                                        for offset, png_data in png_matches:
                                            png_details = self._analyze_png_data(png_data)
                                            if png_details:
                                                # Nested PNGs in image objects are very suspicious
                                                results.append(SteganoDetection(
                                                    method=SteganoMethod.DOCUMENT_COMPONENTS,
                                                    location=f"Image XObject (page {page_idx + 1})",
                                                    confidence=0.9,
                                                    size_bytes=len(png_data),
                                                    page_number=page_idx + 1,
                                                    object_id=xobj.idnum,
                                                    details={
                                                        'xobject_name': xobj_name,
                                                        'offset': offset,
                                                        'png_details': {
                                                            'width': png_details.width,
                                                            'height': png_details.height,
                                                            'bit_depth': png_details.bit_depth,
                                                            'unusual_chunks': png_details.has_unusual_chunks
                                                        }
                                                    },
                                                    extracted_data=png_data,
                                                    sample=stream_data[:100] if len(stream_data) > 100 else stream_data
                                                ))
                                except Exception as e:
                                    if self.verbose:
                                        logger.debug(f"Error extracting image data: {e}")
                                
                                # Add detection if suspicious
                                if suspicious and confidence > 0.4:
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.DOCUMENT_COMPONENTS,
                                        location=f"Image XObject (page {page_idx + 1})",
                                        confidence=min(confidence, 0.8),  # Cap at 0.8
                                        page_number=page_idx + 1,
                                        object_id=xobj.idnum,
                                        details=details
                                    ))
                
                # Check fonts for hidden data
                # Use pdfminer for more detailed analysis
                for lt_obj in extract_pages(io.BytesIO(self.raw_pdf_data), page_numbers=[page_idx]):
                    if isinstance(lt_obj, LTTextBox):
                        # Look for unusual characters or patterns in the text
                        text = lt_obj.get_text()
                        
                        # Check for unusual Unicode ranges that might hide data
                        if any(ord(c) > 0x1000 for c in text):
                            # Unusual Unicode characters (excluding common CJK ranges)
                            if not all(0x4E00 <= ord(c) <= 0x9FFF for c in text if ord(c) > 0x1000):
                                results.append(SteganoDetection(
                                    method=SteganoMethod.DOCUMENT_COMPONENTS,
                                    location=f"Text (page {page_idx + 1})",
                                    confidence=0.5,
                                    page_number=page_idx + 1,
                                    details={
                                        'contains_unusual_unicode': True,
                                        'text_sample': text[:100]
                                    },
                                    sample=text[:100].encode('utf-8', errors='replace')
                                ))
                                
                        # Check for patterns that might indicate binary data encoded as text
                        if len(text) > 100:
                            # Look for long sequences with only 2-4 unique characters
                            char_set = set(text)
                            if 2 <= len(char_set) <= 4 and not all(c.isspace() for c in char_set):
                                results.append(SteganoDetection(
                                    method=SteganoMethod.DOCUMENT_COMPONENTS,
                                    location=f"Text (page {page_idx + 1})",
                                    confidence=0.7,
                                    page_number=page_idx + 1,
                                    details={
                                        'limited_charset': True,
                                        'charset_size': len(char_set),
                                        'text_sample': text[:100]
                                    },
                                    sample=text[:100].encode('utf-8', errors='replace')
                                ))
            
            # Also check for unusual font embedding
            try:
                for obj_id in self.pdf_reader.xref:
                    obj = self.pdf_reader.get_object(obj_id)
                    if isinstance(obj, dict) and obj.get('/Type') == '/Font':
                        if '/FontDescriptor' in obj and '/FontFile2' in obj['/FontDescriptor']:
                            # TrueType fonts can contain hidden data
                            font_file = obj['/FontDescriptor']['/FontFile2']
                            if isinstance(font_file, PyPDF2.generic.IndirectObject):
                                font_obj = self.pdf_reader.get_object(font_file.idnum)
                                if isinstance(font_obj, PyPDF2.generic.StreamObject):
                                    try:
                                        font_data = font_obj.get_data()
                                        entropy = self._calculate_entropy(font_data)
                                        
                                        # Check for unusual entropy in font data
                                        if entropy > 7.5:  # TrueType fonts have high entropy normally
                                            # Look for embedded file signatures in font data
                                            for sig_name, signature in self.signatures.items():
                                                if signature in font_data and sig_name != 'ttf':
                                                    results.append(SteganoDetection(
                                                        method=SteganoMethod.DOCUMENT_COMPONENTS,
                                                        location=f"Embedded font (obj {obj_id})",
                                                        confidence=0.85,
                                                        object_id=obj_id,
                                                        details={
                                                            'font_name': obj.get('/BaseFont', 'Unknown'),
                                                            'contains': sig_name,
                                                            'entropy': entropy
                                                        },
                                                        sample=font_data[:100] if len(font_data) > 100 else font_data
                                                    ))
                                    except Exception as e:
                                        if self.verbose:
                                            logger.debug(f"Error examining font data: {e}")
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error checking font objects: {e}")
            
            logger.info(f"Detected {len(results)} potential instances of document component steganography")
            return results
        except Exception as e:
            logger.error(f"Error in document component steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_javascript_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in PDF JavaScript
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Look for JavaScript objects in PDF
            for obj_id in self.pdf_reader.xref:
                obj = self.pdf_reader.get_object(obj_id)
                
                # Check for /JS and /JavaScript objects
                if isinstance(obj, dict) and ('/JS' in obj or '/JavaScript' in obj):
                    js_obj = None
                    if '/JS' in obj:
                        js_obj = obj['/JS']
                    elif '/JavaScript' in obj:
                        js_obj = obj['/JavaScript']
                    
                    # Extract JavaScript code
                    js_code = None
                    if isinstance(js_obj, PyPDF2.generic.TextStringObject):
                        js_code = js_obj
                    elif isinstance(js_obj, PyPDF2.generic.IndirectObject):
                        js_stream = self.pdf_reader.get_object(js_obj.idnum)
                        if isinstance(js_stream, PyPDF2.generic.StreamObject):
                            try:
                                js_code = js_stream.get_data().decode('latin-1', errors='replace')
                            except Exception as e:
                                if self.verbose:
                                    logger.debug(f"Error decoding JavaScript stream: {e}")
                    
                    if js_code:
                        # Look for suspicious JavaScript patterns that might indicate steganography
                        
                        # Check for large encoded strings
                        b64_pattern = re.compile(r'["\']([A-Za-z0-9+/]{50,}={0,2})["\']')
                        b64_matches = b64_pattern.findall(js_code)
                        
                        for b64_str in b64_matches:
                            try:
                                # Try to decode as base64
                                decoded = base64.b64decode(b64_str)
                                
                                # Check for known file signatures
                                for sig_name, signature in self.signatures.items():
                                    if signature in decoded:
                                        results.append(SteganoDetection(
                                            method=SteganoMethod.JAVASCRIPT,
                                            location=f"JavaScript (obj {obj_id})",
                                            confidence=0.9,
                                            size_bytes=len(decoded),
                                            object_id=obj_id,
                                            details={
                                                'encoding': 'base64',
                                                'contains': sig_name,
                                                'length': len(b64_str)
                                            },
                                            extracted_data=decoded,
                                            sample=decoded[:100] if len(decoded) > 100 else decoded
                                        ))
                                        break
                                
                                # If no known signature but high entropy
                                if len(decoded) > 100:
                                    entropy = self._calculate_entropy(decoded)
                                    if entropy > 7.0:
                                        results.append(SteganoDetection(
                                            method=SteganoMethod.JAVASCRIPT,
                                            location=f"JavaScript (obj {obj_id})",
                                            confidence=0.7,
                                            size_bytes=len(decoded),
                                            object_id=obj_id,
                                            details={
                                                'encoding': 'base64',
                                                'entropy': entropy,
                                                'length': len(b64_str)
                                            },
                                            sample=decoded[:100] if len(decoded) > 100 else decoded
                                        ))
                            except Exception:
                                # Not valid base64
                                pass
                        
                        # Check for hex-encoded data
                        hex_pattern = re.compile(r'["\']([0-9A-Fa-f]{50,})["\']')
                        hex_matches = hex_pattern.findall(js_code)
                        
                        for hex_str in hex_matches:
                            if len(hex_str) % 2 == 0:  # Valid hex strings have even length
                                try:
                                    # Try to decode as hex
                                    decoded = binascii.unhexlify(hex_str)
                                    
                                    # Check for known file signatures
                                    for sig_name, signature in self.signatures.items():
                                        if signature in decoded:
                                            results.append(SteganoDetection(
                                                method=SteganoMethod.JAVASCRIPT,
                                                location=f"JavaScript (obj {obj_id})",
                                                confidence=0.9,
                                                size_bytes=len(decoded),
                                                object_id=obj_id,
                                                details={
                                                    'encoding': 'hex',
                                                    'contains': sig_name,
                                                    'length': len(hex_str)
                                                },
                                                extracted_data=decoded,
                                                sample=decoded[:100] if len(decoded) > 100 else decoded
                                            ))
                                            break
                                    
                                    # If no known signature but high entropy
                                    if len(decoded) > 100:
                                        entropy = self._calculate_entropy(decoded)
                                        if entropy > 7.0:
                                            results.append(SteganoDetection(
                                                method=SteganoMethod.JAVASCRIPT,
                                                location=f"JavaScript (obj {obj_id})",
                                                confidence=0.7,
                                                size_bytes=len(decoded),
                                                object_id=obj_id,
                                                details={
                                                    'encoding': 'hex',
                                                    'entropy': entropy,
                                                    'length': len(hex_str)
                                                },
                                                sample=decoded[:100] if len(decoded) > 100 else decoded
                                            ))
                                except Exception:
                                    # Not valid hex
                                    pass
                        
                        # Check for suspicious JavaScript functions commonly used in steganography
                        suspicious_functions = [
                            'unescape', 'String.fromCharCode', 'atob', 'btoa',
                            'eval', 'fromCharCode', 'charCodeAt', 'createElement'
                        ]
                        
                        for func in suspicious_functions:
                            if func in js_code:
                                # Find where the function is used
                                func_pattern = re.compile(fr'{func}\s*\(([^)]+)\)')
                                func_matches = func_pattern.findall(js_code)
                                
                                if func_matches:
                                    results.append(SteganoDetection(
                                        method=SteganoMethod.JAVASCRIPT,
                                        location=f"JavaScript (obj {obj_id})",
                                        confidence=0.6,  # Lower confidence, needs more validation
                                        object_id=obj_id,
                                        details={
                                            'suspicious_function': func,
                                            'occurrences': len(func_matches),
                                            'example': func_matches[0][:100] if func_matches[0] else ''
                                        },
                                        sample=js_code[:100].encode('utf-8', errors='replace')
                                    ))
                        
                        # Check for suspicious comments
                        if '/*' in js_code and '*/' in js_code:
                            comment_pattern = re.compile(r'/\*(.*?)\*/', re.DOTALL)
                            comments = comment_pattern.findall(js_code)
                            
                            for comment in comments:
                                if len(comment) > 100:  # Long comments might hide data
                                    # Calculate entropy of comment
                                    entropy = self._calculate_entropy(comment.encode('utf-8', errors='replace'))
                                    if entropy > 6.0:  # Lower threshold for text
                                        results.append(SteganoDetection(
                                            method=SteganoMethod.JAVASCRIPT,
                                            location=f"JavaScript comment (obj {obj_id})",
                                            confidence=0.5,
                                            size_bytes=len(comment),
                                            object_id=obj_id,
                                            details={
                                                'type': 'comment',
                                                'entropy': entropy,
                                                'length': len(comment)
                                            },
                                            sample=comment[:100].encode('utf-8', errors='replace')
                                        ))
            
            logger.info(f"Detected {len(results)} potential instances of JavaScript steganography")
            return results
        except Exception as e:
            logger.error(f"Error in JavaScript steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_image_data_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in image data, especially focusing on PNG images
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Extract all image data from the PDF
            images = []
            for i, page in enumerate(self.pdf_reader.pages):
                # Process images through PyPDF2
                if '/Resources' in page and '/XObject' in page['/Resources']:
                    for xobj_name, xobj in page['/Resources']['/XObject'].items():
                        if isinstance(xobj, PyPDF2.generic.IndirectObject):
                            obj = self.pdf_reader.get_object(xobj.idnum)
                            if isinstance(obj, dict) and obj.get('/Subtype') == '/Image':
                                try:
                                    stream_data = None
                                    if isinstance(obj, PyPDF2.generic.StreamObject):
                                        stream_data = obj.get_data()
                                    
                                    if stream_data:
                                        images.append({
                                            'data': stream_data,
                                            'page': i + 1,
                                            'obj_id': xobj.idnum,
                                            'name': xobj_name,
                                            'width': obj.get('/Width', 0),
                                            'height': obj.get('/Height', 0),
                                            'filter': obj.get('/Filter', 'Unknown')
                                        })
                                except Exception as e:
                                    if self.verbose:
                                        logger.debug(f"Error extracting image data: {e}")
                
                # Process images through pdfminer (might find additional images)
                try:
                    with io.BytesIO(self.raw_pdf_data) as pdf_stream:
                        parser = PDFParser(pdf_stream)
                        document = PDFDocument(parser)
                        rsrcmgr = PDFResourceManager()
                        device = PDFPageAggregator(rsrcmgr, laparams=self.laparams)
                        interpreter = PDFPageInterpreter(rsrcmgr, device)
                        
                        for j, page in enumerate(extract_pages(pdf_stream)):
                            if j == i:  # Match the current page
                                for lt_obj in page:
                                    if isinstance(lt_obj, LTImage) or isinstance(lt_obj, LTFigure):
                                        # Check if it's an actual image
                                        if hasattr(lt_obj, 'stream') and lt_obj.stream:
                                            # Extract raw image data
                                            img_data = lt_obj.stream.get_data()
                                            if img_data and len(img_data) > 100:
                                                # Avoid duplicates by checking if we already have this image
                                                if not any(img['data'] == img_data for img in images):
                                                    images.append({
                                                        'data': img_data,
                                                        'page': i + 1,
                                                        'obj_id': None,  # Unknown from pdfminer
                                                        'name': f"pdfminer_img_{len(images)}",
                                                        'width': lt_obj.width if hasattr(lt_obj, 'width') else 0,
                                                        'height': lt_obj.height if hasattr(lt_obj, 'height') else 0,
                                                        'filter': 'Unknown'
                                                    })
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Error extracting images via pdfminer: {e}")
            
            # Process all collected images
            for img_info in images:
                img_data = img_info['data']
                
                # Extract and analyze PNGs
                png_matches = self._extract_potential_png(img_data)
                for offset, png_data in png_matches:
                    png_details = self._analyze_png_data(png_data)
                    
                    if png_details:
                        # Check for LSB steganography
                        has_lsb = self._check_lsb_steganography(png_data) if self.deep_scan else False
                        
                        confidence = 0.5  # Base confidence
                        
                        # Adjust confidence based on various factors
                        if png_details.has_unusual_chunks:
                            confidence += 0.15
                        if png_details.has_suspicious_data:
                            confidence += 0.15
                        if has_lsb:
                            confidence += 0.2
                        if png_details.entropy > 7.0:
                            confidence += 0.1
                        
                        # Create detection entry
                        detection = SteganoDetection(
                            method=SteganoMethod.IMAGE_DATA,
                            location=f"Image on page {img_info['page']}",
                            confidence=min(confidence, 0.95),  # Cap at 0.95
                            size_bytes=len(png_data),
                            page_number=img_info['page'],
                            object_id=img_info['obj_id'],
                            details={
                                'image_name': img_info['name'],
                                'png_details': {
                                    'width': png_details.width,
                                    'height': png_details.height,
                                    'bit_depth': png_details.bit_depth,
                                    'color_type': png_details.color_type,
                                    'unusual_chunks': png_details.has_unusual_chunks,
                                    'suspicious_data': png_details.has_suspicious_data,
                                    'entropy': png_details.entropy,
                                    'has_lsb_steganography': has_lsb,
                                },
                                'offset_in_stream': offset
                            },
                            extracted_data=png_data,
                            sample=png_data[:100] if len(png_data) > 100 else png_data
                        )
                        
                        results.append(detection)
                
                # If no PNG found, check other image formats and patterns
                if not png_matches and len(img_data) > 1000:
                    # Calculate entropy
                    entropy = self._calculate_entropy(img_data)
                    
                    # Check for unusual entropy (very high or suspiciously balanced)
                    if entropy > 7.5 or (6.8 < entropy < 7.2):  # Near 7.0 can indicate manipulated data
                        # If image has JPEG signature
                        if img_data.startswith(b'\xff\xd8\xff'):
                            # Check for additional data after JPEG EOF marker
                            eof_marker = b'\xff\xd9'
                            eof_pos = img_data.find(eof_marker)
                            
                            if eof_pos != -1 and eof_pos + 2 < len(img_data):
                                # Data after EOF
                                appended_data = img_data[eof_pos + 2:]
                                
                                results.append(SteganoDetection(
                                    method=SteganoMethod.IMAGE_DATA,
                                    location=f"JPEG image on page {img_info['page']}",
                                    confidence=0.9,  # High confidence for data after EOF
                                    size_bytes=len(appended_data),
                                    page_number=img_info['page'],
                                    object_id=img_info['obj_id'],
                                    details={
                                        'image_name': img_info['name'],
                                        'technique': 'data_after_eof',
                                        'eof_position': eof_pos,
                                        'appended_data_size': len(appended_data)
                                    },
                                    extracted_data=appended_data,
                                    sample=appended_data[:100] if len(appended_data) > 100 else appended_data
                                ))
                            else:
                                # Check for unusual entropy in JPEG data
                                results.append(SteganoDetection(
                                    method=SteganoMethod.IMAGE_DATA,
                                    location=f"JPEG image on page {img_info['page']}",
                                    confidence=0.6,
                                    size_bytes=len(img_data),
                                    page_number=img_info['page'],
                                    object_id=img_info['obj_id'],
                                    details={
                                        'image_name': img_info['name'],
                                        'entropy': entropy,
                                        'format': 'JPEG'
                                    },
                                    sample=img_data[:100]
                                ))
                        else:
                            # Other image format or unknown
                            results.append(SteganoDetection(
                                method=SteganoMethod.IMAGE_DATA,
                                location=f"Image on page {img_info['page']}",
                                confidence=0.5,
                                size_bytes=len(img_data),
                                page_number=img_info['page'],
                                object_id=img_info['obj_id'],
                                details={
                                    'image_name': img_info['name'],
                                    'entropy': entropy,
                                    'format': 'Unknown'
                                },
                                sample=img_data[:100]
                            ))
            
            logger.info(f"Detected {len(results)} potential instances of image data steganography")
            return results
        except Exception as e:
            logger.error(f"Error in image data steganography detection: {e}", exc_info=True)
            return []
    
    def _detect_xref_table_steganography(self) -> List[SteganoDetection]:
        """
        Detect steganography in PDF cross-reference table
        
        Returns:
            List[SteganoDetection]: List of detected steganography instances
        """
        results = []
        
        try:
            # Extract the xref table from raw PDF data
            xref_pattern = re.compile(rb'xref\s+\d+\s+\d+\s+(.*?)(?:trailer|\Z)', re.DOTALL)
            xref_matches = xref_pattern.findall(self.raw_pdf_data)
            
            if not xref_matches:
                return results
                
            # Analyze xref entries
            for xref_data in xref_matches:
                # Regular xref entries are 20 bytes each: 10 digits offset, 5 digits generation, 1 char flag (f/n), 2 spaces, 2 EOL chars
                # Check if any entries have unusual patterns
                
                # Split into lines
                xref_lines = xref_data.split(b'\n')
                for line in xref_lines:
                    if not line.strip():
                        continue
                        
                    # Check if line matches expected format
                    if not re.match(rb'\d{10}\s\d{5}\s[fn]\s', line):
                        continue
                        
                    # Extract offset and generation
                    try:
                        offset = int(line[:10])
                        generation = int(line[11:16])
                        is_free = line[17:18] == b'f'
                        
                        # Check for highly suspicious patterns
                        if offset > len(self.raw_pdf_data) and not is_free:
                            # References outside the file
                            results.append(SteganoDetection(
                                method=SteganoMethod.XREF_TABLE,
                                location="PDF xref table",
                                confidence=0.8,
                                details={
                                    'invalid_offset': True,
                                    'offset': offset,
                                    'generation': generation,
                                    'file_size': len(self.raw_pdf_data)
                                },
                                sample=line
                            ))
                        elif generation > 0 and not is_free:
                            # Unusual generation number for non-free object
                            # This is uncommon but not necessarily steganography
                            if generation > 65535:  # Very high generation number
                                results.append(SteganoDetection(
                                    method=SteganoMethod.XREF_TABLE,
                                    location="PDF xref table",
                                    confidence=0.7,
                                    details={
                                        'unusual_generation': True,
                                        'offset': offset,
                                        'generation': generation
                                    },
                                    sample=line
                                ))
                    except Exception:
                        # Invalid xref entry
                        pass
            
            # Check for objects that are in the PDF but not in the xref table
            trailer_pattern = re.compile(rb'trailer\s*<<(.*?)>>', re.DOTALL)
            trailer_matches = trailer_pattern.findall(self.raw_pdf_data)
            
            if trailer_matches:
                for trailer_data in trailer_matches:
                    # Extract Size entry from trailer
                    size_pattern = re.compile(rb'/Size\s+(\d+)')
                    size_match = size_pattern.search(trailer_data)
                    
                    if size_match:
                        expected_size = int(size_match.group(1))
                        actual_objects = len(self.pdf_reader.xref)
                        
                        # Check for significant mismatch
                        if actual_objects > expected_size * 1.1:  # More than 10% extra objects
                            results.append(SteganoDetection(
                                method=SteganoMethod.XREF_TABLE,
                                location="PDF trailer",
                                confidence=0.8,
                                details={
                                    'object_count_mismatch': True,
                                    'expected_size': expected_size,
                                    'actual_objects': actual_objects,
                                    'extra_objects': actual_objects - expected_size
                                },
                                sample=trailer_data[:100]
                            ))
            
            # Look for hidden objects (references in raw data that don't match xref table)
            obj_pattern = re.compile(rb'(\d+)\s+(\d+)\s+obj')
            obj_matches = obj_pattern.findall(self.raw_pdf_data)
            
            # Create a set of objects from the xref table
            xref_objects = set()
            for obj_id in self.pdf_reader.xref:
                if obj_id != 0:  # Skip the special zero object
                    xref_objects.add(obj_id)
            
            # Check for objects not in the xref table
            hidden_objects = []
            for obj_id_bytes, gen_bytes in obj_matches:
                obj_id = int(obj_id_bytes)
                gen = int(gen_bytes)
                
                if obj_id not in xref_objects and obj_id != 0:
                    hidden_objects.append((obj_id, gen))
            
            if hidden_objects:
                results.append(SteganoDetection(
                    method=SteganoMethod.XREF_TABLE,
                    location="PDF hidden objects",
                    confidence=0.9,
                    details={
                        'hidden_objects': True,
                        'object_count': len(hidden_objects),
                        'objects': hidden_objects[:10]  # List first 10 hidden objects
                    }
                ))
            
            logger.info(f"Detected {len(results)} potential instances of xref table steganography")
            return results
        except Exception as e:
            logger.error(f"Error in xref table steganography detection: {e}", exc_info=True)
            return []
    
    def detect(self) -> List[SteganoDetection]:
        """
        Run all steganography detection methods
        
        Returns:
            List[SteganoDetection]: Combined list of all detected steganography instances
        """
        all_results = []
        
        # Dictionary of detection methods and their names for logging
        detection_methods = {
            self._detect_object_stream_steganography: "object stream",
            self._detect_metadata_steganography: "metadata",
            self._detect_whitespace_comment_steganography: "whitespace/comment",
            self._detect_document_component_steganography: "document components",
            self._detect_javascript_steganography: "JavaScript",
            self._detect_image_data_steganography: "image data",
            self._detect_xref_table_steganography: "xref table"
        }
        
        if self.enable_threading and self.max_workers > 1:
            # Run detection methods in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_method = {
                    executor.submit(method): name
                    for method, name in detection_methods.items()
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_method):
                    method_name = future_to_method[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        logger.info(f"Completed {method_name} detection with {len(results)} findings")
                    except Exception as e:
                        logger.error(f"Error running {method_name} detection: {e}", exc_info=True)
        else:
            # Run detection methods sequentially
            for method, name in detection_methods.items():
                try:
                    start_time = time.time()
                    results = method()
                    all_results.extend(results)
                    logger.info(f"Completed {name} detection with {len(results)} findings in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error running {name} detection: {e}", exc_info=True)
        
        # Sort results by confidence (highest first)
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Detection complete. Found {len(all_results)} potential steganography instances")
        return all_results
    
    def extract_hidden_data(self, detection: SteganoDetection) -> Optional[bytes]:
        """
        Attempt to extract hidden data from a detected steganography instance
        
        Args:
            detection (SteganoDetection): The detection to extract data from
            
        Returns:
            Optional[bytes]: Extracted data if successful, None otherwise
        """
        # If data was already extracted during detection
        if detection.extracted_data:
            return detection.extracted_data
            
        try:
            if detection.method == SteganoMethod.OBJECT_STREAM:
                # Extract from object stream
                if detection.object_id is not None:
                    obj = self.pdf_reader.get_object(detection.object_id)
                    if isinstance(obj, PyPDF2.generic.StreamObject):
                        stream_data = obj.get_data()
                        
                        # If PNG detected
                        if 'png_details' in detection.details:
                            offset = detection.details.get('stream_offset', 0)
                            # Find PNG signature
                            png_sig = self.png_signature
                            pos = stream_data.find(png_sig, offset)
                            if pos != -1:
                                # Find PNG end (IEND chunk)
                                end_marker = b'IEND'
                                end_pos = stream_data.find(end_marker, pos)
                                if end_pos != -1:
                                    # Include the CRC after IEND (4 bytes)
                                    return stream_data[pos:end_pos + len(end_marker) + 4]
                        
                        return stream_data
                        
            elif detection.method == SteganoMethod.METADATA:
                # Extract from metadata
                field = detection.details.get('metadata_field')
                if field:
                    metadata = self.pdf_reader.metadata
                    if metadata and field in metadata:
                        value = metadata[field]
                        if isinstance(value, str):
                            # Try different decodings
                            if detection.details.get('encoded_type') == 'base64':
                                try:
                                    return base64.b64decode(value)
                                except Exception:
                                    pass
                            elif detection.details.get('encoded_type') == 'hex':
                                try:
                                    return binascii.unhexlify(value)
                                except Exception:
                                    pass
                            return value.encode('utf-8', errors='replace')
                
            elif detection.method == SteganoMethod.IMAGE_DATA:
                # Extract from image data
                if 'png_details' in detection.details:
                    return detection.sample  # We already have the PNG data
                elif 'technique' in detection.details and detection.details['technique'] == 'data_after_eof':
                    return detection.extracted_data  # Data after EOF already extracted
                
                # Try to get the object data
                if detection.object_id is not None:
                    obj = self.pdf_reader.get_object(detection.object_id)
                    if isinstance(obj, PyPDF2.generic.StreamObject):
                        return obj.get_data()
                
            elif detection.method == SteganoMethod.JAVASCRIPT:
                # Extract from JavaScript
                if detection.object_id is not None:
                    obj = self.pdf_reader.get_object(detection.object_id)
                    
                    # Handle different JavaScript representations
                    if isinstance(obj, dict):
                        if '/JS' in obj:
                            js_obj = obj['/JS']
                            if isinstance(js_obj, PyPDF2.generic.TextStringObject):
                                return js_obj.encode('utf-8', errors='replace')
                            elif isinstance(js_obj, PyPDF2.generic.IndirectObject):
                                js_stream = self.pdf_reader.get_object(js_obj.idnum)
                                if isinstance(js_stream, PyPDF2.generic.StreamObject):
                                    return js_stream.get_data()
                        elif '/JavaScript' in obj:
                            js_obj = obj['/JavaScript']
                            if isinstance(js_obj, PyPDF2.generic.TextStringObject):
                                return js_obj.encode('utf-8', errors='replace')
                            elif isinstance(js_obj, PyPDF2.generic.IndirectObject):
                                js_stream = self.pdf_reader.get_object(js_obj.idnum)
                                if isinstance(js_stream, PyPDF2.generic.StreamObject):
                                    return js_stream.get_data()
            
            # For other methods or if specific extraction failed
            return detection.sample
            
        except Exception as e:
            logger.error(f"Error extracting hidden data: {e}", exc_info=True)
            return None
    
    def generate_report(self, detections: List[SteganoDetection]) -> Dict[str, Any]:
        """
        Generate a detailed report from detected steganography instances
        
        Args:
            detections (List[SteganoDetection]): List of detections
            
        Returns:
            Dict[str, Any]: Structured report data
        """
        report = {
            'filename': self.pdf_path.name,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pdf_info': {
                'pages': len(self.pdf_reader.pages),
                'filesize': os.path.getsize(self.pdf_path),
                'version': self.pdf_reader.pdf_header if hasattr(self.pdf_reader, 'pdf_header') else 'Unknown'
            },
            'detection_summary': {
                'total_detections': len(detections),
                'detections_by_method': {},
                'detections_by_confidence': {
                    'high': 0,    # 0.8-1.0
                    'medium': 0,  # 0.5-0.79
                    'low': 0      # 0.0-0.49
                }
            },
            'detections': []
        }
        
        # Count detections by method
        methods_count = {}
        for detection in detections:
            method_name = detection.method.name
            if method_name not in methods_count:
                methods_count[method_name] = 0
            methods_count[method_name] += 1
            
            # Count by confidence level
            if detection.confidence >= 0.8:
                report['detection_summary']['detections_by_confidence']['high'] += 1
            elif detection.confidence >= 0.5:
                report['detection_summary']['detections_by_confidence']['medium'] += 1
            else:
                report['detection_summary']['detections_by_confidence']['low'] += 1
                
        report['detection_summary']['detections_by_method'] = methods_count
        
        # Add detailed information for each detection
        for detection in detections:
            # Create a clean copy of detection details
            details = dict(detection.details) if detection.details else {}
            
            # Remove potentially large binary data
            if 'sample' in details:
                del details['sample']
                
            # Format detection for report
            detection_report = {
                'method': detection.method.name,
                'location': detection.location,
                'confidence': detection.confidence,
                'size_bytes': detection.size_bytes,
                'page_number': detection.page_number,
                'object_id': detection.object_id,
                'details': details,
                # Include a small sample if available, encoded as hex
                'sample_hex': binascii.hexlify(detection.sample).decode('ascii') if detection.sample else None
            }
            
            report['detections'].append(detection_report)
            
        return report

    def save_report(self, report: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save detection report to a JSON file
        
        Args:
            report (Dict[str, Any]): Report dictionary
            output_path (Optional[str]): Path to save the report, if None, uses same name as PDF
            
        Returns:
            str: Path to the saved report file
        """
        if output_path is None:
            output_path = self.pdf_path.with_suffix('.stegano-report.json')
        else:
            output_path = Path(output_path)
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving report: {e}", exc_info=True)
            # Fallback to saving in current directory
            fallback_path = Path(f"{self.pdf_path.stem}_stegano_report.json")
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to fallback location: {fallback_path}")
                return str(fallback_path)
            except Exception as e2:
                logger.error(f"Error saving report to fallback location: {e2}", exc_info=True)
                return ""
    
    def save_extracted_data(self, detection: SteganoDetection, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Save extracted hidden data to a file
        
        Args:
            detection (SteganoDetection): Detection containing the data to save
            output_dir (Optional[str]): Directory to save the data, if None uses current directory
            
        Returns:
            Optional[str]: Path to the saved file or None if failed
        """
        # Extract the data
        data = self.extract_hidden_data(detection)
        if not data:
            logger.warning("No data to save")
            return None
            
        # Create output directory if needed
        if output_dir is None:
            output_dir = self.pdf_path.parent / 'extracted_data'
        else:
            output_dir = Path(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine file extension based on data content
        extension = '.bin'  # Default
        for sig_name, signature in self.signatures.items():
            if data.startswith(signature):
                extension = f'.{sig_name}'
                break
                
        # If PNG, validate it's a complete PNG
        if extension == '.png' and not b'IEND' in data:
            extension = '.incomplete_png'
            
        # Create filename based on detection information
        method_str = detection.method.name.lower()
        loc_str = detection.location.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        
        if detection.page_number:
            filename = f"{self.pdf_path.stem}_{method_str}_{loc_str}_p{detection.page_number}{extension}"
        else:
            filename = f"{self.pdf_path.stem}_{method_str}_{loc_str}{extension}"
            
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Save the file
        try:
            output_path = output_dir / filename
            with open(output_path, 'wb') as f:
                f.write(data)
            logger.info(f"Extracted data saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving extracted data: {e}", exc_info=True)
            return None


def main():
    """
    Main function for running the PDF steganography detector
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Steganography Detector")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--output", "-o", help="Output directory for extracted files and reports")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--deep-scan", "-d", action="store_true", help="Perform deeper analysis (slower)")
    parser.add_argument("--entropy", "-e", type=float, default=7.5, help="Entropy threshold (0-8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--extract-all", "-x", action="store_true", help="Extract all detected hidden data")
    
    args = parser.parse_args()
    
    try:
        # Create detector instance
        detector = PDFSteganographyDetector(
            args.pdf_path,
            enable_threading=args.threads > 1,
            max_workers=args.threads,
            deep_scan=args.deep_scan,
            entropy_threshold=args.entropy,
            verbose=args.verbose
        )
        
        print(f"Analyzing PDF: {args.pdf_path}")
        print(f"Using settings: threads={args.threads}, deep_scan={args.deep_scan}, entropy={args.entropy}")
        
        # Run detection
        start_time = time.time()
        detections = detector.detect()
        elapsed = time.time() - start_time
        
        # Display results
        print(f"\nCompleted analysis in {elapsed:.2f} seconds")
        print(f"Found {len(detections)} potential steganography instances:")
        
        # Create output directory if needed
        output_dir = args.output
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            extract_dir = os.path.join(output_dir, "extracted")
        else:
            extract_dir = "extracted"
            
        # Generate and save report
        report = detector.generate_report(detections)
        report_path = detector.save_report(report, 
                                          output_path=os.path.join(output_dir, f"{os.path.basename(args.pdf_path)}.stegano-report.json") if output_dir else None)
        
        # Display detections by confidence level
        high_conf = [d for d in detections if d.confidence >= 0.8]
        med_conf = [d for d in detections if 0.5 <= d.confidence < 0.8]
        low_conf = [d for d in detections if d.confidence < 0.5]
        
        print(f"  High confidence: {len(high_conf)}")
        print(f"  Medium confidence: {len(med_conf)}")
        print(f"  Low confidence: {len(low_conf)}")
        
        # Display top detections
        if detections:
            print("\nTop 5 detections:")
            for i, detection in enumerate(detections[:5]):
                print(f"{i+1}. [{detection.method.name}] {detection.location} (Confidence: {detection.confidence:.2f})")
                if detection.details:
                    print(f"   Details: {detection.details}")
            
            # Extract data if requested
            if args.extract_all:
                print("\nExtracting hidden data...")
                os.makedirs(extract_dir, exist_ok=True)
                
                extracted_count = 0
                for detection in detections:
                    if detection.confidence >= 0.6:  # Only extract medium-high confidence
                        path = detector.save_extracted_data(detection, output_dir=extract_dir)
                        if path:
                            extracted_count += 1
                
                print(f"Extracted {extracted_count} files to {extract_dir}")
                
            print(f"\nDetailed report saved to: {report_path}")
                
        else:
            print("No steganography detected in the PDF.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())