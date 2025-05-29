import base64
import io
import os
import sys
import json
import struct
import zlib
import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy import stats
import pandas as pd
import argparse

# PDF Processing Libraries
try:
    import fitz  # PyMuPDF
    import PyPDF2
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Install with: pip install PyMuPDF PyPDF2 pdfminer.six")

# ML Libraries
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    import joblib
except ImportError:
    print("ML libraries not found. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SuspiciousIndicator:
    """Data class for storing suspicious findings"""
    category: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    confidence: float
    technical_details: Dict[str, Any]
    location: Optional[str] = None

class PDFSteganoDetector:
    """
    Advanced PDF steganography detection system using ML and forensic analysis
    """
    def __init__(self, ml_model_path: Optional[str] = None):
        self.indicators: List[SuspiciousIndicator] = []
        self.features: Dict[str, Any] = {}
        self.ml_model = None
        self.scaler = None

        # PNG signature for binary detection
        self.PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

        # Initialize ML components
        self._initialize_ml_components(ml_model_path)

    def _initialize_ml_components(self, model_path: Optional[str]):
        """Initialize or create ML models for anomaly detection"""
        if model_path and os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
                logger.info("Loaded pre-trained ML model")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")

        # Fallback: Create new isolation forest for anomaly detection
        if self.ml_model is None:
            self.ml_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()

    def analyze_pdf(self, pdf_path: str, focus_technique: str = 'auto') -> Dict[str, Any]:
        """
        Main analysis function that orchestrates all detection techniques
        """
        self.indicators.clear()
        self.features.clear()

        try:
            # Load PDF content
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()

            # Decode base64 if needed (for the provided example)
            if pdf_content.startswith(b'JVBERi0'):
                try:
                    pdf_content = base64.b64decode(pdf_content)
                except:
                    pass

            # Apply detection techniques based on focus
            if focus_technique == 'auto' or focus_technique == 'object_stream':
                self._analyze_object_streams(pdf_content, pdf_path)

            if focus_technique == 'auto' or focus_technique == 'metadata':
                self._analyze_metadata(pdf_path)

            if focus_technique == 'auto' or focus_technique == 'font_glyph':
                self._analyze_fonts_glyphs(pdf_path)

            if focus_technique == 'auto' or focus_technique == 'entropy':
                self._analyze_entropy_patterns(pdf_content)

            if focus_technique == 'auto' or focus_technique == 'embedded':
                self._scan_embedded_files(pdf_path)

            if focus_technique == 'auto' or focus_technique == 'layers':
                self._detect_invisible_layers(pdf_path)

            # Binary PNG detection
            self._detect_concealed_pngs(pdf_content)

            # ML-based anomaly detection
            self._ml_anomaly_detection()

            # Generate final report
            return self._generate_report()

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e), "indicators": [], "ml_score": None}

    def _analyze_object_streams(self, pdf_content: bytes, pdf_path: str):
        """Analyze PDF object streams for anomalies and enhanced PNG/structural features"""
        try:
            doc = fitz.open("pdf", pdf_content)

            xref_count = doc.xref_length()
            suspicious_objects = []
            large_streams = []
            unused_objects = []
            stream_object_count = 0
            suspicious_filter_count = 0
            high_entropy_streams = 0
            partial_png_signature_count = 0
            high_compression_count = 0
            low_compression_count = 0
            rare_reference_count = 0
            entropy_threshold = 7.5
            partial_png_signatures = [b'PNG', b'IHDR', b'IDAT', b'IEND']

            for i in range(xref_count):
                try:
                    obj_type = doc.xref_get_key(i, "Type")
                    obj_length = doc.xref_get_key(i, "Length")
                    # Check for stream objects
                    if doc.xref_is_stream(i):
                        stream_object_count += 1
                        stream = doc.xref_stream(i)
                        # Entropy of stream
                        if stream:
                            entropy = self._calculate_entropy(stream)
                            if entropy > entropy_threshold:
                                high_entropy_streams += 1
                        # Check for partial PNG signatures
                        for sig in partial_png_signatures:
                            if stream and sig in stream:
                                partial_png_signature_count += 1
                                break
                        # Compression ratio (if possible)
                        try:
                            orig_len = len(stream)
                            comp_len = int(obj_length) if obj_length else orig_len
                            if comp_len > 0:
                                ratio = orig_len / comp_len
                                if ratio > 2.0:
                                    high_compression_count += 1
                                elif ratio < 0.5:
                                    low_compression_count += 1
                        except:
                            pass
                        # Suspicious filters (e.g., unusual or multiple filters)
                        filters = doc.xref_get_key(i, "Filter")
                        if filters and (isinstance(filters, list) and len(filters) > 1 or (isinstance(filters, str) and filters not in ["FlateDecode", "DCTDecode", "JPXDecode"])):
                            suspicious_filter_count += 1
                    # Check for objects with no references (rare/unusual)
                    if not doc.xref_is_stream(i):
                        obj_content = doc.xref_stream(i)
                        if obj_content and len(obj_content) > 1000:
                            unused_objects.append(i)
                        # Rare reference: not referenced by any other object
                        if doc.xref_get_key(i, "Parent") is None and doc.xref_get_key(i, "Prev") is None:
                            rare_reference_count += 1
                    # Large objects
                    if obj_length and isinstance(obj_length, (int, str)):
                        try:
                            length = int(obj_length)
                            if length > 100000:
                                large_streams.append({
                                    "object_id": i,
                                    "length": length,
                                    "type": obj_type
                                })
                        except:
                            pass
                except Exception as e:
                    continue

            # Record findings (existing logic)
            if large_streams:
                self.indicators.append(SuspiciousIndicator(
                    category="object_stream",
                    severity="MEDIUM",
                    description=f"Found {len(large_streams)} unusually large objects",
                    confidence=0.7,
                    technical_details={"large_objects": large_streams}
                ))
            if unused_objects:
                self.indicators.append(SuspiciousIndicator(
                    category="object_stream",
                    severity="HIGH",
                    description=f"Found {len(unused_objects)} potentially unused objects",
                    confidence=0.8,
                    technical_details={"unused_objects": unused_objects}
                ))

            # Store enhanced features for ML
            self.features.update({
                "total_objects": xref_count,
                "large_objects_count": len(large_streams),
                "unused_objects_count": len(unused_objects),
                "avg_object_size": np.mean([obj["length"] for obj in large_streams]) if large_streams else 0,
                "stream_object_count": stream_object_count,
                "suspicious_filter_count": suspicious_filter_count,
                "high_entropy_streams": high_entropy_streams,
                "partial_png_signature_count": partial_png_signature_count,
                "high_compression_count": high_compression_count,
                "low_compression_count": low_compression_count,
                "rare_reference_count": rare_reference_count
            })

            doc.close()
        except Exception as e:
            logger.error(f"Object stream analysis failed: {e}")

    def _analyze_metadata(self, pdf_path: str):
        """Analyze PDF metadata for anomalies and enhanced forensic features"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata

            suspicious_metadata = []
            empty_fields = 0
            long_fields = 0
            nonprintable_fields = 0
            duplicate_value_count = 0
            png_signature_in_metadata = 0
            value_seen = set()
            partial_png_signatures = ["PNG", "IHDR", "IDAT", "IEND"]

            # Check for unusual metadata fields
            standard_fields = {'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate'}

            for key, value in metadata.items():
                if key.lower() not in standard_fields:
                    suspicious_metadata.append({key: value})

                # Check for binary data in metadata
                if isinstance(value, str) and any(ord(c) > 127 for c in value):
                    suspicious_metadata.append({f"binary_in_{key}": value})

                # Count empty fields
                if not value:
                    empty_fields += 1
                # Count long fields
                if isinstance(value, str) and len(value) > 100:
                    long_fields += 1
                # Count non-printable/control characters
                if isinstance(value, str) and any(ord(c) < 32 or ord(c) == 127 for c in value):
                    nonprintable_fields += 1
                # Check for duplicate values
                if value in value_seen:
                    duplicate_value_count += 1
                else:
                    value_seen.add(value)
                # Check for partial PNG signatures
                if isinstance(value, str) and any(sig in value for sig in partial_png_signatures):
                    png_signature_in_metadata += 1

            if suspicious_metadata:
                self.indicators.append(SuspiciousIndicator(
                    category="metadata",
                    severity="MEDIUM",
                    description="Suspicious metadata fields detected",
                    confidence=0.6,
                    technical_details={"suspicious_fields": suspicious_metadata}
                ))

            # Store enhanced features
            self.features.update({
                "metadata_fields_count": len(metadata),
                "suspicious_metadata_count": len(suspicious_metadata),
                "has_creation_date": 'creationDate' in metadata,
                "has_modification_date": 'modDate' in metadata,
                "empty_metadata_fields": empty_fields,
                "long_metadata_fields": long_fields,
                "nonprintable_metadata_fields": nonprintable_fields,
                "duplicate_metadata_values": duplicate_value_count,
                "png_signature_in_metadata": png_signature_in_metadata
            })

            doc.close()

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")

    def _analyze_fonts_glyphs(self, pdf_path: str):
        """Analyze fonts and glyphs for steganographic use"""
        try:
            doc = fitz.open(pdf_path)

            font_anomalies = []
            total_fonts = 0
            embedded_fonts = 0

            for page_num in range(doc.page_count):
                page = doc[page_num]
                font_list = page.get_fonts()

                for font in font_list:
                    total_fonts += 1
                    font_ref, font_ext, font_type, font_basename, font_name, font_encoding = font

                    # Check for embedded fonts (potential hiding place)
                    if font_ext:
                        embedded_fonts += 1

                    # Check for unusual font names or encodings
                    if font_name and (len(font_name) > 50 or any(ord(c) > 127 for c in font_name)):
                        font_anomalies.append({
                            "page": page_num,
                            "font_name": font_name,
                            "encoding": font_encoding,
                            "type": font_type
                        })

            if font_anomalies:
                self.indicators.append(SuspiciousIndicator(
                    category="font_glyph",
                    severity="MEDIUM",
                    description=f"Found {len(font_anomalies)} suspicious fonts",
                    confidence=0.5,
                    technical_details={"font_anomalies": font_anomalies}
                ))

            # Store features
            self.features.update({
                "total_fonts": total_fonts,
                "embedded_fonts": embedded_fonts,
                "font_anomalies_count": len(font_anomalies),
                "embedded_font_ratio": embedded_fonts / max(total_fonts, 1)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Font analysis failed: {e}")

    def _analyze_entropy_patterns(self, pdf_content: bytes):
        """Analyze entropy patterns to detect hidden data"""
        try:
            # Calculate entropy for different sections
            chunk_size = 1024
            entropies = []

            for i in range(0, len(pdf_content), chunk_size):
                chunk = pdf_content[i:i+chunk_size]
                if len(chunk) > 0:
                    entropy = self._calculate_entropy(chunk)
                    entropies.append(entropy)

            if entropies:
                avg_entropy = np.mean(entropies)
                max_entropy = np.max(entropies)
                entropy_variance = np.var(entropies)

                # High entropy might indicate compressed/encrypted data
                if max_entropy > 7.5:  # Very high entropy threshold
                    self.indicators.append(SuspiciousIndicator(
                        category="entropy",
                        severity="MEDIUM",
                        description=f"High entropy section detected (max: {max_entropy:.2f})",
                        confidence=0.6,
                        technical_details={
                            "max_entropy": max_entropy,
                            "avg_entropy": avg_entropy,
                            "entropy_variance": entropy_variance
                        }
                    ))

                # Store features
                self.features.update({
                    "avg_entropy": avg_entropy,
                    "max_entropy": max_entropy,
                    "entropy_variance": entropy_variance,
                    "high_entropy_chunks": sum(1 for e in entropies if e > 7.0)
                })

        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0

        # Count byte frequencies
        byte_counts = Counter(data)
        length = len(data)

        # Calculate entropy
        entropy = 0
        for count in byte_counts.values():
            if count > 0:
                probability = count / length
                entropy -= probability * np.log2(probability)

        return entropy

    def _scan_embedded_files(self, pdf_path: str):
        """Scan for embedded files that might contain hidden PNGs"""
        try:
            doc = fitz.open(pdf_path)

            embedded_files = []

            # Check for file attachments
            attachment_count = doc.embfile_count()

            for i in range(attachment_count):
                file_info = doc.embfile_info(i)
                file_content = doc.embfile_get(i)

                # Check if embedded file contains PNG data
                if file_content and self.PNG_SIGNATURE in file_content:
                    embedded_files.append({
                        "index": i,
                        "info": file_info,
                        "contains_png": True,
                        "size": len(file_content)
                    })
                else:
                    embedded_files.append({
                        "index": i,
                        "info": file_info,
                        "contains_png": False,
                        "size": len(file_content) if file_content else 0
                    })

            if embedded_files:
                png_files = [f for f in embedded_files if f["contains_png"]]

                if png_files:
                    self.indicators.append(SuspiciousIndicator(
                        category="embedded_files",
                        severity="HIGH",
                        description=f"Found {len(png_files)} embedded files containing PNG data",
                        confidence=0.9,
                        technical_details={"png_files": png_files}
                    ))

                self.indicators.append(SuspiciousIndicator(
                    category="embedded_files",
                    severity="LOW",
                    description=f"Found {len(embedded_files)} embedded files",
                    confidence=0.4,
                    technical_details={"all_files": embedded_files}
                ))

            # Store features
            self.features.update({
                "embedded_files_count": len(embedded_files),
                "embedded_png_count": len([f for f in embedded_files if f["contains_png"]]),
                "total_embedded_size": sum(f["size"] for f in embedded_files)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Embedded file scan failed: {e}")

    def _detect_invisible_layers(self, pdf_path: str):
        """Detect invisible layers or optional content groups"""
        try:
            doc = fitz.open(pdf_path)

            invisible_elements = []

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # Get optional content groups
                try:
                    # This is a simplified check - more sophisticated OCG analysis needed
                    text_dict = page.get_text("dict")

                    # Look for invisible text or objects
                    for block in text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    # Check for white text on white background or zero-sized text
                                    if (span.get("size", 0) < 0.1 or
                                        span.get("color", 0) == 16777215):  # White color
                                        invisible_elements.append({
                                            "page": page_num,
                                            "type": "invisible_text",
                                            "details": span
                                        })

                except Exception as e:
                    continue

            if invisible_elements:
                self.indicators.append(SuspiciousIndicator(
                    category="invisible_layers",
                    severity="MEDIUM",
                    description=f"Found {len(invisible_elements)} potentially invisible elements",
                    confidence=0.6,
                    technical_details={"invisible_elements": invisible_elements}
                ))

            # Store features
            self.features.update({
                "invisible_elements_count": len(invisible_elements),
                "pages_with_invisible": len(set(elem["page"] for elem in invisible_elements))
            })

            doc.close()

        except Exception as e:
            logger.error(f"Invisible layer detection failed: {e}")

    def _detect_concealed_pngs(self, pdf_content: bytes):
        """Direct binary search for PNG signatures and data"""
        try:
            png_matches = []

            # Search for PNG signatures
            offset = 0
            while True:
                pos = pdf_content.find(self.PNG_SIGNATURE, offset)
                if pos == -1:
                    break

                # Analyze the PNG data
                png_data = self._extract_png_at_offset(pdf_content, pos)
                if png_data:
                    png_matches.append({
                        "offset": pos,
                        "size": len(png_data),
                        "valid_png": self._validate_png(png_data)
                    })

                offset = pos + 1

            # Search for PNG data patterns (IHDR, IDAT chunks)
            png_chunks = []
            for chunk_type in [b'IHDR', b'IDAT', b'IEND']:
                offset = 0
                while True:
                    pos = pdf_content.find(chunk_type, offset)
                    if pos == -1:
                        break
                    png_chunks.append({
                        "chunk_type": chunk_type.decode(),
                        "offset": pos
                    })
                    offset = pos + 1

            if png_matches:
                valid_pngs = [p for p in png_matches if p["valid_png"]]

                self.indicators.append(SuspiciousIndicator(
                    category="concealed_png",
                    severity="CRITICAL" if valid_pngs else "HIGH",
                    description=f"Found {len(png_matches)} PNG signatures ({len(valid_pngs)} valid)",
                    confidence=0.95 if valid_pngs else 0.7,
                    technical_details={
                        "png_matches": png_matches,
                        "png_chunks": png_chunks[:10]  # Limit output
                    }
                ))

            elif png_chunks:
                self.indicators.append(SuspiciousIndicator(
                    category="concealed_png",
                    severity="MEDIUM",
                    description=f"Found {len(png_chunks)} PNG chunk signatures",
                    confidence=0.5,
                    technical_details={"png_chunks": png_chunks[:10]}
                ))

            # Store features
            self.features.update({
                "png_signatures_count": len(png_matches),
                "valid_png_count": len([p for p in png_matches if p["valid_png"]]),
                "png_chunks_count": len(png_chunks),
                "total_png_size": sum(p["size"] for p in png_matches)
            })

        except Exception as e:
            logger.error(f"PNG detection failed: {e}")

    def _extract_png_at_offset(self, data: bytes, offset: int) -> Optional[bytes]:
        """Extract PNG data starting at given offset"""
        try:
            if offset + 8 >= len(data):
                return None

            # PNG files end with IEND chunk followed by CRC
            end_pattern = b'IEND\xae\x42\x60\x82'
            end_pos = data.find(end_pattern, offset)

            if end_pos != -1:
                return data[offset:end_pos + 8]

            # Fallback: extract reasonable amount of data
            return data[offset:min(offset + 10000, len(data))]

        except Exception:
            return None

    def _validate_png(self, png_data: bytes) -> bool:
        """Validate if data is a proper PNG file"""
        try:
            if len(png_data) < 33:  # Minimum PNG size
                return False

            if not png_data.startswith(self.PNG_SIGNATURE):
                return False

            # Check for IHDR chunk
            if b'IHDR' not in png_data[8:29]:
                return False

            # Check for IEND chunk
            if not png_data.endswith(b'IEND\xae\x42\x60\x82'):
                return False

            return True

        except Exception:
            return False

    def _ml_anomaly_detection(self):
        """Apply machine learning for anomaly detection"""
        try:
            if not self.features:
                return

            # Prepare feature vector
            feature_names = [
                'total_objects', 'large_objects_count', 'unused_objects_count',
                'avg_object_size', 'metadata_fields_count', 'suspicious_metadata_count',
                'total_fonts', 'embedded_fonts', 'font_anomalies_count',
                'embedded_font_ratio', 'avg_entropy', 'max_entropy', 'entropy_variance',
                'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
                'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
                'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size',
                'stream_object_count', 'suspicious_filter_count', 'high_entropy_streams',
                'partial_png_signature_count', 'high_compression_count', 'low_compression_count',
                'rare_reference_count', 'empty_metadata_fields', 'long_metadata_fields',
                'nonprintable_metadata_fields', 'duplicate_metadata_values', 'png_signature_in_metadata'
            ]

            # Create feature vector with defaults for missing features
            feature_vector = []
            for name in feature_names:
                feature_vector.append(self.features.get(name, 0))

            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Handle infinite or NaN values
            feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=0, neginf=0)

            # Apply scaling if we have a fitted scaler
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform(feature_vector)
            else:
                # Fit scaler on single sample (not ideal, but functional)
                feature_vector = self.scaler.fit_transform(feature_vector)

            # Get anomaly score
            if hasattr(self.ml_model, 'decision_function'):
                anomaly_score = self.ml_model.decision_function(feature_vector)[0]
                is_anomaly = self.ml_model.predict(feature_vector)[0] == -1
            else:
                # Fit model on single sample for demonstration
                self.ml_model.fit(feature_vector)
                anomaly_score = -0.5  # Neutral score
                is_anomaly = False

            if is_anomaly or anomaly_score < -0.5:
                self.indicators.append(SuspiciousIndicator(
                    category="ml_anomaly",
                    severity="HIGH",
                    description=f"ML model detected anomalous patterns (score: {anomaly_score:.3f})",
                    confidence=min(abs(anomaly_score), 1.0),
                    technical_details={
                        "anomaly_score": anomaly_score,
                        "feature_vector": feature_vector.tolist(),
                        "feature_names": feature_names
                    }
                ))

            # Store ML results
            self.features.update({
                "ml_anomaly_score": anomaly_score,
                "ml_is_anomaly": is_anomaly
            })

        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""

        # Calculate overall risk score
        severity_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 7, "CRITICAL": 10}
        risk_score = sum(
            severity_weights.get(indicator.severity, 0) * indicator.confidence
            for indicator in self.indicators
        )

        # Categorize indicators
        indicators_by_category = {}
        for indicator in self.indicators:
            if indicator.category not in indicators_by_category:
                indicators_by_category[indicator.category] = []
            indicators_by_category[indicator.category].append({
                "severity": indicator.severity,
                "description": indicator.description,
                "confidence": indicator.confidence,
                "technical_details": indicator.technical_details,
                "location": indicator.location
            })

        # Determine overall assessment
        if risk_score > 20:
            assessment = "HIGH RISK - Strong evidence of steganography"
        elif risk_score > 10:
            assessment = "MEDIUM RISK - Suspicious patterns detected"
        elif risk_score > 5:
            assessment = "LOW RISK - Minor anomalies found"
        else:
            assessment = "CLEAN - No significant suspicious indicators"

        return {
            "assessment": assessment,
            "risk_score": risk_score,
            "total_indicators": len(self.indicators),
            "indicators_by_category": indicators_by_category,
            "ml_analysis": {
                "anomaly_score": self.features.get("ml_anomaly_score"),
                "is_anomaly": self.features.get("ml_is_anomaly")
            },
            "features_extracted": self.features,
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on findings"""
        recommendations = []

        # Check for critical indicators
        critical_indicators = [i for i in self.indicators if i.severity == "CRITICAL"]
        if critical_indicators:
            recommendations.append("URGENT: Manual forensic examination required - potential steganography detected")

        # Check for PNG-related findings
        png_indicators = [i for i in self.indicators if "png" in i.category.lower()]
        if png_indicators:
            recommendations.append("Extract and analyze suspected PNG data using specialized tools")
            recommendations.append("Verify PNG data integrity and examine for additional hidden layers")

        # Check for metadata issues
        metadata_indicators = [i for i in self.indicators if i.category == "metadata"]
        if metadata_indicators:
            recommendations.append("Examine PDF metadata for hidden information or unusual encoding")

        # Check for object anomalies
        object_indicators = [i for i in self.indicators if i.category == "object_stream"]
        if object_indicators:
            recommendations.append("Analyze PDF object structure and examine large/unused objects")

        # ML recommendations
        if self.features.get("ml_is_anomaly"):
            recommendations.append("Statistical analysis indicates anomalous patterns - deeper investigation recommended")

        if not recommendations:
            recommendations.append("No immediate action required - continue routine monitoring")

        return recommendations

def main():
    parser = argparse.ArgumentParser(description="PDF Steganography Analysis Tool")
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    args = parser.parse_args()

    detector = PDFSteganoDetector(ml_model_path='/home/d3bugger/Projects/FINAL YEAR PROJECT/src/detector_app/ai/ml/model.pkl')
    pdf_path = args.pdf_path

    try:
        print("Analyzing PDF...")
        result = detector.analyze_pdf(pdf_path, focus_technique='auto')

        # Print results
        print(f"\n{'='*60}")
        print(f"PDF STEGANOGRAPHY ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"Assessment: {result['assessment']}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Total Indicators: {result['total_indicators']}")

        if result['indicators_by_category']:
            print(f"\n{'='*40}")
            print("DETECTED INDICATORS:")
            print(f"{'='*40}")

            for category, indicators in result['indicators_by_category'].items():
                print(f"\n[{category.upper()}]")
                for indicator in indicators:
                    print(f"  • {indicator['severity']}: {indicator['description']}")
                    print(f"    Confidence: {indicator['confidence']:.1%}")

        if result['recommendations']:
            print(f"\n{'='*40}")
            print("RECOMMENDATIONS:")
            print(f"{'='*40}")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")

        # Print ML analysis if available
        if result['ml_analysis']['anomaly_score'] is not None:
            print(f"\n{'='*40}")
            print("MACHINE LEARNING ANALYSIS:")
            print(f"{'='*40}")
            print(f"Anomaly Score: {result['ml_analysis']['anomaly_score']:.3f}")
            print(f"Is Anomaly: {result['ml_analysis']['is_anomaly']}")

    except Exception as e:
        print(f"Error analyzing PDF: {e}")

if __name__ == "__main__":
    main()

