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
import multiprocessing as mp
import yaml
import time
import datetime

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

# Image Processing Libraries
try:
    from PIL import Image
except ImportError:
    print("Image processing libraries not found. Install with: pip install pillow")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    def __init__(self, config_path: str = 'config.yaml', ml_model_path: Optional[str] = None):
        self.indicators: List[SuspiciousIndicator] = []
        self.features: Dict[str, Any] = {}
        self.ml_model = None
        self.scaler = None
        self.config = self._load_config(config_path)
        self.PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
        self._initialize_ml_components(ml_model_path)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                "entropy_threshold": 8.0,
                "object_size_threshold": 100000,
                "font_name_length_threshold": 50,
                "metadata_field_length_threshold": 100,
                "invisible_text_size_threshold": 0.1,
                "ml_contamination": 0.1,
                "ml_n_estimators": 100
            }

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
                contamination=self.config.get('ml_contamination', 0.1),
                random_state=42,
                n_estimators=self.config.get('ml_n_estimators', 100)
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

            # Decode base64 if needed
            if pdf_content.startswith(b'JVBERi0'):
                try:
                    pdf_content = base64.b64decode(pdf_content)
                except:
                    pass

            # Apply detection techniques based on focus
            analysis_methods = {
                'object_stream': self._analyze_object_streams,
                'metadata': self._analyze_metadata,
                'font_glyph': self._analyze_fonts_glyphs,
                'entropy': self._analyze_entropy_patterns,
                'embedded': self._scan_embedded_files,
                'layers': self._detect_invisible_layers,
                'png': self._detect_concealed_pngs,
                'color_space': self._analyze_color_space,
                'javascript': self._analyze_javascript,
                'annotations': self._analyze_annotations,
                'ml': self._ml_anomaly_detection
            }

            if focus_technique == 'auto':
                for method in analysis_methods.values():
                    method(pdf_content if 'content' in method.__code__.co_varnames else pdf_path)
            elif focus_technique in analysis_methods:
                analysis_methods[focus_technique](pdf_content if 'content' in analysis_methods[focus_technique].__code__.co_varnames else pdf_path)

            # Generate final report
            return self._generate_report()

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e), "indicators": [], "ml_score": None}

    def _analyze_object_streams(self, pdf_content: bytes):
        """Analyze PDF object streams for anomalies"""
        try:
            doc = fitz.open("pdf", pdf_content)
            xref_count = doc.xref_length()
            large_streams = []
            unused_objects = []

            # Use multiprocessing for object analysis
            with mp.Pool() as pool:
                results = pool.map(self._process_object, [(doc, i) for i in range(xref_count)])

            for result in results:
                if result:
                    if result.get('large'):
                        large_streams.append(result)
                    if result.get('unused'):
                        unused_objects.append(result)

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

            # Store features
            self.features.update({
                "total_objects": xref_count,
                "large_objects_count": len(large_streams),
                "unused_objects_count": len(unused_objects)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Object stream analysis failed: {e}")

    def _process_object(self, args):
        """Helper function for multiprocessing object analysis"""
        doc, i = args
        try:
            obj_length = int(doc.xref_get_key(i, "Length") or 0)
            if obj_length > self.config['object_size_threshold']:
                return {"large": True, "object_id": i, "length": obj_length}
            if not doc.xref_is_stream(i) and not doc.xref_get_key(i, "Parent"):
                return {"unused": True, "object_id": i}
            return None
        except:
            return None

    def _analyze_metadata(self, pdf_path: str):
        """Analyze PDF metadata for anomalies"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            suspicious_metadata = []

            for key, value in metadata.items():
                if key.lower() not in {'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationdate', 'moddate'}:
                    suspicious_metadata.append({key: value})

                if isinstance(value, str) and len(value) > self.config['metadata_field_length_threshold']:
                    suspicious_metadata.append({f"long_{key}": value})

            if suspicious_metadata:
                self.indicators.append(SuspiciousIndicator(
                    category="metadata",
                    severity="MEDIUM",
                    description="Suspicious metadata fields detected",
                    confidence=0.6,
                    technical_details={"suspicious_fields": suspicious_metadata}
                ))

            # Store features
            self.features.update({
                "metadata_fields_count": len(metadata),
                "suspicious_metadata_count": len(suspicious_metadata)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")

    def _analyze_fonts_glyphs(self, pdf_path: str):
        """Analyze fonts and glyphs for steganographic use"""
        try:
            doc = fitz.open(pdf_path)
            font_anomalies = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                font_list = page.get_fonts()

                for font in font_list:
                    font_name = font[4]
                    if font_name and len(font_name) > self.config['font_name_length_threshold']:
                        font_anomalies.append({
                            "page": page_num,
                            "font_name": font_name
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
                "total_fonts": sum(len(page.get_fonts()) for page in doc),
                "font_anomalies_count": len(font_anomalies)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Font analysis failed: {e}")

    def _analyze_entropy_patterns(self, pdf_content: bytes):
        """Analyze entropy patterns to detect hidden data"""
        try:
            chunk_size = 1024
            entropies = [self._calculate_entropy(pdf_content[i:i+chunk_size]) for i in range(0, len(pdf_content), chunk_size)]

            if entropies:
                max_entropy = max(entropies)
                if max_entropy > self.config['entropy_threshold']:
                    self.indicators.append(SuspiciousIndicator(
                        category="entropy",
                        severity="MEDIUM",
                        description=f"High entropy detected (max: {max_entropy:.2f})",
                        confidence=0.6,
                        technical_details={"max_entropy": max_entropy}
                    ))

                # Store features
                self.features.update({
                    "avg_entropy": np.mean(entropies),
                    "max_entropy": max_entropy,
                    "entropy_variance": np.var(entropies)
                })

        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0
        byte_counts = Counter(data)
        length = len(data)
        return -sum((count / length) * np.log2(count / length) for count in byte_counts.values() if count > 0)

    def _scan_embedded_files(self, pdf_path: str):
        """Scan for embedded files that might contain hidden data"""
        try:
            doc = fitz.open(pdf_path)
            embedded_files = []

            for i in range(doc.embfile_count()):
                file_content = doc.embfile_get(i)
                if file_content and self.PNG_SIGNATURE in file_content:
                    embedded_files.append({
                        "index": i,
                        "contains_png": True
                    })
                else:
                    embedded_files.append({
                        "index": i,
                        "contains_png": False
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

            # Store features
            self.features.update({
                "embedded_files_count": len(embedded_files),
                "embedded_png_count": len(png_files) if 'png_files' in locals() else 0
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
                text_dict = page.get_text("dict")

                for block in text_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("size", 0) < self.config['invisible_text_size_threshold']:
                                invisible_elements.append({
                                    "page": page_num,
                                    "type": "invisible_text"
                                })

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
                "invisible_elements_count": len(invisible_elements)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Invisible layer detection failed: {e}")

    def _detect_concealed_pngs(self, pdf_content: bytes):
        """Direct binary search for PNG signatures and data"""
        try:
            png_matches = []
            offset = 0
            while True:
                pos = pdf_content.find(self.PNG_SIGNATURE, offset)
                if pos == -1:
                    break
                png_data = self._extract_png_at_offset(pdf_content, pos)
                if png_data:
                    png_matches.append({
                        "offset": pos,
                        "valid_png": self._validate_png(png_data)
                    })
                offset = pos + 1

            if png_matches:
                valid_pngs = [p for p in png_matches if p["valid_png"]]
                self.indicators.append(SuspiciousIndicator(
                    category="concealed_png",
                    severity="CRITICAL" if valid_pngs else "HIGH",
                    description=f"Found {len(png_matches)} PNG signatures ({len(valid_pngs)} valid)",
                    confidence=0.95 if valid_pngs else 0.7,
                    technical_details={"png_matches": png_matches}
                ))

            # Store features
            self.features.update({
                "png_signatures_count": len(png_matches),
                "valid_png_count": len(valid_pngs) if 'valid_pngs' in locals() else 0
            })

        except Exception as e:
            logger.error(f"PNG detection failed: {e}")

    def _extract_png_at_offset(self, data: bytes, offset: int) -> Optional[bytes]:
        """Extract PNG data starting at given offset"""
        try:
            end_pattern = b'IEND\xae\x42\x60\x82'
            end_pos = data.find(end_pattern, offset)
            if end_pos != -1:
                return data[offset:end_pos + 8]
            return None
        except:
            return None

    def _validate_png(self, png_data: bytes) -> bool:
        """Validate if data is a proper PNG file"""
        try:
            if png_data and png_data.startswith(self.PNG_SIGNATURE) and b'IEND\xae\x42\x60\x82' in png_data:
                return True
            return False
        except:
            return False

    def _analyze_color_space(self, pdf_path: str):
        """Analyze color space for potential steganography"""
        try:
            doc = fitz.open(pdf_path)
            color_space_anomalies = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                for img in page.get_images():
                    try:
                        xref = img[0]
                        img_info = doc.extract_image(xref)
                        if img_info:
                            color_space = img_info.get("colorspace", "Unknown")
                            if color_space not in ["DeviceRGB", "DeviceGray", "DeviceCMYK"]:
                                color_space_anomalies.append({
                                    "page": page_num,
                                    "xref": xref,
                                    "color_space": color_space
                                })
                    except:
                        continue

            if color_space_anomalies:
                self.indicators.append(SuspiciousIndicator(
                    category="color_space",
                    severity="MEDIUM",
                    description=f"Found {len(color_space_anomalies)} unusual color spaces",
                    confidence=0.6,
                    technical_details={"color_space_anomalies": color_space_anomalies}
                ))

            # Store features
            self.features.update({
                "color_space_anomalies_count": len(color_space_anomalies)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Color space analysis failed: {e}")

    def _analyze_javascript(self, pdf_path: str):
        """Analyze JavaScript for potential steganography"""
        try:
            doc = fitz.open(pdf_path)
            js_anomalies = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                annots = page.annots()
                for annot in annots:
                    if annot.info.get("js"):
                        js_code = annot.info["js"]
                        if len(js_code) > 100:  # Arbitrary threshold
                            js_anomalies.append({
                                "page": page_num,
                                "js_length": len(js_code)
                            })

            if js_anomalies:
                self.indicators.append(SuspiciousIndicator(
                    category="javascript",
                    severity="HIGH",
                    description=f"Found {len(js_anomalies)} suspicious JavaScript entries",
                    confidence=0.8,
                    technical_details={"js_anomalies": js_anomalies}
                ))

            # Store features
            self.features.update({
                "js_anomalies_count": len(js_anomalies)
            })

            doc.close()

        except Exception as e:
            logger.error(f"JavaScript analysis failed: {e}")

    def _analyze_annotations(self, pdf_path: str):
        """Analyze annotations for potential steganography"""
        try:
            doc = fitz.open(pdf_path)
            annotation_anomalies = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                annots = page.annots()
                for annot in annots:
                    if annot.info.get("contents"):
                        contents = annot.info["contents"]
                        if len(contents) > 100:  # Arbitrary threshold
                            annotation_anomalies.append({
                                "page": page_num,
                                "annotation_type": annot.type[1],
                                "contents_length": len(contents)
                            })

            if annotation_anomalies:
                self.indicators.append(SuspiciousIndicator(
                    category="annotations",
                    severity="MEDIUM",
                    description=f"Found {len(annotation_anomalies)} suspicious annotations",
                    confidence=0.6,
                    technical_details={"annotation_anomalies": annotation_anomalies}
                ))

            # Store features
            self.features.update({
                "annotation_anomalies_count": len(annotation_anomalies)
            })

            doc.close()

        except Exception as e:
            logger.error(f"Annotation analysis failed: {e}")

    def _ml_anomaly_detection(self):
        """Apply machine learning for anomaly detection"""
        try:
            if not self.features:
                return

            # Prepare feature vector
            feature_names = [
                'total_objects', 'large_objects_count', 'unused_objects_count',
                'metadata_fields_count', 'suspicious_metadata_count',
                'total_fonts', 'font_anomalies_count',
                'avg_entropy', 'max_entropy', 'entropy_variance',
                'embedded_files_count', 'embedded_png_count',
                'invisible_elements_count',
                'png_signatures_count', 'valid_png_count',
                'color_space_anomalies_count',
                'js_anomalies_count',
                'annotation_anomalies_count'
            ]

            feature_vector = np.array([self.features.get(name, 0) for name in feature_names]).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=0, neginf=0)

            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform(feature_vector)
            else:
                feature_vector = self.scaler.fit_transform(feature_vector)

            if hasattr(self.ml_model, 'decision_function'):
                anomaly_score = self.ml_model.decision_function(feature_vector)[0]
                is_anomaly = self.ml_model.predict(feature_vector)[0] == -1
            else:
                self.ml_model.fit(feature_vector)
                anomaly_score = -0.5
                is_anomaly = False

            if is_anomaly or anomaly_score < -0.5:
                self.indicators.append(SuspiciousIndicator(
                    category="ml_anomaly",
                    severity="HIGH",
                    description=f"ML model detected anomalous patterns (score: {anomaly_score:.3f})",
                    confidence=min(abs(anomaly_score), 1.0),
                    technical_details={"anomaly_score": anomaly_score}
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
        severity_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 7, "CRITICAL": 10}
        risk_score = sum(severity_weights.get(i.severity, 0) * i.confidence for i in self.indicators)

        indicators_by_category = {}
        for i in self.indicators:
            if i.category not in indicators_by_category:
                indicators_by_category[i.category] = []
            indicators_by_category[i.category].append({
                "severity": i.severity,
                "description": i.description,
                "confidence": i.confidence,
                "technical_details": i.technical_details,
                "location": i.location
            })

        assessment = "CLEAN"
        if risk_score > 20:
            assessment = "HIGH RISK"
        elif risk_score > 10:
            assessment = "MEDIUM RISK"
        elif risk_score > 5:
            assessment = "LOW RISK"

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
        if any(i.severity == "CRITICAL" for i in self.indicators):
            recommendations.append("URGENT: Manual forensic examination required")
        if "concealed_png" in [i.category for i in self.indicators]:
            recommendations.append("Extract and analyze suspected PNG data")
        if "metadata" in [i.category for i in self.indicators]:
            recommendations.append("Examine PDF metadata for hidden information")
        if "object_stream" in [i.category for i in self.indicators]:
            recommendations.append("Analyze PDF object structure")
        if self.features.get("ml_is_anomaly"):
            recommendations.append("Deeper investigation recommended due to anomalous patterns")
        if not recommendations:
            recommendations.append("No immediate action required")
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="PDF Steganography Analysis Tool")
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    args = parser.parse_args()

    detector = PDFSteganoDetector(config_path='config.yaml', ml_model_path='model.pkl')
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