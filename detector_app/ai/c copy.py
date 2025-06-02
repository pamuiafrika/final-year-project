import base64
import io
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any
from collections import Counter
from dataclasses import dataclass
import numpy as np
import fitz  # PyMuPDF
import argparse

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    print("ML libraries not found. Install with: pip install scikit-learn")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class SuspiciousIndicator:
    """Data class for storing suspicious findings in PDF analysis."""
    category: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    confidence: float
    technical_details: Dict[str, Any]
    location: Optional[str] = None

class PDFSteganoDetector:
    """Advanced PDF steganography detection system using ML and forensic analysis."""
    
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

    def __init__(self, ml_model_path: Optional[str] = None):
        """Initialize the PDF steganography detector."""
        self.indicators: List[SuspiciousIndicator] = []
        self.features: Dict[str, Any] = {}
        self.ml_model = None
        self.scaler = None
        self._initialize_ml_components(ml_model_path)

    def _initialize_ml_components(self, model_path: Optional[str]) -> None:
        """Initialize or load machine learning models for anomaly detection."""
        if model_path and os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
                logger.info("Loaded pre-trained ML model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load ML model: %s", e)
        
        if self.ml_model is None:
            self.ml_model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
            self.scaler = StandardScaler()
            logger.info("Initialized new IsolationForest model")

    def analyze_pdf(self, pdf_path: str, focus_technique: str = 'auto') -> Dict[str, Any]:
        """Analyze a PDF file for steganography using specified or all techniques."""
        self.indicators.clear()
        self.features.clear()

        try:
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()

            if pdf_content.startswith(b'JVBERi0'):
                try:
                    pdf_content = base64.b64decode(pdf_content, validate=True)
                except ValueError:
                    logger.debug("PDF content is not base64 encoded, proceeding with raw content")

            analyses = [
                ('object_stream', self._analyze_object_streams),
                ('metadata', self._analyze_metadata),
                ('font_glyph', self._analyze_fonts_glyphs),
                ('entropy', self._analyze_entropy_patterns),
                ('embedded', self._scan_embedded_files),
                ('layers', self._detect_invisible_layers),
            ]

            for technique, analysis_func in analyses:
                if focus_technique == 'auto' or focus_technique == technique:
                    analysis_func(pdf_content, pdf_path)

            self._detect_concealed_pngs(pdf_content)
            self._ml_anomaly_detection()

            return self._generate_report()

        except Exception as e:
            logger.error("PDF analysis failed: %s", e)
            return {"error": str(e), "indicators": [], "ml_score": None}

    def _analyze_object_streams(self, pdf_content: bytes, pdf_path: str) -> None:
        """Analyze PDF object streams for anomalies."""
        try:
            doc = fitz.open("pdf", pdf_content)
            xref_count = doc.xref_length()
            large_streams = []
            unused_objects = []

            for xref in range(xref_count):
                try:
                    obj_length = doc.xref_get_key(xref, "Length")
                    if obj_length and isinstance(obj_length, (int, str)):
                        length = int(obj_length)
                        if length > 100_000:  # 100KB threshold
                            large_streams.append({
                                "object_id": xref,
                                "length": length,
                                "type": doc.xref_get_key(xref, "Type")
                            })

                    if not doc.xref_is_stream(xref) and (obj_content := doc.xref_stream(xref)):
                        if len(obj_content) > 1000:
                            unused_objects.append(xref)
                except Exception:
                    continue

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

            self.features.update({
                "total_objects": xref_count,
                "large_objects_count": len(large_streams),
                "unused_objects_count": len(unused_objects),
                "avg_object_size": np.mean([obj["length"] for obj in large_streams]) if large_streams else 0
            })

            doc.close()

        except Exception as e:
            logger.error("Object stream analysis failed: %s", e)

    def _analyze_metadata(self, pdf_content: bytes, pdf_path: str) -> None:
        """Analyze PDF metadata for anomalies.
        
        Args:
            pdf_content: Raw PDF bytes content
            pdf_path: Path to the PDF file
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            suspicious_metadata = []
            standard_fields = {'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate'}

            for key, value in metadata.items():
                if key.lower() not in standard_fields:
                    suspicious_metadata.append({key: value})
                if isinstance(value, str) and any(ord(c) > 127 for c in value):
                    suspicious_metadata.append({f"binary_in_{key}": value})

            if suspicious_metadata:
                self.indicators.append(SuspiciousIndicator(
                    category="metadata",
                    severity="MEDIUM",
                    description="Suspicious metadata fields detected",
                    confidence=0.6,
                    technical_details={"suspicious_fields": suspicious_metadata}
                ))

            self.features.update({
                "metadata_fields_count": len(metadata),
                "suspicious_metadata_count": len(suspicious_metadata),
                "has_creation_date": 'creationDate' in metadata,
                "has_modification_date": 'modDate' in metadata
            })

            doc.close()

        except Exception as e:
            logger.error("Metadata analysis failed: %s", e)

    def _analyze_fonts_glyphs(self, pdf_path: str) -> None:
        """Analyze fonts and glyphs for steganographic use."""
        try:
            doc = fitz.open(pdf_path)
            font_anomalies = []
            total_fonts = 0
            embedded_fonts = 0

            for page_num in range(doc.page_count):
                for font in doc[page_num].get_fonts():
                    total_fonts += 1
                    font_ref, font_ext, font_type, font_basename, font_name, font_encoding = font

                    if font_ext:
                        embedded_fonts += 1
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

            self.features.update({
                "total_fonts": total_fonts,
                "embedded_fonts": embedded_fonts,
                "font_anomalies_count": len(font_anomalies),
                "embedded_font_ratio": embedded_fonts / max(total_fonts, 1)
            })

            doc.close()

        except Exception as e:
            logger.error("Font analysis failed: %s", e)

    def _analyze_entropy_patterns(self, pdf_content: bytes) -> None:
        """Analyze entropy patterns to detect hidden data."""
        try:
            chunk_size = 1024
            entropies = [
                self._calculate_entropy(pdf_content[i:i + chunk_size])
                for i in range(0, len(pdf_content), chunk_size)
                if len(pdf_content[i:i + chunk_size]) > 0
            ]

            if entropies:
                avg_entropy = np.mean(entropies)
                max_entropy = np.max(entropies)
                entropy_variance = np.var(entropies)

                if max_entropy > 7.5:
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

                self.features.update({
                    "avg_entropy": avg_entropy,
                    "max_entropy": max_entropy,
                    "entropy_variance": entropy_variance,
                    "high_entropy_chunks": sum(1 for e in entropies if e > 7.0)
                })

        except Exception as e:
            logger.error("Entropy analysis failed: %s", e)

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte data."""
        if not data:
            return 0.0
        byte_counts = Counter(data)
        length = len(data)
        return -sum(
            (count / length) * np.log2(count / length)
            for count in byte_counts.values() if count > 0
        )

    def _scan_embedded_files(self, pdf_path: str) -> None:
        """Scan for embedded files that might contain hidden PNGs."""
        try:
            doc = fitz.open(pdf_path)
            embedded_files = []

            for i in range(doc.embfile_count()):
                file_info = doc.embfile_info(i)
                file_content = doc.embfile_get(i)
                embedded_files.append({
                    "index": i,
                    "info": file_info,
                    "contains_png": bool(file_content and self.PNG_SIGNATURE in file_content),
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

            self.features.update({
                "embedded_files_count": len(embedded_files),
                "embedded_png_count": len([f for f in embedded_files if f["contains_png"]]),
                "total_embedded_size": sum(f["size"] for f in embedded_files)
            })

            doc.close()

        except Exception as e:
            logger.error("Embedded file scan failed: %s", e)

    def _detect_invisible_layers(self, pdf_path: str) -> None:
        """Detect invisible layers or optional content groups."""
        try:
            doc = fitz.open(pdf_path)
            invisible_elements = []

            for page_num in range(doc.page_count):
                try:
                    text_dict = doc[page_num].get_text("dict")
                    for block in text_dict.get("blocks", []):
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                if span.get("size", 0) < 0.1 or span.get("color", 0) == 16777215:
                                    invisible_elements.append({
                                        "page": page_num,
                                        "type": "invisible_text",
                                        "details": span
                                    })
                except Exception:
                    continue

            if invisible_elements:
                self.indicators.append(SuspiciousIndicator(
                    category="invisible_layers",
                    severity="MEDIUM",
                    description=f"Found {len(invisible_elements)} potentially invisible elements",
                    confidence=0.6,
                    technical_details={"invisible_elements": invisible_elements}
                ))

            self.features.update({
                "invisible_elements_count": len(invisible_elements),
                "pages_with_invisible": len({elem["page"] for elem in invisible_elements})
            })

            doc.close()

        except Exception as e:
            logger.error("Invisible layer detection failed: %s", e)

    def _detect_concealed_pngs(self, pdf_content: bytes) -> None:
        """Search for concealed PNG signatures and data."""
        try:
            png_matches = []
            offset = 0
            while (pos := pdf_content.find(self.PNG_SIGNATURE, offset)) != -1:
                png_data = self._extract_png_at_offset(pdf_content, pos)
                if png_data:
                    png_matches.append({
                        "offset": pos,
                        "size": len(png_data),
                        "valid_png": self._validate_png(png_data)
                    })
                offset = pos + 1

            png_chunks = []
            for chunk_type in [b'IHDR', b'IDAT', b'IEND']:
                offset = 0
                while (pos := pdf_content.find(chunk_type, offset)) != -1:
                    png_chunks.append({"chunk_type": chunk_type.decode(), "offset": pos})
                    offset = pos + 1

            if png_matches:
                valid_pngs = [p for p in png_matches if p["valid_png"]]
                self.indicators.append(SuspiciousIndicator(
                    category="concealed_png",
                    severity="CRITICAL" if valid_pngs else "HIGH",
                    description=f"Found {len(png_matches)} PNG signatures ({len(valid_pngs)} valid)",
                    confidence=0.95 if valid_pngs else 0.7,
                    technical_details={"png_matches": png_matches, "png_chunks": png_chunks[:10]}
                ))
            elif png_chunks:
                self.indicators.append(SuspiciousIndicator(
                    category="concealed_png",
                    severity="MEDIUM",
                    description=f"Found {len(png_chunks)} PNG chunk signatures",
                    confidence=0.5,
                    technical_details={"png_chunks": png_chunks[:10]}
                ))

            self.features.update({
                "png_signatures_count": len(png_matches),
                "valid_png_count": len([p for p in png_matches if p["valid_png"]]),
                "png_chunks_count": len(png_chunks),
                "total_png_size": sum(p["size"] for p in png_matches)
            })

        except Exception as e:
            logger.error("PNG detection failed: %s", e)

    def _extract_png_at_offset(self, data: bytes, offset: int) -> Optional[bytes]:
        """Extract PNG data starting at given offset."""
        try:
            if offset + 8 >= len(data):
                return None
            end_pattern = b'IEND\xae\x42\x60\x82'
            if (end_pos := data.find(end_pattern, offset)) != -1:
                return data[offset:end_pos + 8]
            return data[offset:min(offset + 10_000, len(data))]
        except Exception:
            return None

    def _validate_png(self, png_data: bytes) -> bool:
        """Validate if data is a proper PNG file."""
        try:
            if len(png_data) < 33 or not png_data.startswith(self.PNG_SIGNATURE):
                return False
            if b'IHDR' not in png_data[8:29] or not png_data.endswith(b'IEND\xae\x42\x60\x82'):
                return False
            return True
        except Exception:
            return False

    def _ml_anomaly_detection(self) -> None:
        """Apply machine learning for anomaly detection."""
        try:
            if not self.features:
                return

            feature_names = [
                'total_objects', 'large_objects_count', 'unused_objects_count', 'avg_object_size',
                'metadata_fields_count', 'suspicious_metadata_count', 'total_fonts', 'embedded_fonts',
                'font_anomalies_count', 'embedded_font_ratio', 'avg_entropy', 'max_entropy',
                'entropy_variance', 'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
                'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
                'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size'
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
                    technical_details={
                        "anomaly_score": anomaly_score,
                        "feature_vector": feature_vector.tolist(),
                        "feature_names": feature_names
                    }
                ))

            self.features.update({"ml_anomaly_score": anomaly_score, "ml_is_anomaly": is_anomaly})

        except Exception as e:
            logger.error("ML anomaly detection failed: %s", e)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        severity_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 7, "CRITICAL": 10}
        risk_score = sum(
            severity_weights.get(indicator.severity, 0) * indicator.confidence
            for indicator in self.indicators
        )

        indicators_by_category = {}
        for indicator in self.indicators:
            indicators_by_category.setdefault(indicator.category, []).append({
                "severity": indicator.severity,
                "description": indicator.description,
                "confidence": indicator.confidence,
                "technical_details": indicator.technical_details,
                "location": indicator.location
            })

        assessment = (
            "HIGH RISK - Strong evidence of steganography" if risk_score > 20 else
            "MEDIUM RISK - Suspicious patterns detected" if risk_score > 10 else
            "LOW RISK - Minor anomalies found" if risk_score > 5 else
            "CLEAN - No significant suspicious indicators"
        )

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
        """Generate actionable recommendations based on findings."""
        recommendations = []

        if any(i.severity == "CRITICAL" for i in self.indicators):
            recommendations.append("URGENT: Manual forensic examination required - potential steganography detected")

        if any("png" in i.category.lower() for i in self.indicators):
            recommendations.extend([
                "Extract and analyze suspected PNG data using specialized tools",
                "Verify PNG data integrity and examine for additional hidden layers"
            ])

        if any(i.category == "metadata" for i in self.indicators):
            recommendations.append("Examine PDF metadata for hidden information or unusual encoding")

        if any(i.category == "object_stream" for i in self.indicators):
            recommendations.append("Analyze PDF object structure and examine large/unused objects")

        if self.features.get("ml_is_anomaly"):
            recommendations.append("Statistical analysis indicates anomalous patterns - deeper investigation recommended")

        return recommendations or ["No immediate action required - continue routine monitoring"]

def main():
    parser = argparse.ArgumentParser(description="PDF Steganography Analysis Tool")
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    args = parser.parse_args()

    detector = PDFSteganoDetector(ml_model_path='/home/d3bugger/Projects/FINAL YEAR PROJECT/src/detector_app/ai/ml/model.pkl')
    pdf_path = args.pdf_path

    try:
        print("Analyzing sample PDF...")
        result = detector.analyze_pdf(pdf_path, focus_technique='auto')

        print("\n" + "=" * 60)
        print("PDF STEGANOGRAPHY ANALYSIS REPORT")
        print("=" * 60)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
            
        if not result.get('assessment'):
            print("WARNING: Analysis completed but no assessment was generated")
            return
            
        print(f"Assessment: {result['assessment']}")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Total Indicators: {result['total_indicators']}")

        if result['indicators_by_category']:
            print("\n" + "=" * 40)
            print("DETECTED INDICATORS:")
            print("=" * 40)
            for category, indicators in result['indicators_by_category'].items():
                print(f"\n[{category.upper()}]")
                for indicator in indicators:
                    print(f"  • {indicator['severity']}: {indicator['description']}")
                    print(f"    Confidence: {indicator['confidence']:.1%}")

        if result['recommendations']:
            print("\n" + "=" * 40)
            print("RECOMMENDATIONS:")
            print("=" * 40)
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")

        if result['ml_analysis']['anomaly_score'] is not None:
            print("\n" + "=" * 40)
            print("MACHINE LEARNING ANALYSIS:")
            print("=" * 40)
            print(f"Anomaly Score: {result['ml_analysis']['anomaly_score']:.3f}")
            print(f"Is Anomaly: {result['ml_analysis']['is_anomaly']}")

    except Exception as e:
        logger.error(f"Analysis failed for {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()