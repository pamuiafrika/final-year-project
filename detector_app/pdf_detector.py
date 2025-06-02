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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import hashlib

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
    
    # RECOMMENDATION 1: Standardized feature names for consistent ML training
    FEATURE_NAMES = [
        'total_objects', 'large_objects_count', 'unused_objects_count', 'avg_object_size',
        'metadata_fields_count', 'suspicious_metadata_count', 'total_fonts', 'embedded_fonts',
        'font_anomalies_count', 'embedded_font_ratio', 'avg_entropy', 'max_entropy',
        'entropy_variance', 'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
        'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
        'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size'
    ]

    def __init__(self, ml_model_path: Optional[str] = None, cache_dir: str = "./cache"):
        """Initialize the PDF steganography detector."""
        self.indicators: List[SuspiciousIndicator] = []
        self.features: Dict[str, Any] = {}
        self.ml_model = None
        self.scaler = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._initialize_ml_components(ml_model_path)

    def _initialize_ml_components(self, model_path: Optional[str]) -> None:
        """Initialize or load machine learning models for anomaly detection."""
        if model_path and os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                scaler_path = model_path.replace('.pkl', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                logger.info("Loaded pre-trained ML model from %s", model_path)
            except Exception as e:
                logger.warning("Failed to load ML model: %s", e)
        
        if self.ml_model is None:
            # RECOMMENDATION: Better IsolationForest parameters
            self.ml_model = IsolationForest(
                contamination=0.05,  # Expect 5% anomalies
                random_state=42, 
                n_estimators=200,    # More trees for stability
                max_samples='auto',
                bootstrap=True
            )
            self.scaler = StandardScaler()
            logger.info("Initialized new IsolationForest model")

    # RECOMMENDATION 2: Training baseline model on clean PDFs
    def train_baseline_model(self, clean_pdf_directory: str, output_model_path: str = "trained_model.pkl") -> Dict[str, Any]:
        """
        Train the ML model on clean PDFs to establish baseline.
        
        Args:
            clean_pdf_directory: Directory containing clean PDF files
            output_model_path: Path to save the trained model
            
        Returns:
            Training statistics and model information
        """
        logger.info("Starting baseline model training on clean PDFs...")
        
        if not os.path.exists(clean_pdf_directory):
            raise ValueError(f"Clean PDF directory not found: {clean_pdf_directory}")
        
        features_list = []
        processed_files = []
        failed_files = []
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(clean_pdf_directory) if f.lower().endswith('.pdf')]
        
        if len(pdf_files) < 10:
            logger.warning(f"Only {len(pdf_files)} PDFs found. Recommend at least 100 clean PDFs for training.")
        
        # Process each PDF
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(clean_pdf_directory, pdf_file)
                logger.info(f"Processing clean PDF: {pdf_file}")
                
                # Reset state for each file
                self.indicators.clear()
                self.features.clear()
                
                # Analyze the PDF
                self.analyze_pdf(pdf_path, focus_technique='auto', skip_ml=True)
                
                # Extract feature vector
                feature_vector = [self.features.get(name, 0) for name in self.FEATURE_NAMES]
                features_list.append(feature_vector)
                processed_files.append(pdf_file)
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                failed_files.append(pdf_file)
                continue
        
        if len(features_list) < 5:
            raise ValueError(f"Too few successful extractions ({len(features_list)}). Need at least 5 clean PDFs.")
        
        # Train the model
        X_train = np.array(features_list)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        # Fit scaler and model
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        self.ml_model.fit(X_train_scaled)
        
        # Save trained models
        joblib.dump(self.ml_model, output_model_path)
        joblib.dump(self.scaler, output_model_path.replace('.pkl', '_scaler.pkl'))
        
        # Save training metadata
        training_info = {
            'training_date': datetime.now().isoformat(),
            'total_files': len(pdf_files),
            'processed_files': len(processed_files),
            'failed_files': len(failed_files),
            'feature_names': self.FEATURE_NAMES,
            'model_parameters': self.ml_model.get_params(),
            'training_statistics': {
                'mean_features': X_train.mean(axis=0).tolist(),
                'std_features': X_train.std(axis=0).tolist(),
                'min_features': X_train.min(axis=0).tolist(),
                'max_features': X_train.max(axis=0).tolist()
            }
        }
        
        with open(output_model_path.replace('.pkl', '_training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Training completed. Model saved to {output_model_path}")
        logger.info(f"Processed: {len(processed_files)}/{len(pdf_files)} files successfully")
        
        return training_info

    # RECOMMENDATION 3: Fix argument compatibility issue
    def analyze_pdf(self, pdf_path: str, focus_technique: str = 'auto', skip_ml: bool = False) -> Dict[str, Any]:
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

            # FIXED: Argument compatibility for all analysis methods
            analyses = [
                ('object_stream', lambda: self._analyze_object_streams(pdf_content, pdf_path)),
                ('metadata', lambda: self._analyze_metadata(pdf_content, pdf_path)),
                ('font_glyph', lambda: self._analyze_fonts_glyphs(pdf_content, pdf_path)),  # FIXED
                ('entropy', lambda: self._analyze_entropy_patterns(pdf_content)),
                ('embedded', lambda: self._scan_embedded_files(pdf_content, pdf_path)),
                ('layers', lambda: self._detect_invisible_layers(pdf_content, pdf_path)),
            ]

            # RECOMMENDATION 4: Parallel processing for performance
            if focus_technique == 'auto':
                self._parallel_analysis(analyses)
            else:
                for technique, analysis_func in analyses:
                    if focus_technique == technique:
                        analysis_func()

            self._detect_concealed_pngs(pdf_content)
            self._enhanced_png_detection(pdf_content)  # RECOMMENDATION: Enhanced PNG detection
            
            if not skip_ml:
                self._ml_anomaly_detection()

            return self._generate_report()

        except Exception as e:
            logger.error("PDF analysis failed: %s", e)
            return {"error": str(e), "indicators": [], "ml_score": None}

    # RECOMMENDATION 4: Parallel processing implementation
    def _parallel_analysis(self, analyses: List) -> None:
        """Execute analysis methods in parallel for better performance."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_analysis = {executor.submit(analysis_func): technique 
                                for technique, analysis_func in analyses}
            
            for future in as_completed(future_to_analysis):
                technique = future_to_analysis[future]
                try:
                    future.result()
                    logger.debug(f"Completed {technique} analysis")
                except Exception as e:
                    logger.error(f"Analysis {technique} failed: {e}")

    # RECOMMENDATION 5: Enhanced PNG detection with LSB analysis
    def _enhanced_png_detection(self, pdf_content: bytes) -> None:
        """Enhanced PNG detection with LSB steganography analysis."""
        try:
            png_matches = []
            offset = 0
            
            while (pos := pdf_content.find(self.PNG_SIGNATURE, offset)) != -1:
                png_data = self._extract_png_at_offset(pdf_content, pos)
                if png_data and len(png_data) > 33:
                    # Analyze PNG for LSB patterns
                    lsb_suspicious = self._analyze_png_lsb_patterns(png_data)
                    chunk_anomalies = self._analyze_png_chunk_ordering(png_data)
                    
                    png_matches.append({
                        "offset": pos,
                        "size": len(png_data),
                        "valid_png": self._validate_png(png_data),
                        "lsb_suspicious": lsb_suspicious,
                        "chunk_anomalies": chunk_anomalies
                    })
                offset = pos + 8  # Skip ahead more efficiently

            if png_matches:
                high_risk_pngs = [p for p in png_matches 
                                if p.get("lsb_suspicious", False) or p.get("chunk_anomalies", False)]
                
                if high_risk_pngs:
                    self.indicators.append(SuspiciousIndicator(
                        category="enhanced_png",
                        severity="CRITICAL",
                        description=f"Found {len(high_risk_pngs)} PNGs with steganographic indicators",
                        confidence=0.9,
                        technical_details={"high_risk_pngs": high_risk_pngs}
                    ))

            self.features.update({
                "enhanced_png_count": len(png_matches),
                "high_risk_png_count": len([p for p in png_matches 
                                          if p.get("lsb_suspicious") or p.get("chunk_anomalies")])
            })

        except Exception as e:
            logger.error("Enhanced PNG detection failed: %s", e)

    def _analyze_png_lsb_patterns(self, png_data: bytes) -> bool:
        """Analyze PNG for LSB steganography patterns."""
        try:
            # Look for unusual patterns in the last bits of image data
            if len(png_data) < 100:
                return False
            
            # Simple heuristic: check for non-random patterns in LSBs
            lsb_bits = []
            for i in range(50, min(len(png_data), 1000)):  # Sample middle section
                lsb_bits.append(png_data[i] & 1)
            
            if len(lsb_bits) < 50:
                return False
            
            # Calculate chi-square test for randomness
            ones = sum(lsb_bits)
            zeros = len(lsb_bits) - ones
            expected = len(lsb_bits) / 2
            
            if expected > 0:
                chi_square = ((ones - expected) ** 2 + (zeros - expected) ** 2) / expected
                return chi_square > 10  # Threshold for non-randomness
            
            return False
        except Exception:
            return False

    def _analyze_png_chunk_ordering(self, png_data: bytes) -> bool:
        """Analyze PNG chunk ordering for anomalies."""
        try:
            chunks = []
            offset = 8  # Skip PNG signature
            
            while offset < len(png_data) - 8:
                if offset + 8 > len(png_data):
                    break
                    
                # Read chunk length and type
                chunk_length = int.from_bytes(png_data[offset:offset+4], 'big')
                chunk_type = png_data[offset+4:offset+8]
                chunks.append(chunk_type)
                
                offset += 8 + chunk_length + 4  # length + type + data + crc
            
            # Check for unusual chunk ordering
            chunk_sequence = [c.decode('ascii', errors='ignore') for c in chunks]
            
            # Standard PNG should start with IHDR and end with IEND
            if chunk_sequence and chunk_sequence[0] != 'IHDR':
                return True
            if chunk_sequence and chunk_sequence[-1] != 'IEND':
                return True
            
            # Look for unusual chunks
            standard_chunks = {'IHDR', 'PLTE', 'IDAT', 'IEND', 'tRNS', 'gAMA', 'cHRM', 'sRGB', 'iCCP'}
            unusual_chunks = [c for c in chunk_sequence if c not in standard_chunks]
            
            return len(unusual_chunks) > 2  # More than 2 unusual chunks is suspicious
            
        except Exception:
            return False

    # FIXED: Argument compatibility for font analysis
    def _analyze_fonts_glyphs(self, pdf_content: bytes, pdf_path: str) -> None:
        """Analyze fonts and glyphs for steganographic use."""
        try:
            doc = fitz.open("pdf", pdf_content)  # Use pdf_content instead of pdf_path for consistency
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

    # RECOMMENDATION 6: Improved caching for repeated analysis
    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key for PDF analysis results."""
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{os.path.basename(pdf_path)}_{file_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load analysis results from cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # RECOMMENDATION 7: Continuous learning capability
    def update_model_with_feedback(self, pdf_path: str, is_clean: bool, model_path: str) -> None:
        """
        Update the model with new feedback data.
        
        Args:
            pdf_path: Path to the PDF that was analyzed
            is_clean: True if confirmed clean, False if confirmed malicious
            model_path: Path to save updated model
        """
        try:
            # Analyze the PDF to get features
            self.indicators.clear()
            self.features.clear()
            self.analyze_pdf(pdf_path, skip_ml=True)
            
            feature_vector = np.array([self.features.get(name, 0) for name in self.FEATURE_NAMES]).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=0, neginf=0)
            
            if is_clean:
                # Add to clean dataset and retrain
                logger.info(f"Adding {pdf_path} to clean training set")
                # In production, you'd maintain a database of clean samples
                # and periodically retrain the model
            else:
                logger.info(f"Flagged {pdf_path} as malicious for future reference")
                # Store malicious samples for analysis but don't train on them
                # (since we're using unsupervised learning)
            
        except Exception as e:
            logger.error(f"Failed to update model with feedback: {e}")

    # Rest of the original methods remain the same...
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
        """Analyze PDF metadata for anomalies."""
        try:
            doc = fitz.open("pdf", pdf_content)
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

    def _scan_embedded_files(self, pdf_content: bytes, pdf_path: str) -> None:
        """Scan for embedded files that might contain hidden PNGs."""
        try:
            doc = fitz.open("pdf", pdf_content)
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

    def _detect_invisible_layers(self, pdf_content: bytes, pdf_path: str) -> None:
        """Detect invisible layers or optional content groups."""
        try:
            doc = fitz.open("pdf", pdf_content)
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
            logger.error("ML anomaly detection failedi: %s", e)

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

    def generate_detailed_report(self, result: dict, pdf_path: str) -> str:
        """Generate a comprehensive PDF steganography analysis report with technical details.
        
        Args:
            result: The analysis result dictionary from analyze_pdf()
            pdf_path: Path to the analyzed PDF file
            
        Returns:
            Formatted multi-line string with the complete report
        """
        report = []
        
        # Header section
        report.append("=" * 80)
        report.append(f"PDF STEGANOGRAPHY ANALYSIS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append(f"File: {pdf_path}")
        report.append(f"Assessment: {result['assessment']}")
        report.append(f"Risk Score: {result['risk_score']:.2f} (Thresholds: LOW<5, MEDIUM<10, HIGH<20, CRITICAL>=20)")
        report.append(f"Total Indicators Found: {result['total_indicators']}")
        report.append("\n")
        
        # Detailed findings section
        report.append("=" * 80)
        report.append("DETAILED FINDINGS")
        report.append("=" * 80)
        
        for category, indicators in result['indicators_by_category'].items():
            report.append(f"\n[{category.upper()} ANALYSIS]")
            for idx, indicator in enumerate(indicators, 1):
                report.append(f"\n{idx}. {indicator['severity']} SEVERITY: {indicator['description']}")
                report.append(f"   Confidence Level: {indicator['confidence']:.1%}")
                
                # Add location information if available
                if indicator.get('location'):
                    report.append(f"   Location in PDF: {indicator['location']}")
                
                # Add detailed technical findings
                tech_details = indicator['technical_details']
                if tech_details:
                    report.append("\n   Technical Details:")
                    for key, value in tech_details.items():
                        if isinstance(value, list):
                            report.append(f"   - {key}:")
                            for item in value[:25]:  # Limit to 5 items to avoid clutter
                                report.append(f"     • {str(item)[:120]}{'...' if len(str(item)) > 120 else ''}")
                        elif isinstance(value, dict):
                            report.append(f"   - {key}:")
                            for k, v in value.items():
                                report.append(f"     • {k}: {str(v)[:120]}{'...' if len(str(v)) > 120 else ''}")
                        else:
                            report.append(f"   - {key}: {str(value)[:120]}{'...' if len(str(value)) > 120 else ''}")
        
        # Machine Learning Analysis section
        if result['ml_analysis']:
            report.append("\n" + "=" * 80)
            report.append("MACHINE LEARNING ANALYSIS")
            report.append("=" * 80)
            report.append(f"Anomaly Score: {result['ml_analysis']['anomaly_score']:.3f}")
            report.append(f"Is Anomaly: {result['ml_analysis']['is_anomaly']}")
            
            if 'feature_vector' in result['ml_analysis']:
                report.append("\nTop Anomalous Features:")
                features = result['ml_analysis'].get('feature_names', [])
                values = result['ml_analysis'].get('feature_vector', [[]])[0]
                
                # Get top 10 most anomalous features
                feature_values = sorted(zip(features, values), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:10]
                
                for feature, value in feature_values:
                    report.append(f" - {feature}: {value:.4f}")
        
        # Recommendations section
        if result['recommendations']:
            report.append("\n" + "=" * 80)
            report.append("RECOMMENDED ACTIONS")
            report.append("=" * 80)
            for i, rec in enumerate(result['recommendations'], 1):
                report.append(f"{i}. {rec}")
    
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="PDF Steganography Analysis Tool")
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    args = parser.parse_args()

    detector = PDFSteganoDetector(ml_model_path='/home/d3bugger/Projects/FINAL YEAR PROJECT/src/detector_app/ai/ml/models/01_model.pkl')
    pdf_path = args.pdf_path

    try:
        print("Analyzing sample PDF...")
        result = detector.analyze_pdf(pdf_path, focus_technique='auto')

        # After getting the analysis result
        detailed_report = detector.generate_detailed_report(result, pdf_path)

        # Save to file or print
        with open('detailed_report.txt', 'w') as f:
            f.write(detailed_report)
        print(detailed_report)

    except Exception as e:
        logger.error(f"Analysis failed for {pdf_path}: {str(e)}")

if __name__ == "__main__":
    main()
