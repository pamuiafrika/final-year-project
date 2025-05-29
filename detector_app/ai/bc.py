import base64
import io
import os
import sys
import json
import struct
import zlib
import re
import logging
import hashlib
import threading
import concurrent.futures
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, kstest, normaltest
import pandas as pd
import argparse
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    import fitz
    import PyPDF2
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    HAS_PDF_LIBS = True
except ImportError as e:
    HAS_PDF_LIBS = False
    print(f"Warning: PDF libraries missing: {e}")

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    import joblib
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    print("Warning: ML libraries missing")

try:
    import cv2
    from PIL import Image
    HAS_IMAGE_LIBS = True
except ImportError:
    HAS_IMAGE_LIBS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SuspiciousIndicator:
    category: str
    severity: str
    description: str
    confidence: float
    technical_details: Dict[str, Any]
    location: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    validation_status: str = "PENDING"
    false_positive_probability: float = 0.0

@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    reason: str
    additional_checks: Dict[str, Any] = field(default_factory=dict)

class PDFStructureValidator:
    def __init__(self):
        self.known_legitimate_patterns = self._load_legitimate_patterns()
        self.suspicious_threshold = 0.7

    def _load_legitimate_patterns(self) -> Dict[str, Any]:
        return {
            'common_metadata_fields': {
                'title', 'author', 'subject', 'keywords', 'creator', 
                'producer', 'creationdate', 'moddate', 'trapped'
            },
            'common_producers': {
                'microsoft', 'adobe', 'libreoffice', 'openoffice',
                'ghostscript', 'pdftk', 'reportlab', 'fpdf'
            },
            'standard_filters': {
                'flatedecode', 'dctdecode', 'jpxdecode', 'lzwdecode',
                'asciihexdecode', 'ascii85decode', 'runlengthdecode'
            },
            'legitimate_encodings': {
                'winansiencoding', 'macromanencoding', 'utf-8', 'utf-16'
            }
        }

    def validate_metadata_anomaly(self, metadata: Dict[str, Any], anomaly_details: Dict[str, Any]) -> ValidationResult:
        false_positive_score = 0.0
        checks = {}
        
        producer = metadata.get('producer', '').lower()
        for known_producer in self.known_legitimate_patterns['common_producers']:
            if known_producer in producer:
                false_positive_score += 0.3
                checks['legitimate_producer'] = True
                break
        
        suspicious_fields = anomaly_details.get('suspicious_fields', [])
        legitimate_field_count = 0
        for field_dict in suspicious_fields:
            for field_name in field_dict.keys():
                if field_name.lower() in self.known_legitimate_patterns['common_metadata_fields']:
                    legitimate_field_count += 1
        
        if legitimate_field_count > len(suspicious_fields) * 0.7:
            false_positive_score += 0.4
            checks['mostly_legitimate_fields'] = True
        
        creation_date = metadata.get('creationdate', '')
        mod_date = metadata.get('moddate', '')
        if creation_date and mod_date:
            try:
                if abs(len(creation_date) - len(mod_date)) < 5:
                    false_positive_score += 0.2
                    checks['consistent_dates'] = True
            except:
                pass
        
        is_valid = false_positive_score > 0.5
        confidence = min(false_positive_score, 1.0)
        
        reason = "Legitimate metadata pattern detected" if is_valid else "Suspicious metadata confirmed"
        
        return ValidationResult(is_valid, confidence, reason, checks)

    def validate_object_anomaly(self, object_details: Dict[str, Any]) -> ValidationResult:
        false_positive_score = 0.0
        checks = {}
        
        large_objects = object_details.get('large_objects', [])
        for obj in large_objects:
            obj_type = obj.get('type', '').lower()
            if obj_type in ['xobject', 'image', 'form']:
                false_positive_score += 0.3
                checks['legitimate_large_object_type'] = True
        
        unused_objects = object_details.get('unused_objects', [])
        if len(unused_objects) < 5:
            false_positive_score += 0.4
            checks['reasonable_unused_count'] = True
        
        is_valid = false_positive_score > 0.5
        confidence = min(false_positive_score, 1.0)
        reason = "Legitimate object structure" if is_valid else "Suspicious object structure confirmed"
        
        return ValidationResult(is_valid, confidence, reason, checks)

    def validate_entropy_anomaly(self, entropy_details: Dict[str, Any], pdf_content: bytes) -> ValidationResult:
        false_positive_score = 0.0
        checks = {}
        
        max_entropy = entropy_details.get('max_entropy', 0)
        avg_entropy = entropy_details.get('avg_entropy', 0)
        
        if 7.0 <= max_entropy <= 7.8:
            false_positive_score += 0.3
            checks['normal_high_entropy'] = True
        
        if pdf_content:
            jpeg_markers = pdf_content.count(b'\xff\xd8\xff')
            png_markers = pdf_content.count(b'\x89PNG')
            if jpeg_markers > 0 or png_markers > 0:
                false_positive_score += 0.4
                checks['contains_images'] = True
        
        variance = entropy_details.get('entropy_variance', 0)
        if variance < 0.5:
            false_positive_score += 0.2
            checks['consistent_entropy'] = True
        
        is_valid = false_positive_score > 0.6
        confidence = min(false_positive_score, 1.0)
        reason = "High entropy from legitimate content" if is_valid else "Suspicious entropy pattern confirmed"
        
        return ValidationResult(is_valid, confidence, reason, checks)

class AdvancedEntropyAnalyzer:
    def __init__(self):
        self.block_sizes = [256, 512, 1024, 2048]
        self.entropy_thresholds = {
            'low': 4.0,
            'medium': 6.0,
            'high': 7.0,
            'critical': 7.8
        }

    def analyze_multiscale_entropy(self, data: bytes) -> Dict[str, Any]:
        results = {}
        
        for block_size in self.block_sizes:
            entropies = []
            for i in range(0, len(data), block_size):
                block = data[i:i+block_size]
                if len(block) >= block_size // 2:
                    entropy = self._calculate_shannon_entropy(block)
                    entropies.append(entropy)
            
            if entropies:
                results[f'block_{block_size}'] = {
                    'mean': np.mean(entropies),
                    'std': np.std(entropies),
                    'max': np.max(entropies),
                    'min': np.min(entropies),
                    'high_entropy_ratio': sum(1 for e in entropies if e > 7.0) / len(entropies)
                }
        
        return results

    def detect_compression_artifacts(self, data: bytes) -> Dict[str, Any]:
        artifacts = {
            'deflate_streams': 0,
            'jpeg_markers': 0,
            'suspicious_patterns': []
        }
        
        artifacts['deflate_streams'] = data.count(b'\x78\x9c') + data.count(b'\x78\xda')
        artifacts['jpeg_markers'] = data.count(b'\xff\xd8\xff')
        
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if len(chunk) < 100:
                continue
            
            byte_freq = Counter(chunk)
            if len(byte_freq) < 10:
                artifacts['suspicious_patterns'].append({
                    'offset': i,
                    'type': 'low_diversity',
                    'unique_bytes': len(byte_freq)
                })
            
            if any(count > len(chunk) * 0.7 for count in byte_freq.values()):
                artifacts['suspicious_patterns'].append({
                    'offset': i,
                    'type': 'repeated_pattern',
                    'max_frequency': max(byte_freq.values()) / len(chunk)
                })
        
        return artifacts

    def _calculate_shannon_entropy(self, data: bytes) -> float:
        if len(data) == 0:
            return 0.0
        
        byte_counts = Counter(data)
        length = len(data)
        entropy = 0.0
        
        for count in byte_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy

class StatisticalAnalyzer:
    def __init__(self):
        self.test_threshold = 0.05

    def chi_square_test(self, data: bytes) -> Dict[str, Any]:
        if len(data) < 256:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        observed = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        expected = np.full(256, len(data) / 256)
        
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'valid': True,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'is_random': p_value > self.test_threshold,
            'uniformity_score': 1.0 - min(p_value * 10, 1.0)
        }

    def kolmogorov_smirnov_test(self, data: bytes) -> Dict[str, Any]:
        if len(data) < 100:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        normalized_data = np.array(list(data)) / 255.0
        statistic, p_value = kstest(normalized_data, 'uniform')
        
        return {
            'valid': True,
            'ks_statistic': statistic,
            'p_value': p_value,
            'is_uniform': p_value > self.test_threshold
        }

    def autocorrelation_analysis(self, data: bytes, max_lag: int = 100) -> Dict[str, Any]:
        if len(data) < max_lag * 2:
            return {'valid': False, 'reason': 'insufficient_data'}
        
        data_array = np.array(list(data), dtype=np.float64)
        data_array = (data_array - np.mean(data_array)) / np.std(data_array)
        
        autocorr = np.correlate(data_array, data_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        significant_lags = []
        for lag in range(1, min(max_lag, len(autocorr))):
            if abs(autocorr[lag]) > 2 / np.sqrt(len(data_array)):
                significant_lags.append((lag, autocorr[lag]))
        
        return {
            'valid': True,
            'significant_lags': significant_lags,
            'max_autocorr': np.max(np.abs(autocorr[1:max_lag])) if len(autocorr) > max_lag else 0,
            'has_pattern': len(significant_lags) > 0
        }

class ModernSteganoDetector:
    def __init__(self):
        self.known_techniques = {
            'content_stream_injection': self._detect_content_stream_injection,
            'xmp_metadata_hiding': self._detect_xmp_metadata_hiding,
            'font_substitution': self._detect_font_substitution,
            'annotation_hiding': self._detect_annotation_hiding,
            'form_field_hiding': self._detect_form_field_hiding,
            'javascript_hiding': self._detect_javascript_hiding,
            'optional_content_groups': self._detect_ocg_hiding,
            'incremental_updates': self._detect_incremental_updates
        }

    def detect_all_techniques(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        for technique_name, detector_func in self.known_techniques.items():
            try:
                result = detector_func(pdf_path, pdf_content)
                if result:
                    indicators.extend(result)
            except Exception as e:
                logger.warning(f"Detection technique {technique_name} failed: {e}")
        
        return indicators

    def _detect_content_stream_injection(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        try:
            doc = fitz.open("pdf", pdf_content)
            
            suspicious_operators = [b'Tf', b'TJ', b'Tj', b'Do', b'q', b'Q']
            hidden_content_count = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                try:
                    content_streams = page.get_contents()
                    if not isinstance(content_streams, list):
                        content_streams = [content_streams] if content_streams else []
                    
                    for stream_ref in content_streams:
                        if stream_ref:
                            stream_data = doc.xref_stream(stream_ref)
                            if stream_data:
                                for operator in suspicious_operators:
                                    if operator in stream_data:
                                        context_start = max(0, stream_data.find(operator) - 50)
                                        context_end = min(len(stream_data), stream_data.find(operator) + 100)
                                        context = stream_data[context_start:context_end]
                                        
                                        if self._is_suspicious_context(context, operator):
                                            hidden_content_count += 1
                
                except Exception as e:
                    continue
            
            if hidden_content_count > 0:
                indicators.append(SuspiciousIndicator(
                    category="content_stream_injection",
                    severity="MEDIUM",
                    description=f"Detected {hidden_content_count} suspicious content stream operations",
                    confidence=0.6,
                    technical_details={"suspicious_operations": hidden_content_count}
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Content stream injection detection failed: {e}")
        
        return indicators

    def _is_suspicious_context(self, context: bytes, operator: bytes) -> bool:
        if len(context) < 20:
            return False
        
        if operator == b'Do' and b'Im' in context:
            return True
        
        if operator in [b'Tj', b'TJ'] and context.count(b'(') != context.count(b')'):
            return True
        
        if operator in [b'q', b'Q'] and context.count(b'q') != context.count(b'Q'):
            return True
        
        return False

    def _detect_xmp_metadata_hiding(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        xmp_pattern = re.compile(rb'<x:xmpmeta.*?</x:xmpmeta>', re.DOTALL | re.IGNORECASE)
        xmp_matches = xmp_pattern.findall(pdf_content)
        
        for match in xmp_matches:
            if len(match) > 10000:
                indicators.append(SuspiciousIndicator(
                    category="xmp_metadata_hiding",
                    severity="MEDIUM",
                    description=f"Unusually large XMP metadata block ({len(match)} bytes)",
                    confidence=0.7,
                    technical_details={"xmp_size": len(match)}
                ))
            
            if b'base64' in match.lower() or b'encoded' in match.lower():
                indicators.append(SuspiciousIndicator(
                    category="xmp_metadata_hiding",
                    severity="HIGH",
                    description="XMP metadata contains encoding references",
                    confidence=0.8,
                    technical_details={"xmp_content": match.decode('utf-8', errors='ignore')[:500]}
                ))
        
        return indicators

    def _detect_font_substitution(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        try:
            doc = fitz.open("pdf", pdf_content)
            
            font_analysis = {
                'total_fonts': 0,
                'embedded_fonts': 0,
                'suspicious_fonts': [],
                'font_name_anomalies': 0
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                fonts = page.get_fonts()
                
                for font in fonts:
                    font_analysis['total_fonts'] += 1
                    font_ref, font_ext, font_type, font_basename, font_name, font_encoding = font
                    
                    if font_ext:
                        font_analysis['embedded_fonts'] += 1
                    
                    if font_name:
                        if len(font_name) > 100 or any(ord(c) > 127 for c in font_name if isinstance(c, str)):
                            font_analysis['font_name_anomalies'] += 1
                            font_analysis['suspicious_fonts'].append({
                                'page': page_num,
                                'name': font_name,
                                'type': font_type,
                                'encoding': font_encoding
                            })
                        
                        if 'hidden' in font_name.lower() or 'steganography' in font_name.lower():
                            indicators.append(SuspiciousIndicator(
                                category="font_substitution",
                                severity="HIGH",
                                description="Font name suggests steganographic use",
                                confidence=0.9,
                                technical_details={"font_name": font_name}
                            ))
            
            if font_analysis['font_name_anomalies'] > 2:
                indicators.append(SuspiciousIndicator(
                    category="font_substitution",
                    severity="MEDIUM",
                    description=f"Multiple fonts with suspicious names ({font_analysis['font_name_anomalies']})",
                    confidence=0.6,
                    technical_details=font_analysis
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Font substitution detection failed: {e}")
        
        return indicators

    def _detect_annotation_hiding(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        try:
            doc = fitz.open("pdf", pdf_content)
            
            hidden_annotations = 0
            total_annotations = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                annotations = page.annots()
                
                for annot in annotations:
                    total_annotations += 1
                    annot_dict = annot.info
                    
                    if annot_dict.get('opacity', 1.0) == 0.0:
                        hidden_annotations += 1
                    
                    if annot.rect.width < 1 or annot.rect.height < 1:
                        hidden_annotations += 1
                    
                    content = annot_dict.get('content', '')
                    if len(content) > 1000:
                        indicators.append(SuspiciousIndicator(
                            category="annotation_hiding",
                            severity="MEDIUM",
                            description=f"Annotation with large content ({len(content)} chars)",
                            confidence=0.6,
                            technical_details={"content_length": len(content), "page": page_num}
                        ))
            
            if hidden_annotations > 0:
                indicators.append(SuspiciousIndicator(
                    category="annotation_hiding",
                    severity="HIGH",
                    description=f"Found {hidden_annotations} hidden annotations",
                    confidence=0.8,
                    technical_details={"hidden_count": hidden_annotations, "total_count": total_annotations}
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Annotation hiding detection failed: {e}")
        
        return indicators

    def _detect_form_field_hiding(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        form_pattern = re.compile(rb'/Type\s*/Widget', re.IGNORECASE)
        hidden_pattern = re.compile(rb'/F\s*(\d+)', re.IGNORECASE)
        
        form_matches = form_pattern.findall(pdf_content)
        hidden_flags = hidden_pattern.findall(pdf_content)
        
        invisible_forms = 0
        for flag in hidden_flags:
            try:
                flag_value = int(flag)
                if flag_value & 2:
                    invisible_forms += 1
            except:
                continue
        
        if invisible_forms > 0:
            indicators.append(SuspiciousIndicator(
                category="form_field_hiding",
                severity="MEDIUM",
                description=f"Found {invisible_forms} invisible form fields",
                confidence=0.7,
                technical_details={"invisible_forms": invisible_forms, "total_forms": len(form_matches)}
            ))
        
        return indicators

    def _detect_javascript_hiding(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        js_patterns = [
            rb'/JavaScript',
            rb'/JS\s*\(',
            rb'app\.alert',
            rb'eval\s*\(',
            rb'unescape\s*\('
        ]
        
        js_detections = []
        for pattern in js_patterns:
            matches = re.findall(pattern, pdf_content, re.IGNORECASE)
            if matches:
                js_detections.extend(matches)
        
        if js_detections:
            indicators.append(SuspiciousIndicator(
                category="javascript_hiding",
                severity="HIGH",
                description=f"JavaScript code detected ({len(js_detections)} instances)",
                confidence=0.8,
                technical_details={"js_patterns": [m.decode('utf-8', errors='ignore') for m in js_detections]}
            ))
        
        return indicators

    def _detect_ocg_hiding(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        ocg_pattern = re.compile(rb'/Type\s*/OCG', re.IGNORECASE)
        ocmd_pattern = re.compile(rb'/Type\s*/OCMD', re.IGNORECASE)
        
        ocg_matches = len(ocg_pattern.findall(pdf_content))
        ocmd_matches = len(ocmd_pattern.findall(pdf_content))
        
        if ocg_matches > 0 or ocmd_matches > 0:
            indicators.append(SuspiciousIndicator(
                category="optional_content_groups",
                severity="MEDIUM",
                description=f"Optional Content Groups detected (OCG: {ocg_matches}, OCMD: {ocmd_matches})",
                confidence=0.6,
                technical_details={"ocg_count": ocg_matches, "ocmd_count": ocmd_matches}
            ))
        
        return indicators

    def _detect_incremental_updates(self, pdf_path: str, pdf_content: bytes) -> List[SuspiciousIndicator]:
        indicators = []
        
        trailer_pattern = re.compile(rb'trailer\s*<<.*?>>\s*startxref', re.DOTALL | re.IGNORECASE)
        trailers = trailer_pattern.findall(pdf_content)
        
        if len(trailers) > 1:
            indicators.append(SuspiciousIndicator(
                category="incremental_updates",
                severity="MEDIUM",
                description=f"Multiple incremental updates detected ({len(trailers)})",
                confidence=0.5,
                technical_details={"update_count": len(trailers)}
            ))
        
        return indicators

class EnhancedMLModel:
    def __init__(self):
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.pca = PCA(n_components=0.95)
        self.is_fitted = False
        self.feature_importance = {}
        self.performance_metrics = {}

    def extract_advanced_features(self, pdf_analysis: Dict[str, Any]) -> np.ndarray:
        features = []
        
        basic_features = [
            'total_objects', 'large_objects_count', 'unused_objects_count',
            'avg_object_size', 'metadata_fields_count', 'suspicious_metadata_count',
            'total_fonts', 'embedded_fonts', 'font_anomalies_count',
            'embedded_font_ratio', 'avg_entropy', 'max_entropy', 'entropy_variance',
            'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
            'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
            'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size'
        ]
        
        for feature in basic_features:
            features.append(pdf_analysis.get(feature, 0))
        
        statistical_features = self._extract_statistical_features(pdf_analysis)
        features.extend(statistical_features)
        
        structural_features = self._extract_structural_features(pdf_analysis)
        features.extend(structural_features)
        
        return np.array(features, dtype=np.float64)

    def _extract_statistical_features(self, analysis: Dict[str, Any]) -> List[float]:
        features = []
        
        entropy_stats = analysis.get('entropy_analysis', {})
        features.extend([
            entropy_stats.get('chi2_uniformity', 0),
            entropy_stats.get('ks_uniformity', 0),
            entropy_stats.get('autocorrelation_strength', 0),
            len(entropy_stats.get('significant_patterns', [])),
        ])
        
        compression_stats = analysis.get('compression_analysis', {})
        features.extend([
            compression_stats.get('deflate_ratio', 0),
            compression_stats.get('jpeg_ratio', 0),
            compression_stats.get('compression_efficiency', 0),
        ])
        
        return features

    def _extract_structural_features(self, analysis: Dict[str, Any]) -> List[float]:
        features = []
        
        structure_stats = analysis.get('structure_analysis', {})
        features.extend([
            structure_stats.get('xref_density', 0),
            structure_stats.get('object_reference_ratio', 0),
            structure_stats.get('stream_to_object_ratio', 0),
            structure_stats.get('indirect_object_ratio', 0),
        ])
        
        version_stats = analysis.get('version_analysis', {})
        features.extend([
            version_stats.get('pdf_version', 1.4),
            version_stats.get('incremental_updates', 0),
            version_stats.get('linearized', 0),
        ])
        
        return features

    def train_ensemble(self, training_data: List[np.ndarray], labels: List[int]):
        if not training_data:
            logger.warning("No training data provided")
            return
        
        X = np.vstack(training_data)
        y = np.array(labels)
        
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        X_scaled_standard = self.scalers['standard'].fit_transform(X)
        X_scaled_robust = self.scalers['robust'].fit_transform(X)
        
        if X_scaled_standard.shape[1] > 10:
            X_pca = self.pca.fit_transform(X_scaled_standard)
        else:
            X_pca = X_scaled_standard
        
        self.models['isolation_forest'].fit(X_scaled_robust)
        
        if len(np.unique(y)) > 1:
            self.models['random_forest'].fit(X_pca, y)
            
            feature_names = [f'feature_{i}' for i in range(X_pca.shape[1])]
            self.feature_importance = dict(zip(
                feature_names,
                self.models['random_forest'].feature_importances_
            ))
        
        self.is_fitted = True
        logger.info("ML ensemble training completed")

    def predict_anomaly(self, features: np.ndarray) -> Dict[str, Any]:
        if not self.is_fitted:
            logger.warning("Models not fitted, using default prediction")
            return {
                'ensemble_score': 0.0,
                'isolation_score': 0.0,
                'classification_score': 0.5,
                'is_anomaly': False,
                'confidence': 0.1
            }
        
        features = np.nan_to_num(features.reshape(1, -1), nan=0, posinf=0, neginf=0)
        
        X_standard = self.scalers['standard'].transform(features)
        X_robust = self.scalers['robust'].transform(features)
        
        if hasattr(self.pca, 'components_'):
            X_pca = self.pca.transform(X_standard)
        else:
            X_pca = X_standard
        
        isolation_score = self.models['isolation_forest'].decision_function(X_robust)[0]
        is_anomaly_isolation = self.models['isolation_forest'].predict(X_robust)[0] == -1
        
        classification_prob = 0.5
        if hasattr(self.models['random_forest'], 'predict_proba'):
            try:
                classification_prob = self.models['random_forest'].predict_proba(X_pca)[0][1]
            except:
                classification_prob = 0.5
        
        ensemble_score = (abs(isolation_score) + classification_prob) / 2
        confidence = min(ensemble_score, 1.0)
        
        is_anomaly = is_anomaly_isolation or classification_prob > 0.7
        
        return {
            'ensemble_score': ensemble_score,
            'isolation_score': isolation_score,
            'classification_score': classification_prob,
            'is_anomaly': is_anomaly,
            'confidence': confidence
        }

class PerformanceOptimizer:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.cache = {}
        self.cache_size_limit = 100

    def parallel_object_analysis(self, doc, object_ids: List[int]) -> Dict[str, Any]:
        def analyze_object(obj_id):
            try:
                return {
                    'id': obj_id,
                    'is_stream': doc.xref_is_stream(obj_id),
                    'length': doc.xref_get_key(obj_id, "Length"),
                    'type': doc.xref_get_key(obj_id, "Type"),
                    'subtype': doc.xref_get_key(obj_id, "Subtype")
                }
            except:
                return {'id': obj_id, 'error': True}
        
        chunk_size = max(1, len(object_ids) // self.max_workers)
        chunks = [object_ids[i:i + chunk_size] for i in range(0, len(object_ids), chunk_size)]
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(analyze_object, obj_id): obj_id 
                              for chunk in chunks for obj_id in chunk}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Object analysis timeout for object {future_to_chunk[future]}")
                except Exception as e:
                    logger.error(f"Object analysis error: {e}")
        
        return {'analyzed_objects': results}

    def cached_entropy_calculation(self, data: bytes, cache_key: str = None) -> float:
        if cache_key is None:
            cache_key = hashlib.md5(data[:1000]).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        entropy = self._calculate_entropy(data)
        
        if len(self.cache) >= self.cache_size_limit:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = entropy
        return entropy

    def _calculate_entropy(self, data: bytes) -> float:
        if len(data) == 0:
            return 0.0
        
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]
        
        return -np.sum(probabilities * np.log2(probabilities))

    def memory_efficient_scan(self, file_path: str, pattern: bytes, chunk_size: int = 8192) -> List[int]:
        positions = []
        
        with open(file_path, 'rb') as f:
            overlap = len(pattern) - 1
            pos = 0
            previous_chunk = b''
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                search_data = previous_chunk + chunk
                
                start = 0
                while True:
                    found = search_data.find(pattern, start)
                    if found == -1:
                        break
                    positions.append(pos - len(previous_chunk) + found)
                    start = found + 1
                
                pos += len(chunk)
                previous_chunk = chunk[-overlap:] if len(chunk) >= overlap else chunk
        
        return positions

class PDFSteganoDetectorEnhanced:
    def __init__(self, ml_model_path: Optional[str] = None):
        self.indicators: List[SuspiciousIndicator] = []
        self.features: Dict[str, Any] = {}
        self.validator = PDFStructureValidator()
        self.entropy_analyzer = AdvancedEntropyAnalyzer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.modern_detector = ModernSteganoDetector()
        self.ml_model = EnhancedMLModel()
        self.optimizer = PerformanceOptimizer()
        
        self.PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
        self.JPEG_SIGNATURE = b'\xff\xd8\xff'
        
        if ml_model_path and os.path.exists(ml_model_path):
            self._load_pretrained_model(ml_model_path)

    def _load_pretrained_model(self, model_path: str):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if 'models' in model_data:
                self.ml_model.models = model_data['models']
            if 'scalers' in model_data:
                self.ml_model.scalers = model_data['scalers']
            if 'pca' in model_data:
                self.ml_model.pca = model_data['pca']
            
            self.ml_model.is_fitted = True
            logger.info("Loaded pretrained ML model")
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")

    def analyze_pdf(self, pdf_path: str, quick_scan: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        self.indicators.clear()
        self.features.clear()
        
        try:
            if not os.path.exists(pdf_path):
                return {"error": "File not found", "indicators": []}
            
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            if pdf_content.startswith(b'JVBERi0'):
                try:
                    pdf_content = base64.b64decode(pdf_content)
                except:
                    pass
            
            if not pdf_content.startswith(b'%PDF'):
                return {"error": "Invalid PDF format", "indicators": []}
            
            # Basic structural analysis
            self._analyze_pdf_structure(pdf_content, pdf_path)
            
            # Modern steganography detection
            modern_indicators = self.modern_detector.detect_all_techniques(pdf_path, pdf_content)
            self.indicators.extend(modern_indicators)
            
            if not quick_scan:
                # Advanced entropy analysis
                self._advanced_entropy_analysis(pdf_content)
                
                # Statistical analysis
                self._statistical_analysis(pdf_content)
                
                # Object stream analysis with performance optimization
                self._optimized_object_analysis(pdf_path, pdf_content)
                
                # Enhanced metadata analysis
                self._enhanced_metadata_analysis(pdf_path)
            
            # PNG and binary detection
            self._comprehensive_binary_detection(pdf_content)
            
            # ML anomaly detection
            self._enhanced_ml_analysis()
            
            # Validate findings to reduce false positives
            self._validate_indicators(pdf_content)
            
            analysis_time = time.time() - start_time
            
            return self._generate_comprehensive_report(analysis_time)
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {"error": str(e), "indicators": [], "analysis_time": 0}

    def _analyze_pdf_structure(self, pdf_content: bytes, pdf_path: str):
        try:
            # Extract PDF version
            version_match = re.search(rb'%PDF-(\d+\.\d+)', pdf_content)
            pdf_version = float(version_match.group(1)) if version_match else 1.4
            
            # Count cross-reference tables
            xref_count = pdf_content.count(b'xref')
            
            # Analyze trailer sections
            trailer_count = pdf_content.count(b'trailer')
            
            # Check for linearization
            is_linearized = b'/Linearized' in pdf_content
            
            # Object counting
            obj_pattern = re.compile(rb'(\d+)\s+(\d+)\s+obj')
            objects = obj_pattern.findall(pdf_content)
            
            self.features.update({
                'pdf_version': pdf_version,
                'xref_count': xref_count,
                'trailer_count': trailer_count,
                'is_linearized': int(is_linearized),
                'total_objects_regex': len(objects),
                'file_size': len(pdf_content)
            })
            
        except Exception as e:
            logger.error(f"PDF structure analysis failed: {e}")

    def _advanced_entropy_analysis(self, pdf_content: bytes):
        try:
            # Multi-scale entropy analysis
            entropy_results = self.entropy_analyzer.analyze_multiscale_entropy(pdf_content)
            
            # Compression artifact detection
            compression_results = self.entropy_analyzer.detect_compression_artifacts(pdf_content)
            
            # Update features
            for block_size, stats in entropy_results.items():
                for stat_name, value in stats.items():
                    self.features[f'entropy_{block_size}_{stat_name}'] = value
            
            self.features.update({
                'compression_deflate_streams': compression_results['deflate_streams'],
                'compression_jpeg_markers': compression_results['jpeg_markers'],
                'compression_suspicious_patterns': len(compression_results['suspicious_patterns'])
            })
            
            # Detect anomalous entropy patterns
            if entropy_results:
                max_entropy_1024 = entropy_results.get('block_1024', {}).get('max', 0)
                if max_entropy_1024 > 7.9:
                    self.indicators.append(SuspiciousIndicator(
                        category="advanced_entropy",
                        severity="HIGH",
                        description=f"Extremely high entropy detected ({max_entropy_1024:.2f})",
                        confidence=0.8,
                        technical_details=entropy_results
                    ))
            
        except Exception as e:
            logger.error(f"Advanced entropy analysis failed: {e}")

    def _statistical_analysis(self, pdf_content: bytes):
        try:
            # Chi-square test for randomness
            chi2_result = self.statistical_analyzer.chi_square_test(pdf_content)
            
            # Kolmogorov-Smirnov test
            ks_result = self.statistical_analyzer.kolmogorov_smirnov_test(pdf_content)
            
            # Autocorrelation analysis
            autocorr_result = self.statistical_analyzer.autocorrelation_analysis(pdf_content)
            
            self.features.update({
                'chi2_p_value': chi2_result.get('p_value', 1.0),
                'chi2_is_random': int(chi2_result.get('is_random', True)),
                'ks_p_value': ks_result.get('p_value', 1.0),
                'ks_is_uniform': int(ks_result.get('is_uniform', True)),
                'autocorr_max': autocorr_result.get('max_autocorr', 0),
                'autocorr_has_pattern': int(autocorr_result.get('has_pattern', False))
            })
            
            # Generate indicators based on statistical tests
            if chi2_result.get('valid') and not chi2_result.get('is_random'):
                self.indicators.append(SuspiciousIndicator(
                    category="statistical_analysis",
                    severity="MEDIUM",
                    description=f"Chi-square test indicates non-random data (p={chi2_result['p_value']:.4f})",
                    confidence=0.6,
                    technical_details=chi2_result
                ))
            
            if autocorr_result.get('has_pattern'):
                self.indicators.append(SuspiciousIndicator(
                    category="statistical_analysis",
                    severity="MEDIUM",
                    description="Autocorrelation analysis detected hidden patterns",
                    confidence=0.7,
                    technical_details=autocorr_result
                ))
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")

    def _optimized_object_analysis(self, pdf_path: str, pdf_content: bytes):
        try:
            doc = fitz.open("pdf", pdf_content)
            
            total_objects = doc.xref_length()
            object_ids = list(range(1, total_objects))
            
            # Parallel object analysis for performance
            if len(object_ids) > 100:
                analysis_results = self.optimizer.parallel_object_analysis(doc, object_ids)
            else:
                analysis_results = {'analyzed_objects': []}
                for obj_id in object_ids:
                    try:
                        obj_info = {
                            'id': obj_id,
                            'is_stream': doc.xref_is_stream(obj_id),
                            'length': doc.xref_get_key(obj_id, "Length"),
                            'type': doc.xref_get_key(obj_id, "Type")
                        }
                        analysis_results['analyzed_objects'].append(obj_info)
                    except:
                        continue
            
            # Analyze results
            stream_objects = [obj for obj in analysis_results['analyzed_objects'] if obj.get('is_stream')]
            large_objects = []
            suspicious_objects = []
            
            for obj in stream_objects:
                try:
                    length = int(obj.get('length', 0)) if obj.get('length') else 0
                    if length > 100000:
                        large_objects.append(obj)
                    
                    # Additional suspicious patterns
                    if obj.get('type') and 'hidden' in str(obj['type']).lower():
                        suspicious_objects.append(obj)
                    
                except:
                    continue
            
            self.features.update({
                'optimized_total_objects': total_objects,
                'optimized_stream_objects': len(stream_objects),
                'optimized_large_objects': len(large_objects),
                'optimized_suspicious_objects': len(suspicious_objects)
            })
            
            if len(large_objects) > 5:
                self.indicators.append(SuspiciousIndicator(
                    category="optimized_object_analysis",
                    severity="MEDIUM",
                    description=f"Multiple large objects detected ({len(large_objects)})",
                    confidence=0.6,
                    technical_details={"large_objects": large_objects[:5]}
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Optimized object analysis failed: {e}")

    def _enhanced_metadata_analysis(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Standard metadata validation
            validation_result = self.validator.validate_metadata_anomaly(metadata, {"suspicious_fields": []})
            
            # Enhanced metadata checks
            metadata_anomalies = {
                'empty_required_fields': 0,
                'oversized_fields': 0,
                'encoding_anomalies': 0,
                'timestamp_anomalies': 0
            }
            
            required_fields = ['title', 'author', 'creator', 'producer']
            for field in required_fields:
                if not metadata.get(field):
                    metadata_anomalies['empty_required_fields'] += 1
            
            for key, value in metadata.items():
                if isinstance(value, str):
                    if len(value) > 1000:
                        metadata_anomalies['oversized_fields'] += 1
                    
                    # Check for unusual encoding
                    try:
                        value.encode('ascii')
                    except UnicodeEncodeError:
                        metadata_anomalies['encoding_anomalies'] += 1
                    
                    # Check for embedded data patterns
                    if any(pattern in value.lower() for pattern in ['base64', 'encoded', 'hidden', 'data:']):
                        self.indicators.append(SuspiciousIndicator(
                            category="enhanced_metadata",
                            severity="HIGH",
                            description=f"Suspicious content in metadata field '{key}'",
                            confidence=0.8,
                            technical_details={"field": key, "content": value[:200]}
                        ))
            
            self.features.update({
                'metadata_validation_confidence': validation_result.confidence,
                'metadata_empty_required': metadata_anomalies['empty_required_fields'],
                'metadata_oversized': metadata_anomalies['oversized_fields'],
                'metadata_encoding_anomalies': metadata_anomalies['encoding_anomalies']
            })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Enhanced metadata analysis failed: {e}")

    def _comprehensive_binary_detection(self, pdf_content: bytes):
        try:
            # PNG detection with validation
            png_positions = self.optimizer.memory_efficient_scan(
                StringIO(pdf_content).name if hasattr(StringIO, 'name') else 'temp',
                self.PNG_SIGNATURE
            )
            
            # Direct memory scan for performance
            png_count = pdf_content.count(self.PNG_SIGNATURE)
            jpeg_count = pdf_content.count(self.JPEG_SIGNATURE)
            
            # Advanced PNG validation
            valid_pngs = 0
            png_analysis = []
            
            for i, pos in enumerate(png_positions[:10]):  # Limit analysis for performance
                try:
                    if pos + 33 < len(pdf_content):
                        png_data = pdf_content[pos:pos + min(10000, len(pdf_content) - pos)]
                        if self._validate_png_structure(png_data):
                            valid_pngs += 1
                            png_analysis.append({
                                'position': pos,
                                'size': len(png_data),
                                'valid': True
                            })
                except:
                    continue
            
            # Check for steganography-specific patterns
            stego_patterns = [
                b'LSB',  # Least Significant Bit
                b'DCT',  # Discrete Cosine Transform
                b'JPEG-Jsteg',
                b'OutGuess',
                b'F5',
                b'StegHide'
            ]
            
            stego_indicators = []
            for pattern in stego_patterns:
                if pattern in pdf_content:
                    stego_indicators.append(pattern.decode('utf-8', errors='ignore'))
            
            self.features.update({
                'binary_png_count': png_count,
                'binary_jpeg_count': jpeg_count,
                'binary_valid_pngs': valid_pngs,
                'binary_stego_patterns': len(stego_indicators)
            })
            
            if valid_pngs > 0:
                self.indicators.append(SuspiciousIndicator(
                    category="comprehensive_binary",
                    severity="CRITICAL",
                    description=f"Valid PNG files found embedded in PDF ({valid_pngs})",
                    confidence=0.95,
                    technical_details={
                        "png_analysis": png_analysis,
                        "stego_patterns": stego_indicators
                    }
                ))
            
            if stego_indicators:
                self.indicators.append(SuspiciousIndicator(
                    category="comprehensive_binary",
                    severity="HIGH",
                    description="Steganography tool signatures detected",
                    confidence=0.9,
                    technical_details={"patterns": stego_indicators}
                ))
            
        except Exception as e:
            logger.error(f"Comprehensive binary detection failed: {e}")

    def _validate_png_structure(self, png_data: bytes) -> bool:
        try:
            if len(png_data) < 33:
                return False
            
            if not png_data.startswith(self.PNG_SIGNATURE):
                return False
            
            # Check IHDR chunk
            ihdr_pos = 8
            if png_data[ihdr_pos:ihdr_pos + 4] != b'\x00\x00\x00\x0d':
                return False
            
            if png_data[ihdr_pos + 4:ihdr_pos + 8] != b'IHDR':
                return False
            
            # Basic dimension validation
            width = struct.unpack('>I', png_data[ihdr_pos + 8:ihdr_pos + 12])[0]
            height = struct.unpack('>I', png_data[ihdr_pos + 12:ihdr_pos + 16])[0]
            
            if width == 0 or height == 0 or width > 100000 or height > 100000:
                return False
            
            return True
            
        except:
            return False

    def _enhanced_ml_analysis(self):
        try:
            if not self.features:
                return
            
            # Extract comprehensive features
            feature_vector = self.ml_model.extract_advanced_features(self.features)
            
            # Get ML prediction
            ml_result = self.ml_model.predict_anomaly(feature_vector)
            
            # Update features with ML results
            self.features.update({
                'ml_ensemble_score': ml_result['ensemble_score'],
                'ml_isolation_score': ml_result['isolation_score'],
                'ml_classification_score': ml_result['classification_score'],
                'ml_is_anomaly': ml_result['is_anomaly'],
                'ml_confidence': ml_result['confidence']
            })
            
            # Generate ML-based indicator
            if ml_result['is_anomaly'] or ml_result['ensemble_score'] > 0.7:
                severity = "HIGH" if ml_result['confidence'] > 0.8 else "MEDIUM"
                
                self.indicators.append(SuspiciousIndicator(
                    category="enhanced_ml_analysis",
                    severity=severity,
                    description=f"ML ensemble detected anomalous patterns (score: {ml_result['ensemble_score']:.3f})",
                    confidence=ml_result['confidence'],
                    technical_details=ml_result
                ))
            
        except Exception as e:
            logger.error(f"Enhanced ML analysis failed: {e}")

    def _validate_indicators(self, pdf_content: bytes):
        validated_indicators = []
        
        for indicator in self.indicators:
            try:
                # Apply appropriate validation based on category
                if indicator.category == "metadata":
                    # This would require additional context, simplified for demo
                    validation = ValidationResult(True, 0.8, "Validation passed", {})
                
                elif indicator.category == "entropy" or "entropy" in indicator.category:
                    validation = self.validator.validate_entropy_anomaly(
                        indicator.technical_details, pdf_content
                    )
                
                elif "object" in indicator.category:
                    validation = self.validator.validate_object_anomaly(
                        indicator.technical_details
                    )
                
                else:
                    # Default validation
                    validation = ValidationResult(True, indicator.confidence, "No specific validation", {})
                
                # Update indicator with validation results
                indicator.validation_status = "VALIDATED" if validation.is_valid else "SUSPICIOUS"
                indicator.false_positive_probability = 1.0 - validation.confidence
                
                # Adjust confidence based on validation
                if not validation.is_valid:
                    indicator.confidence = min(indicator.confidence * 1.2, 1.0)
                else:
                    indicator.confidence = max(indicator.confidence * 0.8, 0.1)
                
                validated_indicators.append(indicator)
                
            except Exception as e:
                logger.warning(f"Validation failed for indicator {indicator.category}: {e}")
                validated_indicators.append(indicator)
        
        self.indicators = validated_indicators

    def _generate_comprehensive_report(self, analysis_time: float) -> Dict[str, Any]:
        # Calculate risk scores
        severity_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 7, "CRITICAL": 10}
        
        # Raw risk score
        raw_risk_score = sum(
            severity_weights.get(indicator.severity, 0) * indicator.confidence
            for indicator in self.indicators
        )
        
        # Adjusted risk score (considering false positive probability)
        adjusted_risk_score = sum(
            severity_weights.get(indicator.severity, 0) * 
            indicator.confidence * 
            (1 - indicator.false_positive_probability)
            for indicator in self.indicators
        )
        
        # Categorize indicators
        indicators_by_category = defaultdict(list)
        indicators_by_severity = defaultdict(list)
        
        for indicator in self.indicators:
            indicators_by_category[indicator.category].append({
                "severity": indicator.severity,
                "description": indicator.description,
                "confidence": indicator.confidence,
                "false_positive_probability": indicator.false_positive_probability,
                "validation_status": indicator.validation_status,
                "technical_details": indicator.technical_details,
                "location": indicator.location
            })
            
            indicators_by_severity[indicator.severity].append(indicator)
        
        # Determine overall assessment
        critical_count = len(indicators_by_severity["CRITICAL"])
        high_count = len(indicators_by_severity["HIGH"])
        
        if critical_count > 0 or adjusted_risk_score > 25:
            assessment = "CRITICAL RISK - Strong evidence of steganography detected"
            risk_level = "CRITICAL"
        elif high_count > 2 or adjusted_risk_score > 15:
            assessment = "HIGH RISK - Multiple suspicious patterns detected"
            risk_level = "HIGH"
        elif adjusted_risk_score > 8:
            assessment = "MEDIUM RISK - Suspicious patterns detected"
            risk_level = "MEDIUM"
        elif adjusted_risk_score > 3:
            assessment = "LOW RISK - Minor anomalies found"
            risk_level = "LOW"
        else:
            assessment = "CLEAN - No significant suspicious indicators"
            risk_level = "CLEAN"
        
        # Generate comprehensive recommendations
        recommendations = self._generate_enhanced_recommendations()
        
        # Performance metrics
        performance_metrics = {
            "analysis_time": analysis_time,
            "total_features_extracted": len(self.features),
            "indicators_generated": len(self.indicators),
            "validated_indicators": len([i for i in self.indicators if i.validation_status == "VALIDATED"])
        }
        
        return {
            "assessment": assessment,
            "risk_level": risk_level,
            "raw_risk_score": raw_risk_score,
            "adjusted_risk_score": adjusted_risk_score,
            "total_indicators": len(self.indicators),
            "indicators_by_category": dict(indicators_by_category),
            "indicators_by_severity": {k: len(v) for k, v in indicators_by_severity.items()},
            "ml_analysis": {
                "ensemble_score": self.features.get("ml_ensemble_score"),
                "is_anomaly": self.features.get("ml_is_anomaly"),
                "confidence": self.features.get("ml_confidence")
            },
            "statistical_analysis": {
                "chi2_random": self.features.get("chi2_is_random"),
                "ks_uniform": self.features.get("ks_is_uniform"),
                "autocorr_patterns": self.features.get("autocorr_has_pattern")
            },
            "binary_analysis": {
                "png_count": self.features.get("binary_png_count", 0),
                "valid_pngs": self.features.get("binary_valid_pngs", 0),
                "stego_patterns": self.features.get("binary_stego_patterns", 0)
            },
            "features_extracted": self.features,
            "recommendations": recommendations,
            "performance_metrics": performance_metrics,
            "validation_summary": {
                "total_validated": len([i for i in self.indicators if i.validation_status == "VALIDATED"]),
                "total_suspicious": len([i for i in self.indicators if i.validation_status == "SUSPICIOUS"]),
                "avg_false_positive_prob": np.mean([i.false_positive_probability for i in self.indicators]) if self.indicators else 0
            }
        }

    def _generate_enhanced_recommendations(self) -> List[str]:
        recommendations = []
        
        # Critical findings
        critical_indicators = [i for i in self.indicators if i.severity == "CRITICAL"]
        if critical_indicators:
            recommendations.append("URGENT: Immediate forensic examination required - potential steganography detected")
            recommendations.append("Isolate the document and preserve for detailed analysis")
        
        # Binary findings
        binary_indicators = [i for i in self.indicators if "binary" in i.category or "png" in i.category.lower()]
        if binary_indicators:
            recommendations.append("Extract and analyze embedded binary data using specialized tools")
            recommendations.append("Verify integrity of embedded images and check for hidden layers")
            recommendations.append("Use steganography detection tools like StegExpose or StegSolve")
        
        # Statistical anomalies
        statistical_indicators = [i for i in self.indicators if "statistical" in i.category]
        if statistical_indicators:
            recommendations.append("Perform detailed statistical analysis of data distribution")
            recommendations.append("Check for hidden channels using frequency analysis")
        
        # ML anomalies
        ml_score = self.features.get("ml_ensemble_score", 0)
        if ml_score > 0.7:
            recommendations.append("Machine learning detected anomalous patterns - investigate unusual file structures")
            recommendations.append("Compare with known clean documents from the same source")
        
        # Modern techniques
        modern_indicators = [i for i in self.indicators if i.category in [
            'content_stream_injection', 'xmp_metadata_hiding', 'annotation_hiding', 
            'javascript_hiding', 'optional_content_groups'
        ]]
        if modern_indicators:
            recommendations.append("Analyze PDF structure for modern steganography techniques")
            recommendations.append("Check JavaScript code and optional content groups for hidden functionality")
        
        # Validation concerns
        high_fp_indicators = [i for i in self.indicators if i.false_positive_probability > 0.5]
        if high_fp_indicators:
            recommendations.append("Some indicators may be false positives - verify findings manually")
            recommendations.append("Cross-reference with document creation context and legitimate use cases")
        
        # General recommendations
        if len(self.indicators) > 5:
            recommendations.append("Multiple anomalies detected - consider comprehensive document reconstruction")
        
        if not recommendations:
            recommendations.append("No immediate action required - continue routine monitoring")
            recommendations.append("Consider periodic re-scanning with updated detection signatures")
        
        return recommendations

    def _analyze_object_streams(self, pdf_content: bytes, pdf_path: str):
        try:
            doc = fitz.open("pdf", pdf_content)
            stream_objects = 0
            suspicious_filters = 0
            high_entropy_streams = 0
            
            for xref in range(1, doc.xref_length()):
                if doc.xref_is_stream(xref):
                    stream_objects += 1
                    try:
                        stream_data = doc.xref_stream(xref)
                        if stream_data:
                            # Check filters
                            filters = doc.xref_get_key(xref, 'Filter')
                            if filters and any(f.lower() not in self.validator.known_legitimate_patterns['standard_filters'] 
                                            for f in str(filters).lower().split()):
                                suspicious_filters += 1
                            
                            # Check entropy
                            if len(stream_data) > 100:
                                entropy = self.optimizer.cached_entropy_calculation(stream_data)
                                if entropy > 7.5:
                                    high_entropy_streams += 1
                    except:
                        continue
            
            self.features.update({
                'stream_object_count': stream_objects,
                'suspicious_filter_count': suspicious_filters,
                'high_entropy_streams': high_entropy_streams
            })
            
            doc.close()
        except Exception as e:
            logger.error(f"Object stream analysis failed: {e}")

    def _analyze_metadata(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            metadata_fields = len(metadata) if metadata else 0
            suspicious_fields = 0
            
            for key, value in (metadata or {}).items():
                if isinstance(value, str):
                    if len(value) > 1000 or any(pattern in value.lower() 
                                              for pattern in ['base64', 'encoded', 'hidden']):
                        suspicious_fields += 1
            
            self.features.update({
                'metadata_fields_count': metadata_fields,
                'suspicious_metadata_count': suspicious_fields
            })
            
            doc.close()
        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")

    def _analyze_fonts_glyphs(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            total_fonts = 0
            embedded_fonts = 0
            font_anomalies = 0
            
            for page in doc:
                fonts = page.get_fonts()
                for font in fonts:
                    total_fonts += 1
                    if font[1]:  # font is embedded
                        embedded_fonts += 1
                    if len(font[3]) > 50 or any(ord(c) > 127 for c in font[3]):
                        font_anomalies += 1
            
            self.features.update({
                'total_fonts': total_fonts,
                'embedded_fonts': embedded_fonts,
                'font_anomalies_count': font_anomalies,
                'embedded_font_ratio': embedded_fonts / total_fonts if total_fonts > 0 else 0
            })
            
            doc.close()
        except Exception as e:
            logger.error(f"Font analysis failed: {e}")

    def _analyze_entropy_patterns(self, pdf_content: bytes):
        try:
            chunk_size = 1024
            entropies = []
            
            for i in range(0, len(pdf_content), chunk_size):
                chunk = pdf_content[i:i + chunk_size]
                if len(chunk) >= chunk_size // 2:
                    entropy = self.optimizer.cached_entropy_calculation(chunk)
                    entropies.append(entropy)
            
            if entropies:
                avg_entropy = np.mean(entropies)
                max_entropy = np.max(entropies)
                entropy_variance = np.var(entropies)
                high_entropy_chunks = sum(1 for e in entropies if e > 7.0)
                
                self.features.update({
                    'avg_entropy': avg_entropy,
                    'max_entropy': max_entropy,
                    'entropy_variance': entropy_variance,
                    'high_entropy_chunks': high_entropy_chunks
                })
        except Exception as e:
            logger.error(f"Entropy pattern analysis failed: {e}")

    def _scan_embedded_files(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            embedded_count = 0
            embedded_size = 0
            png_count = 0
            
            for xref in range(1, doc.xref_length()):
                if doc.xref_is_stream(xref):
                    try:
                        stream_data = doc.xref_stream(xref)
                        if stream_data:
                            embedded_size += len(stream_data)
                            if self.PNG_SIGNATURE in stream_data:
                                png_count += 1
                            embedded_count += 1
                    except:
                        continue
            
            self.features.update({
                'embedded_files_count': embedded_count,
                'embedded_png_count': png_count,
                'total_embedded_size': embedded_size
            })
            
            doc.close()
        except Exception as e:
            logger.error(f"Embedded file scan failed: {e}")

    def _detect_invisible_layers(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            invisible_count = 0
            pages_with_invisible = 0
            
            for page in doc:
                has_invisible = False
                for annot in page.annots():
                    if annot.opacity == 0 or (annot.rect.width < 1 and annot.rect.height < 1):
                        invisible_count += 1
                        has_invisible = True
                if has_invisible:
                    pages_with_invisible += 1
            
            self.features.update({
                'invisible_elements_count': invisible_count,
                'pages_with_invisible': pages_with_invisible
            })
            
            doc.close()
        except Exception as e:
            logger.error(f"Invisible layer detection failed: {e}")

    def _detect_concealed_pngs(self, pdf_content: bytes):
        try:
            png_positions = []
            pos = 0
            while True:
                pos = pdf_content.find(self.PNG_SIGNATURE, pos)
                if pos == -1:
                    break
                png_positions.append(pos)
                pos += 8
            
            valid_pngs = 0
            total_png_size = 0
            png_chunks = 0
            
            for pos in png_positions:
                try:
                    if pos + 33 < len(pdf_content):
                        png_data = pdf_content[pos:pos + min(10000, len(pdf_content) - pos)]
                        if self._validate_png_structure(png_data):
                            valid_pngs += 1
                            total_png_size += len(png_data)
                            png_chunks += png_data.count(b'IDAT') + png_data.count(b'IEND')
                except:
                    continue
            
            self.features.update({
                'png_signatures_count': len(png_positions),
                'valid_png_count': valid_pngs,
                'png_chunks_count': png_chunks,
                'total_png_size': total_png_size
            })
            
        except Exception as e:
            logger.error(f"Concealed PNG detection failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Steganography Analysis Tool")
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    parser.add_argument('--quick', '-q', action='store_true', help='Perform quick scan only')
    parser.add_argument('--output', '-o', help='Output file for detailed report (JSON)')
    parser.add_argument('--model', '-m', help='Path to pre-trained ML model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    detector = PDFSteganoDetectorEnhanced(ml_model_path=args.model)

    try:
        print("Analyzing PDF with enhanced detection system...")
        print(f"Quick scan: {'Yes' if args.quick else 'No'}")
        
        result = detector.analyze_pdf(args.pdf_path, quick_scan=args.quick)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        # Print results
        print(f"\n{'='*80}")
        print(f"ENHANCED PDF STEGANOGRAPHY ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Assessment: {result['assessment']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Raw Risk Score: {result['raw_risk_score']:.1f}")
        print(f"Adjusted Risk Score: {result['adjusted_risk_score']:.1f}")
        print(f"Total Indicators: {result['total_indicators']}")
        print(f"Analysis Time: {result['performance_metrics']['analysis_time']:.2f} seconds")
        
        # Validation summary
        validation = result['validation_summary']
        print(f"\nValidation Summary:")
        print(f"  Validated: {validation['total_validated']}")
        print(f"  Suspicious: {validation['total_suspicious']}")
        print(f"  Avg False Positive Probability: {validation['avg_false_positive_prob']:.1%}")
        
        if result['indicators_by_category']:
            print(f"\n{'='*50}")
            print("DETECTED INDICATORS BY CATEGORY:")
            print(f"{'='*50}")
            
            for category, indicators in result['indicators_by_category'].items():
                print(f"\n[{category.upper().replace('_', ' ')}]")
                for indicator in indicators:
                    print(f"  • {indicator['severity']}: {indicator['description']}")
                    print(f"    Confidence: {indicator['confidence']:.1%}")
                    print(f"    False Positive Probability: {indicator['false_positive_probability']:.1%}")
                    print(f"    Validation: {indicator['validation_status']}")
        
        # Analysis summaries
        if result['binary_analysis']['valid_pngs'] > 0:
            print(f"\n{'='*50}")
            print("BINARY ANALYSIS:")
            print(f"{'='*50}")
            binary = result['binary_analysis']
            print(f"PNG Signatures: {binary['png_count']}")
            print(f"Valid PNGs: {binary['valid_pngs']}")
            print(f"Steganography Patterns: {binary['stego_patterns']}")
        
        if result['statistical_analysis']['chi2_random'] is not None:
            print(f"\n{'='*50}")
            print("STATISTICAL ANALYSIS:")
            print(f"{'='*50}")
            stats = result['statistical_analysis']
            print(f"Chi-square randomness: {stats['chi2_random']}")
            print(f"Kolmogorov-Smirnov uniformity: {stats['ks_uniform']}")
            print(f"Autocorrelation patterns: {stats['autocorr_patterns']}")
        
        if result['ml_analysis']['ensemble_score'] is not None:
            print(f"\n{'='*50}")
            print("MACHINE LEARNING ANALYSIS:")
            print(f"{'='*50}")
            ml = result['ml_analysis']
            print(f"Ensemble Score: {ml['ensemble_score']:.3f}")
            print(f"Is Anomaly: {ml['is_anomaly']}")
            print(f"Confidence: {ml['confidence']:.1%}")
        
        if result['recommendations']:
            print(f"\n{'='*50}")
            print("RECOMMENDATIONS:")
            print(f"{'='*50}")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Save detailed report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
