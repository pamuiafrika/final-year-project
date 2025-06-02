#!/usr/bin/env python3
"""
PDF Malware Detection Inference Script
Uses the trained ensemble model to classify PDF files
Extracts features from PDF files and makes predictions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional
import PyPDF2
import pdfplumber
from pathlib import Path
import re
import hashlib
import struct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFFeatureExtractor:
    """Extract features from PDF files for malware detection"""
    
    def __init__(self):
        self.feature_names = [
            'pdf_size', 'metadata_size', 'pages', 'xref_length', 'title', 'author',
            'subject', 'producer', 'creator', 'creation_date', 'modification_date',
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
            'catalog', 'font', 'fontname', 'page', 'parent', 'next', 'first',
            'last', 'count', 'kids', 'type', 'annot', 'AA', 'OpenAction',
            'JS', 'Javascript', 'launch', 'EmbeddedFile'
        ]
    
    def extract_features(self, pdf_path: str) -> Dict:
        """Extract all features from a PDF file"""
        features = {}
        
        try:
            # Basic file information
            features.update(self._extract_basic_info(pdf_path))
            
            # PDF structure features
            features.update(self._extract_structure_features(pdf_path))
            
            # Metadata features
            features.update(self._extract_metadata_features(pdf_path))
            
            # Security and suspicious features
            features.update(self._extract_security_features(pdf_path))
            
            # Ensure all required features are present
            for feature_name in self.feature_names:
                if feature_name not in features:
                    features[feature_name] = 0
                    
        except Exception as e:
            logger.warning(f"Error extracting features from {pdf_path}: {str(e)}")
            # Return default features if extraction fails
            features = {name: 0 for name in self.feature_names}
            
        return features
    
    def _extract_basic_info(self, pdf_path: str) -> Dict:
        """Extract basic file information"""
        features = {}
        
        # File size
        features['pdf_size'] = os.path.getsize(pdf_path)
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                features['pages'] = len(pdf_reader.pages)
                
                # Try to get metadata size
                metadata = pdf_reader.metadata
                if metadata:
                    metadata_str = str(metadata)
                    features['metadata_size'] = len(metadata_str.encode('utf-8'))
                else:
                    features['metadata_size'] = 0
                    
        except Exception as e:
            logger.warning(f"Error reading PDF basic info: {str(e)}")
            features['pages'] = 0
            features['metadata_size'] = 0
            
        return features
    
    def _extract_structure_features(self, pdf_path: str) -> Dict:
        """Extract PDF structure-related features by parsing raw content"""
        features = {}
        structure_keywords = [
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 
            'startxref', 'catalog', 'font', 'fontname', 'page', 'parent',
            'next', 'first', 'last', 'count', 'kids', 'type', 'annot'
        ]
        
        try:
            with open(pdf_path, 'rb') as file:
                content = file.read()
                content_str = content.decode('latin-1', errors='ignore').lower()
                
                for keyword in structure_keywords:
                    # Count occurrences of each keyword
                    count = len(re.findall(rf'\b{keyword}\b', content_str))
                    features[keyword] = count
                
                # Special handling for xref_length
                xref_matches = re.findall(r'xref\s+(\d+)\s+(\d+)', content_str)
                features['xref_length'] = sum(int(match[1]) for match in xref_matches)
                
        except Exception as e:
            logger.warning(f"Error extracting structure features: {str(e)}")
            for keyword in structure_keywords:
                features[keyword] = 0
            features['xref_length'] = 0
            
        return features
    
    def _extract_metadata_features(self, pdf_path: str) -> Dict:
        """Extract metadata-related features"""
        features = {
            'title': 0, 'author': 0, 'subject': 0, 'producer': 0,
            'creator': 0, 'creation_date': 0, 'modification_date': 0
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                if metadata:
                    # Check presence of metadata fields (binary: 1 if present, 0 if not)
                    features['title'] = 1 if metadata.get('/Title') else 0
                    features['author'] = 1 if metadata.get('/Author') else 0
                    features['subject'] = 1 if metadata.get('/Subject') else 0
                    features['producer'] = 1 if metadata.get('/Producer') else 0
                    features['creator'] = 1 if metadata.get('/Creator') else 0
                    features['creation_date'] = 1 if metadata.get('/CreationDate') else 0
                    features['modification_date'] = 1 if metadata.get('/ModDate') else 0
                    
        except Exception as e:
            logger.warning(f"Error extracting metadata features: {str(e)}")
            
        return features
    
    def _extract_security_features(self, pdf_path: str) -> Dict:
        """Extract security and potentially malicious features"""
        features = {}
        suspicious_keywords = [
            'AA', 'OpenAction', 'JS', 'Javascript', 'launch', 'EmbeddedFile'
        ]
        
        try:
            with open(pdf_path, 'rb') as file:
                content = file.read()
                content_str = content.decode('latin-1', errors='ignore')
                
                for keyword in suspicious_keywords:
                    # Count occurrences (case-insensitive)
                    count = len(re.findall(rf'{keyword}', content_str, re.IGNORECASE))
                    features[keyword] = count
                    
        except Exception as e:
            logger.warning(f"Error extracting security features: {str(e)}")
            for keyword in suspicious_keywords:
                features[keyword] = 0
                
        return features

class PDFMalwareInference:
    """PDF Malware Detection Inference System"""
    
    def __init__(self, model_path: str = 'pdf_detector_ensemble'):
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.feature_names = None
        self.metadata = None
        self.feature_extractor = PDFFeatureExtractor()
        
        self._load_model_components()
    
    def _load_model_components(self):
        """Load all model components"""
        logger.info(f"Loading model components from {self.model_path}...")
        
        try:
            # Load metadata
            with open(f'{self.model_path}/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            
            # Load preprocessors
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoder = joblib.load(f'{self.model_path}/label_encoder.pkl')
            self.feature_selector = joblib.load(f'{self.model_path}/feature_selector.pkl')
            
            # Load models
            for model_name in self.metadata['model_names']:
                model_file = f'{self.model_path}/{model_name}_model.h5'
                if os.path.exists(model_file):
                    self.models[model_name] = keras.models.load_model(model_file)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_file}")
            
            logger.info("All model components loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as during training"""
        X_eng = X.copy()
        
        # Create ratio features
        if 'pdf_size' in X_eng.columns and 'metadata_size' in X_eng.columns:
            X_eng['metadata_ratio'] = X_eng['metadata_size'] / (X_eng['pdf_size'] + 1)
        
        if 'pages' in X_eng.columns and 'pdf_size' in X_eng.columns:
            X_eng['size_per_page'] = X_eng['pdf_size'] / (X_eng['pages'] + 1)
        
        # Create complexity score
        complexity_features = ['obj', 'endobj', 'stream', 'endstream', 'xref']
        if all(feat in X_eng.columns for feat in complexity_features):
            X_eng['complexity_score'] = X_eng[complexity_features].sum(axis=1)
        
        # Create suspicious features count
        suspicious_features = ['JS', 'Javascript', 'AA', 'OpenAction', 'launch', 'EmbeddedFile']
        if all(feat in X_eng.columns for feat in suspicious_features):
            X_eng['suspicious_count'] = X_eng[suspicious_features].sum(axis=1)
        
        # Log transformations for skewed features
        skewed_features = ['pdf_size', 'metadata_size', 'xref_length']
        for feature in skewed_features:
            if feature in X_eng.columns:
                X_eng[f'{feature}_log'] = np.log1p(X_eng[feature])
        
        return X_eng
    
    def preprocess_features(self, features_dict: Dict) -> np.ndarray:
        """Preprocess extracted features for prediction"""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Ensure all training features are present
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0
        
        # Select only the features used during training
        df = df[self.feature_names]
        
        # Apply feature engineering
        df_eng = self.feature_engineering(df)
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(df_eng)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def predict_single_file(self, pdf_path: str) -> Dict:
        """Predict whether a single PDF file is malicious"""
        logger.info(f"Analyzing PDF: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract features
        features = self.feature_extractor.extract_features(pdf_path)
        
        # Preprocess features
        X_processed = self.preprocess_features(features)
        
        # Make predictions with each model
        individual_predictions = {}
        for name, model in self.models.items():
            if name == 'wide_deep':
                pred_proba = model.predict([X_processed, X_processed], verbose=0)[0][0]
            else:
                pred_proba = model.predict(X_processed, verbose=0)[0][0]
            
            individual_predictions[name] = {
                'probability': float(pred_proba),
                'prediction': 'malicious' if pred_proba > 0.5 else 'benign'
            }
        
        # Ensemble prediction (weighted average)
        weights = np.array([0.4, 0.35, 0.25])  # attention, deep_ff, wide_deep
        model_probabilities = [individual_predictions[name]['probability'] 
                              for name in ['attention', 'deep_ff', 'wide_deep']]
        ensemble_prob = np.average(model_probabilities, weights=weights)
        
        # Prepare result
        result = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'ensemble_prediction': {
                'probability': float(ensemble_prob),
                'prediction': 'malicious' if ensemble_prob > 0.5 else 'benign',
                'confidence': f"{max(ensemble_prob, 1-ensemble_prob)*100:.2f}%"
            },
            'individual_models': individual_predictions,
            'extracted_features': features,
            'risk_level': self._get_risk_level(ensemble_prob)
        }
        
        return result
    
    def predict_batch(self, pdf_paths: List[str]) -> List[Dict]:
        """Predict multiple PDF files"""
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.predict_single_file(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                results.append({
                    'file_path': pdf_path,
                    'file_name': os.path.basename(pdf_path),
                    'error': str(e)
                })
        
        return results
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.6:
            return "MEDIUM"
        elif probability >= 0.4:
            return "LOW"
        else:
            return "MINIMAL"
    
    def print_detailed_result(self, result: Dict):
        """Print detailed analysis result"""
        print("\n" + "="*60)
        print("PDF MALWARE DETECTION RESULT")
        print("="*60)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"File: {result['file_name']}")
        print(f"Path: {result['file_path']}")
        print(f"\nENSEMBLE PREDICTION:")
        print(f"  Classification: {result['ensemble_prediction']['prediction'].upper()}")
        print(f"  Probability: {result['ensemble_prediction']['probability']:.4f}")
        print(f"  Confidence: {result['ensemble_prediction']['confidence']}")
        print(f"  Risk Level: {result['risk_level']}")
        
        print(f"\nINDIVIDUAL MODEL PREDICTIONS:")
        for model_name, pred in result['individual_models'].items():
            print(f"  {model_name.title()}: {pred['prediction']} ({pred['probability']:.4f})")
        
        print(f"\nKEY FEATURES EXTRACTED:")
        features = result['extracted_features']
        print(f"  File Size: {features['pdf_size']:,} bytes")
        print(f"  Pages: {features['pages']}")
        print(f"  Suspicious Count: {features.get('suspicious_count', 0)}")
        print(f"  JavaScript Elements: {features['JS'] + features['Javascript']}")
        print(f"  Auto Actions: {features['AA'] + features['OpenAction']}")
        print(f"  Embedded Files: {features['EmbeddedFile']}")
        
        print("="*60)

def main():
    """Main function for testing the inference system"""
    # Initialize inference system
    try:
        detector = PDFMalwareInference()
        print("PDF Malware Detection System loaded successfully!")
        
        # Example usage - replace with actual PDF file paths
        pdf_files = [
            "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/stego/tech_13_embedded.pdf",
            "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/stego/ml_34_embedded.pdf", 
            "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/BIOLOGY- 2.docx (1).pdf"
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/stego/stego2.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/stego/Stego_JS_Embedded.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/stego/output_837fd0927db64ba3aae1c036662d7c4f.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/CHEMISTRY-2 - WazaElimu.com (1).pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/clean/TN 412 DSP Proakis & Manolakis book_23_24.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/claude/enc/com/application-form-for-admission-into-seap_stego.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/claude/enc/meta/BARUA 2_stego.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/claude/multi/TEST 1 2 3_stego.pdf',
            '/home/d3bugger/Projects/FINAL YEAR PROJECT/src/datasets/claude/multi/TN MAJIBU_stego.pdf'
        ]
        
        # Uncomment and modify the following lines to test with your PDF files
        
        # Single file prediction
        # result = detector.predict_single_file("/path/to/your/pdf/file.pdf")
        # detector.print_detailed_result(result)
        
        # Batch prediction
        results = detector.predict_batch(pdf_files)
        for result in results:
            detector.print_detailed_result(result)
        
        
        print("\nTo use this script:")
        print("1. Replace model path if different from 'pdf_detector_ensemble'")
        print("2. Call predict_single_file() with your PDF file path")
        print("3. Use predict_batch() for multiple files")
        
    except Exception as e:
        logger.error(f"Failed to initialize detector: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())