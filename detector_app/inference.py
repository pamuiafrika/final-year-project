
# ==============================================
# 2. inference.py - Updated for Django
# ==============================================

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import logging
from typing import Dict, List, Optional
import PyPDF2
import pdfplumber
from pathlib import Path
import re
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

class DjangoPDFFeatureExtractor:
    """Django-compatible PDF Feature Extractor"""
    
    def __init__(self):
        self.feature_names = [
            'pdf_size', 'metadata_size', 'pages', 'xref_length', 'title', 'author',
            'subject', 'producer', 'creator', 'creation_date', 'modification_date',
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref',
            'catalog', 'font', 'fontname', 'page', 'parent', 'next', 'first',
            'last', 'count', 'kids', 'type', 'annot', 'AA', 'OpenAction',
            'JS', 'Javascript', 'launch', 'EmbeddedFile'
        ]
    
    def extract_features(self, file_path_or_obj) -> Dict:
        """Extract features from PDF file path or Django FieldFile"""
        features = {}
        
        try:
            # Handle Django FieldFile
            if hasattr(file_path_or_obj, 'path'):
                pdf_path = file_path_or_obj.path
                file_size = file_path_or_obj.size
            else:
                pdf_path = str(file_path_or_obj)
                file_size = os.path.getsize(pdf_path)
            
            features['pdf_size'] = file_size
            
            # Extract other features
            features.update(self._extract_structure_features(pdf_path))
            features.update(self._extract_metadata_features(pdf_path))
            features.update(self._extract_security_features(pdf_path))
            
            # Ensure all features are present
            for feature_name in self.feature_names:
                if feature_name not in features:
                    features[feature_name] = 0
                    
        except Exception as e:
            logger.warning(f"Error extracting features: {str(e)}")
            features = {name: 0 for name in self.feature_names}
            
        return features
    
    def _extract_structure_features(self, pdf_path: str) -> Dict:
        """Extract PDF structure features"""
        features = {}
        structure_keywords = [
            'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 
            'startxref', 'catalog', 'font', 'fontname', 'page', 'parent',
            'next', 'first', 'last', 'count', 'kids', 'type', 'annot'
        ]
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                features['pages'] = len(pdf_reader.pages)
                
                # Get metadata size
                metadata = pdf_reader.metadata
                if metadata:
                    features['metadata_size'] = len(str(metadata).encode('utf-8'))
                else:
                    features['metadata_size'] = 0
            
            # Parse raw content for structure keywords
            with open(pdf_path, 'rb') as file:
                content = file.read()
                content_str = content.decode('latin-1', errors='ignore').lower()
                
                for keyword in structure_keywords:
                    count = len(re.findall(rf'\b{keyword}\b', content_str))
                    features[keyword] = count
                
                # Handle xref_length
                xref_matches = re.findall(r'xref\s+(\d+)\s+(\d+)', content_str)
                features['xref_length'] = sum(int(match[1]) for match in xref_matches)
                
        except Exception as e:
            logger.warning(f"Error extracting structure features: {str(e)}")
            features.update({name: 0 for name in structure_keywords})
            features['pages'] = 0
            features['metadata_size'] = 0
            features['xref_length'] = 0
            
        return features
    
    def _extract_metadata_features(self, pdf_path: str) -> Dict:
        """Extract metadata features"""
        features = {
            'title': 0, 'author': 0, 'subject': 0, 'producer': 0,
            'creator': 0, 'creation_date': 0, 'modification_date': 0
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                if metadata:
                    features['title'] = 1 if metadata.get('/Title') else 0
                    features['author'] = 1 if metadata.get('/Author') else 0
                    features['subject'] = 1 if metadata.get('/Subject') else 0
                    features['producer'] = 1 if metadata.get('/Producer') else 0
                    features['creator'] = 1 if metadata.get('/Creator') else 0
                    features['creation_date'] = 1 if metadata.get('/CreationDate') else 0
                    features['modification_date'] = 1 if metadata.get('/ModDate') else 0
                    
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            
        return features
    
    def _extract_security_features(self, pdf_path: str) -> Dict:
        """Extract security features"""
        features = {}
        suspicious_keywords = ['AA', 'OpenAction', 'JS', 'Javascript', 'launch', 'EmbeddedFile']
        
        try:
            with open(pdf_path, 'rb') as file:
                content = file.read()
                content_str = content.decode('latin-1', errors='ignore')
                
                for keyword in suspicious_keywords:
                    count = len(re.findall(rf'{keyword}', content_str, re.IGNORECASE))
                    features[keyword] = count
                    
        except Exception as e:
            logger.warning(f"Error extracting security features: {str(e)}")
            features.update({name: 0 for name in suspicious_keywords})
            
        return features

class DjangoPDFMalwareDetector:
    """Django-integrated PDF Malware Detector"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'pdf_detector_ensemble')
        
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.feature_names = None
        self.feature_extractor = DjangoPDFFeatureExtractor()
        
        self._load_models()
    
    def _load_models(self):
        """Load models with caching"""
        cache_key = 'pdf_detector_models'
        cached_models = cache.get(cache_key)
        
        if cached_models:
            logger.info("Loading models from cache")
            self.__dict__.update(cached_models)
            return
        
        logger.info(f"Loading models from {self.model_path}")
        
        try:
            # Load metadata
            with open(f'{self.model_path}/metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            
            # Load preprocessors
            self.scaler = joblib.load(f'{self.model_path}/scaler.pkl')
            self.label_encoder = joblib.load(f'{self.model_path}/label_encoder.pkl')
            self.feature_selector = joblib.load(f'{self.model_path}/feature_selector.pkl')
            
            # Load models
            for model_name in metadata['model_names']:
                model_file = f'{self.model_path}/{model_name}_model.h5'
                if os.path.exists(model_file):
                    self.models[model_name] = keras.models.load_model(model_file)
            
            # Cache models for 1 hour
            cache_data = {
                'models': self.models,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names
            }
            cache.set(cache_key, cache_data, 3600)
            
            logger.info("Models loaded and cached successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        X_eng = X.copy()
        
        # Same feature engineering as training
        if 'pdf_size' in X_eng.columns and 'metadata_size' in X_eng.columns:
            X_eng['metadata_ratio'] = X_eng['metadata_size'] / (X_eng['pdf_size'] + 1)
        
        if 'pages' in X_eng.columns and 'pdf_size' in X_eng.columns:
            X_eng['size_per_page'] = X_eng['pdf_size'] / (X_eng['pages'] + 1)
        
        complexity_features = ['obj', 'endobj', 'stream', 'endstream', 'xref']
        if all(feat in X_eng.columns for feat in complexity_features):
            X_eng['complexity_score'] = X_eng[complexity_features].sum(axis=1)
        
        suspicious_features = ['JS', 'Javascript', 'AA', 'OpenAction', 'launch', 'EmbeddedFile']
        if all(feat in X_eng.columns for feat in suspicious_features):
            X_eng['suspicious_count'] = X_eng[suspicious_features].sum(axis=1)
        
        skewed_features = ['pdf_size', 'metadata_size', 'xref_length']
        for feature in skewed_features:
            if feature in X_eng.columns:
                X_eng[f'{feature}_log'] = np.log1p(X_eng[feature])
        
        return X_eng
    
    def predict(self, pdf_file) -> Dict:
        """Make prediction on PDF file"""
        logger.info("Starting PDF analysis")
        
        # Extract features
        features = self.feature_extractor.extract_features(pdf_file)
        
        # Preprocess
        df = pd.DataFrame([features])
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0
        
        df = df[self.feature_names]
        df_eng = self.feature_engineering(df)
        X_selected = self.feature_selector.transform(df_eng)
        X_scaled = self.scaler.transform(X_selected)
        
        # Make predictions
        individual_predictions = {}
        probabilities = []
        
        for name, model in self.models.items():
            if name == 'wide_deep':
                pred_proba = model.predict([X_scaled, X_scaled], verbose=0)[0][0]
            else:
                pred_proba = model.predict(X_scaled, verbose=0)[0][0]
            
            individual_predictions[name] = {
                'probability': float(pred_proba),
                'prediction': 'malicious' if pred_proba > 0.5 else 'benign'
            }
            probabilities.append(pred_proba)
        
        # Ensemble prediction
        weights = np.array([0.4, 0.35, 0.25])
        ensemble_prob = np.average(probabilities, weights=weights)
        
        return {
            'ensemble_probability': float(ensemble_prob),
            'is_malicious': bool(ensemble_prob > 0.5),
            'confidence': float(max(ensemble_prob, 1-ensemble_prob) * 100),
            'risk_level': self._get_risk_level(ensemble_prob),
            'individual_predictions': individual_predictions,
            'extracted_features': features
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level"""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.6:
            return "MEDIUM"
        elif probability >= 0.4:
            return "LOW"
        else:
            return "MINIMAL"
