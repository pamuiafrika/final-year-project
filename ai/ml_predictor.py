
# ============================================================================
# AI/ml_predictor.py
# ============================================================================

import os
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from django.conf import settings

class MLPredictor:
    """ML Predictor class for loading models and making predictions"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(settings.BASE_DIR, 'ai', 'models')
        
        self.models_dir = models_dir
        self.preprocessor = None
        self.xgb_model = None
        self.keras_model = None
        self.models_loaded = False
        
        # Load models on initialization
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessor"""
        try:
            # Load preprocessor
            preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
            
            # Load XGBoost model
            xgb_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
            
            # Load Keras model
            keras_path = os.path.join(self.models_dir, 'wide_deep_model.h5')
            if os.path.exists(keras_path):
                self.keras_model = keras.models.load_model(keras_path)
            
            if self.preprocessor and self.xgb_model and self.keras_model:
                self.models_loaded = True
                print("All models loaded successfully!")
            else:
                print("Warning: Some models could not be loaded")
                print(f"Preprocessor: {'✓' if self.preprocessor else '✗'}")
                print(f"XGBoost: {'✓' if self.xgb_model else '✗'}")
                print(f"Keras: {'✓' if self.keras_model else '✗'}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def predict(self, features_df):
        """Make predictions using both models"""
        if not self.models_loaded:
            raise ValueError("Models not loaded properly")
        
        # Prepare features (remove non-feature columns)
        feature_columns = [
            'file_size_bytes', 'pdf_version', 'num_pages', 'num_objects',
            'num_stream_objects', 'num_embedded_files', 'num_annotation_objects',
            'num_form_fields', 'creation_date_ts', 'mod_date_ts', 'creation_mod_date_diff',
            'avg_entropy_per_stream', 'max_entropy_per_stream', 'min_entropy_per_stream',
            'std_entropy_per_stream', 'num_streams_entropy_gt_threshold',
            'num_encrypted_streams', 'num_corrupted_objects', 'num_objects_with_random_markers',
            'has_broken_name_trees', 'num_suspicious_filters', 'has_javascript',
            'has_launch_actions', 'avg_file_size_per_page', 'compression_ratio',
            'num_eof_markers', 'extraction_success', 'extraction_time_ms', 'error_count'
        ]
        
        # Select only feature columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in features_df.columns]
        X = features_df[available_features].copy()
        
        # Handle missing columns by adding them with default values
        for col in feature_columns:
            if col not in X.columns:
                if col in ['has_broken_name_trees', 'has_javascript', 'has_launch_actions', 'extraction_success']:
                    X[col] = False
                else:
                    X[col] = 0
        
        # Reorder columns to match training data
        X = X[feature_columns]
        
        # Preprocess data
        X_processed = self.preprocessor.transform(X)
        
        # XGBoost predictions
        xgb_prob = self.xgb_model.predict_proba(X_processed)[:, 1]
        xgb_pred = (xgb_prob > 0.5).astype(int)
        
        # Keras predictions
        keras_prob = self.keras_model.predict(X_processed, verbose=0).flatten()
        keras_pred = (keras_prob > 0.5).astype(int)
        
        return {
            'xgboost_prediction': xgb_pred,
            'xgboost_probability': xgb_prob,
            'wide_deep_prediction': keras_pred,
            'wide_deep_probability': keras_prob
        }
    
    def predict_single(self, features_dict):
        """Make prediction for a single sample"""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        return self.predict(df)
