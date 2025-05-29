# detector.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from django.conf import settings
from .data_processor import DataProcessor
import fitz  # PyMuPDF
import io
from PIL import Image
import cv2
import xgboost as xgb
import tempfile
import shutil
import logging
import math
from sklearn.preprocessing import StandardScaler

class StegoPDFDetector:
    """Detector class for steganography in PDF files"""
    
    def __init__(self, model_path=None):
        """Initialize the detector with a model path"""
        self.model_path = model_path
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose logging
        
        # Add a handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        if self.model_path:
            self.load_model(self.model_path)
    
    def load_model(self, model_path):
        """Load a trained model from path"""
        self.logger.info(f"Loading model: {model_path}")
        try:
            if model_path.endswith('.h5'):
                self.model = load_model(model_path)
                self.model_type = 'keras'
                self.logger.info(f"Model input shape: {self.model.input_shape}")
                
                # Load model metadata which should include feature normalization parameters
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    self.metadata = joblib.load(metadata_path)
                    self.logger.info(f"Loaded metadata: {self.metadata}")
                    
                    # Check if feature scaler exists in metadata
                    if 'feature_scaler' in self.metadata:
                        self.feature_scaler = self.metadata['feature_scaler']
                        self.logger.info("Loaded feature scaler from metadata")
                    else:
                        self.feature_scaler = None
                        self.logger.warning("No feature scaler found in metadata")
                        
                    # Get expected feature shape from metadata
                    if 'feature_shape' in self.metadata:
                        self.expected_feature_shape = self.metadata['feature_shape']
                        self.logger.info(f"Expected feature shape: {self.expected_feature_shape}")
                    else:
                        # Infer from model input shape
                        if len(self.model.input_shape) == 3:  # LSTM/RNN
                            self.expected_feature_shape = (self.model.input_shape[1], self.model.input_shape[2])
                        else:  # Dense/CNN flattened input
                            self.expected_feature_shape = (self.model.input_shape[1],)
                        self.logger.info(f"Inferred feature shape from model: {self.expected_feature_shape}")
                else:
                    self.metadata = {}
                    self.feature_scaler = None
                    self.logger.warning(f"No metadata file found at {metadata_path}")
                    
                    # Infer expected feature shape from model input
                    if len(self.model.input_shape) == 3:  # LSTM/RNN
                        self.expected_feature_shape = (self.model.input_shape[1], self.model.input_shape[2])
                    else:  # Dense/CNN flattened input
                        self.expected_feature_shape = (self.model.input_shape[1],)
                    self.logger.info(f"Inferred feature shape from model: {self.expected_feature_shape}")
                    
            elif model_path.endswith('.pkl'):
                model = joblib.load(model_path)
                if isinstance(model, dict) and 'blend_model' in model:
                    self.model = model
                    self.model_type = 'ensemble'
                    self.xgb_model = joblib.load(model['xgb_model_path'])
                    self.cnn_model_path = model['cnn_model_path']
                    self.lstm_model_path = model['lstm_model_path']
                    
                    # Load feature scaler if available
                    if 'feature_scaler' in model:
                        self.feature_scaler = model['feature_scaler']
                        self.logger.info("Loaded feature scaler from ensemble model")
                    else:
                        self.feature_scaler = None
                        self.logger.warning("No feature scaler found in ensemble model")
                        
                    self.logger.info("Loaded ensemble model components")
                else:
                    self.model = model
                    self.model_type = 'xgboost'
                    
                    # Check if we have a feature scaler in the model
                    if hasattr(model, 'feature_scaler') or (isinstance(model, dict) and 'feature_scaler' in model):
                        self.feature_scaler = model.feature_scaler if hasattr(model, 'feature_scaler') else model['feature_scaler']
                        self.logger.info("Loaded feature scaler from XGBoost model")
                    else:
                        self.feature_scaler = None
                        self.logger.warning("No feature scaler found in XGBoost model")
                        
                    self.logger.info("Loaded XGBoost model")
            
            self.logger.info(f"Successfully loaded model: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def detect(self, pdf_path):
        """Detect if a PDF contains steganographic content"""
        self.logger.info(f"Starting detection for PDF: {pdf_path}")
        print(f"Starting detection for PDF: {pdf_path}")
        
        try:
            # Extract features with detailed logging
            self.logger.info("Beginning feature extraction")
            features, feature_extraction_details = self.data_processor.extract_features_from_file(pdf_path, return_details=True)
            
            if features is None:
                self.logger.error("Feature extraction failed")
                return {
                    'success': False,
                    'error': 'Feature extraction failed',
                    'is_stego': False,
                    'confidence': 0.0,
                    'details': feature_extraction_details if isinstance(feature_extraction_details, dict) else {}
                }
            
            self.logger.info(f"Extracted features shape: {features.shape}")
            self.logger.debug(f"Feature extraction details: {feature_extraction_details}")
            
            # Create a copy of the features to avoid modifying the original
            processed_features = features.copy()
            
            # Apply normalization if we have a feature scaler
            if self.feature_scaler is not None:
                self.logger.info("Applying stored feature scaler")
                try:
                    processed_features = self.feature_scaler.transform(processed_features.reshape(1, -1))
                    self.logger.info("Feature normalization applied successfully")
                except Exception as e:
                    self.logger.error(f"Error applying feature scaler: {str(e)}", exc_info=True)
                    # Fall back to simple normalization if scaler fails
                    self.logger.warning("Falling back to simple feature standardization")
                    # Ensure features are in 2D format for StandardScaler
                    processed_features = processed_features.reshape(1, -1)
                    scaler = StandardScaler()
                    processed_features = scaler.fit_transform(processed_features)
            else:
                self.logger.warning("No feature scaler available, using simple standardization")
                # Simple standardization as fallback
                # First ensure the array is 2D
                if processed_features.ndim == 1:
                    processed_features = processed_features.reshape(1, -1)
                    self.logger.info(f"Reshaped features to {processed_features.shape}")
                
                # Now apply standardization along the correct axis
                mean = np.mean(processed_features, axis=0, keepdims=True)
                std = np.std(processed_features, axis=0, keepdims=True) + 1e-10
                processed_features = (processed_features - mean) / std
            
            # Process based on model type
            if self.model_type == 'keras':
                self.logger.debug("Calling _predict_with_keras")
                prediction, confidence, prediction_details = self._predict_with_keras(processed_features)
            elif self.model_type == 'xgboost':
                prediction, confidence, prediction_details = self._predict_with_xgboost(processed_features)
            elif self.model_type == 'ensemble':
                prediction, confidence, prediction_details = self._predict_with_ensemble(processed_features)
            else:
                self.logger.error("Unknown model type")
                return {
                    'success': False,
                    'error': 'Unknown model type',
                    'is_stego': False,
                    'confidence': 0.0,
                    'details': {}
                }
            
            self.logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
            details = self._extract_detection_details(pdf_path)
            
            # Add the prediction details to the overall details
            details.update(prediction_details)
            
            return {
                'success': True,
                'is_stego': bool(prediction),
                'confidence': float(confidence),
                'details': details
            }
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'is_stego': False,
                'confidence': 0.0,
                'details': {}
            }
    
    def _predict_with_keras(self, features):
        """Make prediction using a Keras model (CNN or LSTM)"""
        self.logger.info("Predicting with Keras model")
        prediction_details = {}
        
        try:
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            feature_size = features.shape[1]
            self.logger.info(f"Raw features shape: {features.shape}")
            
            # Get expected input shape from the model
            input_shape = self.model.input_shape
            self.logger.info(f"Model input shape: {input_shape}")
            
            if len(input_shape) == 4:  # CNN
                # For CNN models, reshape as required
                expected_side_length = input_shape[1]
                expected_size = expected_side_length * expected_side_length
                
                # Log the transformation we're about to do
                self.logger.info(f"Preparing features for CNN model with expected size: {expected_size}")
                
                # Handle feature size mismatch
                features_processed = self._prepare_features_for_cnn(features, expected_side_length)
                
                # Validate feature size
                if features_processed.shape[1] != expected_side_length * expected_side_length:
                    self.logger.error(f"Feature size mismatch: got {features_processed.shape[1]}, expected {expected_size}")
                    raise ValueError(f"Feature size mismatch: got {features_processed.shape[1]}, expected {expected_size}")
                
                # Reshape features for CNN input
                features_reshaped = features_processed.reshape(-1, expected_side_length, expected_side_length, 1)
                prediction_details['reshape_method'] = f"CNN: {features.shape} → {features_reshaped.shape}"
                
            elif len(input_shape) == 3:  # LSTM
                # For LSTM models, properly reshape to (samples, time_steps, features)
                expected_time_steps = input_shape[1]
                expected_features_per_step = input_shape[2]
                expected_size = expected_time_steps * expected_features_per_step
                
                self.logger.info(f"Preparing features for LSTM model: need {expected_time_steps} time steps with {expected_features_per_step} features each")
                
                # Prepare features for LSTM - this maintains feature relationships
                features_reshaped = self._prepare_features_for_lstm(
                    features, 
                    expected_time_steps, 
                    expected_features_per_step
                )
                
                prediction_details['reshape_method'] = f"LSTM: {features.shape} → {features_reshaped.shape}"
                self.logger.info(f"Reshaped features for LSTM: {features_reshaped.shape}")
                
            else:  # Dense model or other
                expected_size = input_shape[1]
                self.logger.info(f"Preparing features for Dense model with expected size: {expected_size}")
                
                # Handle feature size mismatch
                if feature_size < expected_size:
                    padding = expected_size - feature_size
                    features_processed = np.pad(features, ((0, 0), (0, padding)), 'constant')
                    self.logger.info(f"Padded features from {feature_size} to {expected_size}")
                elif feature_size > expected_size:
                    features_processed = features[:, :expected_size]
                    self.logger.info(f"Truncated features from {feature_size} to {expected_size}")
                else:
                    features_processed = features
                    self.logger.info("Feature size matches expected size")
                
                features_reshaped = features_processed
                prediction_details['reshape_method'] = f"Dense: {features.shape} → {features_reshaped.shape}"
            
            # Make prediction with verbose logging
            self.logger.info(f"Final input shape for model: {features_reshaped.shape}")
            pred_probas = self.model.predict(features_reshaped, verbose=0)
            self.logger.info(f"Raw prediction output: {pred_probas}")
            
            # Store raw prediction probabilities
            prediction_details['raw_prediction'] = pred_probas.tolist()
            
            # Handle different output formats
            if len(pred_probas.shape) > 1 and pred_probas.shape[1] > 1:
                # Multi-class output
                prediction = np.argmax(pred_probas, axis=1)[0]
                confidence = float(pred_probas[0][prediction]*100)
                prediction_details['prediction_type'] = 'multi-class'
                prediction_details['class_probabilities'] = {
                    i: float(prob) for i, prob in enumerate(pred_probas[0])
                }
            else:
                # Binary output - ensure consistent handling
                pred_prob = float(pred_probas[0][0]) if len(pred_probas.shape) > 1 else float(pred_probas[0])
                prediction = 1 if pred_prob >= 0.5 else 0
                confidence = pred_prob if prediction == 1 else 1 - pred_prob
                prediction_details['prediction_type'] = 'binary'
                prediction_details['stego_probability'] = pred_prob
                
            self.logger.info(f"Returning prediction: {prediction}, confidence: {confidence}")
            return prediction, confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"Error in Keras prediction: {str(e)}", exc_info=True)
            raise
    
    def _prepare_features_for_cnn(self, features, side_length):
        """Prepare features for CNN input by preserving feature relationships"""
        feature_size = features.shape[1]
        expected_size = side_length * side_length
        
        if feature_size < expected_size:
            # Pad features to required size
            padding = expected_size - feature_size
            features_processed = np.pad(features, ((0, 0), (0, padding)), 'constant')
            self.logger.info(f"Padded features from {feature_size} to {expected_size}")
        elif feature_size > expected_size:
            # Use dimensionality reduction or selection of most important features
            # For simplicity, we'll truncate, but ideally you would use PCA or feature importance
            features_processed = features[:, :expected_size]
            self.logger.info(f"Truncated features from {feature_size} to {expected_size}")
        else:
            features_processed = features
            self.logger.info("Feature size matches expected size")
            
        return features_processed
    
    def _prepare_features_for_lstm(self, features, time_steps, features_per_step):
        """Prepare features for LSTM input by maintaining meaningful sequential relationships"""
        feature_size = features.shape[1]
        expected_size = time_steps * features_per_step
        
        # Create a properly sized feature array
        if feature_size < expected_size:
            # Pad with zeros at the end
            padding = expected_size - feature_size
            features_padded = np.pad(features, ((0, 0), (0, padding)), 'constant')
            self.logger.info(f"Padded features from {feature_size} to {expected_size}")
        elif feature_size > expected_size:
            # Keep the most significant features (ideally, this would be based on feature importance)
            features_padded = features[:, :expected_size]
            self.logger.info(f"Truncated features from {feature_size} to {expected_size}")
        else:
            features_padded = features
            self.logger.info("Feature size matches expected size")
        
        # Now reshape to (samples, time_steps, features_per_step)
        # This reshape maintains feature grouping rather than arbitrary splitting
        features_reshaped = features_padded.reshape(-1, time_steps, features_per_step)
        
        return features_reshaped
    
    def _predict_with_xgboost(self, features):
        """Make prediction using an XGBoost model"""
        self.logger.info("Predicting with XGBoost model")
        prediction_details = {}
        
        try:
            # Get feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                # Save top 10 feature importances for diagnostics
                feature_importances = self.model.feature_importances_
                top_indices = np.argsort(feature_importances)[-10:]
                top_importances = {int(idx): float(feature_importances[idx]) for idx in top_indices}
                prediction_details['top_feature_importances'] = top_importances
            
            # Make prediction
            prediction = int(self.model.predict(features)[0])
            pred_probas = self.model.predict_proba(features)[0]
            confidence = float(pred_probas[prediction])
            
            # Store probabilities
            prediction_details['raw_probabilities'] = pred_probas.tolist()
            prediction_details['stego_probability'] = float(pred_probas[1])  # Probability of being stego
            
            self.logger.info(f"XGBoost prediction: {prediction}, Confidence: {confidence}")
            return prediction, confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"Error in XGBoost prediction: {str(e)}", exc_info=True)
            raise
    
    def _predict_with_ensemble(self, features):
        """Make prediction using an ensemble model"""
        self.logger.info("Predicting with ensemble model")
        prediction_details = {}
        
        try:
            # XGBoost prediction
            xgb_pred_proba = self.xgb_model.predict_proba(features)[:, 1]
            prediction_details['xgb_stego_probability'] = float(xgb_pred_proba[0])
            
            # Create blended features
            blend_features = np.column_stack([xgb_pred_proba])
            
            # Final prediction
            prediction = int(self.model['blend_model'].predict(blend_features)[0])
            pred_probas = self.model['blend_model'].predict_proba(blend_features)[0]
            confidence = float(pred_probas[prediction])
            
            # Store ensemble details
            prediction_details['raw_ensemble_probabilities'] = pred_probas.tolist()
            prediction_details['stego_probability'] = float(pred_probas[1])  # Probability of being stego
            
            self.logger.info(f"Ensemble prediction: {prediction}, Confidence: {confidence}")
            return prediction, confidence, prediction_details
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {str(e)}", exc_info=True)
            raise
    
    def _extract_detection_details(self, pdf_path):
        """Extract details from the PDF for reporting"""
        self.logger.info(f"Extracting detection details for: {pdf_path}")
        details = {
            'filename': os.path.basename(pdf_path),
            'filesize': os.path.getsize(pdf_path),
            'potential_stego_locations': []
        }
        
        doc = None
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            details['num_pages'] = len(doc)
            
            # Check for JavaScript in the document
            has_js = False
            js_count = 0
            
            # Process each page
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    
                    # Process images on the page
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            
                            suspicious = False
                            reasons = []
                            
                            # Enhanced image property checks
                            if base_image.get("colorspace") == 3:  # RGB colorspace
                                suspicious = True
                                reasons.append("RGB colorspace")
                                
                            img_width = base_image.get("width", 0)
                            img_height = base_image.get("height", 0)
                            img_size = img_width * img_height
                            
                            # Check image dimensions
                            if img_size > 500000:
                                suspicious = True
                                reasons.append("Large image size")
                            
                            # Check for unusual compression ratio
                            img_bytes = len(base_image.get("image", b""))
                            if img_size > 0 and img_bytes > 0:
                                compression_ratio = img_size / img_bytes
                                if compression_ratio < 0.5:  # Unusually low compression (possible hidden data)
                                    suspicious = True
                                    reasons.append(f"Unusual compression ratio: {compression_ratio:.2f}")
                            
                            # Add entropy check for image data
                            try:
                                img_data = np.frombuffer(base_image.get("image", b""), dtype=np.uint8)
                                if len(img_data) > 0:
                                    hist = np.histogram(img_data, bins=256, range=(0, 256))[0]
                                    hist = hist / len(img_data)  # Normalize
                                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                                    
                                    # High entropy can indicate encrypted/hidden data
                                    if entropy > 7.5:  # Close to maximum entropy for 8-bit data
                                        suspicious = True
                                        reasons.append(f"High entropy: {entropy:.2f}")
                                    
                                    details.setdefault('image_entropies', []).append({
                                        'page': page_num + 1,
                                        'image': img_index + 1,
                                        'entropy': float(entropy)
                                    })
                            except Exception as e:
                                self.logger.warning(f"Error calculating image entropy: {str(e)}")
                                
                            if suspicious:
                                details['potential_stego_locations'].append({
                                    'page': page_num + 1,
                                    'image_index': img_index + 1,
                                    'reasons': reasons,
                                    'dimensions': f"{img_width}x{img_height}"
                                })
                        except Exception as e:
                            self.logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
                    
                    # More thorough check for JavaScript
                    try:
                        page_text = page.get_text()
                        js_indicators = ['/JavaScript', '/JS', 'function(', 'eval(']
                        for indicator in js_indicators:
                            if indicator in page_text:
                                has_js = True
                                js_count += 1
                                break
                    except Exception as e:
                        self.logger.warning(f"Error checking page text for JavaScript on page {page_num}: {str(e)}")
                    
                    # Check annotations more thoroughly
                    try:
                        annotations = page.annots()
                        if annotations:
                            for annot in annotations:
                                try:
                                    # Check annotation actions for JavaScript
                                    if annot.xref > 0:  # Valid xref
                                        action_key = doc.xref_get_key(annot.xref, "A/S")
                                        js_key = doc.xref_get_key(annot.xref, "A/JS")
                                        
                                        if (action_key and "/JavaScript" in action_key[1]) or (js_key is not None):
                                            has_js = True
                                            js_count += 1
                                            
                                            # Extract JavaScript content if possible
                                            if js_key:
                                                js_content = doc.xref_get_key(annot.xref, "A/JS")[1]
                                                details.setdefault('javascript_snippets', []).append({
                                                    'page': page_num + 1,
                                                    'type': 'annotation',
                                                    'content': js_content[:100] + '...' if len(js_content) > 100 else js_content
                                                })
                                except Exception as e:
                                    self.logger.warning(f"Error checking annotation on page {page_num}: {str(e)}")
                    except Exception as e:
                        self.logger.warning(f"Error processing annotations on page {page_num}: {str(e)}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing page {page_num}: {str(e)}")
            
            # Check document catalog for JavaScript and embedded files
            try:
                root = doc.pdf_catalog()
                if root and isinstance(root, dict):
                    # Check for JavaScript in Names dictionary
                    names_ref = root.get("Names")
                    if names_ref:
                        names_obj = doc.xref_object(doc.xref_get_key(0, "Root")[1])
                        if names_obj and isinstance(names_obj, dict) and "JavaScript" in str(names_obj):
                            has_js = True
                            js_count += 1
                            
                    # Check for embedded files
                    embedded_files = []
                    try:
                        embedded_files_count = 0
                        for i in range(1, doc.xref_length()):
                            if doc.xref_is_stream(i):
                                obj = doc.xref_object(i)
                                if obj and isinstance(obj, dict) and obj.get("Type") == "/EmbeddedFile":
                                    embedded_files_count += 1
                                    embedded_files.append({
                                        'xref': i,
                                        'size': doc.xref_length(i) if hasattr(doc, 'xref_length') else 0
                                    })
                        if embedded_files_count > 0:
                            details['embedded_files'] = {
                                'count': embedded_files_count,
                                'files': embedded_files[:10]  # Limit to first 10 for brevity
                            }
                    except Exception as e:
                        self.logger.warning(f"Error checking for embedded files: {str(e)}")
                        
            except Exception as e:
                self.logger.warning(f"Error checking document catalog: {str(e)}")
            
            details['has_javascript'] = has_js
            details['javascript_count'] = js_count
            
            self.logger.info("Successfully extracted detection details")
        except Exception as e:
            self.logger.error(f"Error extracting detection details: {str(e)}")
            details['extraction_error'] = str(e)
        finally:
            # Always close the document
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
                    
        return details
    
    def batch_detect(self, pdf_paths):
        """Run detection on multiple PDFs"""
        self.logger.info(f"Starting batch detection for {len(pdf_paths)} PDFs")
        results = []
        for pdf_path in pdf_paths:
            result = self.detect(pdf_path)
            result['pdf_path'] = pdf_path
            results.append(result)
        self.logger.info("Batch detection completed")
        return results
      
    def extract_hidden_data(self, pdf_path, output_dir=None):
        """Attempt to extract hidden PNG images from a PDF"""
        self.logger.info(f"Extracting hidden data from: {pdf_path}")
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        extraction_results = {
            'success': False,
            'extracted_files': [],
            'output_dir': output_dir
        }
        try:
            detection = self.detect(pdf_path)
            if not detection['success'] or not detection['is_stego']:
                self.logger.warning("No steganography detected in this PDF")
                extraction_results['message'] = "No steganography detected in this PDF"
                return extraction_results
            doc = fitz.open(pdf_path)
            extracted_count = 0
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is None:
                            self.logger.warning(f"Failed to decode image {img_index} on page {page_num}")
                            continue
                        extracted_png = self._extract_png_from_lsb(img)
                        if extracted_png is not None:
                            output_file = os.path.join(output_dir, f"extracted_{page_num+1}_{img_index+1}.png")
                            cv2.imwrite(output_file, extracted_png)
                            extracted_count += 1
                            extraction_results['extracted_files'].append(output_file)
                            self.logger.info(f"Extracted PNG: {output_file}")
                    except Exception as e:
                        self.logger.warning(f"Error processing image {img_index} on page {page_num}: {str(e)}")
            doc.close()
            if extracted_count > 0:
                extraction_results['success'] = True
                extraction_results['message'] = f"Successfully extracted {extracted_count} hidden images"
            else:
                extraction_results['message'] = "No hidden data could be extracted"
            self.logger.info(f"Hidden data extraction completed: {extraction_results['message']}")
            return extraction_results
        except Exception as e:
            self.logger.error(f"Error extracting hidden data: {str(e)}")
            extraction_results['error'] = str(e)
            return extraction_results
    
    def _extract_png_from_lsb(self, img):
        """Extract PNG from least significant bits of image"""
        self.logger.info("Attempting to extract PNG from LSB")
        try:
            height, width = img.shape[:2]
            blue, green, red = cv2.split(img)
            max_bytes = width * height // 8
            buffer = np.zeros(max_bytes, dtype=np.uint8)
            bit_index = 0
            byte_index = 0
            for y in range(height):
                for x in range(width):
                    if byte_index >= max_bytes:
                        break
                    if bit_index < 8:
                        buffer[byte_index] |= ((blue[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                    if bit_index < 8:
                        buffer[byte_index] |= ((green[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                    if bit_index < 8:
                        buffer[byte_index] |= ((red[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                    if bit_index >= 8:
                        bit_index = 0
                        byte_index += 1
                if byte_index >= max_bytes:
                    break
            png_signature = np.array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], dtype=np.uint8)
            signature_found = False
            png_start_index = -1
            for i in range(min(1000, len(buffer) - 8)):
                if np.array_equal(buffer[i:i+8], png_signature):
                    signature_found = True
                    png_start_index = i
                    self.logger.info(f"PNG signature found at byte offset {i}")
                    break
            if not signature_found:
                self.logger.info("No PNG signature found")
                return None
            if png_start_index + 24 < len(buffer):
                chunk_size = int.from_bytes(buffer[png_start_index+8:png_start_index+12].tobytes(), byteorder='big')
                chunk_type = buffer[png_start_index+12:png_start_index+16].tobytes().decode('ascii', errors='ignore')
                if chunk_type == "IHDR" and chunk_size == 13:
                    width_bytes = buffer[png_start_index+16:png_start_index+20].tobytes()
                    height_bytes = buffer[png_start_index+20:png_start_index+24].tobytes()
                    hidden_width = int.from_bytes(width_bytes, byteorder='big')
                    hidden_height = int.from_bytes(height_bytes, byteorder='big')
                    if 0 < hidden_width <= 2000 and 0 < hidden_height <= 2000:
                        estimated_size = min(png_start_index + hidden_width * hidden_height * 3 + 1000, len(buffer))
                        end_marker = np.array([0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82], dtype=np.uint8)
                        iend_pos = -1
                        for i in range(png_start_index + 24, min(png_start_index + estimated_size, len(buffer) - 8)):
                            if np.array_equal(buffer[i:i+8], end_marker):
                                iend_pos = i + 8
                                break
                        png_data = buffer[png_start_index:iend_pos if iend_pos > 0 else png_start_index + estimated_size].tobytes()
                        img_array = np.frombuffer(png_data, dtype=np.uint8)
                        extracted_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if extracted_img is not None:
                            self.logger.info("Successfully decoded PNG")
                            return extracted_img
                        else:
                            self.logger.warning("Failed to decode PNG data")
            self.logger.info("No valid PNG could be extracted")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting PNG from LSB: {str(e)}")
            return None
    
    def get_available_models(self):
        """Get a list of available trained models"""
        self.logger.info("Retrieving available models")
        models = []
        try:
            model_dir = settings.ML_MODEL_DIR
            cnn_models = [f for f in os.listdir(model_dir) if f.startswith('cnn_model_') and f.endswith('.h5')]
            for model_file in cnn_models:
                model_path = os.path.join(model_dir, model_file)
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    try:
                        metadata = joblib.load(metadata_path)
                        models.append({
                            'id': os.path.splitext(model_file)[0],
                            'path': model_path,
                            'type': 'CNN',
                            'accuracy': metadata.get('accuracy', 0.0),
                            'timestamp': metadata.get('timestamp', 0)
                        })
                        self.logger.info(f"Found CNN model: {model_file}")
                    except Exception as e:
                        self.logger.warning(f"Error loading CNN metadata for {model_file}: {str(e)}")
            lstm_models = [f for f in os.listdir(model_dir) if f.startswith('lstm_model_') and f.endswith('.h5')]
            for model_file in lstm_models:
                model_path = os.path.join(model_dir, model_file)
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    try:
                        metadata = joblib.load(metadata_path)
                        models.append({
                            'id': os.path.splitext(model_file)[0],
                            'path': model_path,
                            'type': 'LSTM',
                            'accuracy': metadata.get('accuracy', 0.0),
                            'timestamp': metadata.get('timestamp', 0)
                        })
                        self.logger.info(f"Found LSTM model: {model_file}")
                    except Exception as e:
                        self.logger.warning(f"Error loading LSTM metadata for {model_file}: {str(e)}")
            xgb_models = [f for f in os.listdir(model_dir) if f.startswith('xgboost_model_') and f.endswith('.pkl')]
            for model_file in xgb_models:
                model_path = os.path.join(model_dir, model_file)
                try:
                    model = joblib.load(model_path)
                    accuracy = getattr(model, 'best_score_', 0.0)
                    models.append({
                        'id': os.path.splitext(model_file)[0],
                        'path': model_path,
                        'type': 'XGBoost',
                        'accuracy': accuracy,
                        'timestamp': int(model_file.split('_')[-1].split('.')[0])
                    })
                    self.logger.info(f"Found XGBoost model: {model_file}")
                except Exception as e:
                    self.logger.warning(f"Error loading XGBoost model {model_file}: {str(e)}")
            ensemble_models = [f for f in os.listdir(model_dir) if f.startswith('ensemble_model_') and f.endswith('.pkl')]
            for model_file in ensemble_models:
                model_path = os.path.join(model_dir, model_file)
                try:
                    model = joblib.load(model_path)
                    models.append({
                        'id': os.path.splitext(model_file)[0],
                        'path': model_path,
                        'type': 'Ensemble',
                        'accuracy': model.get('accuracy', 0.0),
                        'timestamp': int(model_file.split('_')[-1].split('.')[0])
                    })
                    self.logger.info(f"Found Ensemble model: {model_file}")
                except Exception as e:
                    self.logger.warning(f"Error loading Ensemble model {model_file}: {str(e)}")
            models.sort(key=lambda x: x['accuracy'], reverse=True)
            self.logger.info(f"Found {len(models)} available models")
        except Exception as e:
            self.logger.error(f"Error getting available models: {str(e)}")
        return models