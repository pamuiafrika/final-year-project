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

class StegoPDFDetector:
    """Detector class for steganography in PDF files"""
    
    def __init__(self, model_path=None):
        """Initialize the detector with a model path"""
        self.model_path = model_path
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Load model if provided
        if self.model_path:
            self.load_model(self.model_path)
    
    def load_model(self, model_path):
        """Load a trained model from path"""
        try:
            # Determine model type based on file extension
            if model_path.endswith('.h5'):
                # Keras model (CNN or LSTM)
                self.model = load_model(model_path)
                self.model_type = 'keras'
                
                # Try to load metadata
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    self.metadata = joblib.load(metadata_path)
                else:
                    # Default metadata if not found
                    self.metadata = {}
                
            elif model_path.endswith('.pkl'):
                # Check if it's an ensemble model
                model = joblib.load(model_path)
                
                if isinstance(model, dict) and 'blend_model' in model:
                    # It's an ensemble model
                    self.model = model
                    self.model_type = 'ensemble'
                    
                    # Load component models
                    self.xgb_model = joblib.load(model['xgb_model_path'])
                    
                    # CNN and LSTM models will be loaded on demand to save memory
                    self.cnn_model_path = model['cnn_model_path']
                    self.lstm_model_path = model['lstm_model_path']
                    
                else:
                    # Regular XGBoost model
                    self.model = model
                    self.model_type = 'xgboost'
            
            self.logger.info(f"Successfully loaded model: {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def detect(self, pdf_path):
        """Detect if a PDF contains steganographic content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict: Detection results with confidence score and details
        """
        try:
            # Extract features
            features = self.data_processor.extract_features_from_file(pdf_path)
            
            if features is None:
                return {
                    'success': False,
                    'error': 'Feature extraction failed',
                    'is_stego': False,
                    'confidence': 0.0,
                    'details': {}
                }
            
            # Make prediction based on model type
            if self.model_type == 'keras':
                # Handle CNN or LSTM model
                prediction, confidence = self._predict_with_keras(features)
            
            elif self.model_type == 'xgboost':
                # Handle XGBoost model
                prediction, confidence = self._predict_with_xgboost(features)
            
            elif self.model_type == 'ensemble':
                # Handle ensemble model
                prediction, confidence = self._predict_with_ensemble(features)
            
            else:
                return {
                    'success': False,
                    'error': 'Unknown model type',
                    'is_stego': False,
                    'confidence': 0.0,
                    'details': {}
                }
            
            # Extract additional details from the PDF for reporting
            details = self._extract_detection_details(pdf_path)
            
            return {
                'success': True,
                'is_stego': bool(prediction),
                'confidence': float(confidence),
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'is_stego': False,
                'confidence': 0.0,
                'details': {}
            }
    
    def _predict_with_keras_fixed(self, features):
        """Make prediction using a Keras model (CNN or LSTM) - Fixed to handle size mismatches"""
        # Get metadata
        side_length = self.metadata.get('side_length', int(np.sqrt(features.shape[1])))
        sequence_length = self.metadata.get('sequence_length', 100)
        n_features = self.metadata.get('n_features', features.shape[1] // sequence_length)
        
        # Log the shape information for debugging
        self.logger.info(f"Feature shape: {features.shape}, Expected side length: {side_length}")
        
        # Prepare features based on model architecture
        if 'padding' in self.metadata:
            # It's a CNN model
            # Calculate the expected size for the CNN model
            expected_size = side_length * side_length
            feature_size = features.shape[1]
            
            # Handle size mismatch by either padding or truncating
            if feature_size < expected_size:
                # Need to pad
                padding = expected_size - feature_size
                self.logger.info(f"Padding features from size {feature_size} to {expected_size}")
                features_processed = np.pad(features, ((0, 0), (0, padding)), 'constant')
            elif feature_size > expected_size:
                # Need to truncate
                self.logger.info(f"Truncating features from size {feature_size} to {expected_size}")
                features_processed = features[:, :expected_size]
            else:
                features_processed = features
                
            # Reshape to match expected dimensions
            features_reshaped = features_processed.reshape(-1, side_length, side_length, 1)
        
        elif 'sequence_length' in self.metadata:
            # It's an LSTM model
            feature_size = features.shape[1]
            if feature_size % sequence_length != 0:
                padding = sequence_length - (feature_size % sequence_length)
                features_padded = np.pad(features, ((0, 0), (0, padding)), 'constant')
                n_features = (feature_size + padding) // sequence_length
            else:
                features_padded = features
            features_reshaped = features_padded.reshape(-1, sequence_length, n_features)
        
        else:
            # Assume CNN if metadata is incomplete
            feature_size = features.shape[1]
            side_length = int(np.sqrt(feature_size))
            if side_length * side_length != feature_size:
                # Adjust side_length to ensure perfect square
                side_length = int(np.sqrt(feature_size)) + 1
                expected_size = side_length * side_length
                padding = expected_size - feature_size
                self.logger.info(f"Auto-adjusting side length to {side_length}, padding {padding} zeros")
                features_padded = np.pad(features, ((0, 0), (0, padding)), 'constant')
            else:
                features_padded = features
            features_reshaped = features_padded.reshape(-1, side_length, side_length, 1)
        
        # Log the reshaped dimensions
        self.logger.info(f"Reshaped features to: {features_reshaped.shape}")
        
        # Make prediction
        pred_probas = self.model.predict(features_reshaped)
        prediction = np.argmax(pred_probas, axis=1)[0]
        confidence = pred_probas[0][prediction]
        
        return prediction, confidence
    
    
    def _predict_with_xgboost(self, features):
        """Make prediction using an XGBoost model"""
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get confidence score
        pred_probas = self.model.predict_proba(features)[0]
        confidence = pred_probas[int(prediction)]
        
        return prediction, confidence
    
    def _predict_with_ensemble(self, features):
        """Make prediction using an ensemble model"""
        # Get XGBoost prediction probability
        xgb_pred_proba = self.xgb_model.predict_proba(features)[:, 1]
        
        # Create blend features
        blend_features = np.column_stack([xgb_pred_proba])
        
        # Make final prediction using the blend model
        prediction = self.model['blend_model'].predict(blend_features)[0]
        
        # Get confidence score
        pred_probas = self.model['blend_model'].predict_proba(blend_features)[0]
        confidence = pred_probas[int(prediction)]
        
        return prediction, confidence
    
    def _extract_detection_details(self, pdf_path):
        """Extract details from the PDF for reporting"""
        details = {
            'filename': os.path.basename(pdf_path),
            'filesize': os.path.getsize(pdf_path),
            'potential_stego_locations': []
        }
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract basic details
            details['num_pages'] = len(doc)
            
            # Look for potential steganographic indicators
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Look for images that might contain steganography
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        # Check image properties that might indicate steganography
                        suspicious = False
                        reasons = []
                        
                        # Example of suspicious indicators (customize based on knowledge)
                        if base_image["colorspace"] == 3:  # RGB images are more common for steganography
                            suspicious = True
                            reasons.append("RGB colorspace")
                        
                        if base_image.get("height", 0) * base_image.get("width", 0) > 500000:
                            # Large images might be used to hide more data
                            suspicious = True
                            reasons.append("Large image size")
                        
                        # Add to potential locations if suspicious
                        if suspicious:
                            details['potential_stego_locations'].append({
                                'page': page_num + 1,
                                'image_index': img_index + 1,
                                'reasons': reasons
                            })
                    except Exception as e:
                        continue
            
            # Check for JavaScript (sometimes used in PDF steganography)
            has_js = False
            js_count = 0
            
            for i in range(len(doc)):
                page = doc[i]
                if '/JavaScript' in page or '/JS' in page:
                    has_js = True
                    js_count += 1
            
            details['has_javascript'] = has_js
            details['javascript_count'] = js_count
            
            doc.close()
            
        except Exception as e:
            details['extraction_error'] = str(e)
        
        return details
    
    def batch_detect(self, pdf_paths):
        """Run detection on multiple PDFs
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            list: Results for each PDF
        """
        results = []
        
        for pdf_path in pdf_paths:
            result = self.detect(pdf_path)
            result['pdf_path'] = pdf_path
            results.append(result)
        
        return results
    
    def extract_hidden_data(self, pdf_path, output_dir=None):
        """Attempt to extract hidden PNG images from a PDF identified as containing steganography
        
        Args:
            pdf_path: Path to the stego PDF
            output_dir: Directory to save extracted images (optional)
            
        Returns:
            dict: Results of extraction attempt
        """
        if output_dir is None:
            # Create temporary directory
            output_dir = tempfile.mkdtemp()
        
        extraction_results = {
            'success': False,
            'extracted_files': [],
            'output_dir': output_dir
        }
        
        try:
            # Detect if this PDF contains steganography
            detection = self.detect(pdf_path)
            
            if not detection['success'] or not detection['is_stego']:
                extraction_results['message'] = "No steganography detected in this PDF"
                return extraction_results
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract and analyze images
            extracted_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to numpy array for analysis
                        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            continue
                        
                        # Check for potential PNG signatures hidden in the image
                        # This is a simplified approach - real steganography extraction would be more complex
                        # and would need to implement the specific steganography algorithm used
                        
                        # Example implementation: check for PNG signature in LSB
                        extracted_png = self._extract_png_from_lsb(img)
                        
                        if extracted_png is not None:
                            # Save extracted PNG
                            output_file = os.path.join(output_dir, f"extracted_{page_num+1}_{img_index+1}.png")
                            cv2.imwrite(output_file, extracted_png)
                            extracted_count += 1
                            extraction_results['extracted_files'].append(output_file)
                    except Exception as e:
                        continue
            
            doc.close()
            
            if extracted_count > 0:
                extraction_results['success'] = True
                extraction_results['message'] = f"Successfully extracted {extracted_count} hidden images"
            else:
                extraction_results['message'] = "No hidden data could be extracted"
            
            return extraction_results
            
        except Exception as e:
            extraction_results['error'] = str(e)
            return extraction_results
        
    
    def _extract_png_from_lsb(self, img):
        """Extract PNG from least significant bits of image
        
        This implementation extracts hidden PNG data using LSB steganography techniques.
        It can detect and extract PNGs hidden in the least significant bits of image color channels.
        
        Args:
            img: OpenCV image (numpy array)
                
        Returns:
            extracted_img or None if no PNG found
        """
        try:
            height, width = img.shape[:2]
            
            # Split the image into color channels
            blue, green, red = cv2.split(img)
            
            # Create buffer to store extracted data
            # We'll extract enough bytes to potentially find a complete PNG
            max_bytes = width * height // 8  # Each pixel can store up to 3 bits (one per channel)
            buffer = np.zeros(max_bytes, dtype=np.uint8)
            
            # Extract LSB from all channels and combine them
            bit_index = 0
            byte_index = 0
            
            # PNG files start with a signature, so we need to extract enough bytes to check for it
            for y in range(height):
                for x in range(width):
                    if byte_index >= max_bytes:
                        break
                        
                    # Extract one bit from each channel (blue, green, red)
                    # and combine them into bytes
                    if bit_index < 8:
                        # Blue channel LSB
                        buffer[byte_index] |= ((blue[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                        
                    if bit_index < 8:
                        # Green channel LSB
                        buffer[byte_index] |= ((green[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                        
                    if bit_index < 8:
                        # Red channel LSB
                        buffer[byte_index] |= ((red[y, x] & 1) << (7 - bit_index))
                        bit_index += 1
                    
                    # If we've collected 8 bits (1 byte), move to the next byte
                    if bit_index >= 8:
                        bit_index = 0
                        byte_index += 1
                
                if byte_index >= max_bytes:
                    break
            
            # Check if we found a PNG signature
            # PNG signature is 89 50 4E 47 0D 0A 1A 0A in hex
            png_signature = np.array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], dtype=np.uint8)
            
            # Look for PNG signature in the extracted data
            signature_found = False
            png_start_index = -1
            
            # We need at least 8 bytes for the signature
            for i in range(min(1000, len(buffer) - 8)):  # Limit search to first 1000 bytes for efficiency
                if np.array_equal(buffer[i:i+8], png_signature):
                    signature_found = True
                    png_start_index = i
                    self.logger.info(f"PNG signature found at byte offset {i}")
                    break
            
            if not signature_found:
                # Try different bit extraction patterns
                # Some steganography methods use different bit orders or patterns
                
                # Reset buffer
                buffer = np.zeros(max_bytes, dtype=np.uint8)
                
                # Alternative bit extraction: take LSB from sequential pixels rather than sequential channels
                bit_index = 0
                byte_index = 0
                
                for y in range(height):
                    for x in range(width):
                        if byte_index >= max_bytes:
                            break
                            
                        # Extract LSB from the blue channel of this pixel
                        if bit_index < 8:
                            buffer[byte_index] |= ((blue[y, x] & 1) << (7 - bit_index))
                            bit_index += 1
                        
                        # If we've collected 8 bits (1 byte), move to the next byte
                        if bit_index >= 8:
                            bit_index = 0
                            byte_index += 1
                    
                    if byte_index >= max_bytes:
                        break
                
                # Check for PNG signature again with this alternative extraction
                for i in range(min(1000, len(buffer) - 8)):
                    if np.array_equal(buffer[i:i+8], png_signature):
                        signature_found = True
                        png_start_index = i
                        self.logger.info(f"PNG signature found with alternative extraction at byte offset {i}")
                        break
            
            if not signature_found:
                # Try another extraction pattern: 2 LSBs instead of 1
                # Reset buffer (we'll need more space since we're extracting 2 bits per channel)
                buffer = np.zeros(max_bytes * 2, dtype=np.uint8)
                
                bit_index = 0
                byte_index = 0
                
                for y in range(height):
                    for x in range(width):
                        if byte_index >= max_bytes * 2:
                            break
                            
                        # Extract 2 LSBs from each channel
                        if bit_index < 8:
                            buffer[byte_index] |= ((blue[y, x] & 3) << (6 - bit_index))  # 2 bits
                            bit_index += 2
                        
                        if bit_index < 8:
                            buffer[byte_index] |= ((green[y, x] & 3) << (6 - bit_index))  # 2 bits
                            bit_index += 2
                        
                        if bit_index < 8:
                            buffer[byte_index] |= ((red[y, x] & 3) << (6 - bit_index))  # 2 bits
                            bit_index += 2
                        
                        # If we've collected 8 bits (1 byte), move to the next byte
                        if bit_index >= 8:
                            bit_index = 0
                            byte_index += 1
                    
                    if byte_index >= max_bytes * 2:
                        break
                
                # Check for PNG signature with this 2-LSB extraction
                for i in range(min(1000, len(buffer) - 8)):
                    if np.array_equal(buffer[i:i+8], png_signature):
                        signature_found = True
                        png_start_index = i
                        self.logger.info(f"PNG signature found with 2-LSB extraction at byte offset {i}")
                        break
            
            if signature_found:
                # We found a PNG signature! Now, extract the complete PNG
                # We need to parse the PNG structure to determine its size
                
                # Start with a reasonable buffer size and try to extract the full PNG
                # After the 8-byte signature, the PNG format has an 8-byte chunk header
                # with a 4-byte length field followed by a 4-byte chunk type
                
                # IHDR chunk must follow signature and contains image dimensions
                if png_start_index + 24 < len(buffer):  # We need at least signature(8) + chunk size(4) + "IHDR"(4) + width(4) + height(4)
                    chunk_size = int.from_bytes(buffer[png_start_index+8:png_start_index+12].tobytes(), byteorder='big')
                    chunk_type = buffer[png_start_index+12:png_start_index+16].tobytes().decode('ascii', errors='ignore')
                    
                    if chunk_type == "IHDR" and chunk_size == 13:  # Valid IHDR chunk
                        # Extract width and height from IHDR
                        width_bytes = buffer[png_start_index+16:png_start_index+20].tobytes()
                        height_bytes = buffer[png_start_index+20:png_start_index+24].tobytes()
                        
                        hidden_width = int.from_bytes(width_bytes, byteorder='big')
                        hidden_height = int.from_bytes(height_bytes, byteorder='big')
                        
                        # Sanity check - reject unreasonable dimensions
                        if 0 < hidden_width <= 2000 and 0 < hidden_height <= 2000:
                            self.logger.info(f"Found hidden PNG with dimensions {hidden_width}x{hidden_height}")
                            
                            # Estimate the total PNG size needed
                            # This is a rough estimate; actual PNG size depends on compression
                            estimated_size = png_start_index + hidden_width * hidden_height * 3 + 1000  # Add extra for headers/chunks
                            
                            # Ensure we don't exceed buffer size
                            estimated_size = min(estimated_size, len(buffer))
                            
                            # Look for IEND chunk which marks the end of PNG
                            end_marker = np.array([0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82], dtype=np.uint8)
                            
                            iend_pos = -1
                            for i in range(png_start_index + 24, min(png_start_index + estimated_size, len(buffer) - 8)):
                                if np.array_equal(buffer[i:i+8], end_marker):
                                    iend_pos = i + 8  # Include the IEND marker itself
                                    break
                            
                            if iend_pos > 0:
                                # We found the complete PNG from signature to IEND
                                png_data = buffer[png_start_index:iend_pos].tobytes()
                            else:
                                # We didn't find IEND, use our estimation
                                png_data = buffer[png_start_index:png_start_index + estimated_size].tobytes()
                            
                            # Try to decode the PNG
                            try:
                                img_array = np.frombuffer(png_data, dtype=np.uint8)
                                extracted_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                
                                if extracted_img is not None:
                                    return extracted_img
                                else:
                                    self.logger.warning("Found PNG signature but couldn't decode to image")
                            except Exception as e:
                                self.logger.error(f"Error decoding extracted PNG: {str(e)}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting hidden data: {str(e)}")
            return None
    
    
    def get_available_models(self):
        """Get a list of available trained models
        
        Returns:
            list: Information about available models
        """
        models = []
        
        try:
            model_dir = settings.ML_MODEL_DIR
            
            # Look for CNN models (.h5 files with metadata)
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
                    except:
                        pass
            
            # Look for LSTM models
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
                    except:
                        pass
            
            # Look for XGBoost models
            xgb_models = [f for f in os.listdir(model_dir) if f.startswith('xgboost_model_') and f.endswith('.pkl')]
            for model_file in xgb_models:
                model_path = os.path.join(model_dir, model_file)
                try:
                    model = joblib.load(model_path)
                    # Assuming the model object has an 'attribute' called score_
                    accuracy = getattr(model, 'best_score_', 0.0)
                    models.append({
                        'id': os.path.splitext(model_file)[0],
                        'path': model_path,
                        'type': 'XGBoost',
                        'accuracy': accuracy,
                        'timestamp': int(model_file.split('_')[-1].split('.')[0])
                    })
                except:
                    pass
            
            # Look for ensemble models
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
                except:
                    pass
            
            # Sort by accuracy (descending)
            models.sort(key=lambda x: x['accuracy'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting available models: {str(e)}")
        
        return models