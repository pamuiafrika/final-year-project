# data-proessor.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from django.conf import settings
import PyPDF2
import fitz  # PyMuPDF
import cv2
import io
from PIL import Image

class DataProcessor:
    def __init__(self, dataset_dir=None):
        """Initialize data processor with dataset directory"""
        self.dataset_dir = dataset_dir or os.path.join(settings.DATASET_DIR)
        
        # Ensure directories exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'clean'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, 'stego'), exist_ok=True)
    
    def process_dataset(self, dataset_id):
        """Process dataset for model training"""
        from detector_app.models import Dataset
        
        dataset = Dataset.objects.get(id=dataset_id)
        clean_dir = os.path.join(self.dataset_dir, 'clean')
        stego_dir = os.path.join(self.dataset_dir, 'stego')
        
        # Extract features from PDFs
        clean_features = self._extract_features_from_directory(clean_dir)
        stego_features = self._extract_features_from_directory(stego_dir)
        
        # Handle empty directories
        if not clean_features and not stego_features:
            raise ValueError("No valid PDFs found in clean or stego directories")
        
        # Create labeled dataset
        X_clean = np.array(clean_features) if clean_features else np.array([])
        y_clean = np.zeros(len(clean_features)) if clean_features else np.array([])
        
        X_stego = np.array(stego_features) if stego_features else np.array([])
        y_stego = np.ones(len(stego_features)) if stego_features else np.array([])
        
        # Combine datasets
        X = np.vstack((X_clean, X_stego)) if X_clean.size and X_stego.size else (X_clean if X_clean.size else X_stego)
        y = np.concatenate((y_clean, y_stego)) if y_clean.size and y_stego.size else (y_clean if y_clean.size else y_stego)
        
        if X.size == 0:
            raise ValueError("No features extracted from dataset")
        
        # Reshape features for LSTM model (samples, timesteps, features)
        X = self._reshape_features_for_lstm(X)
        
        # Save the feature length as metadata for future use
        feature_metadata = {
            'original_feature_length': X.shape[1] * X.shape[2],
            'timesteps': X.shape[1],
            'features_per_timestep': X.shape[2]
        }
        
        # Store metadata information
        self._save_feature_metadata(feature_metadata)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Update dataset counts
        dataset.clean_pdf_count = len(X_clean)
        dataset.stego_pdf_count = len(X_stego)
        dataset.save()
        
        return X_train, X_test, y_train, y_test
    
    def _save_feature_metadata(self, metadata):
        """Save feature metadata for future use during detection"""
        metadata_path = os.path.join(settings.MODELS_DIR, 'feature_metadata.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def _load_feature_metadata(self):
        """Load feature metadata for feature reshaping"""
        metadata_path = os.path.join(settings.MODELS_DIR, 'feature_metadata.json')
        
        try:
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load feature metadata: {str(e)}")
            # Default values based on typical LSTM configuration
            return {
                'timesteps': 100,
                'features_per_timestep': 2,
                'original_feature_length': 200
            }
    
    def _reshape_features_for_lstm(self, features):
        """Reshape features to fit LSTM model input (samples, timesteps, features)"""
        if isinstance(features, list):
            features = np.array(features)
        
        num_samples = features.shape[0]
        
        # Get feature dimensions from metadata or use defaults
        metadata = self._load_feature_metadata()
        timesteps = metadata.get('timesteps', 100)
        features_per_timestep = metadata.get('features_per_timestep', 2)
        target_feature_length = timesteps * features_per_timestep
        
        # Make sure our features match the expected length
        reshaped_features = []
        for i in range(num_samples):
            feature_vector = features[i].flatten()
            current_length = len(feature_vector)
            
            # Pad or truncate to target length
            if current_length < target_feature_length:
                feature_vector = np.pad(feature_vector, 
                                       (0, target_feature_length - current_length),
                                       mode='constant')
            elif current_length > target_feature_length:
                feature_vector = feature_vector[:target_feature_length]
                
            # Reshape to (timesteps, features_per_timestep)
            feature_vector = feature_vector.reshape(timesteps, features_per_timestep)
            reshaped_features.append(feature_vector)
        
        return np.array(reshaped_features)
    
    def _extract_features_from_directory(self, directory):
        """Extract features from all PDFs in a directory"""
        features = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDFs found in {directory}")
            return []
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing {os.path.basename(directory)}"):
            pdf_path = os.path.join(directory, pdf_file)
            try:
                # Validate PDF
                with fitz.open(pdf_path) as doc:
                    if doc.page_count == 0:
                        print(f"Skipping empty PDF: {pdf_file}")
                        continue
                
                # Extract features from the file
                file_features = self.extract_features_from_file(pdf_path)
                if file_features is not None:
                    features.append(file_features.flatten())  # Flatten to 1D array
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        return features
    
    def _extract_statistical_features(self, pdf_path):
        """Extract statistical features from PDF byte structure"""
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Convert bytes to numpy array for analysis
        byte_array = np.frombuffer(pdf_bytes, dtype=np.uint8)
        
        # Calculate statistical features
        features = [
            len(byte_array),                 # File size
            np.mean(byte_array),             # Mean byte value
            np.std(byte_array),              # Standard deviation
            np.median(byte_array),           # Median
            np.percentile(byte_array, 25),   # 1st quartile
            np.percentile(byte_array, 75),   # 3rd quartile
            np.max(byte_array),              # Maximum value
            np.min(byte_array),              # Minimum value
            np.sum(byte_array == 0),         # Number of null bytes
            np.sum(byte_array == 255),       # Number of 0xFF bytes
            *np.histogram(byte_array, bins=32, range=(0, 256))[0]  # Byte histogram (32 bins)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_image_features(self, pdf_path):
        """Extract features from images that might be hidden in PDF"""
        image_features = []
        FIXED_IMAGE_FEATURE_LENGTH = 128  # Define a fixed length for image features
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            
            # Process each page to find and analyze images
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get all images on the page
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # Image reference
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to OpenCV format for analysis
                        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            # Extract DCT coefficients (common in steganography analysis)
                            dct_features = self._extract_dct_features(img)
                            
                            # Extract noise features
                            noise_features = self._extract_noise_features(img)
                            
                            # Combine image-based features
                            img_features = np.concatenate([dct_features, noise_features])
                            image_features.append(img_features)
                    except Exception as e:
                        # Skip problematic images
                        print(f"Error processing image {img_index} in {pdf_path}: {str(e)}")
                        continue
                        
            pdf_document.close()
            
            # If images were found, aggregate their features
            if image_features:
                # Convert to numpy array
                image_features = np.array(image_features)
                # Calculate mean and std features across all images
                mean_features = np.mean(image_features, axis=0)
                std_features = np.std(image_features, axis=0)
                combined = np.concatenate([mean_features, std_features])
                
                # Pad or truncate to fixed length
                if len(combined) < FIXED_IMAGE_FEATURE_LENGTH:
                    # Pad with zeros
                    combined = np.pad(combined, (0, FIXED_IMAGE_FEATURE_LENGTH - len(combined)), mode='constant')
                elif len(combined) > FIXED_IMAGE_FEATURE_LENGTH:
                    # Truncate to fixed length
                    combined = combined[:FIXED_IMAGE_FEATURE_LENGTH]
                return combined.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing images in {pdf_path}: {str(e)}")
        
        # Return zero features if no images or processing failed
        return np.zeros(FIXED_IMAGE_FEATURE_LENGTH, dtype=np.float32)
    
    def _extract_dct_features(self, img):
        """Extract DCT coefficient features for steganography detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to fixed dimensions for consistent feature extraction
        resized = cv2.resize(gray, (128, 128))
        
        # Apply DCT transform
        dct = cv2.dct(np.float32(resized))
        
        # Extract DCT features (low and high frequency coefficients)
        low_freq = dct[:8, :8].flatten()
        high_freq = dct[-8:, -8:].flatten()
        
        # Calculate statistical measures of DCT coefficients
        dct_features = [
            np.mean(np.abs(low_freq)),
            np.std(low_freq),
            np.mean(np.abs(high_freq)),
            np.std(high_freq),
            *np.histogram(low_freq, bins=16, range=(-1000, 1000))[0],
            *np.histogram(high_freq, bins=16, range=(-1000, 1000))[0]
        ]
        
        return np.array(dct_features, dtype=np.float32)
    
    def _extract_noise_features(self, img):
        """Extract noise features for steganography detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to remove noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Extract noise by subtracting blurred image
        noise = cv2.absdiff(gray, blur)
        
        # Calculate noise statistics
        noise_features = [
            np.mean(noise),
            np.std(noise),
            np.median(noise),
            np.max(noise),
            np.min(noise),
            *np.histogram(noise, bins=16, range=(0, 256))[0]
        ]
        
        return np.array(noise_features, dtype=np.float32)
    
    def _extract_metadata_features(self, pdf_path):
        """Extract metadata features from PDF"""
        metadata_features = []
        
        try:
            # Use PyPDF2 to extract metadata
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                info = pdf_reader.metadata if pdf_reader.metadata else {}
                
                # Count number of pages
                num_pages = len(pdf_reader.pages)
                
                # Check for certain metadata fields (convert to binary features)
                has_author = 1 if info.get('/Author') else 0
                has_creator = 1 if info.get('/Creator') else 0
                has_producer = 1 if info.get('/Producer') else 0
                has_subject = 1 if info.get('/Subject') else 0
                has_title = 1 if info.get('/Title') else 0
                
                # Extract text length (might indicate hidden content)
                total_text_length = 0
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text() or ""
                    total_text_length += len(text)
                
                # Extract file structure information
                file_size = os.path.getsize(pdf_path)
                avg_bytes_per_page = file_size / max(1, num_pages)
                
                # Check for Javascript (can be used in steganography)
                has_javascript = 0
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    if '/JavaScript' in page or '/JS' in page:
                        has_javascript = 1
                        break
                
                # Compile metadata features
                metadata_features = [
                    num_pages,
                    has_author,
                    has_creator,
                    has_producer,
                    has_subject,
                    has_title,
                    has_javascript,
                    total_text_length,
                    avg_bytes_per_page,
                    file_size,
                ]
                
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {str(e)}")
            # If metadata extraction fails, use zeros
            pass
        
        # Ensure we have a fixed-length feature vector even if extraction fails
        if not metadata_features:
            metadata_features = [0] * 10
            
        return np.array(metadata_features, dtype=np.float32)
    
    def extract_features_from_file(self, file_path, return_details=False):
        """Extract features from a single PDF file for detection"""
        try:
            # Create a dictionary to store feature extraction details
            feature_details = {
                'file_path': file_path,
                'feature_components': {}
            }
            
            # Extract features using the same methods as for training
            statistical_features = self._extract_statistical_features(file_path)
            feature_details['feature_components']['statistical'] = {
                'shape': statistical_features.shape,
                'mean': float(np.mean(statistical_features)),
                'std': float(np.std(statistical_features))
            }
            
            image_features = self._extract_image_features(file_path)
            feature_details['feature_components']['image'] = {
                'shape': image_features.shape,
                'mean': float(np.mean(image_features)),
                'std': float(np.std(image_features))
            }
            
            metadata_features = self._extract_metadata_features(file_path)
            feature_details['feature_components']['metadata'] = {
                'shape': metadata_features.shape,
                'mean': float(np.mean(metadata_features)),
                'std': float(np.std(metadata_features))
            }
            
            # Combine all features
            combined_features = np.concatenate([
                statistical_features,
                image_features,
                metadata_features
            ])
            
            feature_details['combined_shape'] = combined_features.shape
            feature_details['combined_mean'] = float(np.mean(combined_features))
            feature_details['combined_std'] = float(np.std(combined_features))
            
            # Log the feature dimensions for debugging
            print(f"Extracted features shape: {combined_features.shape}")
            
            # Apply feature scaling
            combined_features = (combined_features - np.mean(combined_features)) / (np.std(combined_features) + 1e-10)
            
            if return_details:
                return combined_features, feature_details
            else:
                return combined_features
                
        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")
            if return_details:
                return None, {'error': str(e)}
            else:
                return None
    
    def prepare_features_for_prediction(self, features):
        """Prepare extracted features for model prediction"""
        if features is None:
            return None
            
        # Get model input shape requirements from metadata
        metadata = self._load_feature_metadata()
        timesteps = metadata.get('timesteps', 100)
        features_per_timestep = metadata.get('features_per_timestep', 2)
        expected_length = timesteps * features_per_timestep
        
        # Make sure our features match the expected length
        features = features.flatten()
        current_length = len(features)
        
        print(f"Raw features shape: {features.shape}")
        print(f"Expected feature size for LSTM: {expected_length}")
        
        # Pad or truncate to match expected length
        if current_length < expected_length:
            print(f"Padded features from {current_length} to {expected_length}")
            features = np.pad(features, (0, expected_length - current_length), mode='constant')
        elif current_length > expected_length:
            print(f"Truncated features from {current_length} to {expected_length}")
            features = features[:expected_length]
        
        # Reshape to (samples, timesteps, features_per_timestep)
        reshaped = features.reshape(1, timesteps, features_per_timestep)
        print(f"Reshaped features shape: {reshaped.shape}")
        
        return reshaped