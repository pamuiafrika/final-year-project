#!/usr/bin/env python3
"""
PDF Feature Extractor
Extracts features from PDF files for steganography detection
"""

import os
import hashlib
import time
import pandas as pd
import numpy as np
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PDFFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'pdf_name', 'file_hash', 'file_size_bytes', 'pdf_version', 'num_pages',
            'num_objects', 'num_stream_objects', 'num_embedded_files',
            'num_annotation_objects', 'num_form_fields', 'creation_date_ts',
            'mod_date_ts', 'creation_mod_date_diff', 'avg_entropy_per_stream',
            'max_entropy_per_stream', 'min_entropy_per_stream', 'std_entropy_per_stream',
            'num_streams_entropy_gt_threshold', 'num_encrypted_streams',
            'num_corrupted_objects', 'num_objects_with_random_markers',
            'has_broken_name_trees', 'num_suspicious_filters', 'has_javascript',
            'has_launch_actions', 'avg_file_size_per_page', 'compression_ratio',
            'num_eof_markers', 'extraction_success', 'extraction_time_ms', 'error_count'
        ]
    
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count frequency of each byte
        byte_counts = Counter(data)
        length = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def get_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return "error_calculating_hash"
    
    def extract_basic_features(self, file_path):
        """Extract basic file features"""
        features = {}
        
        try:
            # Basic file info
            file_path = Path(file_path)
            features['pdf_name'] = file_path.name
            features['file_hash'] = self.get_file_hash(file_path)
            features['file_size_bytes'] = file_path.stat().st_size
            
            return features
        except Exception as e:
            print(f"Error extracting basic features: {e}")
            return {
                'pdf_name': file_path.name if hasattr(file_path, 'name') else 'unknown',
                'file_hash': 'error',
                'file_size_bytes': 0
            }
    
    def safe_get_indirect_object(self, obj, key, default=None):
        """Safely get value from IndirectObject or dict"""
        try:
            if hasattr(obj, 'get_object'):
                # It's an IndirectObject, resolve it first
                resolved_obj = obj.get_object()
                if isinstance(resolved_obj, dict):
                    return resolved_obj.get(key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        except Exception:
            return default
    
    def safe_check_key_in_object(self, obj, key):
        """Safely check if key exists in IndirectObject or dict"""
        try:
            if hasattr(obj, 'get_object'):
                # It's an IndirectObject, resolve it first
                resolved_obj = obj.get_object()
                if isinstance(resolved_obj, dict):
                    return key in resolved_obj
            elif isinstance(obj, dict):
                return key in obj
            return False
        except Exception:
            return False
    
    def extract_pdf_metadata(self, file_path):
        """Extract PDF metadata using PyPDF2"""
        features = {
            'pdf_version': 1.4,
            'num_pages': 0,
            'creation_date_ts': 0,
            'mod_date_ts': 0,
            'creation_mod_date_diff': 0,
            'has_javascript': False,
            'has_launch_actions': False,
            'num_form_fields': 0
        }
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic info
                features['num_pages'] = len(reader.pages)
                
                # Version
                if hasattr(reader, 'pdf_header'):
                    version_str = reader.pdf_header
                    if isinstance(version_str, bytes):
                        version_str = version_str.decode('utf-8', errors='ignore')
                    if 'PDF-' in str(version_str):
                        try:
                            features['pdf_version'] = float(str(version_str).split('PDF-')[1][:3])
                        except:
                            features['pdf_version'] = 1.4
                
                # Metadata
                if reader.metadata:
                    # Creation and modification dates
                    creation_date = reader.metadata.get('/CreationDate')
                    mod_date = reader.metadata.get('/ModDate')
                    
                    if creation_date:
                        features['creation_date_ts'] = hash(str(creation_date)) % 1000000
                    if mod_date:
                        features['mod_date_ts'] = hash(str(mod_date)) % 1000000
                    
                    features['creation_mod_date_diff'] = abs(
                        features['mod_date_ts'] - features['creation_date_ts']
                    )
                
                # Check for JavaScript and actions
                for page in reader.pages:
                    try:
                        # Check for Additional Actions
                        if self.safe_check_key_in_object(page, '/AA'):
                            features['has_launch_actions'] = True
                        
                        # Check annotations for JavaScript
                        if self.safe_check_key_in_object(page, '/Annots'):
                            features['has_javascript'] = True
                            
                    except Exception as e:
                        print(f"Error processing page: {e}")
                        continue
                
                # Form fields - safely check for AcroForm
                try:
                    if hasattr(reader, 'trailer') and reader.trailer:
                        root_obj = self.safe_get_indirect_object(reader.trailer, '/Root')
                        if root_obj and self.safe_check_key_in_object(root_obj, '/AcroForm'):
                            features['num_form_fields'] = 1  # Simplified
                except Exception as e:
                    print(f"Error checking AcroForm: {e}")
                
        except Exception as e:
            print(f"Error extracting PDF metadata: {e}")
        
        return features
    
    def extract_structure_features(self, file_path):
        """Extract PDF structure features using PyMuPDF"""
        features = {
            'num_objects': 0,
            'num_stream_objects': 0,
            'num_embedded_files': 0,
            'num_annotation_objects': 0,
            'avg_entropy_per_stream': 0.0,
            'max_entropy_per_stream': 0.0,
            'min_entropy_per_stream': 0.0,
            'std_entropy_per_stream': 0.0,
            'num_streams_entropy_gt_threshold': 0,
            'num_encrypted_streams': 0,
            'num_corrupted_objects': 0,
            'num_objects_with_random_markers': 0,
            'has_broken_name_trees': False,
            'num_suspicious_filters': 0,
            'compression_ratio': 1.0,
            'num_eof_markers': 1
        }
        
        try:
            doc = fitz.open(file_path)
            
            # Basic counts
            features['num_pages'] = doc.page_count
            
            # Embedded files
            try:
                embedded_files = doc.embfile_names()
                features['num_embedded_files'] = len(embedded_files) if embedded_files else 0
            except Exception as e:
                print(f"Error getting embedded files: {e}")
                features['num_embedded_files'] = 0
            
            # Annotations
            try:
                total_annotations = 0
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    annotations = page.annots()
                    total_annotations += len(list(annotations))
                
                features['num_annotation_objects'] = total_annotations
            except Exception as e:
                print(f"Error counting annotations: {e}")
                features['num_annotation_objects'] = 0
            
            # Calculate compression ratio
            try:
                compressed_size = os.path.getsize(file_path)
                if features['num_pages'] > 0:
                    features['compression_ratio'] = compressed_size / features['num_pages']
                else:
                    features['compression_ratio'] = compressed_size
            except Exception as e:
                print(f"Error calculating compression ratio: {e}")
                features['compression_ratio'] = 1.0
            
            # Simulate other structure features (would need more advanced PDF parsing)
            features['num_objects'] = features['num_pages'] * 6  # Approximation
            features['num_stream_objects'] = max(0, features['num_objects'] - features['num_pages'])
            
            # Entropy calculations (simplified)
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                    if len(data) > 1000:  # Only calculate for larger files
                        # Sample entropy from different parts
                        chunk_size = max(1000, len(data) // 10)
                        entropies = []
                        for i in range(0, len(data), chunk_size):
                            chunk = data[i:i+chunk_size]
                            if chunk:
                                entropies.append(self.calculate_entropy(chunk))
                        
                        if entropies:
                            features['avg_entropy_per_stream'] = np.mean(entropies)
                            features['max_entropy_per_stream'] = np.max(entropies)
                            features['min_entropy_per_stream'] = np.min(entropies)
                            features['std_entropy_per_stream'] = np.std(entropies)
                            features['num_streams_entropy_gt_threshold'] = sum(1 for e in entropies if e > 7.0)
            except Exception as e:
                print(f"Error calculating entropy: {e}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting structure features: {e}")
        
        return features
    
    def extract_security_features(self, file_path):
        """Extract security-related features"""
        features = {
            'num_encrypted_streams': 0,
            'num_suspicious_filters': 0,
            'has_javascript': False,
            'has_launch_actions': False
        }
        
        try:
            # Read raw PDF content
            with open(file_path, 'rb') as f:
                content = f.read()
                # Try to decode as text for pattern matching
                try:
                    content_str = content.decode('latin-1', errors='ignore')
                except:
                    content_str = str(content)
            
            # Check for encryption
            if '/Encrypt' in content_str:
                features['num_encrypted_streams'] = 1
            
            # Check for suspicious filters
            suspicious_filters = ['/ASCIIHexDecode', '/ASCII85Decode', '/LZWDecode', '/FlateDecode']
            for filter_name in suspicious_filters:
                features['num_suspicious_filters'] += content_str.count(filter_name)
            
            # Check for JavaScript
            js_indicators = ['/JavaScript', '/JS', 'app.alert', 'eval(']
            for indicator in js_indicators:
                if indicator in content_str:
                    features['has_javascript'] = True
                    break
            
            # Check for launch actions
            action_indicators = ['/Launch', '/GoToR', '/URI']
            for indicator in action_indicators:
                if indicator in content_str:
                    features['has_launch_actions'] = True
                    break
        
        except Exception as e:
            print(f"Error extracting security features: {e}")
        
        return features
    
    def extract_all_features(self, file_path):
        """Extract all features from a PDF file"""
        start_time = time.time()
        
        # Initialize features dictionary
        features = {name: 0 for name in self.feature_names}
        error_count = 0
        
        try:
            # Extract different feature groups
            basic_features = self.extract_basic_features(file_path)
            features.update(basic_features)
            
            try:
                pdf_features = self.extract_pdf_metadata(file_path)
                features.update(pdf_features)
            except Exception as e:
                print(f"Error in PDF metadata extraction: {e}")
                error_count += 1
            
            try:
                structure_features = self.extract_structure_features(file_path)
                features.update(structure_features)
            except Exception as e:
                print(f"Error in structure extraction: {e}")
                error_count += 1
            
            try:
                security_features = self.extract_security_features(file_path)
                features.update(security_features)
            except Exception as e:
                print(f"Error in security extraction: {e}")
                error_count += 1
            
            # Calculate derived features
            if features['num_pages'] > 0:
                features['avg_file_size_per_page'] = features['file_size_bytes'] / features['num_pages']
            else:
                features['avg_file_size_per_page'] = features['file_size_bytes']
            
            # Set extraction success
            features['extraction_success'] = True
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            error_count += 1
            features['extraction_success'] = False
        
        # Set timing and error info
        extraction_time = time.time() - start_time
        features['extraction_time_ms'] = int(extraction_time * 1000)
        features['error_count'] = error_count
        
        # Ensure all boolean values are properly set
        bool_fields = ['has_broken_name_trees', 'has_javascript', 'has_launch_actions', 'extraction_success']
        for field in bool_fields:
            if field not in features:
                features[field] = False
        
        return features
    
    def extract_from_directory(self, directory_path, output_csv=None):
        """Extract features from all PDF files in a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Find all PDF files
        pdf_files = list(directory_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory_path}")
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        # Extract features from each file
        all_features = []
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                features = self.extract_all_features(pdf_file)
                all_features.append(features)
            except Exception as e:
                print(f"Failed to process {pdf_file.name}: {e}")
                # Add error entry
                error_features = {name: 0 for name in self.feature_names}
                error_features['pdf_name'] = pdf_file.name
                error_features['extraction_success'] = False
                error_features['error_count'] = 1
                all_features.append(error_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Add label column 
        df['label'] = 1
        
        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Features saved to {output_csv}")
        
        return df
    
    def extract_single_file(self, file_path):
        """Extract features from a single PDF file"""
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        
        features = self.extract_all_features(file_path)
        
        # Convert to DataFrame for consistency
        df = pd.DataFrame([features])
        df['label'] = 1
        
        return df

# Usage examples and testing
if __name__ == "__main__":
    # Initialize extractor
    extractor = PDFFeatureExtractor()
    
    # Example usage:
    # 1. Extract from single file
    # features_df = extractor.extract_single_file("sample.pdf")
    # print(features_df.head())
    
    # 2. Extract from directory
    features_df = extractor.extract_from_directory("/home/d3bugger/Documents/steg00", "features.csv")
    print(f"Extracted features for {len(features_df)} files")
    
    # 3. Process specific file
    # features = extractor.extract_all_features("example.pdf")
    # print("Extracted features:", features)
    
    print("PDF Feature Extractor Ready!")
    print("Available methods:")
    print("- extract_single_file(file_path)")
    print("- extract_from_directory(directory_path, output_csv)")
    print("- extract_all_features(file_path)")