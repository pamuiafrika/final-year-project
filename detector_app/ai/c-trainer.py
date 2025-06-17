import os
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")
    raise

logger = logging.getLogger(__name__)

class PDFSteganoTrainer:
    """
    Unsupervised training module for PDF steganography detection.
    Learns patterns from clean PDFs to establish baseline for anomaly detection.
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 random_state: int = 42,
                 feature_selection_threshold: float = 0.1):
        """
        Initialize the PDF steganography trainer.
        
        Args:
            contamination: Expected proportion of anomalies in dataset (0.1 = 10%)
            n_estimators: Number of trees in IsolationForest
            random_state: Random seed for reproducibility
            feature_selection_threshold: Minimum variance threshold for feature selection
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_selection_threshold = feature_selection_threshold
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_stats = {}
        self.training_history = []
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ML components."""
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        self.scaler = StandardScaler()
        
        # Define feature names (should match your PDFSteganoDetector)
        self.feature_names = [
            'total_objects', 'large_objects_count', 'unused_objects_count', 'avg_object_size',
            'metadata_fields_count', 'suspicious_metadata_count', 'total_fonts', 'embedded_fonts',
            'font_anomalies_count', 'embedded_font_ratio', 'avg_entropy', 'max_entropy',
            'entropy_variance', 'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
            'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
            'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size'
        ]
    
    def extract_features_from_pdfs(self, pdf_paths: List[str], 
                                  output_cache: Optional[str] = None) -> np.ndarray:
        """
        Extract features from a list of PDF files using a modified PDFSteganoDetector.
        
        Args:
            pdf_paths: List of paths to PDF files
            output_cache: Optional path to cache extracted features
            
        Returns:
            numpy array of extracted features
        """
        logger.info(f"Extracting features from {len(pdf_paths)} PDF files...")
        
        # Check if cached features exist
        if output_cache and os.path.exists(output_cache):
            logger.info(f"Loading cached features from {output_cache}")
            return np.load(output_cache)
        
        features_list = []
        failed_files = []
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            try:
                # Extract features directly without ML anomaly detection
                features = self._extract_features_from_single_pdf(pdf_path)
                
                if features is not None:
                    features_list.append(features)
                else:
                    logger.warning(f"Failed to extract features from {pdf_path}")
                    failed_files.append(pdf_path)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                failed_files.append(pdf_path)
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files[:5]}...")
        
        if not features_list:
            raise ValueError("No features could be extracted from any PDF files")
        
        features_array = np.array(features_list)
        
        # Handle NaN and infinite values
        features_array = np.nan_to_num(features_array, nan=0, posinf=1e6, neginf=-1e6)
        
        # Cache features if requested
        if output_cache:
            np.save(output_cache, features_array)
            logger.info(f"Features cached to {output_cache}")
        
        logger.info(f"Successfully extracted features from {len(features_list)} PDFs")
        logger.info(f"Feature matrix shape: {features_array.shape}")
        
        return features_array
    
    def _extract_features_from_single_pdf(self, pdf_path: str) -> Optional[List[float]]:
        """
        Extract features from a single PDF file without ML anomaly detection.
        This is a streamlined version that avoids the ML model fitting issue.
        """
        import fitz  # PyMuPDF
        from collections import Counter
        
        try:
            features = {}
            
            # Read PDF content
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # 1. Object Stream Analysis
            try:
                xref_count = doc.xref_length()
                large_streams = []
                unused_objects = []
                
                for xref in range(xref_count):
                    try:
                        obj_length = doc.xref_get_key(xref, "Length")
                        if obj_length and isinstance(obj_length, (int, str)):
                            length = int(obj_length)
                            if length > 100_000:  # 100KB threshold
                                large_streams.append(length)
                        
                        if not doc.xref_is_stream(xref) and (obj_content := doc.xref_stream(xref)):
                            if len(obj_content) > 1000:
                                unused_objects.append(xref)
                    except Exception:
                        continue
                
                features.update({
                    "total_objects": xref_count,
                    "large_objects_count": len(large_streams),
                    "unused_objects_count": len(unused_objects),
                    "avg_object_size": np.mean(large_streams) if large_streams else 0
                })
            except Exception:
                features.update({
                    "total_objects": 0, "large_objects_count": 0, 
                    "unused_objects_count": 0, "avg_object_size": 0
                })
            
            # 2. Metadata Analysis
            try:
                metadata = doc.metadata
                suspicious_metadata = []
                standard_fields = {'title', 'author', 'subject', 'keywords', 'creator', 'producer', 'creationDate', 'modDate'}
                
                for key, value in metadata.items():
                    if key.lower() not in standard_fields:
                        suspicious_metadata.append({key: value})
                    if isinstance(value, str) and any(ord(c) > 127 for c in value):
                        suspicious_metadata.append({f"binary_in_{key}": value})
                
                features.update({
                    "metadata_fields_count": len(metadata),
                    "suspicious_metadata_count": len(suspicious_metadata),
                    "has_creation_date": 'creationDate' in metadata,
                    "has_modification_date": 'modDate' in metadata
                })
            except Exception:
                features.update({
                    "metadata_fields_count": 0, "suspicious_metadata_count": 0,
                    "has_creation_date": False, "has_modification_date": False
                })
            
            # 3. Font Analysis
            try:
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
                            font_anomalies.append(font_name)
                
                features.update({
                    "total_fonts": total_fonts,
                    "embedded_fonts": embedded_fonts,
                    "font_anomalies_count": len(font_anomalies),
                    "embedded_font_ratio": embedded_fonts / max(total_fonts, 1)
                })
            except Exception:
                features.update({
                    "total_fonts": 0, "embedded_fonts": 0,
                    "font_anomalies_count": 0, "embedded_font_ratio": 0
                })
            
            # 4. Entropy Analysis
            try:
                chunk_size = 1024
                entropies = []
                
                for i in range(0, len(pdf_content), chunk_size):
                    chunk = pdf_content[i:i + chunk_size]
                    if len(chunk) > 0:
                        entropy = self._calculate_entropy(chunk)
                        entropies.append(entropy)
                
                if entropies:
                    features.update({
                        "avg_entropy": np.mean(entropies),
                        "max_entropy": np.max(entropies),
                        "entropy_variance": np.var(entropies),
                        "high_entropy_chunks": sum(1 for e in entropies if e > 7.0)
                    })
                else:
                    features.update({
                        "avg_entropy": 0, "max_entropy": 0,
                        "entropy_variance": 0, "high_entropy_chunks": 0
                    })
            except Exception:
                features.update({
                    "avg_entropy": 0, "max_entropy": 0,
                    "entropy_variance": 0, "high_entropy_chunks": 0
                })
            
            # 5. Embedded Files Analysis
            try:
                embedded_files = []
                png_signature = b'\x89PNG\r\n\x1a\n'
                
                for i in range(doc.embfile_count()):
                    file_content = doc.embfile_get(i)
                    embedded_files.append({
                        "contains_png": bool(file_content and png_signature in file_content),
                        "size": len(file_content) if file_content else 0
                    })
                
                features.update({
                    "embedded_files_count": len(embedded_files),
                    "embedded_png_count": len([f for f in embedded_files if f["contains_png"]]),
                    "total_embedded_size": sum(f["size"] for f in embedded_files)
                })
            except Exception:
                features.update({
                    "embedded_files_count": 0, "embedded_png_count": 0,
                    "total_embedded_size": 0
                })
            
            # 6. Invisible Elements Analysis
            try:
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
                                        invisible_elements.append(span)
                    except Exception:
                        continue
                
                features.update({
                    "invisible_elements_count": len(invisible_elements),
                    "pages_with_invisible": len({elem.get("page", 0) for elem in invisible_elements})
                })
            except Exception:
                features.update({
                    "invisible_elements_count": 0, "pages_with_invisible": 0
                })
            
            # 7. PNG Detection
            try:
                png_signature = b'\x89PNG\r\n\x1a\n'
                png_matches = []
                offset = 0
                
                while (pos := pdf_content.find(png_signature, offset)) != -1:
                    png_matches.append(pos)
                    offset = pos + 1
                
                # Count PNG chunks
                png_chunks = 0
                for chunk_type in [b'IHDR', b'IDAT', b'IEND']:
                    png_chunks += pdf_content.count(chunk_type)
                
                features.update({
                    "png_signatures_count": len(png_matches),
                    "valid_png_count": len(png_matches),  # Simplified for training
                    "png_chunks_count": png_chunks,
                    "total_png_size": len(png_matches) * 1000  # Estimated
                })
            except Exception:
                features.update({
                    "png_signatures_count": 0, "valid_png_count": 0,
                    "png_chunks_count": 0, "total_png_size": 0
                })
            
            doc.close()
            
            # Convert to feature vector
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {pdf_path}: {str(e)}")
            return None
    
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
    
    def analyze_feature_distributions(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature distributions to understand data characteristics.
        
        Args:
            features: Feature matrix
            
        Returns:
            Dictionary containing feature statistics
        """
        logger.info("Analyzing feature distributions...")
        
        stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_data = features[:, i]
            
            stats[feature_name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'median': np.median(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'variance': np.var(feature_data),
                'zero_ratio': np.sum(feature_data == 0) / len(feature_data)
            }
        
        self.feature_stats = stats
        return stats
    
    def feature_selection(self, features: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Bypass feature selection to use all available features.
        
        Args:
            features: Original feature matrix
            
        Returns:
            Tuple of (all features, all feature names)
        """
        logger.info("Bypassing feature selection: using all features")
        return features, self.feature_names
    
    def train_model(self, features: np.ndarray, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the unsupervised anomaly detection model.
        
        Args:
            features: Feature matrix from clean PDFs
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting model training...")
        
        # Analyze feature distributions
        feature_stats = self.analyze_feature_distributions(features)
        
        # Feature selection
        selected_features, selected_names = self.feature_selection(features)
        self.feature_names = selected_names
        
        # Split data for validation
        if validation_split > 0:
            train_features, val_features = train_test_split(
                selected_features, 
                test_size=validation_split,
                random_state=self.random_state
            )
        else:
            train_features = selected_features
            val_features = None
        
        # Scale features
        logger.info("Scaling features...")
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Train the model
        logger.info("Training IsolationForest model...")
        self.model.fit(train_features_scaled)
        
        # Evaluate on training data
        train_predictions = self.model.predict(train_features_scaled)
        train_scores = self.model.decision_function(train_features_scaled)
        
        # Calculate training metrics
        train_anomaly_rate = np.sum(train_predictions == -1) / len(train_predictions)
        train_score_stats = {
            'mean': np.mean(train_scores),
            'std': np.std(train_scores),
            'min': np.min(train_scores),
            'max': np.max(train_scores)
        }
        
        results = {
            'training_samples': len(train_features),
            'selected_features': len(selected_names),
            'feature_names': selected_names,
            'train_anomaly_rate': train_anomaly_rate,
            'train_score_stats': train_score_stats,
            'feature_statistics': feature_stats
        }
        
        # Validation evaluation if available
        if val_features is not None:
            val_features_scaled = self.scaler.transform(val_features)
            val_predictions = self.model.predict(val_features_scaled)
            val_scores = self.model.decision_function(val_features_scaled)
            
            val_anomaly_rate = np.sum(val_predictions == -1) / len(val_predictions)
            val_score_stats = {
                'mean': np.mean(val_scores),
                'std': np.std(val_scores),
                'min': np.min(val_scores),
                'max': np.max(val_scores)
            }
            
            results.update({
                'validation_samples': len(val_features),
                'val_anomaly_rate': val_anomaly_rate,
                'val_score_stats': val_score_stats
            })
        
        # Store training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'hyperparameters': {
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'random_state': self.random_state,
                'feature_selection_threshold': self.feature_selection_threshold
            }
        }
        self.training_history.append(training_record)
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training anomaly rate: {train_anomaly_rate:.2%}")
        if val_features is not None:
            logger.info(f"Validation anomaly rate: {val_anomaly_rate:.2%}")
        
        return results
    
    def save_model(self, model_path: str, include_metadata: bool = True):
        """
        Save the trained model and associated components.
        
        Args:
            model_path: Path to save the model file
            include_metadata: Whether to save additional metadata
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before saving")
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
        if include_metadata:
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_statistics': self.feature_stats,
                'training_history': self.training_history,
                'hyperparameters': {
                    'contamination': self.contamination,
                    'n_estimators': self.n_estimators,
                    'random_state': self.random_state,
                    'feature_selection_threshold': self.feature_selection_threshold
                }
            }
            
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names', self.feature_names)
            self.feature_stats = metadata.get('feature_statistics', {})
            self.training_history = metadata.get('training_history', [])
        
        logger.info(f"Model loaded from {model_path}")
    
    def create_visualizations(self, features: np.ndarray, output_dir: str):
        """
        Create visualizations of feature distributions and model performance.
        
        Args:
            features: Feature matrix
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature distribution plots
        n_features = min(len(self.feature_names), 12)  # Limit to 12 features for readability
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            feature_data = features[:, i]
            ax.hist(feature_data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{self.feature_names[i]}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature correlation heatmap
        if len(self.feature_names) > 1:
            correlation_matrix = np.corrcoef(features.T)
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, 
                       xticklabels=self.feature_names, 
                       yticklabels=self.feature_names,
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def train_pdf_steganography_model(clean_pdf_directory: str,
                                model_output_path: str,
                                cache_features: bool = True,
                                validation_split: float = 0.2,
                                contamination: float = 0.1,
                                create_plots: bool = True) -> Dict[str, Any]:
    """
    Main training function that processes clean PDFs and trains the model.
    
    Args:
        clean_pdf_directory: Directory containing clean PDF files for training
        model_output_path: Path where the trained model will be saved
        cache_features: Whether to cache extracted features
        validation_split: Fraction of data for validation
        contamination: Expected anomaly rate in training data
        create_plots: Whether to create visualization plots
        
    Returns:
        Training results and metrics
    """
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(clean_pdf_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {clean_pdf_directory}")
    
    logger.info(f"Found {len(pdf_files)} PDF files for training")
    
    # Initialize trainer
    trainer = PDFSteganoTrainer(
        contamination=contamination,
        random_state=42
    )
    
    # Extract features
    cache_path = None
    if cache_features:
        cache_path = os.path.join(os.path.dirname(model_output_path), 'training_features.npy')
    
    features = trainer.extract_features_from_pdfs(pdf_files, cache_path)
    
    # Train model
    results = trainer.train_model(features, validation_split)
    
    # Save model
    trainer.save_model(model_output_path)
    
    # Create visualizations
    if create_plots:
        viz_dir = os.path.join(os.path.dirname(model_output_path), 'training_visualizations')
        trainer.create_visualizations(features, viz_dir)
    
    return results


# Example usage function
def example_training_workflow():
    """
    Example of how to use the training system.
    """
    
    # Configuration
    CLEAN_PDF_DIR = "/home/d3bugger/Documents/Datasets/Pdf"  # Directory with clean PDF files
    MODEL_OUTPUT = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/ml_models/stegov2/v02_model.pkl"
    
    
    try:
        # Train the model
        results = train_pdf_steganography_model(
            clean_pdf_directory=CLEAN_PDF_DIR,
            model_output_path=MODEL_OUTPUT,
            cache_features=True,
            validation_split=0.2,
            contamination=0.05,  # Expect 5% anomalies in training data
            create_plots=True
        )
        
        print("Training completed successfully!")
        print(f"Training samples: {results['training_samples']}")
        print(f"Selected features: {results['selected_features']}")
        print(f"Training anomaly rate: {results['train_anomaly_rate']:.2%}")
        
        if 'val_anomaly_rate' in results:
            print(f"Validation anomaly rate: {results['val_anomaly_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run example workflow
    example_training_workflow()
