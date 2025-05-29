import os
import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import argparse

# Import the detector class from bc.py
from bc import PDFSteganoDetectorEnhanced, EnhancedMLModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(clean_dir: str, stego_dir: str) -> tuple:
    """
    Load and prepare dataset from clean and steganographic PDFs
    """
    detector = PDFSteganoDetectorEnhanced()
    features = []
    labels = []
    
    logger.info("Processing clean PDFs...")
    for pdf_file in Path(clean_dir).glob("*.pdf"):
        try:
            result = detector.analyze_pdf(str(pdf_file), quick_scan=False)
            if "error" not in result:
                features.append(detector.features)
                labels.append(0)  # 0 for clean
        except Exception as e:
            logger.warning(f"Error processing {pdf_file}: {e}")
    
    logger.info("Processing steganographic PDFs...")
    for pdf_file in Path(stego_dir).glob("*.pdf"):
        try:
            result = detector.analyze_pdf(str(pdf_file), quick_scan=False)
            if "error" not in result:
                features.append(detector.features)
                labels.append(1)  # 1 for stego
        except Exception as e:
            logger.warning(f"Error processing {pdf_file}: {e}")
    
    return features, labels

def train_model(features: List[Dict[str, Any]], labels: List[int], output_path: str):
    """
    Train the ML model and save it
    """
    ml_model = EnhancedMLModel()
    
    # Convert features to vectors
    feature_vectors = []
    for feature_dict in features:
        try:
            vector = ml_model.extract_advanced_features(feature_dict)
            feature_vectors.append(vector)
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            continue
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.2, random_state=42
    )
    
    logger.info("Training ML ensemble...")
    ml_model.train_ensemble(X_train, y_train)
    
    # Evaluate model
    predictions = []
    for features in X_test:
        result = ml_model.predict_anomaly(features)
        predictions.append(1 if result['is_anomaly'] else 0)
    
    logger.info("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # Save the trained model
    model_data = {
        'models': ml_model.models,
        'scalers': ml_model.scalers,
        'pca': ml_model.pca,
        'feature_importance': ml_model.feature_importance
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train PDF Steganography Detection Model")
    parser.add_argument('--clean', required=True, help='Directory containing clean PDF files')
    parser.add_argument('--stego', required=True, help='Directory containing steganographic PDF files')
    parser.add_argument('--output', required=True, help='Path to save trained model')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Loading dataset...")
    features, labels = load_dataset(args.clean, args.stego)
    
    if not features:
        logger.error("No valid features extracted from dataset")
        return 1
    
    logger.info(f"Dataset loaded: {len(features)} samples ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
    
    try:
        train_model(features, labels, args.output)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())