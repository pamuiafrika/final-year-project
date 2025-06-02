#!/usr/bin/env python3
"""
PDF Steganography Model Trainer Script
Uses the train_baseline_model() function from c.py to train models
"""

import os
import argparse
from c import PDFSteganoDetector

def train_model(clean_pdf_dir: str, output_model_path: str):
    """
    Train a model using clean PDFs and save it to specified path
    """
    detector = PDFSteganoDetector()
    print(f"Starting model training with PDFs from: {clean_pdf_dir}")
    
    try:
        results = detector.train_baseline_model(
            clean_pdf_directory=clean_pdf_dir,
            output_model_path=output_model_path
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {output_model_path}")
        print(f"Scaler saved to: {output_model_path.replace('.pkl', '_scaler.pkl')}")
        print(f"Metadata saved to: {output_model_path.replace('.pkl', '_training_info.json')}")
        
        print("\nTraining Statistics:")
        print(f"- Processed {results['processed_files']} clean PDFs")
        print(f"- Failed to process {results['failed_files']} files")
        print(f"- Features used: {len(results['feature_names'])}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Train PDF steganography detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'clean_pdf_dir',
        help='Directory containing clean PDF files for training'
    )
    parser.add_argument(
        '-o', '--output',
        default='trained_model.pkl',
        help='Output path for trained model (.pkl)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.clean_pdf_dir):
        print(f"Error: Directory not found: {args.clean_pdf_dir}")
        return
    
    train_model(args.clean_pdf_dir, args.output)

if __name__ == "__main__":
    main()