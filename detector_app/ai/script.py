import os
import argparse
import joblib
import logging
import numpy as np
from tqdm import tqdm
from bc import PDFSteganoDetectorEnhanced

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_pdf_stegano")

def extract_features_from_pdf(detector, pdf_path):
    try:
        # Only extract features, don't run full ML detection
        detector.indicators.clear()
        detector.features.clear()
        # Run all feature extraction steps
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        detector._analyze_object_streams(pdf_content, pdf_path)
        detector._analyze_metadata(pdf_path)
        detector._analyze_fonts_glyphs(pdf_path)
        detector._analyze_entropy_patterns(pdf_content)
        detector._scan_embedded_files(pdf_path)
        detector._detect_invisible_layers(pdf_path)
        detector._detect_concealed_pngs(pdf_content)
        # Return features as a dict
        return dict(detector.features)
    except Exception as e:
        logger.warning(f"Failed to process {pdf_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train PDF steganography anomaly detector.")
    parser.add_argument('--input_dir', required=True, help='Directory containing clean PDF files for training')
    parser.add_argument('--model_out', required=True, help='Path to save trained IsolationForest model (e.g. model.pkl)')
    parser.add_argument('--scaler_out', required=True, help='Path to save trained StandardScaler (e.g. scaler.pkl)')
    args = parser.parse_args()

    pdf_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files for training.")

    detector = PDFSteganoDetectorEnhanced()
    feature_list = []
    feature_names = [
        'total_objects', 'large_objects_count', 'unused_objects_count',
        'avg_object_size', 'metadata_fields_count', 'suspicious_metadata_count',
        'total_fonts', 'embedded_fonts', 'font_anomalies_count',
        'embedded_font_ratio', 'avg_entropy', 'max_entropy', 'entropy_variance',
        'high_entropy_chunks', 'embedded_files_count', 'embedded_png_count',
        'total_embedded_size', 'invisible_elements_count', 'pages_with_invisible',
        'png_signatures_count', 'valid_png_count', 'png_chunks_count', 'total_png_size',
        'stream_object_count', 'suspicious_filter_count', 'high_entropy_streams',
        'partial_png_signature_count', 'high_compression_count', 'low_compression_count',
        'rare_reference_count'
    ]

    for pdf_path in tqdm(pdf_files, desc="Extracting features"):
        features = extract_features_from_pdf(detector, pdf_path)
        if features is not None:
            # Ensure all features are present, fill missing with 0
            row = [features.get(name, 0) for name in feature_names]
            feature_list.append(row)

    if not feature_list:
        logger.error("No features extracted. Training aborted.")
        return

    X = np.array(feature_list)
    logger.info(f"Extracted features from {X.shape[0]} PDFs.")

    # Train scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train IsolationForest
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    model.fit(X_scaled)

    # Save model and scaler
    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)
    logger.info(f"Model saved to {args.model_out}")
    logger.info(f"Scaler saved to {args.scaler_out}")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
