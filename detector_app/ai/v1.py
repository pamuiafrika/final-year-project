#!/usr/bin/env python3
"""
Advanced Deep Learning PDF Malware Detection Algorithm
Optimized for 18K dataset with 34 features
Enhanced with attention mechanisms and ensemble learning
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import joblib
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPDFDetector:
    """Advanced Deep Learning PDF Malware Detection System"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.models = {}
        self.feature_names = None
        self.model_history = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def load_and_preprocess_data(self, csv_file_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess the PDF dataset"""
        logger.info("Loading dataset...")
        
        # Load data
        df = pd.read_csv(csv_file_path)
        logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Handle missing values and data cleaning
        df = self._clean_data(df)
        
        # Separate features and labels
        X = df.drop(['class'], axis=1)
        y = df['class']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        logger.info("Cleaning data...")
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'class':
                # Convert to categorical codes for non-target columns
                df[col] = pd.Categorical(df[col]).codes
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle outliers using IQR method for key features
        key_features = ['pdf_size', 'metadata_size', 'pages', 'xref_length']
        for feature in key_features:
            if feature in df.columns:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Data cleaned: {df.shape[0]} samples remaining")
        return df
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        logger.info("Performing feature engineering...")
        
        X_eng = X.copy()
        
        # Create ratio features
        if 'pdf_size' in X_eng.columns and 'metadata_size' in X_eng.columns:
            X_eng['metadata_ratio'] = X_eng['metadata_size'] / (X_eng['pdf_size'] + 1)
        
        if 'pages' in X_eng.columns and 'pdf_size' in X_eng.columns:
            X_eng['size_per_page'] = X_eng['pdf_size'] / (X_eng['pages'] + 1)
        
        # Create complexity score
        complexity_features = ['obj', 'endobj', 'stream', 'endstream', 'xref']
        if all(feat in X_eng.columns for feat in complexity_features):
            X_eng['complexity_score'] = X_eng[complexity_features].sum(axis=1)
        
        # Create suspicious features count
        suspicious_features = ['JS', 'Javascript', 'AA', 'OpenAction', 'launch', 'EmbeddedFile']
        if all(feat in X_eng.columns for feat in suspicious_features):
            X_eng['suspicious_count'] = X_eng[suspicious_features].sum(axis=1)
        
        # Log transformations for skewed features
        skewed_features = ['pdf_size', 'metadata_size', 'xref_length']
        for feature in skewed_features:
            if feature in X_eng.columns:
                X_eng[f'{feature}_log'] = np.log1p(X_eng[feature])
        
        logger.info(f"Feature engineering complete: {X_eng.shape[1]} features")
        return X_eng
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for training...")
        
        # Feature engineering
        X_eng = self.feature_engineering(X)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(25, X_eng.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_eng, y)
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Data prepared: {X_scaled.shape[1]} features selected")
        return X_scaled, y_encoded
    
    def create_attention_model(self, input_dim: int) -> keras.Model:
        """Create advanced neural network with attention mechanism"""
        inputs = keras.Input(shape=(input_dim,), name='input_features')
        
        # Feature embedding
        x = layers.Dense(128, activation='relu', name='embedding')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-head attention mechanism
        attention_heads = 4
        head_size = 32
        
        attention_outputs = []
        for i in range(attention_heads):
            # Query, Key, Value projections
            query = layers.Dense(head_size, name=f'query_{i}')(x)
            key = layers.Dense(head_size, name=f'key_{i}')(x)
            value = layers.Dense(head_size, name=f'value_{i}')(x)
            
            # Attention scores
            attention_scores = layers.Dot(axes=[1, 1], name=f'attention_scores_{i}')([query, key])
            attention_weights = layers.Softmax(name=f'attention_weights_{i}')(attention_scores)
            
            # Apply attention to values
            attended = layers.Multiply(name=f'attended_{i}')([attention_weights, value])
            attention_outputs.append(attended)
        
        # Concatenate attention heads
        if len(attention_outputs) > 1:
            x = layers.Concatenate(name='multi_head_concat')(attention_outputs)
        else:
            x = attention_outputs[0]
        
        # Deep layers with residual connections
        residual = x
        x = layers.Dense(256, activation='relu', name='deep_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu', name='deep_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual connection
        if x.shape[-1] == residual.shape[-1]:
            x = layers.Add(name='residual_connection')([x, residual])
        
        # Final layers
        x = layers.Dense(64, activation='relu', name='final_hidden')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='AttentionPDFDetector')
        return model
    
    def create_ensemble_models(self, input_dim: int) -> Dict[str, keras.Model]:
        """Create ensemble of different model architectures"""
        models = {}
        
        # Model 1: Attention-based (created above)
        models['attention'] = self.create_attention_model(input_dim)
        
        # Model 2: Deep Feed-Forward
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        models['deep_ff'] = keras.Model(inputs=inputs, outputs=outputs, name='DeepFeedForward')
        
        # Model 3: Wide & Deep
        wide_inputs = keras.Input(shape=(input_dim,), name='wide_input')
        deep_inputs = keras.Input(shape=(input_dim,), name='deep_input')
        
        # Wide part (linear)
        wide = layers.Dense(1, activation='linear', name='wide_part')(wide_inputs)
        
        # Deep part
        deep = layers.Dense(128, activation='relu')(deep_inputs)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(64, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(32, activation='relu', name='deep_part')(deep)
        
        # Combine wide and deep
        combined = layers.Concatenate()([wide, deep])
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        models['wide_deep'] = keras.Model(inputs=[wide_inputs, deep_inputs], outputs=outputs, name='WideDeep')
        
        return models
    
    def compile_models(self, models: Dict[str, keras.Model]):
        """Compile all models with optimized settings"""
        for name, model in models.items():
            if name == 'wide_deep':
                # Special compilation for wide & deep
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                )
            else:
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                )
    
    def train_models(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, any]:
        """Train ensemble of models with advanced techniques"""
        logger.info("Training ensemble models...")
        
        # Create models
        models = self.create_ensemble_models(X.shape[1])
        self.compile_models(models)
        
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train each model
        histories = {}
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            
            model_callbacks = [early_stopping, reduce_lr]
            
            # Model checkpoint
            checkpoint = callbacks.ModelCheckpoint(
                f'best_{name}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            model_callbacks.append(checkpoint)
            
            # Prepare data for wide & deep model
            if name == 'wide_deep':
                train_data = [X, X]
            else:
                train_data = X
            
            # Train model
            history = model.fit(
                train_data, y,
                epochs=100,
                batch_size=32,
                validation_split=validation_split,
                callbacks=model_callbacks,
                verbose=1
            )
            
            histories[name] = history.history
            self.models[name] = model
        
        self.model_history = histories
        logger.info("Model training completed!")
        return histories
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for name, model in self.models.items():
            if name == 'wide_deep':
                pred = model.predict([X, X], verbose=0)
            else:
                pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        # Weighted ensemble (you can adjust weights based on validation performance)
        weights = np.array([0.4, 0.35, 0.25])  # attention, deep_ff, wide_deep
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating models...")
        
        results = {}
        
        # Individual model evaluation
        for name, model in self.models.items():
            if name == 'wide_deep':
                y_pred_proba = model.predict([X_test, X_test], verbose=0)
            else:
                y_pred_proba = model.predict(X_test, verbose=0)
            
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_pred_proba = y_pred_proba.flatten()
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'auc': auc,
                'accuracy': report['accuracy'],
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1': report['macro avg']['f1-score']
            }
        
        # Ensemble evaluation
        ensemble_pred_proba = self.predict_ensemble(X_test)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, ensemble_pred_proba)
        report = classification_report(y_test, ensemble_pred, output_dict=True)
        
        results['ensemble'] = {
            'auc': auc,
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score']
        }
        
        return results
    
    def plot_training_history(self):
        """Plot training history for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (name, history) in enumerate(self.model_history.items()):
            row = i // 2
            col = i % 2
            
            ax = axes[row, col]
            ax.plot(history['loss'], label='Training Loss')
            ax.plot(history['val_loss'], label='Validation Loss')
            ax.set_title(f'{name.title()} Model - Training History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path: str = 'pdf_detector_ensemble'):
        """Save the complete model pipeline"""
        # Create directory
        os.makedirs(model_path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model.save(f'{model_path}/{name}_model.h5')
        
        # Save preprocessors
        joblib.dump(self.scaler, f'{model_path}/scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_path}/label_encoder.pkl')
        joblib.dump(self.feature_selector, f'{model_path}/feature_selector.pkl')
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'model_names': list(self.models.keys()),
            'created_at': datetime.now().isoformat()
        }
        
        with open(f'{model_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def fit(self, csv_file_path: str) -> Dict[str, any]:
        """Complete training pipeline"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(csv_file_path)
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, 
            test_size=0.2, 
            stratify=y_processed, 
            random_state=self.random_state
        )
        
        # Train models
        histories = self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Plot training history
        self.plot_training_history()
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Save model
        self.save_model()
        
        return results

def main():
    """Main execution function"""
    # Initialize detector
    detector = AdvancedPDFDetector(random_state=42)
    
    # Train the model (replace with your CSV file path)
    csv_file_path = "/home/d3bugger/Projects/FINAL YEAR PROJECT/src/detector_app/ai/ml/dataset/Final.csv"  # Update this path
    
    try:
        results = detector.fit(csv_file_path)
        print("\nTraining completed successfully!")
        print("Models saved to 'pdf_detector_ensemble' directory")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()