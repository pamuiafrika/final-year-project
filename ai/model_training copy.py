#!/usr/bin/env python3
"""
PDF Steganography Detection - ML Pipeline
Trains XGBoost and Wide & Deep models for PDF steganography detection
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import warnings
warnings.filterwarnings('ignore')

class PDFSteganographyDetector:
    def __init__(self):
        self.feature_columns = [
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
        
        self.numeric_features = [
            'file_size_bytes', 'pdf_version', 'num_pages', 'num_objects',
            'num_stream_objects', 'num_embedded_files', 'num_annotation_objects',
            'num_form_fields', 'creation_date_ts', 'mod_date_ts', 'creation_mod_date_diff',
            'avg_entropy_per_stream', 'max_entropy_per_stream', 'min_entropy_per_stream',
            'std_entropy_per_stream', 'num_streams_entropy_gt_threshold',
            'num_encrypted_streams', 'num_corrupted_objects', 'num_objects_with_random_markers',
            'num_suspicious_filters', 'avg_file_size_per_page', 'compression_ratio',
            'num_eof_markers', 'extraction_time_ms', 'error_count'
        ]
        
        self.categorical_features = [
            'has_broken_name_trees', 'has_javascript', 'has_launch_actions', 'extraction_success'
        ]
        
        self.preprocessor = None
        self.xgb_model = None
        self.keras_model = None
        self.label_encoder = LabelEncoder()
        
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        # Numeric features: impute missing values and scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical features: convert to string, impute, then encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='False')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Filter out unknown labels for training
        train_data = df[df['label'] != -1].copy()
        
        # Convert boolean columns to string representation for consistent processing
        for col in self.categorical_features:
            if col in train_data.columns:
                # Convert boolean values to strings
                if train_data[col].dtype == bool:
                    train_data[col] = train_data[col].astype(str)
                else:
                    # Handle mixed types by converting to string first
                    train_data[col] = train_data[col].astype(str).str.lower()
                    # Map common boolean representations
                    boolean_mapping = {
                        'true': 'True',
                        'false': 'False', 
                        '1': 'True',
                        '0': 'False',
                        '1.0': 'True',
                        '0.0': 'False',
                        'yes': 'True',
                        'no': 'False'
                    }
                    train_data[col] = train_data[col].map(boolean_mapping).fillna('False')
        
        X = train_data[self.feature_columns]
        y = train_data['label']

        return X, y

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Check class distribution
        unique_classes = np.unique(y_train)
        print(f"Unique classes in training set: {unique_classes}")
        
        if len(unique_classes) < 2:
            print("Warning: Only one class found in training data. Cannot train binary classifier.")
            print("Skipping XGBoost training...")
            return None
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20
        }
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        if self.xgb_model is not None:
            train_pred = self.xgb_model.predict(X_train)
            test_pred = self.xgb_model.predict(X_test)
            train_prob = self.xgb_model.predict_proba(X_train)[:, 1]
            test_prob = self.xgb_model.predict_proba(X_test)[:, 1]
            
            print(f"XGBoost Train AUC: {roc_auc_score(y_train, train_prob):.4f}")
            print(f"XGBoost Test AUC: {roc_auc_score(y_test, test_prob):.4f}")
            print("\nXGBoost Test Classification Report:")
            print(classification_report(y_test, test_pred))
        
        return self.xgb_model
    
    def create_wide_deep_model(self, n_features):
        """Create Wide & Deep model using Keras Functional API"""
        # Input layer
        input_layer = layers.Input(shape=(n_features,), name='features')
        
        # Wide part - linear transformation
        wide = layers.Dense(1, activation='linear', name='wide')(input_layer)
        
        # Deep part - neural network
        deep = layers.Dense(128, activation='relu', name='deep_1')(input_layer)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(64, activation='relu', name='deep_2')(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(32, activation='relu', name='deep_3')(deep)
        deep = layers.Dropout(0.1)(deep)
        deep = layers.Dense(1, activation='linear', name='deep_output')(deep)
        
        # Combine wide and deep
        combined = layers.Add(name='wide_deep_add')([wide, deep])
        output = layers.Dense(1, activation='sigmoid', name='output')(combined)
        
        # Create model
        model = keras.Model(inputs=input_layer, outputs=output, name='wide_deep_model')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'auc']
        )
        
        return model
    
    def train_wide_deep(self, X_train, X_test, y_train, y_test):
        """Train Wide & Deep model"""
        print("\nTraining Wide & Deep model...")
        
        # Check class distribution
        unique_classes = np.unique(y_train)
        print(f"Unique classes in training set: {unique_classes}")
        
        if len(unique_classes) < 2:
            print("Warning: Only one class found in training data. Cannot train binary classifier.")
            print("Skipping Wide & Deep training...")
            return None, None
        
        # Create model
        self.keras_model = self.create_wide_deep_model(X_train.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.keras_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_pred_prob = self.keras_model.predict(X_train)
        test_pred_prob = self.keras_model.predict(X_test)
        train_pred = (train_pred_prob > 0.5).astype(int)
        test_pred = (test_pred_prob > 0.5).astype(int)
        
        print(f"\nWide & Deep Train AUC: {roc_auc_score(y_train, train_pred_prob):.4f}")
        print(f"Wide & Deep Test AUC: {roc_auc_score(y_test, test_pred_prob):.4f}")
        print("\nWide & Deep Test Classification Report:")
        print(classification_report(y_test, test_pred))
        
        return self.keras_model, history
    
    def save_models(self, models_dir='models'):
        """Save trained models and preprocessor"""
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        # Save preprocessor (always available if we got this far)
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, f'{models_dir}/preprocessor.pkl')
            print(f"Preprocessor saved to {models_dir}/preprocessor.pkl")
        
        # Save XGBoost model
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, f'{models_dir}/xgboost_model.pkl')
            print(f"XGBoost model saved to {models_dir}/xgboost_model.pkl")
        
        # Save Keras model
        if self.keras_model is not None:
            self.keras_model.save(f'{models_dir}/wide_deep_model.h5')
            print(f"Wide & Deep model saved to {models_dir}/wide_deep_model.h5")
    
    def load_models(self, models_dir='models'):
        """Load trained models and preprocessor"""
        # Load preprocessor
        self.preprocessor = joblib.load(f'{models_dir}/preprocessor.pkl')
        
        # Load XGBoost model
        self.xgb_model = joblib.load(f'{models_dir}/xgboost_model.pkl')
        
        # Load Keras model
        self.keras_model = keras.models.load_model(f'{models_dir}/wide_deep_model.h5')
        
        print("Models loaded successfully!")
    
    def predict(self, X):
        """Make predictions using both models"""
        if self.preprocessor is None:
            raise ValueError("Models not trained or loaded!")
        
        # Apply the same preprocessing as in prepare_data
        X_processed_input = X.copy()
        for col in self.categorical_features:
            if col in X_processed_input.columns:
                if X_processed_input[col].dtype == bool:
                    X_processed_input[col] = X_processed_input[col].astype(str)
                else:
                    X_processed_input[col] = X_processed_input[col].astype(str).str.lower()
                    boolean_mapping = {
                        'true': 'True',
                        'false': 'False', 
                        '1': 'True',
                        '0': 'False',
                        '1.0': 'True',
                        '0.0': 'False',
                        'yes': 'True',
                        'no': 'False'
                    }
                    X_processed_input[col] = X_processed_input[col].map(boolean_mapping).fillna('False')
        
        # Preprocess data
        X_processed = self.preprocessor.transform(X_processed_input)
        
        # XGBoost predictions
        xgb_prob = self.xgb_model.predict_proba(X_processed)[:, 1]
        xgb_pred = (xgb_prob > 0.5).astype(int)
        
        # Wide & Deep predictions
        keras_prob = self.keras_model.predict(X_processed).flatten()
        keras_pred = (keras_prob > 0.5).astype(int)
        
        return {
            'xgboost_prediction': xgb_pred,
            'xgboost_probability': xgb_prob,
            'wide_deep_prediction': keras_pred,
            'wide_deep_probability': keras_prob
        }
    
    def train_pipeline(self, csv_file_path):
        """Complete training pipeline"""
        print("Loading data...")
        df = pd.read_csv(csv_file_path)
        
        print("Data info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for categorical columns data types
        for col in self.categorical_features:
            if col in df.columns:
                print(f"{col}: dtype={df[col].dtype}, unique values={df[col].unique()[:10]}")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Create preprocessor
        self.create_preprocessor()
        
        # Split data (skip if only one class)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Error: Only one class found in dataset: {unique_classes}")
            print("Cannot train binary classifier. You need both steganographic (1) and clean (0) PDF samples.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit preprocessor
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print(f"Training set shape: {X_train_processed.shape}")
        print(f"Test set shape: {X_test_processed.shape}")
        print(f"Class distribution - 0: {sum(y_train == 0)}, 1: {sum(y_train == 1)}")
        
        # Train models
        xgb_model = self.train_xgboost(X_train_processed, X_test_processed, y_train, y_test)
        keras_model, history = self.train_wide_deep(X_train_processed, X_test_processed, y_train, y_test)
        
        # Only save models if they were successfully trained
        if xgb_model is not None or keras_model is not None:
            self.save_models()
            print("\nTraining completed successfully!")
        else:
            print("\nWarning: No models were trained due to insufficient class diversity in the dataset.")
            print("You need both positive (steganographic) and negative (clean) PDF samples to train the classifier.")

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = PDFSteganographyDetector()
    
    # Train pipeline (assuming you have features.csv)
    detector.train_pipeline('features.csv')
    
    # Example of loading and predicting
    # detector.load_models()
    # predictions = detector.predict(new_data)
    
    print("PDF Steganography Detection Pipeline Ready!")