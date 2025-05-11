import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LSTM, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
from django.conf import settings
import time
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir=None):
        """Initialize model trainer with model directory"""
        self.model_dir = model_dir or os.path.join(settings.ML_MODEL_DIR)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models and return results
        
        Args:
            X_train, X_test: Feature data for training and testing
            y_train, y_test: Labels for training and testing
            
        Returns:
            Dictionary of model results
        """
        results = {}
        
        # Log shapes for debugging
        logger.warning(f"X_train shape: {X_train.shape}")
        logger.warning(f"y_train shape: {y_train.shape}")
        logger.warning(f"X_test shape: {X_test.shape}")
        logger.warning(f"y_test shape: {y_test.shape}")
        
        # Ensure X and y have the same number of samples
        # This is likely the root cause of your error
        assert X_train.shape[0] == y_train.shape[0], "Training data and labels must have the same number of samples"
        assert X_test.shape[0] == y_test.shape[0], "Test data and labels must have the same number of samples"
        
        # Train CNN model
        try:
            cnn_path, cnn_accuracy = self.train_cnn(X_train, y_train, X_test, y_test)
            results['cnn'] = {
                'model_path': cnn_path,
                'accuracy': cnn_accuracy
            }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Error training CNN model: {error_msg}")
            results['cnn'] = {'error': error_msg}
        
        # Train XGBoost model
        try:
            xgb_path, xgb_accuracy = self.train_xgboost(X_train, y_train, X_test, y_test)
            results['xgboost'] = {
                'model_path': xgb_path,
                'accuracy': xgb_accuracy
            }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Error training XGBoost model: {error_msg}")
            results['xgboost'] = {'error': error_msg}
        
        # Train LSTM model
        try:
            lstm_path, lstm_accuracy = self.train_lstm(X_train, y_train, X_test, y_test)
            results['lstm'] = {
                'model_path': lstm_path,
                'accuracy': lstm_accuracy
            }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Error training LSTM model: {error_msg}")
            results['lstm'] = {'error': error_msg}
        
        return results

    def reshape_for_cnn(self, X):
        """Reshape features for CNN input"""
        # Determine feature size
        if len(X.shape) > 2:  # If already in shape (samples, timesteps, features)
            # Flatten the 3D array to 2D first
            samples = X.shape[0]
            X_flattened = X.reshape(samples, -1)
        else:
            X_flattened = X
            
        feature_size = X_flattened.shape[1]
        
        # Calculate dimensions for a proper square reshaping
        side_length = int(np.ceil(np.sqrt(feature_size)))
        perfect_square = side_length * side_length
        
        # Calculate needed padding
        padding = perfect_square - feature_size
        
        # Pad features if needed
        if padding > 0:
            X_padded = np.pad(X_flattened, ((0, 0), (0, padding)), 'constant')
        else:
            X_padded = X_flattened
        
        # Reshape to square "images" for CNN
        X_reshaped = X_padded.reshape(-1, side_length, side_length, 1)
        
        return X_reshaped, side_length, padding

    def train_cnn(self, X_train, y_train, X_test, y_test):
        """Train a CNN model for steganography detection"""
        logger.warning("Training CNN model...")
        
        # Reshape data for CNN
        X_train_reshaped, side_length, padding = self.reshape_for_cnn(X_train)
        X_test_reshaped, _, _ = self.reshape_for_cnn(X_test)
        
        # Print dimensions for debugging
        logger.warning(f"Original feature size: {X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[1] * X_train.shape[2]}")
        logger.warning(f"Padded to: {X_train_reshaped.shape[1] * X_train_reshaped.shape[2]}")
        logger.warning(f"Reshaped to: {X_train_reshaped.shape}")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        # Create CNN model
        inputs = Input(shape=(side_length, side_length, 1), name='input_layer')
        
        # First conv block
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Second conv block
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Flatten and dense layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Set up callbacks
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'cnn_model_{timestamp}.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        
        # Train the model
        history = model.fit(
            X_train_reshaped, y_train_cat,
            batch_size=min(32, len(X_train) // 4),  # Adjust batch size based on dataset size
            epochs=50,
            verbose=1,
            validation_data=(X_test_reshaped, y_test_cat),
            callbacks=[early_stopping, model_checkpoint]
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Save model metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        metadata = {
            'side_length': side_length,
            'padding': padding,
            'accuracy': accuracy,
            'timestamp': timestamp
        }
        joblib.dump(metadata, metadata_path)
        
        return model_path, accuracy

    def reshape_for_xgboost(self, X):
        """Reshape features for XGBoost input"""
        # XGBoost needs 2D data (samples, features)
        if len(X.shape) > 2:  # If in shape (samples, timesteps, features)
            samples = X.shape[0]
            X_reshaped = X.reshape(samples, -1)  # Flatten to 2D
        else:
            X_reshaped = X
            
        return X_reshaped

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost model for steganography detection"""
        logger.warning("Training XGBoost model...")
        
        # Reshape data for XGBoost (2D array)
        X_train_reshaped = self.reshape_for_xgboost(X_train)
        X_test_reshaped = self.reshape_for_xgboost(X_test)
        
        # Define model parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False
        }
        
        # Create and train the model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_reshaped, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_reshaped)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the model
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'xgboost_model_{timestamp}.pkl')
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = model_path.replace('.pkl', '_metadata.pkl')
        metadata = {
            'accuracy': accuracy,
            'timestamp': timestamp,
            'feature_shape': X_train_reshaped.shape
        }
        joblib.dump(metadata, metadata_path)
        
        return model_path, accuracy

    def train_lstm(self, X_train, y_train, X_test, y_test):
        """Train an LSTM model for steganography detection"""
        logger.warning("Training LSTM model...")
        
        # LSTM requires 3D input (samples, timesteps, features)
        if len(X_train.shape) == 2:
            # If input is 2D, we need to reshape it to 3D
            samples = X_train.shape[0]
            features = X_train.shape[1]
            
            # Determine timesteps and features_per_timestep
            timesteps = 100  # Default value
            features_per_timestep = features // timesteps
            
            if features % timesteps != 0:
                # Pad features to make them divisible by timesteps
                padding = timesteps - (features % timesteps)
                X_train_padded = np.pad(X_train, ((0, 0), (0, padding)), 'constant')
                X_test_padded = np.pad(X_test, ((0, 0), (0, padding)), 'constant')
                features = X_train_padded.shape[1]
                features_per_timestep = features // timesteps
                
                # Reshape to 3D: (samples, timesteps, features_per_timestep)
                X_train_reshaped = X_train_padded.reshape(samples, timesteps, features_per_timestep)
                X_test_reshaped = X_test_padded.reshape(X_test.shape[0], timesteps, features_per_timestep)
            else:
                # Reshape directly
                X_train_reshaped = X_train.reshape(samples, timesteps, features_per_timestep)
                X_test_reshaped = X_test.reshape(X_test.shape[0], timesteps, features_per_timestep)
        else:
            # Already in 3D form
            X_train_reshaped = X_train
            X_test_reshaped = X_test
            
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        # Set up callbacks
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'lstm_model_{timestamp}.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        
        # Train the model
        history = model.fit(
            X_train_reshaped, y_train_cat,
            batch_size=min(32, len(X_train) // 4),
            epochs=50,
            verbose=1,
            validation_data=(X_test_reshaped, y_test_cat),
            callbacks=[early_stopping, model_checkpoint]
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Save model metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl')
        metadata = {
            'input_shape': (X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
            'accuracy': accuracy,
            'timestamp': timestamp
        }
        joblib.dump(metadata, metadata_path)
        
        return model_path, accuracy
    
    
    def train_ensemble(self, X_train, y_train, X_test, y_test, model_paths):
        """Train an ensemble model combining CNN, XGBoost, and LSTM
        
        Args:
            X_train, y_train: Training data and labels
            X_test, y_test: Testing data and labels
            model_paths: List of saved model paths [cnn_path, xgb_path, lstm_path]
            
        Returns:
            model_path: Path to saved ensemble model
            accuracy: Test accuracy
        """
        logger.warning("Training ensemble model...")
        
        # Prepare data for all models
        # For XGBoost - need flattened 2D data
        X_train_flat = self.reshape_for_xgboost(X_train)
        X_test_flat = self.reshape_for_xgboost(X_test)
        
        # Load the XGBoost model directly
        try:
            xgb_model = joblib.load(model_paths[1])
            xgb_train_preds = xgb_model.predict_proba(X_train_flat)[:, 1].reshape(-1, 1)
            xgb_test_preds = xgb_model.predict_proba(X_test_flat)[:, 1].reshape(-1, 1)
        except Exception as e:
            logger.warning(f"Error loading XGBoost model: {str(e)}")
            # Fallback if model can't be loaded
            xgb_train_preds = np.zeros((X_train.shape[0], 1))
            xgb_test_preds = np.zeros((X_test.shape[0], 1))
        
        # Load CNN model
        try:
            cnn_model = load_model(model_paths[0])
            metadata_path = model_paths[0].replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                cnn_metadata = joblib.load(metadata_path)
                side_length = cnn_metadata.get('side_length')
                padding = cnn_metadata.get('padding')
            else:
                # Use default values if metadata not found
                feature_size = X_train_flat.shape[1]
                side_length = int(np.ceil(np.sqrt(feature_size)))
                padding = (side_length * side_length) - feature_size

            # Reshape data for CNN
            X_train_cnn, _, _ = self.reshape_for_cnn(X_train)
            X_test_cnn, _, _ = self.reshape_for_cnn(X_test)
            
            # Get CNN predictions
            cnn_train_preds = cnn_model.predict(X_train_cnn)[:, 1].reshape(-1, 1)
            cnn_test_preds = cnn_model.predict(X_test_cnn)[:, 1].reshape(-1, 1)
        except Exception as e:
            logger.warning(f"Error loading CNN model: {str(e)}")
            # Fallback if model can't be loaded
            cnn_train_preds = np.zeros((X_train.shape[0], 1))
            cnn_test_preds = np.zeros((X_test.shape[0], 1))
        
        # Load LSTM model
        try:
            lstm_model = load_model(model_paths[2])
            metadata_path = model_paths[2].replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                lstm_metadata = joblib.load(metadata_path)
                input_shape = lstm_metadata.get('input_shape')
                timesteps = input_shape[0] if input_shape else 100
                features_per_timestep = input_shape[1] if input_shape else X_train_flat.shape[1] // 100
            else:
                # Default values
                timesteps = 100
                features_per_timestep = X_train_flat.shape[1] // timesteps
                
            # Handle LSTM data preparation - making sure we don't hit any dimension issues
            if len(X_train.shape) == 2:
                samples = X_train.shape[0]
                features = X_train.shape[1]
                
                # Ensure features is divisible by timesteps
                if features % timesteps != 0:
                    padding = timesteps - (features % timesteps)
                    X_train_padded = np.pad(X_train, ((0, 0), (0, padding)), 'constant')
                    X_test_padded = np.pad(X_test, ((0, 0), (0, padding)), 'constant')
                    features = X_train_padded.shape[1]
                else:
                    X_train_padded = X_train
                    X_test_padded = X_test
                    
                features_per_timestep = features // timesteps
                X_train_lstm = X_train_padded.reshape(samples, timesteps, features_per_timestep)
                X_test_lstm = X_test_padded.reshape(X_test.shape[0], timesteps, features_per_timestep)
            else:
                # Already in 3D form
                X_train_lstm = X_train
                X_test_lstm = X_test
            
            # Get LSTM predictions
            lstm_train_preds = lstm_model.predict(X_train_lstm)[:, 1].reshape(-1, 1)
            lstm_test_preds = lstm_model.predict(X_test_lstm)[:, 1].reshape(-1, 1)
        except Exception as e:
            logger.warning(f"Error loading LSTM model: {str(e)}")
            # Fallback if model can't be loaded
            lstm_train_preds = np.zeros((X_train.shape[0], 1))
            lstm_test_preds = np.zeros((X_test.shape[0], 1))
        
        # Create meta-features by combining predictions from base models
        meta_train = np.hstack([xgb_train_preds, cnn_train_preds, lstm_train_preds])
        meta_test = np.hstack([xgb_test_preds, cnn_test_preds, lstm_test_preds])
        
        # Train a meta-classifier (blender) on these predictions
        meta_classifier = xgb.XGBClassifier(
            n_estimators=50, 
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False
        )
        
        meta_classifier.fit(meta_train, y_train)
        
        # Make final predictions using the meta-classifier
        y_pred = meta_classifier.predict(meta_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.warning(f"Ensemble Model Accuracy: {accuracy:.4f}")
        
        # Save the ensemble model
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'ensemble_model_{timestamp}.pkl')
        
        # Save the ensemble as a dictionary containing necessary components
        ensemble = {
            'meta_classifier': meta_classifier,
            'model_paths': model_paths,
            'accuracy': accuracy,
            'timestamp': timestamp
        }
        
        joblib.dump(ensemble, model_path)
        
        return model_path, accuracy

    def predict(self, X, model_type='cnn', model_path=None):
        """Make predictions using a trained model
        
        Args:
            X: Features to predict
            model_type: Type of model to use ('cnn', 'xgboost', 'lstm')
            model_path: Optional path to a specific model
            
        Returns:
            Predicted classes and probabilities
        """
        if model_path is None:
            # Use the most recent model of the specified type
            model_files = [f for f in os.listdir(self.model_dir) 
                         if f.startswith(f'{model_type}_model_') and 
                         (f.endswith('.h5') or f.endswith('.pkl'))]
            if not model_files:
                raise ValueError(f"No trained {model_type} model found")
            
            # Sort by timestamp (which is part of the filename)
            model_files.sort(reverse=True)
            model_path = os.path.join(self.model_dir, model_files[0])
        
        # Load model metadata
        metadata_path = model_path.replace('.h5', '_metadata.pkl').replace('.pkl', '_metadata.pkl')
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
        
        if model_type == 'cnn':
            # Reshape input for CNN
            side_length = metadata.get('side_length')
            padding = metadata.get('padding')
            
            if side_length is None:
                # Try to infer shape if not in metadata
                if len(X.shape) > 2:
                    X_flattened = X.reshape(X.shape[0], -1)
                else:
                    X_flattened = X
                feature_size = X_flattened.shape[1]
                side_length = int(np.ceil(np.sqrt(feature_size)))
                padding = side_length * side_length - feature_size
            
            if padding > 0:
                if len(X.shape) > 2:
                    X_flattened = X.reshape(X.shape[0], -1)
                else:
                    X_flattened = X
                X_padded = np.pad(X_flattened, ((0, 0), (0, padding)), 'constant')
            else:
                if len(X.shape) > 2:
                    X_padded = X.reshape(X.shape[0], -1)
                else:
                    X_padded = X
            
            X_reshaped = X_padded.reshape(-1, side_length, side_length, 1)
            
            # Load model and predict
            model = load_model(model_path)
            y_pred_proba = model.predict(X_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        elif model_type == 'xgboost':
            # Reshape input for XGBoost
            if len(X.shape) > 2:
                X_reshaped = X.reshape(X.shape[0], -1)
            else:
                X_reshaped = X
                
            # Load model and predict
            model = joblib.load(model_path)
            y_pred = model.predict(X_reshaped)
            y_pred_proba = model.predict_proba(X_reshaped)
            
        elif model_type == 'lstm':
            # LSTM requires 3D input (samples, timesteps, features)
            input_shape = metadata.get('input_shape')
            
            if input_shape:
                timesteps, features_per_timestep = input_shape
            else:
                # Use default values if not in metadata
                timesteps = 100
                if len(X.shape) > 2:
                    features_per_timestep = X.shape[2]
                else:
                    features = X.shape[1]
                    features_per_timestep = features // timesteps
            
            if len(X.shape) == 2:
                # Reshape 2D to 3D
                samples = X.shape[0]
                features = X.shape[1]
                
                # Pad if necessary
                if features < timesteps * features_per_timestep:
                    padding = (timesteps * features_per_timestep) - features
                    X_padded = np.pad(X, ((0, 0), (0, padding)), 'constant')
                    X_reshaped = X_padded.reshape(samples, timesteps, features_per_timestep)
                elif features > timesteps * features_per_timestep:
                    # Truncate
                    X_truncated = X[:, :timesteps * features_per_timestep]
                    X_reshaped = X_truncated.reshape(samples, timesteps, features_per_timestep)
                else:
                    # Perfect fit
                    X_reshaped = X.reshape(samples, timesteps, features_per_timestep)
            else:
                # Already 3D
                X_reshaped = X
                
            # Load model and predict
            model = load_model(model_path)
            y_pred_proba = model.predict(X_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return y_pred, y_pred_proba
    
    
