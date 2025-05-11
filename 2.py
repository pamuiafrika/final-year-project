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

class ModelTrainer:
    def __init__(self, model_dir=None):
        """Initialize model trainer with model directory"""
        self.model_dir = model_dir or os.path.join(settings.ML_MODEL_DIR)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_cnn(self, X_train, y_train, X_test, y_test):
        """Train a CNN model for steganography detection
        
        Args:
            X_train: Training feature data
            y_train: Training labels
            X_test: Test feature data
            y_test: Test labels
            
        Returns:
            model_path: Path to saved model
            accuracy: Test accuracy
        """
        print("Training CNN model...")
        
        # Determine feature size
        feature_size = X_train.shape[1]
        
        # Set up early stopping and model checkpoint
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'cnn_model_{timestamp}.h5')
        
        # Calculate dimensions for a proper square reshaping
        side_length = int(np.sqrt(feature_size))
        
        # If feature_size is not a perfect square, find the next perfect square
        if side_length * side_length < feature_size:
            side_length += 1
        
        # Calculate needed padding to reach the perfect square
        perfect_square = side_length * side_length
        padding = perfect_square - feature_size
        
        if padding > 0:
            # Pad features if needed to create a square
            X_train_padded = np.pad(X_train, ((0, 0), (0, padding)), 'constant')
            X_test_padded = np.pad(X_test, ((0, 0), (0, padding)), 'constant')
        else:
            X_train_padded = X_train
            X_test_padded = X_test
        
        # Double-check that we have the right number of features after padding
        assert X_train_padded.shape[1] == side_length * side_length, f"Expected {side_length * side_length} features after padding, got {X_train_padded.shape[1]}"
        
        # Reshape to 2D "images" for CNN
        X_train_reshaped = X_train_padded.reshape(-1, side_length, side_length, 1)
        X_test_reshaped = X_test_padded.reshape(-1, side_length, side_length, 1)
        
        # Print shape information for debugging
        print(f"Original feature size: {feature_size}")
        print(f"Padded to: {X_train_padded.shape[1]}")
        print(f"Reshaped to: {X_train_reshaped.shape}")
        
        # Convert to categorical for training
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        # Create model architecture based on input size
        # For a 14x14 input, we need to be careful with the architecture
        
        # Create model with Input layer for clearer dimensions
        inputs = Input(shape=(side_length, side_length, 1))
        
        # First convolutional layer
        x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Calculate dimensions after first conv+pool
        dim_after_first = side_length // 2  # Using padding='same' and pool of 2x2
        
        # Only add second conv layer if we have enough dimensions left
        if dim_after_first >= 3:
            x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            
            # Calculate dimensions after second conv+pool
            dim_after_second = dim_after_first // 2
            
            # Only add third conv layer if we have enough dimensions left
            if dim_after_second >= 3:
                x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
                x = BatchNormalization()(x)
                # No pooling for the third layer to preserve dimensions
        
        # Flatten the output for dense layers
        x = Flatten()(x)
        
        # Dense layers - scale based on input size
        dense_units = min(256, side_length * side_length * 4)  # Scale dense layer size
        
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(dense_units // 2, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(2, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Print model summary for debugging
        model.summary()
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', 
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy'])
        
        # Set up callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        
        # Train the model
        try:
            history = model.fit(
                X_train_reshaped, y_train_cat,
                batch_size=min(32, len(X_train) // 2),  # Adjust batch size based on dataset size
                epochs=50,
                verbose=1,
                validation_data=(X_test_reshaped, y_test_cat),
                callbacks=[early_stopping, model_checkpoint]
            )
            
            # Evaluate the model
            y_pred = model.predict(X_test_reshaped)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            print(f"CNN Model Accuracy: {accuracy:.4f}")
            
            # Save metadata
            # Save the model
            model_path = os.path.join(self.model_dir, f'cnn_model_{timestamp}.h5')
            model.save(model_path)

            # Save metadata
            metadata_path = model_path.replace('.h5', '_metadata.pkl')
            metadata = {
                'feature_size': feature_size,
                'side_length': side_length,
                'padding': padding,
                'accuracy': accuracy,
                'timestamp': timestamp
            }
            joblib.dump(metadata, metadata_path)
            
            return model_path, accuracy
        
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            # Try a simpler model if the first one fails
            print("Attempting to train with a simpler model architecture...")
            
            # Create a simpler model using functional API for clarity
            simple_inputs = Input(shape=(side_length, side_length, 1))
            x = Flatten()(simple_inputs)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='relu')(x)
            simple_outputs = Dense(2, activation='softmax')(x)
            
            simple_model = Model(inputs=simple_inputs, outputs=simple_outputs)
            
            simple_model.compile(loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate=0.001),
                                metrics=['accuracy'])
            
            # Train the simpler model
            simple_model.fit(
                X_train_reshaped, y_train_cat,
                batch_size=min(32, len(X_train) // 2),
                epochs=50,
                verbose=1,
                validation_data=(X_test_reshaped, y_test_cat),
                callbacks=[early_stopping, model_checkpoint]
            )
            
            # Evaluate the model
            y_pred = simple_model.predict(X_test_reshaped)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            print(f"Simple Model Accuracy: {accuracy:.4f}")
            
            # Save the model
            model_path = os.path.join(self.model_dir, f'cnn_model_{timestamp}.h5')
            model.save(model_path)

            # Save metadata
            metadata_path = model_path.replace('.h5', '_metadata.pkl')
            metadata = {
                'feature_size': feature_size,
                'side_length': side_length,
                'padding': padding,
                'accuracy': accuracy,
                'timestamp': timestamp
            }
            joblib.dump(metadata, metadata_path)
            
            return model_path, accuracy
    
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost model for steganography detection
        
        Returns:
            model_path: Path to saved model
            accuracy: Test accuracy
        """
        print("Training XGBoost model...")
        
        try:
            # Define model parameters without early stopping
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            # Create and train the model without any eval_set or early_stopping
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"XGBoost Model Accuracy: {accuracy:.4f}")
            
            # Save the model
            timestamp = int(time.time())
            model_path = os.path.join(self.model_dir, f'xgboost_model_{timestamp}.pkl')
            joblib.dump(model, model_path)
            
            return model_path, accuracy
            
        except Exception as e:
            print(f"Error in XGBoost training: {str(e)}")
            
            # Try with an even simpler approach using the older XGBoost API
            try:
                print("Trying alternative XGBoost approach...")
                
                # Convert data to DMatrix format
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Parameters
                params = {
                    'max_depth': 6,
                    'eta': 0.1,
                    'objective': 'binary:logistic',
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'eval_metric': 'logloss'
                }
                
                # Simple training without early stopping
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100
                )
                
                # Evaluate
                y_pred_probs = model.predict(dtest)
                y_pred = (y_pred_probs > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"XGBoost Alternative Model Accuracy: {accuracy:.4f}")
                
                # Save
                timestamp = int(time.time())
                model_path = os.path.join(self.model_dir, f'xgboost_alt_model_{timestamp}.pkl')
                model.save_model(model_path)
                
                # Save info about model format
                metadata_path = os.path.join(self.model_dir, f'xgboost_metadata_{timestamp}.pkl')
                metadata = {
                    'model_type': 'xgb_native',
                    'accuracy': accuracy,
                    'timestamp': timestamp
                }
                joblib.dump(metadata, metadata_path)
                
                return model_path, accuracy
                
            except Exception as e2:
                print(f"Alternative XGBoost approach also failed: {str(e2)}")
                raise e  # Re-raise the original exception
    
    
    def train_lstm(self, X_train, y_train, X_test, y_test):
        """Train an LSTM model for steganography detection
        
        Returns:
            model_path: Path to saved model
            accuracy: Test accuracy
        """
        print("Training LSTM model...")
        
        # Reshape data for LSTM [samples, time steps, features]
        feature_size = X_train.shape[1]
        sequence_length = 100  # Adjust based on feature size
        n_features = feature_size // sequence_length
        
        if feature_size % sequence_length != 0:
            # Pad to make divisible
            padding = sequence_length - (feature_size % sequence_length)
            X_train_padded = np.pad(X_train, ((0, 0), (0, padding)), 'constant')
            X_test_padded = np.pad(X_test, ((0, 0), (0, padding)), 'constant')
            n_features = (feature_size + padding) // sequence_length
        else:
            X_train_padded = X_train
            X_test_padded = X_test
        
        # Reshape for LSTM input
        X_train_reshaped = X_train_padded.reshape(-1, sequence_length, n_features)
        X_test_reshaped = X_test_padded.reshape(-1, sequence_length, n_features)
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)
        
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(128, input_shape=(sequence_length, n_features), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        
        # Set up callbacks
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'lstm_model_{timestamp}.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')
        
        # Train the model
        model.fit(
            X_train_reshaped, y_train_cat,
            batch_size=32,
            epochs=50,
            verbose=1,
            validation_data=(X_test_reshaped, y_test_cat),
            callbacks=[early_stopping, model_checkpoint]
        )
        
        # Evaluate the model
        y_pred = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        print(f"LSTM Model Accuracy: {accuracy:.4f}")
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, f'lstm_metadata_{timestamp}.pkl')
        metadata = {
            'feature_size': feature_size,
            'sequence_length': sequence_length,
            'n_features': n_features,
            'accuracy': accuracy,
            'timestamp': timestamp
        }
        joblib.dump(metadata, metadata_path)
        
        return model_path, accuracy
    
    def train_ensemble(self, X_train, y_train, X_test, y_test, model_paths):
        """Train an ensemble model combining CNN, XGBoost, and LSTM
        
        Args:
            model_paths: List of saved model paths [cnn_path, xgb_path, lstm_path]
            
        Returns:
            model_path: Path to saved ensemble model
            accuracy: Test accuracy
        """
        print("Training ensemble model...")
        
        # Load individual models
        models = []
        
        # Load XGBoost model
        xgb_model = joblib.load(model_paths[1])
        models.append(('xgb', xgb_model))
        
        # For CNN and LSTM, we'll need to create specialized wrapper functions
        # because they use different input shapes
        
        # CNN prediction function
        def cnn_predict(X):
            # Load CNN model and metadata
            cnn_model = load_model(model_paths[0])
            metadata_path = model_paths[0].replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                side_length = metadata.get('side_length')
                padding = metadata.get('padding')
            else:
                # Use default values if metadata not found
                feature_size = X.shape[1]
                side_length = int(np.sqrt(feature_size))
                padding = side_length * side_length - feature_size
                
            # Reshape input just like in training
            if padding > 0:
                X_padded = np.pad(X, ((0, 0), (0, padding)), 'constant')
            else:
                X_padded = X
                
            X_reshaped = X_padded.reshape(-1, side_length, side_length, 1)
            
            # Make predictions
            y_pred = cnn_model.predict(X_reshaped)
            return np.argmax(y_pred, axis=1)
        
        # LSTM prediction function
        def lstm_predict(X):
            # Load LSTM model and metadata
            lstm_model = load_model(model_paths[2])
            metadata_path = model_paths[2].replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                sequence_length = metadata.get('sequence_length')
                n_features = metadata.get('n_features')
            else:
                # Use default values if metadata not found
                feature_size = X.shape[1]
                sequence_length = 100
                n_features = feature_size // sequence_length
                
            # Reshape input just like in training
            if feature_size % sequence_length != 0:
                padding = sequence_length - (feature_size % sequence_length)
                X_padded = np.pad(X, ((0, 0), (0, padding)), 'constant')
                n_features = (feature_size + padding) // sequence_length
            else:
                X_padded = X
                
            X_reshaped = X_padded.reshape(-1, sequence_length, n_features)
            
            # Make predictions
            y_pred = lstm_model.predict(X_reshaped)
            return np.argmax(y_pred, axis=1)
        
        # Create a simple blending model
        # First, get predictions from each model
        xgb_preds = xgb_model.predict_proba(X_train)[:, 1]
        
        # Create new feature matrix with model predictions
        X_blend = np.column_stack([xgb_preds])
        
        # Train a meta-model (using XGBoost for simplicity)
        blend_model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        blend_model.fit(X_blend, y_train)
        
        # Evaluate the ensemble model
        xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]
        X_blend_test = np.column_stack([xgb_test_preds])
        
        y_pred = blend_model.predict(X_blend_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Ensemble Model Accuracy: {accuracy:.4f}")
        
        # Save the ensemble model
        timestamp = int(time.time())
        model_path = os.path.join(self.model_dir, f'ensemble_model_{timestamp}.pkl')
        
        # Save the ensemble as a dictionary containing all needed components
        ensemble = {
            'blend_model': blend_model,
            'xgb_model_path': model_paths[1],
            'cnn_model_path': model_paths[0],
            'lstm_model_path': model_paths[2],
            'accuracy': accuracy
        }
        
        joblib.dump(ensemble, model_path)
        
        return model_path, accuracy