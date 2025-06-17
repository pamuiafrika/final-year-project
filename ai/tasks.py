# Celery tasks (these would normally be in a separate tasks.py file)
import os
from celery import shared_task
from django.conf import settings
from django.utils import timezone
import joblib
from .model_training import PDFSteganographyDetector
from .models import MLModel, ModelEvaluation, TrainingSession, TrainingVisualization
import pandas as pd
import numpy as np
from django.core.files.base import ContentFile
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

@shared_task
def analyze_dataset_task(session_id):
    """Background task to analyze dataset"""
    try:
        session = TrainingSession.objects.get(id=session_id)
        df = pd.read_csv(session.training_data_file.path)
        
        # Update session with data info
        session.data_shape = {'rows': len(df), 'columns': len(df.columns)}
        
        if 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            session.class_distribution = label_counts
        
        session.save()
        
    except Exception as e:
        session = TrainingSession.objects.get(id=session_id)
        session.status = 'failed'
        session.error_message = str(e)
        session.save()

@shared_task
def train_models_task(session_id):
    """Background task to train models"""
    try:
        session = TrainingSession.objects.get(id=session_id)
        
        # Initialize detector
        detector = PDFSteganographyDetector()
        
        # Load and prepare data
        df = pd.read_csv(session.training_data_file.path)
        X, y = detector.prepare_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessor and preprocess data
        detector.create_preprocessor()
        X_train_processed = detector.preprocessor.fit_transform(X_train)
        X_test_processed = detector.preprocessor.transform(X_test)
        
        # Create models
        xgb_model = MLModel.objects.create(
            name=f"XGBoost - {session.name}",
            model_type='xgboost',
            created_by=session.created_by,
            status='training'
        )
        
        wide_deep_model = MLModel.objects.create(
            name=f"Wide & Deep - {session.name}",
            model_type='wide_deep',
            created_by=session.created_by,
            status='training'
        )
        
        # Train XGBoost model
        xgb_model_obj = detector.train_xgboost(X_train_processed, X_test_processed, y_train, y_test)
        
        # Save XGBoost model
        xgb_model_path = os.path.join(settings.MODELS_DIR, f'xgb/xgb_{session.id}.pkl')
        joblib.dump(xgb_model_obj, xgb_model_path)
        xgb_model.model_file.name = f'xgb/xgb_{session.id}.pkl'
        xgb_model.save()

        # Calculate XGBoost metrics
        xgb_train_pred = xgb_model_obj.predict(X_train_processed)
        xgb_test_pred = xgb_model_obj.predict(X_test_processed)
        xgb_train_prob = xgb_model_obj.predict_proba(X_train_processed)[:, 1]
        xgb_test_prob = xgb_model_obj.predict_proba(X_test_processed)[:, 1]
        
        xgb_metrics = {
            'accuracy': accuracy_score(y_test, xgb_test_pred),
            'precision': precision_score(y_test, xgb_test_pred),
            'recall': recall_score(y_test, xgb_test_pred),
            'f1': f1_score(y_test, xgb_test_pred),
            'auc_roc': roc_auc_score(y_test, xgb_test_prob),
            'confusion_matrix': confusion_matrix(y_test, xgb_test_pred)
        }

        # Calculate ROC curve for XGBoost
        xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_test_prob)
        xgb_metrics['fpr'] = xgb_fpr
        xgb_metrics['tpr'] = xgb_tpr

        # Update XGBoost model with evaluation metrics
        xgb_model.accuracy = xgb_metrics['accuracy']
        xgb_model.metadata = {
            'precision': xgb_metrics['precision'],
            'recall': xgb_metrics['recall'],
            'f1_score': xgb_metrics['f1'],
            'auc_roc': xgb_metrics['auc_roc'],
            'confusion_matrix': xgb_metrics['confusion_matrix'].tolist(),
            'training_session_id': session.id
        }
        xgb_model.save()

        # Create XGBoost visualizations
        create_visualization(
            session=session,
            model=xgb_model,
            chart_type='confusion_matrix',
            data={'confusion_matrix': xgb_metrics['confusion_matrix'].tolist()},
            title='XGBoost Confusion Matrix'
        )

        create_visualization(
            session=session,
            model=xgb_model,
            chart_type='roc_curve',
            data={
                'fpr': xgb_metrics['fpr'].tolist(),
                'tpr': xgb_metrics['tpr'].tolist(),
                'auc': xgb_metrics['auc_roc']
            },
            title='XGBoost ROC Curve'
        )

        # Train Wide & Deep model

        keras_model, history = detector.train_wide_deep(X_train_processed, X_test_processed, y_train, y_test)
        wide_deep_model_path = os.path.join(settings.MODELS_DIR, f'wide_deep/wide_deep_{session.id}.keras')
        keras_model.save(wide_deep_model_path)
        wide_deep_model.model_file.name = f'wide_deep/wide_deep_{session.id}.keras'

        # Calculate metrics for Wide & Deep model
        train_pred_prob = keras_model.predict(X_train_processed)
        test_pred_prob = keras_model.predict(X_test_processed)
        test_pred = (test_pred_prob > 0.5).astype(int)
        
        wide_deep_metrics = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'auc_roc': roc_auc_score(y_test, test_pred_prob),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'fpr': None,  # Will be computed below
            'tpr': None,  # Will be computed below
        }
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, test_pred_prob)
        wide_deep_metrics['fpr'] = fpr
        wide_deep_metrics['tpr'] = tpr

        # Update Wide & Deep model with evaluation metrics
        wide_deep_model.accuracy = wide_deep_metrics['accuracy']
        wide_deep_model.metadata = {
            'precision': wide_deep_metrics['precision'],
            'recall': wide_deep_metrics['recall'],
            'f1_score': wide_deep_metrics['f1'],
            'auc_roc': wide_deep_metrics['auc_roc'],
            'confusion_matrix': wide_deep_metrics['confusion_matrix'].tolist(),
            'training_session_id': session.id,
            'training_history': history.history if history else {}
        }
        wide_deep_model.save()

        # Create Wide & Deep visualizations
        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='confusion_matrix',
            data={'confusion_matrix': wide_deep_metrics['confusion_matrix'].tolist()},
            title='Wide & Deep Confusion Matrix'
        )

        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='roc_curve',
            data={
                'fpr': wide_deep_metrics['fpr'].tolist(),
                'tpr': wide_deep_metrics['tpr'].tolist(),
                'auc': wide_deep_metrics['auc_roc']
            },
            title='Wide & Deep ROC Curve'
        )

        create_visualization(
            session=session,
            model=wide_deep_model,
            chart_type='training_history',
            data={
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            },
            title='Wide & Deep Training History'
        )

        # Update session status
        session.status = 'completed'
        session.completed_at = timezone.now()
        session.save()
        
        # Update model status
        xgb_model.status = 'completed'
        wide_deep_model.status = 'completed'
        xgb_model.save()
        wide_deep_model.save()
        
        
    except Exception as e:
        session = TrainingSession.objects.get(id=session_id)
        session.status = 'failed'
        session.error_message = str(e)
        session.save()


@shared_task
def create_visualization(session, model, chart_type, data, title):
    """Create and save a visualization"""
    plt.figure(figsize=(10, 8))
    
    if chart_type == 'confusion_matrix':
        cm = np.array(data['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
    elif chart_type == 'roc_curve':
        plt.plot(data['fpr'], data['tpr'], label=f'ROC Curve (AUC = {data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend()
        
    elif chart_type == 'feature_importance':
        features = list(data['features'])
        importance = list(data['importance'])
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx[-20:]]  # Top 20
        importance = [importance[i] for i in sorted_idx[-20:]]
        
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {title}')
        plt.tight_layout()
        
    elif chart_type == 'training_history':
        epochs = range(1, len(data['loss']) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(epochs, data['loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, data['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, data['accuracy'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, data['val_accuracy'], 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
    
    # Save plot to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Create visualization record
    visualization = TrainingVisualization.objects.create(
        training_session=session,
        model=model,
        chart_type=chart_type,
        title=title,
        chart_data=data
    )
    
    # Save image
    image_file = ContentFile(image_png)
    visualization.chart_image.save(
        f'{chart_type}_{model.id}_{session.id}.png',
        image_file,
        save=True
    )
    
    return visualization