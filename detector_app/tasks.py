import os
import json
from celery import shared_task
from django.conf import settings
from .models import Dataset, TrainedModel, PDFScan, BulkScan
from .ml.data_processor import DataProcessor
from .ml.model_trainer import ModelTrainer
from .ml.detector import StegoPDFDetector

@shared_task
def process_single_pdf(pdf_scan_id):
    """Process a single PDF file using the active models"""
    pdf_scan = PDFScan.objects.get(id=pdf_scan_id)
    pdf_scan.status = 'processing'
    pdf_scan.save()
    
    print("Processing PDF File")
    
    try:
        # Get active models
        active_models = TrainedModel.objects.filter(is_active=True)
        
        if not active_models.exists():
            pdf_scan.status = 'failed'
            pdf_scan.save()
            return {'error': 'No active models found'}
        
        # Use the first active model for now
        model = active_models.first()
        
        # Initialize detector
        detector = StegoPDFDetector(model_path=model.file_path)
        
        # Predict - detector.detect returns a dictionary, not a tuple
        pdf_path = os.path.join(settings.MEDIA_ROOT, pdf_scan.file.name)
        result_dict = detector.detect(pdf_path)
        
        # Check if detection was successful
        if not result_dict.get('success', False):
            pdf_scan.status = 'failed'
            pdf_scan.error_message = result_dict.get('error', 'Unknown error during detection')
            pdf_scan.save()
            return {'error': pdf_scan.error_message}
        
        # Extract values from the result dictionary
        is_stego = result_dict.get('is_stego', False)
        confidence = result_dict.get('confidence', 0.0)
        details = result_dict.get('details', {})
        
        # Update scan object with results
        pdf_scan.status = 'completed'
        pdf_scan.result = 'stego' if is_stego else 'clean'
        pdf_scan.confidence = confidence
        pdf_scan.model_used = model
        
        # Store additional details if needed
        if details:
            pdf_scan.additional_info = json.dumps(details)
            
        pdf_scan.save()
        
        return {
            'id': pdf_scan.id,
            'status': pdf_scan.status,
            'result': pdf_scan.result,
            'confidence': confidence,
            'details': details
        }
        
    except Exception as e:
        pdf_scan.status = 'failed'
        pdf_scan.error_message = str(e)
        pdf_scan.save()
        return {'error': str(e)}


@shared_task
def process_bulk_pdfs(bulk_scan_id, pdf_ids):
    """Process multiple PDF files in bulk"""
    bulk_scan = BulkScan.objects.get(id=bulk_scan_id)
    bulk_scan.status = 'processing'
    bulk_scan.save()
    
    try:
        # Get active models
        active_models = TrainedModel.objects.filter(is_active=True)
        
        if not active_models.exists():
            bulk_scan.status = 'failed'
            bulk_scan.save()
            return {'error': 'No active models found'}
        
        # Use the first active model for now
        model = active_models.first()
        
        # Initialize detector
        detector = StegoPDFDetector(model_path=model.file_path)
        
        # Process each PDF
        clean_count = 0
        stego_count = 0
        
        for pdf_id in pdf_ids:
            pdf_scan = PDFScan.objects.get(id=pdf_id)
            pdf_scan.status = 'processing'
            pdf_scan.model_used = model
            pdf_scan.save()
            
            try:
                # Predict using fixed method
                pdf_path = os.path.join(settings.MEDIA_ROOT, pdf_scan.file.name)
                result_dict = detector.detect(pdf_path)
                
                if not result_dict.get('success', False):
                    pdf_scan.status = 'failed'
                    pdf_scan.error_message = result_dict.get('error', 'Unknown error during detection')
                    pdf_scan.save()
                    continue
                
                # Extract values from the result dictionary
                is_stego = result_dict.get('is_stego', False)
                confidence = result_dict.get('confidence', 0.0)
                
                # Update scan object with results
                pdf_scan.status = 'completed'
                pdf_scan.result = 'stego' if is_stego else 'clean'
                pdf_scan.confidence = confidence
                pdf_scan.save()
                
                # Update counts
                if pdf_scan.result == 'stego':
                    stego_count += 1
                else:
                    clean_count += 1
                
                # Update bulk scan progress
                bulk_scan.processed_files += 1
                bulk_scan.clean_count = clean_count
                bulk_scan.stego_count = stego_count
                bulk_scan.save()
                
            except Exception as e:
                pdf_scan.status = 'failed'
                pdf_scan.error_message = str(e)
                pdf_scan.save()
        
        # Complete bulk scan
        bulk_scan.status = 'completed'
        bulk_scan.save()
        
        return {
            'id': bulk_scan.id,
            'status': bulk_scan.status,
            'total_files': bulk_scan.total_files,
            'processed_files': bulk_scan.processed_files,
            'clean_count': bulk_scan.clean_count,
            'stego_count': bulk_scan.stego_count
        }
        
    except Exception as e:
        bulk_scan.status = 'failed'
        bulk_scan.save()
        return {'error': str(e)}

@shared_task
def train_model_task(dataset_id):
    """Train a model using specified dataset"""
    dataset = Dataset.objects.get(id=dataset_id)
    
    try:
        # Process data
        data_processor = DataProcessor()
        X_train, X_test, y_train, y_test = data_processor.process_dataset(dataset_id)
        
        # Print shapes for debugging
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Verify data cardinality before proceeding
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError(f"Data cardinality mismatch: X_train={len(X_train)}, y_train={len(y_train)}, X_test={len(X_test)}, y_test={len(y_test)}")
        
        # Train models
        trainer = ModelTrainer()
        
        model_results = {}
        
        # Train and save CNN model
        try:
            cnn_model_path, cnn_accuracy = trainer.train_cnn(X_train, y_train, X_test, y_test)
            cnn_model = TrainedModel.objects.create(
                name=f"CNN Model - {dataset.name}",
                model_type='cnn',
                dataset=dataset,
                accuracy=cnn_accuracy,
                file_path=cnn_model_path
            )
            model_results['cnn'] = {'id': cnn_model.id, 'accuracy': cnn_accuracy}
        except Exception as e:
            print(f"Error training CNN model: {str(e)}")
            model_results['cnn'] = {'error': str(e)}
        
        # Train and save XGBoost model
        try:
            xgb_model_path, xgb_accuracy = trainer.train_xgboost(X_train, y_train, X_test, y_test)
            xgb_model = TrainedModel.objects.create(
                name=f"XGBoost Model - {dataset.name}",
                model_type='xgboost',
                dataset=dataset,
                accuracy=xgb_accuracy,
                file_path=xgb_model_path
            )
            model_results['xgboost'] = {'id': xgb_model.id, 'accuracy': xgb_accuracy}
        except Exception as e:
            print(f"Error training XGBoost model: {str(e)}")
            model_results['xgboost'] = {'error': str(e)}
        
        # Train and save LSTM model
        try:
            lstm_model_path, lstm_accuracy = trainer.train_lstm(X_train, y_train, X_test, y_test)
            lstm_model = TrainedModel.objects.create(
                name=f"LSTM Model - {dataset.name}",
                model_type='lstm',
                dataset=dataset,
                accuracy=lstm_accuracy,
                file_path=lstm_model_path
            )
            model_results['lstm'] = {'id': lstm_model.id, 'accuracy': lstm_accuracy}
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            model_results['lstm'] = {'error': str(e)}
        
        # Only train ensemble if all other models succeeded
        if all(key in model_results and 'id' in model_results[key] for key in ['cnn', 'xgboost', 'lstm']):
            try:
                ensemble_model_path, ensemble_accuracy = trainer.train_ensemble(
                    X_train, y_train, X_test, y_test,
                    [
                        model_results['cnn']['id'],
                        model_results['xgboost']['id'], 
                        model_results['lstm']['id']
                    ]
                )
                ensemble_model = TrainedModel.objects.create(
                    name=f"Ensemble Model - {dataset.name}",
                    model_type='ensemble',
                    dataset=dataset,
                    accuracy=ensemble_accuracy,
                    file_path=ensemble_model_path
                )
                model_results['ensemble'] = {'id': ensemble_model.id, 'accuracy': ensemble_accuracy}
            except Exception as e:
                print(f"Error training Ensemble model: {str(e)}")
                model_results['ensemble'] = {'error': str(e)}
        
        # Set the best model as active if no active models exist
        successful_models = [model_results[key] for key in model_results 
                            if isinstance(model_results[key], dict) and 'id' in model_results[key]]
        
        if successful_models and not TrainedModel.objects.filter(is_active=True).exists():
            best_model_info = max(successful_models, key=lambda m: m.get('accuracy', 0))
            best_model = TrainedModel.objects.get(id=best_model_info['id'])
            best_model.is_active = True
            best_model.save()
        
        return {
            'dataset_id': dataset.id,
            'models': model_results
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in train_model_task: {error_details}")
        return {'error': str(e), 'traceback': error_details}