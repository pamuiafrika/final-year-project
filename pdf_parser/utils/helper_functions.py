import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_analysis_report(pdf_path, analysis_results):
    """
    Create a comprehensive analysis report from all module results
    
    Args:
        pdf_path (str): Path to the analyzed PDF file
        analysis_results (dict): Combined results from all analysis modules
        
    Returns:
        dict: Formatted analysis report
    """
    # Create base report structure
    report = {
        'file_name': os.path.basename(pdf_path),
        'file_size': os.path.getsize(pdf_path),
        'analysis_date': datetime.now().isoformat(),
        'results': analysis_results,
        'summary': {
            'suspicious_score': 0,
            'anomalies_detected': 0,
            'recommendation': ''
        }
    }
    
    # Count total anomalies
    anomalies_count = 0
    if 'metadata' in analysis_results:
        anomalies_count += len(analysis_results['metadata'].get('anomalies', []))
    
    if 'images' in analysis_results:
        if 'summary' in analysis_results['images']:
            anomalies_count += analysis_results['images']['summary'].get('suspicious_embedded_images', 0)
            anomalies_count += analysis_results['images']['summary'].get('suspicious_hidden_pngs', 0)
    
    if 'compression' in analysis_results:
        if 'summary' in analysis_results['compression']:
            anomalies_count += analysis_results['compression']['summary'].get('suspicious_artifacts', 0)
    
    report['summary']['anomalies_detected'] = anomalies_count
    
    # Calculate overall suspicious score
    suspicious_score = 0
    if 'images' in analysis_results and 'summary' in analysis_results['images']:
        suspicious_score += analysis_results['images']['summary'].get('overall_suspicion_score', 0) * 0.5
        
    if 'compression' in analysis_results and 'summary' in analysis_results['compression']:
        suspicious_score += analysis_results['compression']['summary'].get('compression_suspicion_score', 0) * 0.3
        
    # Add score for metadata anomalies
    if 'metadata' in analysis_results:
        metadata_anomalies = analysis_results['metadata'].get('anomalies', [])
        metadata_score = sum(3 if a.get('severity') == 'high' else 
                            (2 if a.get('severity') == 'medium' else 1) 
                            for a in metadata_anomalies)
        suspicious_score += min(metadata_score, 10) * 0.2  # Cap at 5, weight at 20%
    
    report['summary']['suspicious_score'] = round(min(suspicious_score, 10), 2)  # Cap at 10, round to 2 decimal places
    
    # Add recommendation based on suspicious score
    if report['summary']['suspicious_score'] < 2:
        report['summary']['recommendation'] = 'Low risk - No significant steganographic content detected.'
    elif report['summary']['suspicious_score'] < 5:
        report['summary']['recommendation'] = 'Medium risk - Some anomalies detected. Further investigation may be warranted.'
    else:
        report['summary']['recommendation'] = 'High risk - Significant anomalies detected. This file likely contains hidden data.'
    
    return report

def save_report_to_file(report, output_path):
    """
    Save analysis report to a JSON file
    
    Args:
        report (dict): Analysis report to save
        output_path (str): Path to save the report to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return False

def get_file_hash(file_path, hash_algorithm='sha256'):
    """
    Calculate hash for a file
    
    Args:
        file_path (str): Path to the file
        hash_algorithm (str): Hash algorithm to use (md5, sha1, sha256)
        
    Returns:
        str: Calculated hash
    """
    import hashlib
    
    algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256
    }
    
    if hash_algorithm not in algorithms:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
    
    hash_obj = algorithms[hash_algorithm]()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return None