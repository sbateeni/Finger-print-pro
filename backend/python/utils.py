import os
import logging
import json
from datetime import datetime
import numpy as np
import cv2

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, f'fingerprint_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_image(img, path, create_dir=True):
    """
    Save an image to the specified path.
    
    Args:
        img (numpy.ndarray): Image to save
        path (str): Path to save the image
        create_dir (bool): Whether to create the directory if it doesn't exist
    """
    if create_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    cv2.imwrite(path, img)

def load_image(path):
    """
    Load an image from the specified path.
    
    Args:
        path (str): Path to the image
        
    Returns:
        numpy.ndarray: Loaded image
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_features(features, path):
    """
    Save feature vectors to a file.
    
    Args:
        features (numpy.ndarray): Feature vectors to save
        path (str): Path to save the features
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, features)

def load_features(path):
    """
    Load feature vectors from a file.
    
    Args:
        path (str): Path to the features file
        
    Returns:
        numpy.ndarray: Loaded feature vectors
    """
    return np.load(path)

def save_metadata(metadata, path):
    """
    Save metadata to a JSON file.
    
    Args:
        metadata (dict): Metadata to save
        path (str): Path to save the metadata
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata(path):
    """
    Load metadata from a JSON file.
    
    Args:
        path (str): Path to the metadata file
        
    Returns:
        dict: Loaded metadata
    """
    with open(path, 'r') as f:
        return json.load(f)

def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (numpy.ndarray): Model predictions
        ground_truth (numpy.ndarray): Ground truth labels
        
    Returns:
        dict: Dictionary of metrics
    """
    # Calculate confusion matrix
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        }
    }

def visualize_fingerprint(img, minutiae=None, core=None, delta=None, save_path=None):
    """
    Visualize fingerprint with detected features.
    
    Args:
        img (numpy.ndarray): Fingerprint image
        minutiae (list): List of minutiae points
        core (tuple): Core point coordinates
        delta (tuple): Delta point coordinates
        save_path (str): Path to save the visualization
    """
    # Convert to color image for visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw minutiae points
    if minutiae:
        for m in minutiae:
            x, y = int(m['x']), int(m['y'])
            if m['type'] == 'ridge_ending':
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)  # Green for ridge endings
            else:  # bifurcation
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # Blue for bifurcations
    
    # Draw core point
    if core:
        x, y = int(core[0]), int(core[1])
        cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)  # Red for core
    
    # Draw delta point
    if delta:
        x, y = int(delta[0]), int(delta[1])
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)  # Cyan for delta
    
    # Save visualization if path is provided
    if save_path:
        save_image(vis, save_path)
    
    return vis 