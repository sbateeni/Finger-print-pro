import os
import logging
from datetime import datetime
import cv2
import numpy as np

def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    
    Args:
        log_dir (str): Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)
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
    Save an image to a file.
    
    Args:
        img (numpy.ndarray): Image to save
        path (str): Path to save the image
        create_dir (bool): Whether to create directory if it doesn't exist
    """
    if create_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def load_image(path):
    """
    Load an image from a file.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    return img

def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (list): List of predicted values
        ground_truth (list): List of ground truth values
        
    Returns:
        dict: Dictionary of metrics
    """
    tp = sum(1 for p, t in zip(predictions, ground_truth) if p == t == 1)
    fp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    } 