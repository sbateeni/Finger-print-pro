import cv2
import numpy as np

def preprocess_fingerprint(image_path):
    """
    Preprocess a fingerprint image by enhancing contrast and removing noise.
    
    Args:
        image_path (str): Path to the fingerprint image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    
    # Normalize image
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(opening)
    
    return enhanced 