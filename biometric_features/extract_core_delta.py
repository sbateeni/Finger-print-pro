import cv2
import numpy as np

def extract_core_delta(img):
    """
    Extract core and delta points from a fingerprint image.
    
    Args:
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        tuple: (core_point, delta_point) where each point is (x, y)
    """
    # Calculate orientation field
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation angles
    orientation = np.arctan2(sobely, sobelx)
    
    # Find core point (highest curvature)
    curvature = cv2.Laplacian(img, cv2.CV_64F)
    core_idx = np.unravel_index(np.argmax(curvature), curvature.shape)
    
    # Find delta point (lowest curvature)
    delta_idx = np.unravel_index(np.argmin(curvature), curvature.shape)
    
    return core_idx, delta_idx 