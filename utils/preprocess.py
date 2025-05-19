import cv2
import numpy as np
from skimage import exposure

def enhance_contrast(image):
    """Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    # Convert to grayscale if image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def remove_noise(image):
    """Remove noise from the fingerprint image"""
    # Apply bilateral filter to remove noise while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised

def normalize_image(image):
    """Normalize image intensity values"""
    normalized = exposure.equalize_hist(image)
    normalized = (normalized * 255).astype(np.uint8)
    return normalized

def preprocess_fingerprint(image_path):
    """Main preprocessing function for fingerprint images"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply preprocessing steps
    enhanced = enhance_contrast(gray)
    denoised = remove_noise(enhanced)
    normalized = normalize_image(denoised)
    
    return normalized

def save_preprocessed_image(image, output_path):
    """Save preprocessed image to file"""
    cv2.imwrite(output_path, image) 