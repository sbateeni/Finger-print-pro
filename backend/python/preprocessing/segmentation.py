import cv2
import numpy as np
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import label, regionprops

def segment_fingerprint(image_path):
    """
    Segment the fingerprint from the background.
    
    Args:
        image_path (str): Path to the fingerprint image
        
    Returns:
        numpy.ndarray: Segmented fingerprint image
    """
    try:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image")
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = binary_closing(binary, kernel)
        binary = binary_opening(binary, kernel)
        
        # Label connected components
        labeled = label(binary)
        regions = regionprops(labeled)
        
        # Find the largest region (fingerprint)
        largest_region = max(regions, key=lambda x: x.area)
        
        # Create mask for the fingerprint
        mask = np.zeros_like(binary)
        mask[labeled == largest_region.label] = 255
        
        # Apply the mask to the original image
        segmented = cv2.bitwise_and(img, mask)
        
        # Fill holes in the fingerprint region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Final segmentation
        segmented = cv2.bitwise_and(img, mask)
        
        return segmented
        
    except Exception as e:
        print(f"Error in segmentation: {str(e)}")
        return None

def enhance_contrast(img):
    """
    Enhance the contrast of the fingerprint image.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    return enhanced

def remove_background_noise(img):
    """
    Remove background noise from the fingerprint image.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Denoised image
    """
    # Apply bilateral filter to preserve edges while removing noise
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    return denoised

def normalize_intensity(img):
    """
    Normalize the intensity of the fingerprint image.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Normalized image
    """
    # Normalize to [0, 255]
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)

def preprocess_fingerprint(image_path):
    """
    Complete preprocessing pipeline for fingerprint images.
    
    Args:
        image_path (str): Path to the fingerprint image
        
    Returns:
        numpy.ndarray: Preprocessed fingerprint image
    """
    try:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image")
        
        # Apply preprocessing steps
        enhanced = enhance_contrast(img)
        denoised = remove_background_noise(enhanced)
        normalized = normalize_intensity(denoised)
        segmented = segment_fingerprint(image_path)
        
        if segmented is None:
            return normalized
        
        # Combine segmentation with preprocessing
        final = cv2.bitwise_and(normalized, segmented)
        
        return final
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None 