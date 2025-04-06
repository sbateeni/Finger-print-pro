import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def assess_quality(image_path):
    """
    Assess the quality of a fingerprint image based on multiple criteria.
    
    Args:
        image_path (str): Path to the fingerprint image
        
    Returns:
        float: Quality score between 0 and 1
    """
    try:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image")
        
        # Calculate quality metrics
        contrast_score = calculate_contrast(img)
        clarity_score = calculate_clarity(img)
        noise_score = calculate_noise(img)
        orientation_score = calculate_orientation_consistency(img)
        
        # Combine scores with weights
        weights = {
            'contrast': 0.3,
            'clarity': 0.3,
            'noise': 0.2,
            'orientation': 0.2
        }
        
        quality_score = (
            weights['contrast'] * contrast_score +
            weights['clarity'] * clarity_score +
            weights['noise'] * noise_score +
            weights['orientation'] * orientation_score
        )
        
        return quality_score
        
    except Exception as e:
        print(f"Error in quality assessment: {str(e)}")
        return 0.0

def calculate_contrast(img):
    """
    Calculate the contrast of the image using histogram analysis.
    """
    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Calculate contrast as the standard deviation of the histogram
    mean = np.sum(np.arange(256) * hist)
    variance = np.sum((np.arange(256) - mean) ** 2 * hist)
    contrast = np.sqrt(variance)
    
    # Normalize to [0, 1]
    return min(contrast / 128, 1.0)

def calculate_clarity(img):
    """
    Calculate the clarity of ridge patterns using local variance.
    """
    # Apply Sobel operator to detect edges
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Calculate local variance
    local_var = cv2.blur(gradient_mag**2, (5, 5)) - cv2.blur(gradient_mag, (5, 5))**2
    
    # Calculate clarity score
    clarity = np.mean(local_var)
    
    # Normalize to [0, 1]
    return min(clarity / 1000, 1.0)

def calculate_noise(img):
    """
    Calculate the noise level in the image.
    """
    # Apply median filter to get noise-free approximation
    denoised = cv2.medianBlur(img, 3)
    
    # Calculate noise as the difference between original and denoised
    noise = np.abs(img.astype(np.float32) - denoised.astype(np.float32))
    
    # Calculate noise score (inverse of noise level)
    noise_level = np.mean(noise)
    noise_score = 1.0 - min(noise_level / 50, 1.0)
    
    return noise_score

def calculate_orientation_consistency(img):
    """
    Calculate the consistency of ridge orientation.
    """
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(gy, gx)
    
    # Calculate local orientation consistency
    block_size = 16
    h, w = img.shape
    consistency = np.zeros((h // block_size, w // block_size))
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = orientation[i:i+block_size, j:j+block_size]
            if block.size > 0:
                # Calculate circular variance
                mean_angle = np.arctan2(np.mean(np.sin(block)), np.mean(np.cos(block)))
                variance = 1 - np.sqrt(np.mean(np.sin(block - mean_angle))**2 + 
                                     np.mean(np.cos(block - mean_angle))**2)
                consistency[i//block_size, j//block_size] = 1 - variance
    
    # Calculate overall consistency score
    consistency_score = np.mean(consistency)
    
    return consistency_score 