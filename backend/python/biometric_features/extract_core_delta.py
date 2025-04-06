import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def extract_core_delta(img):
    """
    Extract core and delta points from a fingerprint image.
    
    Args:
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        tuple: (core_point, delta_point) where each point is (x, y) coordinates
    """
    try:
        # Calculate orientation field
        orientation = calculate_orientation_field(img)
        
        # Calculate curvature
        curvature = calculate_curvature(orientation)
        
        # Find core point
        core = find_core_point(curvature, orientation)
        
        # Find delta point
        delta = find_delta_point(curvature, orientation)
        
        return core, delta
        
    except Exception as e:
        print(f"Error in core/delta extraction: {str(e)}")
        return None, None

def calculate_orientation_field(img, block_size=16):
    """
    Calculate the orientation field of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input image
        block_size (int): Size of the block for orientation calculation
        
    Returns:
        numpy.ndarray: Orientation field in radians
    """
    # Calculate gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    orientation = np.arctan2(gy, gx)
    
    # Smooth the orientation field
    orientation = gaussian_filter(orientation, sigma=2)
    
    return orientation

def calculate_curvature(orientation, block_size=16):
    """
    Calculate the curvature of the orientation field.
    
    Args:
        orientation (numpy.ndarray): Orientation field
        block_size (int): Size of the block for curvature calculation
        
    Returns:
        numpy.ndarray: Curvature field
    """
    # Calculate second derivatives
    gxx = cv2.Sobel(orientation, cv2.CV_64F, 2, 0, ksize=3)
    gyy = cv2.Sobel(orientation, cv2.CV_64F, 0, 2, ksize=3)
    gxy = cv2.Sobel(orientation, cv2.CV_64F, 1, 1, ksize=3)
    
    # Calculate curvature
    curvature = (gxx * gyy - gxy**2) / (1 + gxx**2 + gyy**2)**2
    
    return curvature

def find_core_point(curvature, orientation):
    """
    Find the core point of the fingerprint.
    
    Args:
        curvature (numpy.ndarray): Curvature field
        orientation (numpy.ndarray): Orientation field
        
    Returns:
        tuple: (x, y) coordinates of the core point
    """
    # Find local maxima in curvature
    peaks = peak_local_max(curvature, min_distance=20, threshold_rel=0.5)
    
    if len(peaks) == 0:
        return None
    
    # Select the point with highest curvature
    max_curvature = -np.inf
    core_point = None
    
    for peak in peaks:
        y, x = peak
        if curvature[y, x] > max_curvature:
            max_curvature = curvature[y, x]
            core_point = (x, y)
    
    return core_point

def find_delta_point(curvature, orientation):
    """
    Find the delta point of the fingerprint.
    
    Args:
        curvature (numpy.ndarray): Curvature field
        orientation (numpy.ndarray): Orientation field
        
    Returns:
        tuple: (x, y) coordinates of the delta point
    """
    # Find local minima in curvature
    peaks = peak_local_max(-curvature, min_distance=20, threshold_rel=0.5)
    
    if len(peaks) == 0:
        return None
    
    # Select the point with lowest curvature
    min_curvature = np.inf
    delta_point = None
    
    for peak in peaks:
        y, x = peak
        if curvature[y, x] < min_curvature:
            min_curvature = curvature[y, x]
            delta_point = (x, y)
    
    return delta_point

def validate_core_delta(core, delta, img_shape):
    """
    Validate the core and delta points.
    
    Args:
        core (tuple): Core point coordinates
        delta (tuple): Delta point coordinates
        img_shape (tuple): Shape of the image
        
    Returns:
        tuple: Validated core and delta points
    """
    h, w = img_shape
    
    # Validate core point
    if core is not None:
        x, y = core
        if x < 0 or x >= w or y < 0 or y >= h:
            core = None
    
    # Validate delta point
    if delta is not None:
        x, y = delta
        if x < 0 or x >= w or y < 0 or y >= h:
            delta = None
    
    return core, delta 