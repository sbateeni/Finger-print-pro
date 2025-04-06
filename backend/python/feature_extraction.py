import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max

def extract_features(img):
    """
    Extract all features from a fingerprint image.
    
    Args:
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    try:
        # Extract minutiae points
        minutiae = extract_minutiae(img)
        
        # Extract core and delta points
        core, delta = extract_core_delta(img)
        
        # Calculate orientation field
        orientation = calculate_orientation_field(img)
        
        # Calculate ridge density
        density = calculate_ridge_density(img)
        
        # Calculate ridge width
        width = calculate_ridge_width(img)
        
        # Extract pores
        pores = extract_pores(img)
        
        return {
            'minutiae': minutiae,
            'core': core,
            'delta': delta,
            'orientation': orientation,
            'ridge_density': density,
            'ridge_width': width,
            'pores': pores
        }
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None

def extract_minutiae(img):
    """
    Extract minutiae points from a fingerprint image.
    
    Args:
        img (numpy.ndarray): Preprocessed fingerprint image
        
    Returns:
        list: List of minutiae points with their properties
    """
    # Apply thinning
    thinned = cv2.ximgproc.thinning(img)
    
    # Find minutiae points
    minutiae = []
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]])
    
    # Find ridge endings and bifurcations
    for y in range(1, thinned.shape[0]-1):
        for x in range(1, thinned.shape[1]-1):
            if thinned[y, x] == 255:
                # Calculate crossing number
                neighbors = thinned[y-1:y+2, x-1:x+2]
                cn = np.sum(neighbors * kernel) // 10
                
                if cn == 1:  # Ridge ending
                    minutiae.append({
                        'x': x,
                        'y': y,
                        'type': 'ridge_ending',
                        'angle': calculate_minutiae_angle(thinned, x, y)
                    })
                elif cn == 3:  # Bifurcation
                    minutiae.append({
                        'x': x,
                        'y': y,
                        'type': 'bifurcation',
                        'angle': calculate_minutiae_angle(thinned, x, y)
                    })
    
    return minutiae

def calculate_minutiae_angle(img, x, y):
    """
    Calculate the orientation angle of a minutiae point.
    
    Args:
        img (numpy.ndarray): Thinned fingerprint image
        x (int): x-coordinate of the minutiae point
        y (int): y-coordinate of the minutiae point
        
    Returns:
        float: Orientation angle in radians
    """
    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate orientation
    angle = np.arctan2(gy[y, x], gx[y, x])
    
    return angle

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

def calculate_ridge_density(img, block_size=32):
    """
    Calculate the ridge density of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input image
        block_size (int): Size of the block for density calculation
        
    Returns:
        float: Average ridge density
    """
    # Apply thinning
    thinned = cv2.ximgproc.thinning(img)
    
    # Calculate density in blocks
    h, w = img.shape
    density = np.zeros((h // block_size, w // block_size))
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = thinned[i:i+block_size, j:j+block_size]
            if block.size > 0:
                density[i//block_size, j//block_size] = np.sum(block == 255) / block.size
    
    # Calculate average density
    avg_density = np.mean(density)
    
    return avg_density

def calculate_ridge_width(img):
    """
    Calculate the average ridge width of the fingerprint.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        float: Average ridge width in pixels
    """
    # Apply distance transform
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    
    # Calculate average ridge width
    avg_width = np.mean(dist[img == 255]) * 2
    
    return avg_width

def extract_pores(img):
    """
    Extract pores from the fingerprint image.
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        list: List of pore coordinates
    """
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Filter pores based on size
    pores = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 10:  # Pores are typically small
            x, y = centroids[i]
            pores.append((int(x), int(y)))
    
    return pores 