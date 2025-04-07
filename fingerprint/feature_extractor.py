import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as sk_skeletonize

class MinutiaePoint:
    def __init__(self, x, y, minutiae_type, angle=0.0):
        self.x = x
        self.y = y
        self.type = minutiae_type  # 'ridge_ending' or 'bifurcation'
        self.angle = angle

def skeletonize(img):
    """
    Convert the input image to a skeleton image using scikit-image
    """
    try:
        # Ensure the image is binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Normalize to 0-1 range for skimage
        binary_normalized = binary.astype(bool)
        
        # Apply skeletonization
        skeleton = sk_skeletonize(binary_normalized)
        
        # Convert back to uint8 format
        return (skeleton * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in skeletonization: {str(e)}")
        return None

def extract_features(image):
    """
    Extract minutiae points from the fingerprint image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create skeleton image
        skeleton_image = skeletonize(image)
        if skeleton_image is None:
            raise Exception("فشل في عملية تنحيف الصورة")
        
        # Ensure the image is binary
        binary = skeleton_image > 0

        # Find minutiae points using crossing number method
        minutiae_points = []
        rows, cols = binary.shape
        
        # Padding to avoid border issues
        padded = np.pad(binary, ((1, 1), (1, 1)), mode='constant')
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if padded[i, j]:  # Only process ridge pixels
                    # Get 3x3 neighborhood
                    neighbors = padded[i-1:i+2, j-1:j+2].astype(np.uint8)
                    cn = compute_crossing_number(neighbors)
                    
                    if cn == 1:  # Ridge ending
                        angle = compute_orientation(binary, i-1, j-1)
                        minutiae_points.append(MinutiaePoint(j-1, i-1, 'ridge_ending', angle))
                    elif cn == 3:  # Bifurcation
                        angle = compute_orientation(binary, i-1, j-1)
                        minutiae_points.append(MinutiaePoint(j-1, i-1, 'bifurcation', angle))
        
        # Filter false minutiae
        filtered_points = filter_minutiae(minutiae_points, binary)
        
        if not filtered_points:
            raise Exception("لم يتم العثور على نقاط مميزة في الصورة")
        
        return filtered_points
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return []

def compute_crossing_number(neighborhood):
    """
    Compute the crossing number for a 3x3 neighborhood
    """
    # Convert to binary pattern
    pattern = neighborhood.flatten()[:-1]  # Exclude last element
    transitions = 0
    
    # Count transitions between 0 and 1
    for k in range(8):
        transitions += abs(int(pattern[k]) - int(pattern[(k+1)%8]))
    
    return transitions // 2

def compute_orientation(image, i, j, window_size=7):
    """
    Compute the orientation of the ridge at point (i,j)
    """
    # Extract local window
    half_size = window_size // 2
    window = image[max(0, i-half_size):min(image.shape[0], i+half_size+1),
                  max(0, j-half_size):min(image.shape[1], j+half_size+1)]
    
    # Compute gradient
    gy, gx = np.gradient(window.astype(float))
    angle = np.arctan2(gy.mean(), gx.mean())
    
    return angle

def filter_minutiae(minutiae_points, binary_image, min_distance=10):
    """
    Filter false minutiae points:
    1. Remove points too close to the border
    2. Remove points too close to each other
    3. Remove points in high-density regions
    """
    filtered = []
    rows, cols = binary_image.shape
    border = 20
    
    for point in minutiae_points:
        # Check border distance
        if (point.x < border or point.x >= cols - border or 
            point.y < border or point.y >= rows - border):
            continue
            
        # Check distance from other points
        too_close = False
        for other in filtered:
            dx = point.x - other.x
            dy = point.y - other.y
            if np.sqrt(dx*dx + dy*dy) < min_distance:
                too_close = True
                break
                
        if not too_close:
            filtered.append(point)
    
    return filtered 