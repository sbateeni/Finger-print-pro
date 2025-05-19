import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter

def compute_orientation_field(image, block_size=16):
    """Compute the orientation field of the fingerprint"""
    # Apply Sobel operators
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute orientation
    orientation = np.arctan2(sobely, sobelx) / 2
    
    # Smooth orientation field
    orientation = gaussian_filter(orientation, sigma=2)
    
    return orientation

def skeletonize(image):
    """Skeletonize binary image using morphological operations"""
    # Convert to binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Initialize skeleton
    skeleton = np.zeros(binary.shape, np.uint8)
    
    # Create structuring element
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Skeletonize
    done = False
    while not done:
        # Erode
        eroded = cv2.erode(binary, element)
        # Dilate
        dilated = cv2.dilate(eroded, element)
        # Subtract
        temp = cv2.subtract(binary, dilated)
        # Union
        skeleton = cv2.bitwise_or(skeleton, temp)
        # Update binary
        binary = eroded.copy()
        # Check if done
        if cv2.countNonZero(binary) == 0:
            done = True
    
    return skeleton

def detect_minutiae(image, orientation_field):
    """Detect minutiae points in the fingerprint"""
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Skeletonize the binary image
    skeleton = skeletonize(binary)
    
    # Find minutiae points
    minutiae = []
    height, width = skeleton.shape
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            if skeleton[y, x] == 255:
                # Count neighbors
                neighbors = skeleton[y-1:y+2, x-1:x+2]
                neighbor_count = np.sum(neighbors) - 255
                
                # Detect ridge endings and bifurcations
                if neighbor_count == 1:  # Ridge ending
                    minutiae.append({
                        'x': x,
                        'y': y,
                        'type': 'ending',
                        'angle': orientation_field[y, x]
                    })
                elif neighbor_count == 3:  # Bifurcation
                    minutiae.append({
                        'x': x,
                        'y': y,
                        'type': 'bifurcation',
                        'angle': orientation_field[y, x]
                    })
    
    return minutiae

def extract_features(image):
    """Main function to extract fingerprint features"""
    # Compute orientation field
    orientation_field = compute_orientation_field(image)
    
    # Detect minutiae
    minutiae = detect_minutiae(image, orientation_field)
    
    return minutiae

def visualize_minutiae(image, minutiae):
    """Visualize minutiae points on the image"""
    # Create a color image for visualization
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw minutiae points
    for m in minutiae:
        x, y = int(m['x']), int(m['y'])
        color = (0, 255, 0) if m['type'] == 'ending' else (0, 0, 255)
        cv2.circle(vis_image, (x, y), 3, color, -1)
    
    return vis_image 