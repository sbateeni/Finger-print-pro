import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from database.models import db, Fingerprint, Minutiae, Match

def match_fingerprints(image_path):
    """
    Match a fingerprint against the database using multiple methods.
    
    Args:
        image_path (str): Path to the fingerprint image to match
        
    Returns:
        list: List of matches with scores and metadata
    """
    try:
        # Preprocess the input fingerprint
        from preprocessing.segmentation import preprocess_fingerprint
        from biometric_features.extract_core_delta import extract_core_delta
        
        img = preprocess_fingerprint(image_path)
        if img is None:
            raise ValueError("Failed to preprocess fingerprint")
        
        # Extract features
        core, delta = extract_core_delta(img)
        minutiae = extract_minutiae(img)
        
        # Get all fingerprints from database
        fingerprints = Fingerprint.query.all()
        
        matches = []
        
        for fp in fingerprints:
            # Load stored fingerprint
            stored_img = cv2.imread(fp.image_path, cv2.IMREAD_GRAYSCALE)
            if stored_img is None:
                continue
            
            # Calculate similarity scores using different methods
            minutiae_score = match_minutiae(minutiae, fp.minutiae)
            deep_learning_score = match_deep_learning(img, stored_img)
            hybrid_score = calculate_hybrid_score(minutiae_score, deep_learning_score)
            
            if hybrid_score > 0.7:  # Threshold for considering a match
                matches.append({
                    'fingerprint_id': fp.id,
                    'score': hybrid_score,
                    'algorithm': 'hybrid',
                    'minutiae_score': minutiae_score,
                    'deep_learning_score': deep_learning_score
                })
        
        # Sort matches by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches
        
    except Exception as e:
        print(f"Error in fingerprint matching: {str(e)}")
        return []

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

def match_minutiae(minutiae1, minutiae2):
    """
    Match two sets of minutiae points.
    
    Args:
        minutiae1 (list): First set of minutiae points
        minutiae2 (list): Second set of minutiae points
        
    Returns:
        float: Matching score between 0 and 1
    """
    if not minutiae1 or not minutiae2:
        return 0.0
    
    # Convert minutiae to feature vectors
    features1 = np.array([[m['x'], m['y'], m['angle']] for m in minutiae1])
    features2 = np.array([[m['x'], m['y'], m['angle']] for m in minutiae2])
    
    # Calculate pairwise distances
    distances = np.zeros((len(features1), len(features2)))
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            # Euclidean distance for position
            pos_dist = np.sqrt(np.sum((f1[:2] - f2[:2])**2))
            # Angular distance
            ang_dist = min(abs(f1[2] - f2[2]), 2*np.pi - abs(f1[2] - f2[2]))
            # Combined distance
            distances[i, j] = pos_dist + 10 * ang_dist
    
    # Find best matches
    matches = []
    used1 = set()
    used2 = set()
    
    while True:
        # Find minimum distance
        min_dist = np.min(distances)
        if min_dist > 20:  # Threshold for matching
            break
        
        # Find indices of minimum distance
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        
        if i not in used1 and j not in used2:
            matches.append((i, j))
            used1.add(i)
            used2.add(j)
        
        # Set used distances to infinity
        distances[i, :] = np.inf
        distances[:, j] = np.inf
    
    # Calculate matching score
    score = len(matches) / min(len(minutiae1), len(minutiae2))
    
    return min(score, 1.0)

def match_deep_learning(img1, img2):
    """
    Match fingerprints using deep learning features.
    
    Args:
        img1 (numpy.ndarray): First fingerprint image
        img2 (numpy.ndarray): Second fingerprint image
        
    Returns:
        float: Matching score between 0 and 1
    """
    try:
        # Load pre-trained model
        model = torch.load('models/fingerprint_cnn.pth')
        model.eval()
        
        # Preprocess images
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        # Convert to tensor
        img1_tensor = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features1 = model(img1_tensor)
            features2 = model(img2_tensor)
        
        # Calculate similarity
        similarity = cosine_similarity(features1.numpy(), features2.numpy())
        
        return float(similarity[0][0])
        
    except Exception as e:
        print(f"Error in deep learning matching: {str(e)}")
        return 0.0

def calculate_hybrid_score(minutiae_score, deep_learning_score):
    """
    Calculate hybrid matching score combining minutiae and deep learning scores.
    
    Args:
        minutiae_score (float): Minutiae matching score
        deep_learning_score (float): Deep learning matching score
        
    Returns:
        float: Hybrid matching score
    """
    # Weighted combination
    weights = {
        'minutiae': 0.6,
        'deep_learning': 0.4
    }
    
    hybrid_score = (
        weights['minutiae'] * minutiae_score +
        weights['deep_learning'] * deep_learning_score
    )
    
    return hybrid_score 