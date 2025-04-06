import cv2
import numpy as np
from database.models import Fingerprint, Minutiae, Match, db
import logging

def match_fingerprints(image_path):
    """
    Match a fingerprint against the database.
    
    Args:
        image_path (str): Path to the fingerprint image to match
        
    Returns:
        list: List of matches with confidence scores
    """
    try:
        logging.info(f"Starting fingerprint matching for image: {image_path}")
        
        # Load and preprocess the input image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        logging.info(f"Successfully loaded input image with shape: {img.shape}")
        
        # Get all fingerprints from database
        fingerprints = Fingerprint.query.all()
        if not fingerprints:
            logging.warning("No fingerprints found in database")
            return []
        
        logging.info(f"Found {len(fingerprints)} fingerprints in database")
        matches = []
        
        for fp in fingerprints:
            try:
                # Load stored fingerprint
                stored_img = cv2.imread(fp.image_path, cv2.IMREAD_GRAYSCALE)
                if stored_img is None:
                    logging.warning(f"Could not read stored fingerprint image: {fp.image_path}")
                    continue
                
                logging.info(f"Processing fingerprint ID: {fp.id}")
                
                # Calculate similarity score
                score = calculate_similarity(img, stored_img)
                logging.info(f"Similarity score for fingerprint {fp.id}: {score}")
                
                if score > 0.7:  # Threshold for matching
                    matches.append({
                        'fingerprint_id': fp.id,
                        'confidence': float(score),
                        'user_id': fp.user_id
                    })
                    logging.info(f"Match found for fingerprint {fp.id} with confidence {score}")
                
            except Exception as e:
                logging.error(f"Error processing fingerprint {fp.id}: {str(e)}")
                continue
        
        logging.info(f"Matching completed. Found {len(matches)} matches")
        return matches
        
    except Exception as e:
        logging.error(f"Error in match_fingerprints: {str(e)}")
        raise

def calculate_similarity(img1, img2):
    """
    Calculate similarity between two fingerprint images.
    
    Args:
        img1 (numpy.ndarray): First fingerprint image
        img2 (numpy.ndarray): Second fingerprint image
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        # Resize images to same size
        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))
        
        # Normalize images
        img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calculate correlation coefficient
        correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        score = correlation[0][0]
        
        # Ensure score is between 0 and 1
        score = max(0, min(1, score))
        
        logging.debug(f"Calculated similarity score: {score}")
        return score
        
    except Exception as e:
        logging.error(f"Error in calculate_similarity: {str(e)}")
        raise 