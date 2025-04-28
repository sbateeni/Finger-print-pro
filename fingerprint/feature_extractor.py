import cv2
import numpy as np
import torch
import gc
from .preprocessor import get_device_info

def extract_features(image):
    """Extract features from fingerprint image using both SIFT and ORB"""
    try:
        # Get device info for optimization
        device_info = get_device_info()
        
        # Initialize feature detectors
        sift = cv2.SIFT_create()
        orb = cv2.ORB_create()
        
        # Extract features using both detectors
        sift_kp, sift_desc = sift.detectAndCompute(image, None)
        orb_kp, orb_desc = orb.detectAndCompute(image, None)
        
        # Merge keypoints and descriptors
        keypoints = sift_kp + orb_kp
        descriptors = np.vstack([sift_desc, orb_desc]) if sift_desc is not None and orb_desc is not None else None
        
        # Filter keypoints based on strength
        if descriptors is not None:
            strength = np.linalg.norm(descriptors, axis=1)
            threshold = np.mean(strength) + np.std(strength)
            strong_indices = strength > threshold
            keypoints = [kp for i, kp in enumerate(keypoints) if strong_indices[i]]
            descriptors = descriptors[strong_indices]
        
        # Clean up memory
        gc.collect()
        if device_info['cuda_available']:
            torch.cuda.empty_cache()
            
        return keypoints, descriptors
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None, None

def classify_minutiae(image, keypoints):
    """
    تصنيف النقاط المميزة في البصمة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        keypoints (list): قائمة النقاط المميزة
        
    Returns:
        list: قائمة تحتوي على أنواع النقاط المميزة
    """
    device_info = get_device_info()
    minutiae_types = []
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size)
        
        # استخراج المنطقة المحيطة بالنقطة
        roi = image[max(0, y-radius):min(image.shape[0], y+radius),
                   max(0, x-radius):min(image.shape[1], x+radius)]
        
        if roi.size == 0:
            minutiae_types.append('unknown')
            continue
        
        # حساب التدرج
        gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # حساب التباين المحلي
        local_contrast = np.std(roi)
        
        # تحديد نوع النقطة المميزة
        if np.mean(gradient_magnitude) > 50 and local_contrast > 30:
            if np.std(gradient_direction) > 1.0:
                minutiae_types.append('bifurcation')
            else:
                minutiae_types.append('ridge_ending')
        else:
            minutiae_types.append('unknown')
    
    return minutiae_types

def match_features(features1, features2):
    """
    مقارنة المميزات بين بصمتين مع تحسينات الأداء
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        
    Returns:
        dict: نتائج المقارنة
    """
    device_info = get_device_info()
    
    # التحقق من وجود واصفات كافية
    if features1['descriptors'] is None or features2['descriptors'] is None:
        return {
            'matches': [],
            'score': 0.0,
            'count': 0
        }
    
    # إنشاء كائن المطابقة
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # مقارنة الواصفات باستخدام FLANN
        matches = flann.knnMatch(
            features1['descriptors'].astype(np.float32),
            features2['descriptors'].astype(np.float32),
            k=2
        )
        
        # تطبيق نسبة Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # حساب نسبة التطابق
        min_features = min(len(features1['keypoints']), len(features2['keypoints']))
        if min_features > 0:
            match_score = len(good_matches) / min_features
        else:
            match_score = 0.0
            
    except Exception as e:
        print(f"Error during matching: {str(e)}")
        return {
            'matches': [],
            'score': 0.0,
            'count': 0
        }
    
    # تنظيف الذاكرة
    gc.collect()
    
    return {
        'matches': [(m.queryIdx, m.trainIdx) for m in good_matches],
        'score': match_score,
        'count': len(good_matches)
    } 