import cv2
import numpy as np
import torch
import gc
from .preprocessor import get_device_info
from scipy.spatial.distance import cosine

def extract_features(image):
    """استخراج المميزات من صورة البصمة"""
    try:
        # إنشاء كائن SIFT
        sift = cv2.SIFT_create()
        
        # استخراج النقاط المميزة والوصف
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        # تنظيف الذاكرة
        gc.collect()
        
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
    """مقارنة المميزات بين بصمتين"""
    try:
        if features1['descriptors'] is None or features2['descriptors'] is None:
            return 0.0
            
        # استخدام FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # مطابقة المميزات
        matches = flann.knnMatch(features1['descriptors'], features2['descriptors'], k=2)
        
        # تطبيق Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # حساب نسبة التطابق
        match_score = len(good_matches) / max(len(features1['keypoints']), len(features2['keypoints'])) * 100
        
        # تنظيف الذاكرة
        gc.collect()
        
        return min(match_score, 100.0)
        
    except Exception as e:
        print(f"Error in feature matching: {str(e)}")
        return 0.0 