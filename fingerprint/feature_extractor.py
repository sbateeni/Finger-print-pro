import cv2
import numpy as np
import torch
import gc
from .preprocessor import get_device_info

def extract_features(image):
    """
    استخراج المميزات من صورة البصمة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة المعالجة
        
    Returns:
        dict: قاموس يحتوي على المميزات المستخرجة
    """
    device_info = get_device_info()
    
    # إنشاء كائنات المستخرجين
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        patchSize=31,
        fastThreshold=20
    )
    
    # استخراج النقاط المميزة والوصف باستخدام SIFT
    sift_keypoints, sift_descriptors = sift.detectAndCompute(image, None)
    
    # استخراج النقاط المميزة والوصف باستخدام ORB
    orb_keypoints, orb_descriptors = orb.detectAndCompute(image, None)
    
    # دمج النقاط المميزة والوصف
    keypoints = sift_keypoints + orb_keypoints
    descriptors = np.vstack([sift_descriptors, orb_descriptors]) if sift_descriptors is not None and orb_descriptors is not None else None
    
    # تصفية النقاط المميزة حسب الجودة
    if len(keypoints) > 0 and descriptors is not None:
        # حساب قوة النقاط المميزة
        strengths = [kp.response for kp in keypoints]
        threshold = np.mean(strengths) * 0.5
        
        # تصفية النقاط المميزة
        filtered_keypoints = []
        filtered_descriptors = []
        for kp, desc in zip(keypoints, descriptors):
            if kp.response > threshold:
                filtered_keypoints.append(kp)
                filtered_descriptors.append(desc)
        
        keypoints = filtered_keypoints
        descriptors = np.array(filtered_descriptors)
    
    # استخراج إحداثيات النقاط المميزة
    minutiae = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    # تصنيف النقاط المميزة
    minutiae_types = classify_minutiae(image, keypoints)
    
    # تنظيف الذاكرة
    gc.collect()
    
    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'minutiae': minutiae,
        'minutiae_types': minutiae_types,
        'count': len(keypoints)
    }

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
    
    # إنشاء كائنات المطابقة
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # مقارنة الواصفات باستخدام FLANN
    flann_matches = flann.knnMatch(features1['descriptors'], features2['descriptors'], k=2)
    
    # مقارنة الواصفات باستخدام BF
    bf_matches = bf.match(features1['descriptors'], features2['descriptors'])
    
    # تطبيق نسبة Lowe's ratio test مع عتبة متغيرة
    good_matches = []
    for m, n in flann_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # إضافة التطابقات الجيدة من BF
    good_matches.extend([m for m in bf_matches if m.distance < 50])
    
    # حساب نسبة التطابق
    min_features = min(len(features1['keypoints']), len(features2['keypoints']))
    if min_features > 0:
        match_score = len(good_matches) / min_features
    else:
        match_score = 0.0
    
    # تنظيف الذاكرة
    gc.collect()
    
    return {
        'matches': [(m.queryIdx, m.trainIdx) for m in good_matches],
        'score': match_score,
        'count': len(good_matches)
    } 