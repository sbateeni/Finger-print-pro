import cv2
import numpy as np

def extract_features(image):
    """
    استخراج المميزات من صورة البصمة
    
    Args:
        image (numpy.ndarray): صورة البصمة المعالجة
        
    Returns:
        dict: قاموس يحتوي على المميزات المستخرجة
    """
    # إنشاء كائن SIFT
    sift = cv2.SIFT_create()
    
    # استخراج النقاط المميزة والوصف
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # استخراج إحداثيات النقاط المميزة
    minutiae = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    # تصنيف النقاط المميزة
    minutiae_types = classify_minutiae(image, keypoints)
    
    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'minutiae': minutiae,
        'minutiae_types': minutiae_types,
        'count': len(keypoints)
    }

def classify_minutiae(image, keypoints):
    """
    تصنيف النقاط المميزة في البصمة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        keypoints (list): قائمة النقاط المميزة
        
    Returns:
        list: قائمة تحتوي على أنواع النقاط المميزة
    """
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
        
        # تحديد نوع النقطة المميزة
        if np.mean(gradient_magnitude) > 50:
            minutiae_types.append('ridge_ending')
        else:
            minutiae_types.append('bifurcation')
    
    return minutiae_types

def match_features(features1, features2):
    """
    مقارنة المميزات بين بصمتين
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        
    Returns:
        dict: نتائج المقارنة
    """
    # إنشاء كائن FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # مقارنة الواصفات
    matches = flann.knnMatch(features1['descriptors'], features2['descriptors'], k=2)
    
    # تطبيق نسبة Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # حساب نسبة التطابق
    match_score = len(good_matches) / min(len(features1['keypoints']), len(features2['keypoints']))
    
    return {
        'matches': [(m.queryIdx, m.trainIdx) for m in good_matches],
        'score': match_score,
        'count': len(good_matches)
    } 