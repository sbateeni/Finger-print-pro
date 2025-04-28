import cv2
import numpy as np
from .feature_extractor import match_features
from scipy.spatial.distance import cosine
import torch
import gc
from .preprocessor import get_device_info

def match_fingerprints(features1, features2, threshold=0.3):
    """
    مقارنة بصمتين وإرجاع نتيجة المطابقة مع تحسينات الأداء
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        threshold (float): الحد الأدنى لنسبة التطابق
        
    Returns:
        dict: نتائج المطابقة
    """
    device_info = get_device_info()
    
    # التحقق من وجود مميزات كافية
    if features1['count'] < 5 or features2['count'] < 5:
        return {
            'is_match': False,
            'score': 0.0,
            'matches': [],
            'match_count': 0,
            'threshold': threshold,
            'error': 'عدد النقاط المميزة غير كافٍ'
        }
    
    # مقارنة المميزات
    match_result = match_features(features1, features2)
    
    # حساب درجة الثقة
    confidence_score = calculate_confidence_score(features1, features2, match_result)
    
    # حساب درجة التشابه
    similarity_score = calculate_similarity_score(features1, features2)
    
    # حساب درجة التطابق المكاني
    spatial_score = calculate_spatial_score(features1, features2, match_result)
    
    # تحديد نتيجة المطابقة
    is_match = (match_result['score'] >= threshold and 
               confidence_score >= 0.4 and 
               similarity_score >= 0.3 and
               spatial_score >= 0.3)
    
    # حساب النتيجة النهائية
    final_score = (match_result['score'] * 0.3 + 
                  confidence_score * 0.2 + 
                  similarity_score * 0.2 +
                  spatial_score * 0.3)
    
    # تنظيف الذاكرة
    gc.collect()
    
    return {
        'is_match': is_match,
        'score': final_score * 100,  # تحويل إلى نسبة مئوية
        'matches': match_result['matches'],
        'match_count': match_result['count'],
        'threshold': threshold,
        'confidence': confidence_score,
        'similarity': similarity_score,
        'spatial': spatial_score
    }

def calculate_confidence_score(features1, features2, match_result):
    """
    حساب درجة الثقة في نتيجة المطابقة مع تحسينات الأداء
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        match_result (dict): نتائج المطابقة
        
    Returns:
        float: درجة الثقة
    """
    device_info = get_device_info()
    
    # حساب نسبة النقاط المتطابقة
    match_ratio = match_result['count'] / min(features1['count'], features2['count'])
    
    # حساب تناسق أنواع النقاط المميزة
    type_consistency = 0
    for match in match_result['matches']:
        idx1, idx2 = match
        if features1['minutiae_types'][idx1] == features2['minutiae_types'][idx2]:
            type_consistency += 1
    
    type_consistency = type_consistency / match_result['count'] if match_result['count'] > 0 else 0
    
    # حساب درجة الثقة النهائية
    confidence = (match_ratio * 0.6 + type_consistency * 0.4)
    
    return confidence

def calculate_similarity_score(features1, features2):
    """
    حساب درجة التشابه بين بصمتين مع تحسينات الأداء
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        
    Returns:
        float: درجة التشابه
    """
    device_info = get_device_info()
    
    # مقارنة المميزات
    match_result = match_features(features1, features2)
    
    # حساب درجة التشابه
    similarity_score = match_result['score']
    
    # تطبيق معامل التصحيح
    if match_result['count'] < 10:
        similarity_score *= 0.7  # تخفيف العقوبة
    
    return similarity_score

def calculate_spatial_score(features1, features2, match_result):
    """
    حساب درجة التطابق المكاني بين البصمتين
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        match_result (dict): نتائج المطابقة
        
    Returns:
        float: درجة التطابق المكاني
    """
    if len(match_result['matches']) < 3:
        return 0.0
    
    # استخراج النقاط المتطابقة
    points1 = np.array([features1['minutiae'][m[0]] for m in match_result['matches']])
    points2 = np.array([features2['minutiae'][m[1]] for m in match_result['matches']])
    
    # حساب المصفوفة الأساسية
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3.0, 0.99)
    
    if F is None:
        return 0.0
    
    # حساب المسافات بين النقاط
    distances1 = np.zeros((len(points1), len(points1)))
    distances2 = np.zeros((len(points2), len(points2)))
    
    for i in range(len(points1)):
        for j in range(i+1, len(points1)):
            distances1[i,j] = distances1[j,i] = np.linalg.norm(points1[i] - points1[j])
            distances2[i,j] = distances2[j,i] = np.linalg.norm(points2[i] - points2[j])
    
    # حساب نسبة التطابق في المسافات
    distance_ratios = []
    for i in range(len(points1)):
        for j in range(i+1, len(points1)):
            if distances1[i,j] > 0 and distances2[i,j] > 0:
                ratio = min(distances1[i,j], distances2[i,j]) / max(distances1[i,j], distances2[i,j])
                distance_ratios.append(ratio)
    
    if not distance_ratios:
        return 0.0
    
    # حساب درجة التطابق المكاني
    spatial_score = np.mean(distance_ratios)
    
    return spatial_score

def find_best_match(query_features, database_features, threshold=0.3):
    """
    البحث عن أفضل تطابق في قاعدة البيانات مع تحسينات الأداء
    
    Args:
        query_features (dict): مميزات البصمة المراد مطابقتها
        database_features (list): قائمة مميزات البصمات في قاعدة البيانات
        threshold (float): الحد الأدنى لنسبة التطابق
        
    Returns:
        dict: أفضل نتيجة مطابقة
    """
    device_info = get_device_info()
    
    best_match = {
        'index': -1,
        'score': 0,
        'is_match': False,
        'confidence': 0
    }
    
    # استخدام GPU إذا كان متاحاً
    if device_info['gpu_available']:
        # تحويل المميزات إلى tensors
        query_desc = torch.from_numpy(query_features['descriptors']).cuda()
        db_desc = torch.stack([torch.from_numpy(f['descriptors']).cuda() for f in database_features])
        
        # حساب المسافات
        distances = torch.cdist(query_desc, db_desc)
        
        # العثور على أفضل تطابق
        min_distances, indices = torch.min(distances, dim=1)
        match_scores = 1 - min_distances / torch.max(distances)
        
        best_idx = torch.argmax(match_scores).item()
        best_score = match_scores[best_idx].item()
        
        if best_score >= threshold:
            best_match = {
                'index': best_idx,
                'score': best_score,
                'is_match': True,
                'confidence': best_score
            }
    else:
        # استخدام CPU
        for i, db_features in enumerate(database_features):
            match_result = match_fingerprints(query_features, db_features, threshold)
            
            if match_result['score'] > best_match['score'] and match_result['is_match']:
                best_match = {
                    'index': i,
                    'score': match_result['score'],
                    'is_match': match_result['is_match'],
                    'confidence': match_result['confidence']
                }
    
    # تنظيف الذاكرة
    gc.collect()
    
    return best_match

def extract_features(image):
    """
    استخراج المميزات من صورة البصمة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة المعالجة
        
    Returns:
        numpy.ndarray: مصفوفة المميزات
    """
    device_info = get_device_info()
    
    # استخدام SIFT لاستخراج النقاط المميزة
    sift = cv2.SIFT_create(
        nfeatures=0,  # عدد غير محدود من النقاط المميزة
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        return np.array([])
    
    # تنظيف الذاكرة
    gc.collect()
    
    return descriptors

def compare_fingerprints(fp1_features, fp2_features):
    """
    مقارنة مميزات بصمتين مع تحسينات الأداء
    
    Args:
        fp1_features (numpy.ndarray): مميزات البصمة الأولى
        fp2_features (numpy.ndarray): مميزات البصمة الثانية
        
    Returns:
        float: درجة التطابق بين البصمتين
    """
    device_info = get_device_info()
    
    if len(fp1_features) == 0 or len(fp2_features) == 0:
        return 0.0
    
    # استخدام GPU إذا كان متاحاً
    if device_info['gpu_available']:
        fp1_tensor = torch.from_numpy(fp1_features).cuda()
        fp2_tensor = torch.from_numpy(fp2_features).cuda()
        
        # حساب متوسط المميزات
        fp1_mean = torch.mean(fp1_tensor, dim=0)
        fp2_mean = torch.mean(fp2_tensor, dim=0)
        
        # حساب درجة التطابق
        similarity = 1 - torch.nn.functional.cosine_similarity(fp1_mean, fp2_mean, dim=0).item()
    else:
        # استخدام CPU
        fp1_mean = np.mean(fp1_features, axis=0)
        fp2_mean = np.mean(fp2_features, axis=0)
        
        # حساب درجة التطابق
        similarity = 1 - cosine(fp1_mean, fp2_mean)
    
    # تحويل درجة التطابق إلى نسبة مئوية
    match_score = max(0, min(100, similarity * 100))
    
    # تنظيف الذاكرة
    gc.collect()
    
    return match_score 