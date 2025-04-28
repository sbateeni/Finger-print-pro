import cv2
import numpy as np
from .feature_extractor import match_features
from scipy.spatial.distance import cosine

def match_fingerprints(features1, features2, threshold=0.3):
    """
    مقارنة بصمتين وإرجاع نتيجة المطابقة
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        threshold (float): الحد الأدنى لنسبة التطابق
        
    Returns:
        dict: نتائج المطابقة
    """
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
    
    # تحديد نتيجة المطابقة
    is_match = (match_result['score'] >= threshold and 
               confidence_score >= 0.4 and 
               similarity_score >= 0.3)
    
    # حساب النتيجة النهائية
    final_score = (match_result['score'] * 0.4 + 
                  confidence_score * 0.3 + 
                  similarity_score * 0.3)
    
    return {
        'is_match': is_match,
        'score': final_score * 100,  # تحويل إلى نسبة مئوية
        'matches': match_result['matches'],
        'match_count': match_result['count'],
        'threshold': threshold,
        'confidence': confidence_score
    }

def calculate_confidence_score(features1, features2, match_result):
    """
    حساب درجة الثقة في نتيجة المطابقة
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        match_result (dict): نتائج المطابقة
        
    Returns:
        float: درجة الثقة
    """
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
    حساب درجة التشابه بين بصمتين
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        
    Returns:
        float: درجة التشابه
    """
    # مقارنة المميزات
    match_result = match_features(features1, features2)
    
    # حساب درجة التشابه
    similarity_score = match_result['score']
    
    # تطبيق معامل التصحيح
    if match_result['count'] < 10:
        similarity_score *= 0.7  # تخفيف العقوبة
    
    return similarity_score

def find_best_match(query_features, database_features, threshold=0.3):
    """
    البحث عن أفضل تطابق في قاعدة البيانات
    
    Args:
        query_features (dict): مميزات البصمة المراد مطابقتها
        database_features (list): قائمة مميزات البصمات في قاعدة البيانات
        threshold (float): الحد الأدنى لنسبة التطابق
        
    Returns:
        dict: أفضل نتيجة مطابقة
    """
    best_match = {
        'index': -1,
        'score': 0,
        'is_match': False,
        'confidence': 0
    }
    
    for i, db_features in enumerate(database_features):
        match_result = match_fingerprints(query_features, db_features, threshold)
        
        if match_result['score'] > best_match['score'] and match_result['is_match']:
            best_match = {
                'index': i,
                'score': match_result['score'],
                'is_match': match_result['is_match'],
                'confidence': match_result['confidence']
            }
    
    return best_match

def extract_features(image):
    """
    استخراج المميزات من صورة البصمة
    
    Args:
        image (numpy.ndarray): صورة البصمة المعالجة
        
    Returns:
        numpy.ndarray: مصفوفة المميزات
    """
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
    
    return descriptors

def compare_fingerprints(fp1_features, fp2_features):
    """
    مقارنة مميزات بصمتين
    
    Args:
        fp1_features (numpy.ndarray): مميزات البصمة الأولى
        fp2_features (numpy.ndarray): مميزات البصمة الثانية
        
    Returns:
        float: درجة التطابق بين البصمتين
    """
    if len(fp1_features) == 0 or len(fp2_features) == 0:
        return 0.0
    
    # حساب متوسط المميزات لكل بصمة
    fp1_mean = np.mean(fp1_features, axis=0)
    fp2_mean = np.mean(fp2_features, axis=0)
    
    # حساب درجة التطابق باستخدام جيب التمام
    similarity = 1 - cosine(fp1_mean, fp2_mean)
    
    # تحويل درجة التطابق إلى نسبة مئوية
    match_score = max(0, min(100, similarity * 100))
    
    return match_score 