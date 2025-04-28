import cv2
import numpy as np
from .feature_extractor import match_features

def match_fingerprints(features1, features2, threshold=0.5):
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
    if features1['count'] < 10 or features2['count'] < 10:
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
    
    # تحديد نتيجة المطابقة
    is_match = match_result['score'] >= threshold and confidence_score >= 0.6
    
    return {
        'is_match': is_match,
        'score': match_result['score'],
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
    confidence = (match_ratio * 0.7 + type_consistency * 0.3)
    
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
        similarity_score *= 0.5
    
    return similarity_score

def find_best_match(query_features, database_features, threshold=0.5):
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