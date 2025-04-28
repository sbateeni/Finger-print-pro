import cv2
import numpy as np
from .feature_extractor import match_features

def match_fingerprints(features1, features2, threshold=0.6):
    """
    مقارنة بصمتين وإرجاع نتيجة المطابقة
    
    Args:
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        threshold (float): الحد الأدنى لنسبة التطابق
        
    Returns:
        dict: نتائج المطابقة
    """
    # مقارنة المميزات
    match_result = match_features(features1, features2)
    
    # تحديد نتيجة المطابقة
    is_match = match_result['score'] >= threshold
    
    return {
        'is_match': is_match,
        'score': match_result['score'],
        'matches': match_result['matches'],
        'match_count': match_result['count'],
        'threshold': threshold
    }

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
    
    return similarity_score

def find_best_match(query_features, database_features, threshold=0.6):
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
        'is_match': False
    }
    
    for i, db_features in enumerate(database_features):
        match_result = match_fingerprints(query_features, db_features, threshold)
        
        if match_result['score'] > best_match['score']:
            best_match = {
                'index': i,
                'score': match_result['score'],
                'is_match': match_result['is_match']
            }
    
    return best_match 