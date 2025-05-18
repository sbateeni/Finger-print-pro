import os
import json
import numpy as np
import cv2
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features

class SourceAFISProcessor:
    def __init__(self):
        self.template_dir = "templates"
        os.makedirs(self.template_dir, exist_ok=True)
        
    def preprocess_image(self, image_path):
        """معالجة الصورة باستخدام OpenCV"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # تحويل إلى تدرج رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # تحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # تخفيف الضوضاء
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # تحسين الحواف
            edges = cv2.Canny(denoised, 100, 200)
            
            # استخراج السمات
            features = extract_features(denoised)
            
            return {
                'processed': denoised,
                'edges': edges,
                'features': features
            }
        except Exception as e:
            print(f"Error processing fingerprint: {str(e)}")
            return None
            
    def extract_minutiae(self, image_path):
        """استخراج النقاط المميزة باستخدام OpenCV"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # تحويل إلى تدرج رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # تحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # تخفيف الضوضاء
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # استخراج السمات
            features = extract_features(denoised)
            
            if features is None or 'minutiae' not in features:
                return None
                
            # تحويل النقاط المميزة إلى التنسيق المطلوب
            formatted_minutiae = []
            for type_name, contours in features['minutiae'].items():
                for contour in contours:
                    try:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            # حساب الزاوية
                            if len(contour) >= 5:
                                ellipse = cv2.fitEllipse(contour)
                                angle = ellipse[2]
                            else:
                                angle = 0
                                
                            formatted_minutiae.append({
                                'x': cX,
                                'y': cY,
                                'theta': float(angle),
                                'type': type_name
                            })
                    except:
                        continue
            
            return formatted_minutiae
        except Exception as e:
            print(f"خطأ في استخراج النقاط المميزة: {str(e)}")
            return None
            
    def save_template(self, minutiae, filename):
        """حفظ قالب البصمة"""
        try:
            template_path = os.path.join(self.template_dir, f"{filename}.json")
            with open(template_path, 'w') as f:
                json.dump(minutiae, f)
            return template_path
        except Exception as e:
            print(f"خطأ في حفظ قالب البصمة: {str(e)}")
            return None
            
    def match_fingerprints(self, features1, features2):
        """مقارنة البصمات"""
        try:
            if features1 is None or features2 is None:
                return 0.0, []
            
            # حساب نسبة التطابق
            match_score, matches = match_features(features1, features2)
            
            return match_score, matches
        except Exception as e:
            print(f"Error matching fingerprints: {str(e)}")
            return 0.0, []

def process_fingerprint(image):
    """معالجة البصمة باستخدام OpenCV"""
    try:
        # تحويل الصورة إلى تدرج رمادي
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # تخفيف الضوضاء
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # تحسين الحواف
        edges = cv2.Canny(denoised, 100, 200)
        
        # استخراج السمات
        features = extract_features(denoised)
        
        return {
            'processed': denoised,
            'edges': edges,
            'features': features
        }
    except Exception as e:
        print(f"Error processing fingerprint: {str(e)}")
        return None

def match_fingerprints(features1, features2):
    """مقارنة البصمات"""
    try:
        if features1 is None or features2 is None:
            return 0.0, []
            
        # حساب نسبة التطابق
        match_score, matches = match_features(features1, features2)
        
        return match_score, matches
    except Exception as e:
        print(f"Error matching fingerprints: {str(e)}")
        return 0.0, [] 