import os
import json
import numpy as np
import cv2
from sourceafis import FingerprintTemplate, FingerprintMatcher

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
            
            return denoised, edges
        except Exception as e:
            print(f"خطأ في معالجة الصورة: {str(e)}")
            return None
            
    def extract_minutiae(self, image_path):
        """استخراج النقاط المميزة باستخدام SourceAFIS"""
        try:
            # قراءة الصورة
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # إنشاء قالب البصمة
            template = FingerprintTemplate(img)
            
            # استخراج النقاط المميزة
            minutiae = template.minutiae
            
            # تحويل النقاط المميزة إلى التنسيق المطلوب
            formatted_minutiae = []
            for m in minutiae:
                formatted_minutiae.append({
                    'x': int(m.position[0]),
                    'y': int(m.position[1]),
                    'theta': float(m.angle),
                    'type': self._classify_minutiae(m.angle)
                })
            
            return formatted_minutiae
        except Exception as e:
            print(f"خطأ في استخراج النقاط المميزة: {str(e)}")
            return None
            
    def _classify_minutiae(self, angle):
        """تصنيف النقاط المميزة بناءً على الزاوية"""
        # تصنيف بسيط بناءً على الزاوية
        if angle < 45 or angle > 315:
            return 'ridge_endings'
        elif 45 <= angle <= 135:
            return 'bifurcations'
        elif 135 < angle <= 225:
            return 'islands'
        elif 225 < angle <= 315:
            return 'dots'
        return 'unknown'
        
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
            
    def match_fingerprints(self, minutiae1, minutiae2):
        """مطابقة البصمات باستخدام SourceAFIS"""
        try:
            # إنشاء قوالب البصمات
            template1 = FingerprintTemplate(minutiae1)
            template2 = FingerprintTemplate(minutiae2)
            
            # إنشاء مطابق
            matcher = FingerprintMatcher()
            
            # إضافة القالب الأول
            matcher.add(template1)
            
            # مطابقة القالب الثاني
            match = matcher.match(template2)
            
            # الحصول على درجة المطابقة
            score = match.score
            
            return score
        except Exception as e:
            print(f"خطأ في مطابقة البصمات: {str(e)}")
            return 0 