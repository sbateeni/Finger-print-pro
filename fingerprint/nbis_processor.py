import os
import subprocess
import json
import numpy as np
import cv2

class NBISProcessor:
    def __init__(self):
        self.mindtct_path = "mindtct"  # مسار تنفيذ MINDTCT
        self.bozorth3_path = "bozorth3"  # مسار تنفيذ Bozorth3
        
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
            
    def extract_minutiae(self, image_path, output_dir):
        """استخراج النقاط المميزة باستخدام MINDTCT"""
        try:
            # إنشاء مجلد المخرجات إذا لم يكن موجوداً
            os.makedirs(output_dir, exist_ok=True)
            
            # تنفيذ MINDTCT
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, base_name)
            
            cmd = [self.mindtct_path, image_path, output_path]
            subprocess.run(cmd, check=True)
            
            # قراءة ملف النقاط المميزة
            minutiae_file = f"{output_path}.xyt"
            if os.path.exists(minutiae_file):
                return self._read_minutiae_file(minutiae_file)
            return None
        except Exception as e:
            print(f"خطأ في استخراج النقاط المميزة: {str(e)}")
            return None
            
    def _read_minutiae_file(self, xyt_file):
        """قراءة ملف النقاط المميزة بتنسيق XYT"""
        minutiae = []
        try:
            with open(xyt_file, 'r') as f:
                for line in f:
                    x, y, theta = map(float, line.strip().split())
                    minutiae.append({
                        'x': int(x),
                        'y': int(y),
                        'theta': theta,
                        'type': self._classify_minutiae(theta)
                    })
            return minutiae
        except Exception as e:
            print(f"خطأ في قراءة ملف النقاط المميزة: {str(e)}")
            return None
            
    def _classify_minutiae(self, theta):
        """تصنيف النقاط المميزة بناءً على الزاوية"""
        # تصنيف بسيط بناءً على الزاوية
        if theta < 45 or theta > 315:
            return 'ridge_endings'
        elif 45 <= theta <= 135:
            return 'bifurcations'
        elif 135 < theta <= 225:
            return 'islands'
        elif 225 < theta <= 315:
            return 'dots'
        return 'unknown'
        
    def match_fingerprints(self, minutiae1, minutiae2):
        """مطابقة البصمات باستخدام Bozorth3"""
        try:
            # حفظ النقاط المميزة في ملفات مؤقتة
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            file1 = os.path.join(temp_dir, "fp1.xyt")
            file2 = os.path.join(temp_dir, "fp2.xyt")
            
            self._save_minutiae_to_xyt(minutiae1, file1)
            self._save_minutiae_to_xyt(minutiae2, file2)
            
            # تنفيذ Bozorth3
            cmd = [self.bozorth3_path, "-m1", file1, file2]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # تحليل النتيجة
            score = self._parse_bozorth3_output(result.stdout)
            
            # تنظيف الملفات المؤقتة
            os.remove(file1)
            os.remove(file2)
            
            return score
        except Exception as e:
            print(f"خطأ في مطابقة البصمات: {str(e)}")
            return 0
            
    def _save_minutiae_to_xyt(self, minutiae, output_file):
        """حفظ النقاط المميزة بتنسيق XYT"""
        with open(output_file, 'w') as f:
            for m in minutiae:
                f.write(f"{m['x']} {m['y']} {m['theta']}\n")
                
    def _parse_bozorth3_output(self, output):
        """تحليل مخرجات Bozorth3"""
        try:
            # البحث عن درجة المطابقة في المخرجات
            for line in output.split('\n'):
                if "MATCH" in line:
                    return int(line.split()[-1])
            return 0
        except:
            return 0 