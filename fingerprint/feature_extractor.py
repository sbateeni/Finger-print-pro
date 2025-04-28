import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize as sk_skeletonize
from skimage.feature import peak_local_max
from typing import List, Dict, Tuple, Union, Any
import logging
from collections import defaultdict
import math

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinutiaePoint:
    def __init__(self, x, y, minutiae_type, angle=0.0):
        self.x = x
        self.y = y
        self.type = minutiae_type  # 'ridge_ending' or 'bifurcation'
        self.angle = angle

class FeatureExtractor:
    def __init__(self):
        self.min_quality = 0.15  # تخفيض الحد الأدنى للجودة
        self.radius = 5
        self.min_minutiae = 3  # تخفيض الحد الأدنى للنقاط المميزة
        self.kernel_size = 3
        self.block_size = 16
        self.orientation_smooth_size = 5

    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """استخراج المميزات من الصورة"""
        try:
            # التحقق من نوع الصورة
            if not isinstance(image, np.ndarray):
                raise ValueError("الصورة يجب أن تكون من نوع numpy array")
            
            # تحويل الصورة إلى تدرج الرمادي
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # التأكد من أن الصورة من نوع uint8
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # تحسين التباين باستخدام CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # إزالة الضوضاء باستخدام مرشح ثنائي
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # تحسين الحواف
            edges = cv2.Canny(denoised, 50, 150)
            
            # تطبيق الترقق
            skeleton = self._skeletonize(edges)
            
            # استخراج النقاط المميزة
            minutiae = self.extract_minutiae(skeleton)
            
            # حساب جودة النقاط
            quality_scores = [self._calculate_quality(denoised, point) for point in minutiae]
            
            # تصفية النقاط حسب الجودة
            filtered_minutiae = []
            filtered_scores = []
            for point, score in zip(minutiae, quality_scores):
                if score >= self.min_quality:
                    filtered_minutiae.append(point)
                    filtered_scores.append(score)
            
            # التحقق من عدد النقاط
            if len(filtered_minutiae) < self.min_minutiae:
                # محاولة ثانية مع معاملات مختلفة
                enhanced = cv2.equalizeHist(gray)
                denoised = cv2.GaussianBlur(enhanced, (5,5), 0)
                edges = cv2.Canny(denoised, 30, 100)
                skeleton = self._skeletonize(edges)
                minutiae = self.extract_minutiae(skeleton)
                quality_scores = [self._calculate_quality(denoised, point) for point in minutiae]
                filtered_minutiae = []
                filtered_scores = []
                for point, score in zip(minutiae, quality_scores):
                    if score >= self.min_quality:
                        filtered_minutiae.append(point)
                        filtered_scores.append(score)
                
                if len(filtered_minutiae) < self.min_minutiae:
                    # محاولة ثالثة مع معاملات مختلفة
                    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2)
                    denoised = cv2.medianBlur(enhanced, 3)
                    edges = cv2.Canny(denoised, 20, 80)
                    skeleton = self._skeletonize(edges)
                    minutiae = self.extract_minutiae(skeleton)
                    quality_scores = [self._calculate_quality(denoised, point) for point in minutiae]
                    filtered_minutiae = []
                    filtered_scores = []
                    for point, score in zip(minutiae, quality_scores):
                        if score >= self.min_quality:
                            filtered_minutiae.append(point)
                            filtered_scores.append(score)
                    
                    if len(filtered_minutiae) < self.min_minutiae:
                        raise ValueError(f"لم يتم العثور على نقاط مميزة كافية. تم العثور على {len(filtered_minutiae)} نقطة فقط.")
            
            # تحويل النقاط المميزة إلى القالب المطلوب
            minutiae_list = []
            for point, score in zip(filtered_minutiae, filtered_scores):
                x, y, point_type, angle = point
                minutiae_list.append({
                    'x': int(x),
                    'y': int(y),
                    'type': point_type,
                    'angle': float(angle),
                    'quality': float(score)
                })
            
            return {
                'minutiae': minutiae_list,
                'count': len(minutiae_list)
            }
            
        except Exception as e:
            logger.error(f'خطأ في استخراج المميزات: {str(e)}')
            raise

    def extract_minutiae(self, image: np.ndarray) -> List[Tuple[int, int, str, float]]:
        """استخراج النقاط المميزة من الصورة"""
        try:
            # التأكد من أن الصورة من نوع uint8
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # تطبيق Otsu's thresholding
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # تطبيق عملية التمدد لتعزيز نمط الخطوط
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # حساب عدد الجيران لكل نقطة
            neighbors = cv2.filter2D(dilated, -1, np.ones((3,3)))
            
            # البحث عن النقاط المميزة
            minutiae = []
            rows, cols = dilated.shape
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if dilated[i,j] == 255:  # نقطة بيضاء
                        # حساب عدد الجيران
                        count = neighbors[i,j] / 255
                        
                        # تحديد نوع النقطة
                        if count == 1:  # نقطة نهاية
                            minutiae.append((j, i, 'endpoint', self._calculate_angle(dilated, j, i)))
                        elif count == 3:  # نقطة تفرع
                            minutiae.append((j, i, 'branch', self._calculate_angle(dilated, j, i)))
            
            # تصفية النقاط القريبة
            filtered_minutiae = self._filter_close_minutiae(minutiae)
            
            # إعادة المحاولة إذا كان عدد النقاط غير كافٍ
            if len(filtered_minutiae) < self.min_minutiae:
                # تعديل المعاملات وإعادة المحاولة
                kernel = np.ones((5,5), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=2)
                neighbors = cv2.filter2D(dilated, -1, np.ones((5,5)))
                
                minutiae = []
                for i in range(1, rows-1):
                    for j in range(1, cols-1):
                        if dilated[i,j] == 255:
                            count = neighbors[i,j] / 255
                            if count == 1:
                                minutiae.append((j, i, 'endpoint', self._calculate_angle(dilated, j, i)))
                            elif count == 3:
                                minutiae.append((j, i, 'branch', self._calculate_angle(dilated, j, i)))
                
                filtered_minutiae = self._filter_close_minutiae(minutiae)
            
            return filtered_minutiae
            
        except Exception as e:
            logger.error(f'خطأ في استخراج النقاط المميزة: {str(e)}')
            raise

    def _filter_close_minutiae(self, minutiae: List[Tuple[int, int, str, float]], min_distance: int = 10) -> List[Tuple[int, int, str, float]]:
        """تصفية النقاط المميزة القريبة"""
        if not minutiae:
            return []
        
        # ترتيب النقاط حسب الإحداثيات
        sorted_minutiae = sorted(minutiae, key=lambda x: (x[0], x[1]))
        filtered = [sorted_minutiae[0]]
        
        for point in sorted_minutiae[1:]:
            # حساب المسافة إلى أقرب نقطة مصفاة
            min_dist = float('inf')
            for filtered_point in filtered:
                dist = np.sqrt((point[0] - filtered_point[0])**2 + (point[1] - filtered_point[1])**2)
                min_dist = min(min_dist, dist)
            
            # إضافة النقطة إذا كانت المسافة كافية
            if min_dist >= min_distance:
                filtered.append(point)
        
        return filtered

    def _calculate_quality(self, image: np.ndarray, point: Tuple[int, int, str, float]) -> float:
        """حساب جودة النقطة المميزة"""
        try:
            x, y = point[0], point[1]
            patch_size = 5
            
            # استخراج المنطقة المحيطة بالنقطة
            x1 = max(0, x - patch_size)
            y1 = max(0, y - patch_size)
            x2 = min(image.shape[1], x + patch_size + 1)
            y2 = min(image.shape[0], y + patch_size + 1)
            
            patch = image[y1:y2, x1:x2]
            
            # حساب التباين
            contrast = np.std(patch)
            
            # حساب التماسك
            coherence = cv2.Laplacian(patch, cv2.CV_64F).var()
            
            # حساب الجودة النهائية
            quality = (contrast / 255.0 + coherence / 1000.0) / 2.0
            
            return min(max(quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f'خطأ في حساب جودة النقطة: {str(e)}')
            return 0.0

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """تحويل الصورة إلى تدرج الرمادي"""
        try:
            # التحقق من أن الصورة صالحة
            if image is None:
                raise ValueError("الصورة فارغة")
            
            # التحقق من نوع البيانات
            if not isinstance(image, np.ndarray):
                raise ValueError("الصورة يجب أن تكون من نوع numpy array")
            
            # التحقق من أبعاد الصورة
            if len(image.shape) < 2:
                raise ValueError("أبعاد الصورة غير صحيحة")
            
            # تحويل الصورة إلى تدرج الرمادي إذا كانت ملونة
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        except Exception as e:
            logger.error(f'خطأ في تحويل الصورة إلى تدرج الرمادي: {str(e)}')
            raise

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """تحسين التباين"""
        try:
            # تطبيق CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
            # تطبيق مرشح متوسط
            enhanced = cv2.medianBlur(enhanced, 3)
            
            # تطبيق معادلة التباين
            enhanced = cv2.equalizeHist(enhanced)
            
            return enhanced
        except Exception as e:
            logger.error(f'خطأ في تحسين التباين: {str(e)}')
            raise

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """إزالة الضوضاء"""
        try:
            # تطبيق إزالة الضوضاء غير المحلية
            denoised = cv2.fastNlMeansDenoising(image, None, 5, 3, 9)
            
            # تطبيق مرشح ثنائي
            denoised = cv2.bilateralFilter(denoised, 5, 75, 75)
            
            # تطبيق مرشح متوسط
            denoised = cv2.medianBlur(denoised, 3)
            
            return denoised
        except Exception as e:
            logger.error(f'خطأ في إزالة الضوضاء: {str(e)}')
            raise

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """كشف الحواف"""
        try:
            # تطبيق كشف الحواف
            edges = cv2.Canny(image, 30, 100)
            
            # تحسين الحواف
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)
            
            return edges
        except Exception as e:
            logger.error(f'خطأ في كشف الحواف: {str(e)}')
            raise

    def _skeletonize(self, image: np.ndarray) -> np.ndarray:
        """تحويل الصورة إلى هيكل عظمي"""
        try:
            # تحويل الصورة إلى ثنائية
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # تطبيق الترقق
            skeleton = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            done = False
            
            while not done:
                eroded = cv2.erode(binary, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(binary, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                binary = eroded.copy()
                
                zeros = cv2.countNonZero(binary)
                if zeros == 0:
                    done = True
            
            # تنظيف الهيكل العظمي
            skeleton = cv2.medianBlur(skeleton, 3)
            
            return skeleton
        except Exception as e:
            logger.error(f'خطأ في تحويل الصورة إلى هيكل عظمي: {str(e)}')
            raise

    def _calculate_angle(self, skeleton: np.ndarray, x: int, y: int) -> float:
        """حساب زاوية النقطة المميزة"""
        try:
            # حساب التدرج
            sobelx = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            
            # حساب الزاوية
            angle = np.arctan2(sobely[y, x], sobelx[y, x]) * 180 / np.pi
            
            # تطبيع الزاوية
            angle = (angle + 360) % 360
            
            return float(angle)
        except Exception as e:
            logger.error(f'خطأ في حساب الزاوية: {str(e)}')
            return 0.0

    def _filter_minutiae(self, minutiae: List[Tuple[int, int, str, float]], skeleton: np.ndarray, min_distance: int = 6) -> List[Tuple[int, int, str, float]]:
        """تصفية النقاط المميزة المتقاربة"""
        if not minutiae:
            return []
            
        filtered = []
        used = set()
        
        # ترتيب النقاط حسب الجودة
        sorted_minutiae = sorted(minutiae, key=lambda x: x[3], reverse=True)
        
        for i, m1 in enumerate(sorted_minutiae):
            if i in used:
                continue
                
            filtered.append(m1)
            used.add(i)
            
            # إزالة النقاط القريبة
            for j, m2 in enumerate(sorted_minutiae[i+1:], i+1):
                if j in used:
                    continue
                    
                dist = math.sqrt((m1[0] - m2[0])**2 + (m1[1] - m2[1])**2)
                if dist < min_distance:
                    used.add(j)
        
        return filtered

def skeletonize(img):
    """
    Convert the input image to a skeleton image using scikit-image
    """
    try:
        # Ensure the image is binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Normalize to 0-1 range for skimage
        binary_normalized = binary.astype(bool)
        
        # Apply skeletonization
        skeleton = sk_skeletonize(binary_normalized)
        
        # Convert back to uint8 format
        return (skeleton * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error in skeletonization: {str(e)}")
        return None

def extract_features(image):
    """
    Extract minutiae points from the fingerprint image
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create skeleton image
        skeleton_image = skeletonize(image)
        if skeleton_image is None:
            raise Exception("فشل في عملية تنحيف الصورة")
        
        # Ensure the image is binary
        binary = skeleton_image > 0

        # Find minutiae points using crossing number method
        minutiae_points = []
        rows, cols = binary.shape
        
        # Padding to avoid border issues
        padded = np.pad(binary, ((1, 1), (1, 1)), mode='constant')
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if padded[i, j]:  # Only process ridge pixels
                    # Get 3x3 neighborhood
                    neighbors = padded[i-1:i+2, j-1:j+2].astype(np.uint8)
                    cn = compute_crossing_number(neighbors)
                    
                    if cn == 1:  # Ridge ending
                        angle = compute_orientation(binary, i-1, j-1)
                        minutiae_points.append(MinutiaePoint(j-1, i-1, 'ridge_ending', angle))
                    elif cn == 3:  # Bifurcation
                        angle = compute_orientation(binary, i-1, j-1)
                        minutiae_points.append(MinutiaePoint(j-1, i-1, 'bifurcation', angle))
        
        # Filter false minutiae
        filtered_points = filter_minutiae(minutiae_points, binary)
        
        if not filtered_points:
            raise Exception("لم يتم العثور على نقاط مميزة في الصورة")
        
        return filtered_points
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return []

def compute_crossing_number(neighborhood):
    """
    Compute the crossing number for a 3x3 neighborhood
    """
    # Convert to binary pattern
    pattern = neighborhood.flatten()[:-1]  # Exclude last element
    transitions = 0
    
    # Count transitions between 0 and 1
    for k in range(8):
        transitions += abs(int(pattern[k]) - int(pattern[(k+1)%8]))
    
    return transitions // 2

def compute_orientation(image, i, j, window_size=7):
    """
    Compute the orientation of the ridge at point (i,j)
    """
    # Extract local window
    half_size = window_size // 2
    window = image[max(0, i-half_size):min(image.shape[0], i+half_size+1),
                  max(0, j-half_size):min(image.shape[1], j+half_size+1)]
    
    # Compute gradient
    gy, gx = np.gradient(window.astype(float))
    angle = np.arctan2(gy.mean(), gx.mean())
    
    return angle

def filter_minutiae(minutiae_points, binary_image, min_distance=10):
    """
    Filter false minutiae points:
    1. Remove points too close to the border
    2. Remove points too close to each other
    3. Remove points in high-density regions
    """
    filtered = []
    rows, cols = binary_image.shape
    border = 20
    
    for point in minutiae_points:
        # Check border distance
        if (point.x < border or point.x >= cols - border or 
            point.y < border or point.y >= rows - border):
            continue
            
        # Check distance from other points
        too_close = False
        for other in filtered:
            dx = point.x - other.x
            dy = point.y - other.y
            if np.sqrt(dx*dx + dy*dy) < min_distance:
                too_close = True
                break
                
        if not too_close:
            filtered.append(point)
    
    return filtered 

def draw_minutiae_points(image, minutiae_points):
    """
    رسم النقاط المميزة على صورة البصمة
    """
    # نسخ الصورة للرسم عليها
    result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    # رسم كل نقطة مع لونها المميز
    for point in minutiae_points:
        x, y = int(point.x), int(point.y)
        if point.type == 'ridge_ending':
            # نقاط النهاية باللون الأحمر
            cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
            # رسم خط يشير إلى الاتجاه
            dx = int(15 * np.cos(point.angle))
            dy = int(15 * np.sin(point.angle))
            cv2.line(result, (x, y), (x + dx, y + dy), (0, 0, 255), 2)
        else:
            # نقاط التفرع باللون الأخضر
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            # رسم خط يشير إلى الاتجاه
            dx = int(15 * np.cos(point.angle))
            dy = int(15 * np.sin(point.angle))
            cv2.line(result, (x, y), (x + dx, y + dy), (0, 255, 0), 2)
    
    return result 