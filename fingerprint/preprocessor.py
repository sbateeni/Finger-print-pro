import cv2
import numpy as np
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.morphology import skeletonize, remove_small_objects
from skimage.transform import rotate, resize
from skimage.feature import canny, peak_local_max
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.color import rgb2gray
import logging
from typing import Dict, Optional, Tuple, Union
import math

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, config: Optional[Dict] = None):
        """
        تهيئة معالج الصور
        
        Args:
            config: قاموس يحتوي على إعدادات المعالجة
        """
        # الإعدادات الافتراضية
        self.config = {
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8),
            'denoising_strength': 10,
            'denoising_template_size': 7,
            'denoising_search_size': 21,
            'canny_threshold1': 100,
            'canny_threshold2': 200,
            'gaussian_sigma': 1.0,
            'binary_threshold': 0,
            'min_ridge_width': 3,
            'min_ridge_length': 10,
            'enhancement_method': 'clahe',  # 'clahe', 'histogram', 'adaptive'
            'denoising_method': 'fastnl',  # 'fastnl', 'bilateral', 'tv'
            'edge_detection_method': 'canny',  # 'canny', 'sobel'
            'skeletonization_method': 'zhang',  # 'zhang', 'morphological'
        }
        
        # تحديث الإعدادات إذا تم توفيرها
        if config:
            self.config.update(config)

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """معالجة الصورة وتحسين جودتها"""
        try:
            # التحقق من نوع الصورة
            if not isinstance(image, np.ndarray):
                raise ValueError("الصورة يجب أن تكون من نوع numpy array")
            
            # التأكد من أن الصورة في التنسيق الصحيح
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # تحويل الصورة إلى تدرج رمادي
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # تحسين التباين باستخدام CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # إزالة الضوضاء باستخدام مرشح متوسط
            denoised = cv2.medianBlur(enhanced, 3)
            
            # تحسين الحواف
            edges = cv2.Canny(denoised, 50, 150)
            
            # تطبيق عملية الترقق
            skeleton = self._skeletonize(denoised)
            
            return {
                'gray': gray,
                'enhanced': enhanced,
                'denoised': denoised,
                'edges': edges,
                'skeleton': skeleton
            }
            
        except Exception as e:
            logger.error(f'خطأ في معالجة الصورة: {str(e)}')
            raise

    def _skeletonize(self, image: np.ndarray) -> np.ndarray:
        """ترقيق الصورة للحصول على الهيكل العظمي"""
        try:
            # تحويل الصورة إلى ثنائية
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # تطبيق عملية الترقق
            skeleton = np.zeros_like(binary)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            
            while True:
                eroded = cv2.erode(binary, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(binary, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                binary = eroded.copy()
                
                if cv2.countNonZero(binary) == 0:
                    break
            
            return skeleton
            
        except Exception as e:
            logger.error(f'خطأ في عملية الترقق: {str(e)}')
            raise

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        تحسين تباين الصورة
        
        Args:
            image: صورة بتدرج الرمادي
            
        Returns:
            صورة محسنة التباين
        """
        try:
            method = self.config['enhancement_method']
            
            if method == 'clahe':
                # تطبيق CLAHE
                clahe = cv2.createCLAHE(
                    clipLimit=self.config['clahe_clip_limit'],
                    tileGridSize=self.config['clahe_tile_size']
                )
                enhanced = clahe.apply(image)
                
            elif method == 'histogram':
                # تطبيق معادلة التدرج الرمادي
                enhanced = cv2.equalizeHist(image)
                
            elif method == 'adaptive':
                # تطبيق معادلة التدرج الرمادي التكيفية
                enhanced = equalize_adapthist(image)
                enhanced = (enhanced * 255).astype(np.uint8)
                
            else:
                # استخدام الطريقة الافتراضية
                clahe = cv2.createCLAHE(
                    clipLimit=self.config['clahe_clip_limit'],
                    tileGridSize=self.config['clahe_tile_size']
                )
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.error(f'خطأ في تحسين التباين: {str(e)}')
            raise

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        إزالة الضوضاء من الصورة
        
        Args:
            image: صورة بتدرج الرمادي
            
        Returns:
            صورة بدون ضوضاء
        """
        try:
            method = self.config['denoising_method']
            
            if method == 'fastnl':
                # تطبيق Non-local Means Denoising
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    self.config['denoising_strength'],
                    self.config['denoising_template_size'],
                    self.config['denoising_search_size']
                )
                
            elif method == 'bilateral':
                # تطبيق Bilateral Filter
                denoised = cv2.bilateralFilter(
                    image,
                    9,
                    75,
                    75
                )
                
            elif method == 'tv':
                # تطبيق Total Variation Denoising
                denoised = denoise_tv_chambolle(image, weight=0.1)
                denoised = (denoised * 255).astype(np.uint8)
                
            else:
                # استخدام الطريقة الافتراضية
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    self.config['denoising_strength'],
                    self.config['denoising_template_size'],
                    self.config['denoising_search_size']
                )
            
            return denoised
            
        except Exception as e:
            logger.error(f'خطأ في إزالة الضوضاء: {str(e)}')
            raise

    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        كشف حواف الصورة
        
        Args:
            image: صورة بتدرج الرمادي
            
        Returns:
            صورة حواف
        """
        try:
            method = self.config['edge_detection_method']
            
            if method == 'canny':
                # تطبيق Canny Edge Detection
                edges = cv2.Canny(
                    image,
                    self.config['canny_threshold1'],
                    self.config['canny_threshold2']
                )
                
            elif method == 'sobel':
                # تطبيق Sobel Edge Detection
                sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
                
            else:
                # استخدام الطريقة الافتراضية
                edges = cv2.Canny(
                    image,
                    self.config['canny_threshold1'],
                    self.config['canny_threshold2']
                )
            
            return edges
            
        except Exception as e:
            logger.error(f'خطأ في كشف الحواف: {str(e)}')
            raise

    def _skeletonize_image(self, image: np.ndarray) -> np.ndarray:
        """
        تحويل الصورة إلى هيكل عظمي
        
        Args:
            image: صورة حواف
            
        Returns:
            صورة هيكل عظمي
        """
        try:
            method = self.config['skeletonization_method']
            
            # تحويل الصورة إلى ثنائية
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if method == 'zhang':
                # تطبيق خوارزمية Zhang-Suen
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
                        
            elif method == 'morphological':
                # تطبيق خوارزمية مورفولوجية
                skeleton = np.zeros(binary.shape, np.uint8)
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                
                while True:
                    eroded = cv2.erode(binary, element)
                    dilated = cv2.dilate(eroded, element)
                    skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary, dilated))
                    binary = eroded.copy()
                    
                    if cv2.countNonZero(binary) == 0:
                        break
                        
            else:
                # استخدام الطريقة الافتراضية
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
            
            return skeleton
        except Exception as e:
            logger.error(f'خطأ في تحويل الصورة إلى هيكل عظمي: {str(e)}')
            raise

    def _enhance_final_image(self, image: np.ndarray) -> np.ndarray:
        """
        تحسين الصورة النهائية
        
        Args:
            image: صورة هيكل عظمي
            
        Returns:
            صورة نهائية محسنة
        """
        try:
            # تطبيق Gaussian Blur
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            
            # تطبيق العتبة
            _, enhanced = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # إزالة النقاط الصغيرة
            min_ridge_width = self.config['min_ridge_width']
            min_ridge_length = self.config['min_ridge_length']
            
            # إزالة النقاط الصغيرة
            kernel = np.ones((min_ridge_width, min_ridge_width), np.uint8)
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
            
            # إزالة الخطوط القصيرة
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced, connectivity=8)
            
            # إزالة المكونات الصغيرة
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_ridge_length:
                    labels[labels == i] = 0
            
            enhanced = (labels > 0).astype(np.uint8) * 255
            
            return enhanced
            
        except Exception as e:
            logger.error(f'خطأ في تحسين الصورة النهائية: {str(e)}')
            raise

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        تدوير الصورة
        
        Args:
            image: صورة البصمة
            angle: زاوية التدوير بالدرجات
            
        Returns:
            صورة مدورة
        """
        try:
            # الحصول على أبعاد الصورة
            height, width = image.shape[:2]
            
            # حساب مركز الصورة
            center = (width / 2, height / 2)
            
            # إنشاء مصفوفة التدوير
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # تطبيق التدوير
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            return rotated
            
        except Exception as e:
            logger.error(f'خطأ في تدوير الصورة: {str(e)}')
            raise

    def scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        تغيير حجم الصورة
        
        Args:
            image: صورة البصمة
            scale_factor: معامل التغيير
            
        Returns:
            صورة بحجم جديد
        """
        try:
            # حساب الأبعاد الجديدة
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            
            # تغيير حجم الصورة
            scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            return scaled
            
        except Exception as e:
            logger.error(f'خطأ في تغيير حجم الصورة: {str(e)}')
            raise

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        تطبيع الصورة
        
        Args:
            image: صورة البصمة
            
        Returns:
            صورة مطبعة
        """
        try:
            # تطبيع الصورة إلى النطاق [0, 255]
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
            return normalized
            
        except Exception as e:
            logger.error(f'خطأ في تطبيع الصورة: {str(e)}')
            raise

    def segment_fingerprint(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        تقسيم البصمة إلى مناطق
        
        Args:
            image: صورة البصمة
            
        Returns:
            صورة البصمة المقسمة وقناع المنطقة
        """
        try:
            # تحويل الصورة إلى تدرج الرمادي
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # تطبيق عتبة Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # تطبيق عمليات مورفولوجية
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # تطبيق القناع على الصورة
            segmented = cv2.bitwise_and(gray, mask)
            
            return segmented, mask
            
        except Exception as e:
            logger.error(f'خطأ في تقسيم البصمة: {str(e)}')
            raise

    def estimate_orientation(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        تقدير اتجاه البصمة
        
        Args:
            image: صورة البصمة
            block_size: حجم الكتلة
            
        Returns:
            مصفوفة الاتجاهات
        """
        try:
            # تحويل الصورة إلى تدرج الرمادي
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # حساب التدرج
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # حساب الاتجاه
            orientation = np.arctan2(gy, gx) / 2
            
            # حساب متوسط الاتجاه لكل كتلة
            height, width = gray.shape
            block_h = height // block_size
            block_w = width // block_size
            
            block_orientation = np.zeros((block_h, block_w))
            
            for i in range(block_h):
                for j in range(block_w):
                    y_start = i * block_size
                    y_end = min((i + 1) * block_size, height)
                    x_start = j * block_size
                    x_end = min((j + 1) * block_size, width)
                    
                    block_orientation[i, j] = np.mean(orientation[y_start:y_end, x_start:x_end])
            
            return block_orientation
            
        except Exception as e:
            logger.error(f'خطأ في تقدير اتجاه البصمة: {str(e)}')
            raise

    def enhance_ridges(self, image: np.ndarray, orientation: np.ndarray, freq: float = 1/9) -> np.ndarray:
        """
        تحسين الخطوط
        
        Args:
            image: صورة البصمة
            orientation: مصفوفة الاتجاهات
            freq: تردد الخطوط
            
        Returns:
            صورة محسنة الخطوط
        """
        try:
            # تحويل الصورة إلى تدرج الرمادي
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # تطبيق فلتر Gabor
            enhanced = np.zeros_like(gray, dtype=np.float32)
            
            for i in range(orientation.shape[0]):
                for j in range(orientation.shape[1]):
                    theta = orientation[i, j]
                    
                    # إنشاء فلتر Gabor
                    kernel = cv2.getGaborKernel(
                        (21, 21),
                        4.0,
                        theta,
                        freq,
                        0.5,
                        0,
                        ktype=cv2.CV_32F
                    )
                    
                    # تطبيق الفلتر
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    
                    # إضافة النتيجة
                    y_start = i * 16
                    y_end = min((i + 1) * 16, gray.shape[0])
                    x_start = j * 16
                    x_end = min((j + 1) * 16, gray.shape[1])
                    
                    enhanced[y_start:y_end, x_start:x_end] = filtered[y_start:y_end, x_start:x_end]
            
            # تطبيع النتيجة
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f'خطأ في تحسين الخطوط: {str(e)}')
            raise

# إنشاء كائن المعالج
preprocessor = Preprocessor()

# تصدير الدالة
def preprocess_image(image):
    return preprocessor.preprocess_image(image)

def enhance_image(image):
    """
    تحسين جودة صورة البصمة
    """
    # تحويل الصورة إلى تدرجات الرمادي
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء باستخدام مرشح ثنائي
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return denoised

def normalize_image(image):
    """
    تطبيع الصورة لتحسين التباين
    """
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def segment_fingerprint(image):
    """
    فصل البصمة عن الخلفية
    """
    # تطبيق عتبة Otsu
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # تطبيق عمليات مورفولوجية لتحسين القناع
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # تطبيق القناع على الصورة
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented

def estimate_orientation(image, block_size=16):
    """
    تقدير اتجاه الخطوط في البصمة
    """
    # حساب التدرجات
    gy, gx = np.gradient(image.astype(float))
    
    # حساب المكونات
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    
    # تطبيق متوسط الكتل
    kernel_size = (block_size, block_size)
    gxx_block = cv2.blur(gxx, kernel_size)
    gyy_block = cv2.blur(gyy, kernel_size)
    gxy_block = cv2.blur(gxy, kernel_size)
    
    # حساب الاتجاه
    orientation = np.arctan2(2 * gxy_block, gxx_block - gyy_block) / 2
    
    return orientation

def enhance_ridges(image, orientation, freq=1/9):
    """
    تحسين خطوط البصمة باستخدام مرشح جابور
    """
    enhanced = np.zeros_like(image)
    rows, cols = image.shape
    
    for i in range(0, rows, 16):
        for j in range(0, cols, 16):
            # إنشاء مرشح جابور للكتلة الحالية
            angle = orientation[i//16, j//16]
            kernel = cv2.getGaborKernel((15, 15), 4.0, angle, 1/freq, 0.5, 0, ktype=cv2.CV_32F)
            
            # تطبيق المرشح على الكتلة
            block = image[i:min(i+16, rows), j:min(j+16, cols)]
            filtered = cv2.filter2D(block, cv2.CV_8UC1, kernel)
            enhanced[i:min(i+16, rows), j:min(j+16, cols)] = filtered
    
    return enhanced

def enhance_contrast(image):
    """
    تحسين تباين الصورة
    """
    try:
        # تحويل إلى تدرج الرمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تطبيق CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    except Exception as e:
        logger.error(f'خطأ في تحسين التباين: {str(e)}')
        raise

def remove_noise(image):
    """
    إزالة الضوضاء من الصورة
    """
    try:
        # تطبيق مرشح Gaussian
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # تطبيق مرشح Median
        denoised = cv2.medianBlur(blurred, 5)
        
        return denoised
    except Exception as e:
        logger.error(f'خطأ في إزالة الضوضاء: {str(e)}')
        raise

def detect_edges(image):
    """
    كشف الحواف في الصورة
    """
    try:
        # تطبيق Canny
        edges = cv2.Canny(image, 100, 200)
        
        return edges
    except Exception as e:
        logger.error(f'خطأ في كشف الحواف: {str(e)}')
        raise

def skeletonize_image(image):
    """
    تحويل الصورة إلى هيكل عظمي
    """
    try:
        # تحويل إلى تدرج الرمادي
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تطبيق العتبة
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # تحويل إلى هيكل عظمي
        skeleton = skeletonize(binary > 0)
        
        return skeleton
    except Exception as e:
        logger.error(f'خطأ في تحويل الصورة إلى هيكل عظمي: {str(e)}')
        raise 