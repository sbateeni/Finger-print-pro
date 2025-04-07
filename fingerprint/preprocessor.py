import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage

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

def preprocess_image(image_path):
    """
    المعالجة الأولية الكاملة لصورة البصمة
    """
    # قراءة الصورة
    if isinstance(image_path, str):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين جودة الصورة
    enhanced = enhance_image(image)
    
    # تطبيع الصورة
    normalized = normalize_image(enhanced)
    
    # فصل البصمة
    segmented = segment_fingerprint(normalized)
    
    # تقدير اتجاه الخطوط
    orientation = estimate_orientation(segmented)
    
    # تحسين الخطوط
    final = enhance_ridges(segmented, orientation)
    
    return final

def enhance_contrast(image):
    """
    Enhance the contrast of the image using CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced

def remove_background(image):
    """
    Remove the background noise from the image
    """
    # Calculate the local variance
    local_std = ndimage.gaussian_filter(image, sigma=2)
    mask = local_std > np.mean(local_std)
    return image * mask 