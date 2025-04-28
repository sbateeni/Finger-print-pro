import cv2
import numpy as np

def calculate_quality(image):
    """
    حساب جودة صورة البصمة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        float: درجة الجودة (0-100)
    """
    # تحويل الصورة إلى تدرج رمادي إذا كانت ملونة
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # حساب التباين
    contrast = np.std(gray)
    
    # حساب الوضوح باستخدام لابلاس
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # حساب نسبة الضوضاء
    noise = cv2.fastNlMeansDenoising(gray)
    noise_ratio = np.mean(np.abs(gray - noise)) / 255.0
    
    # حساب نسبة المناطق السوداء (الخلفية)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = np.sum(binary == 0) / binary.size
    
    # حساب درجة الجودة النهائية
    quality_score = (
        (contrast / 128.0) * 30 +  # التباين يساهم بنسبة 30%
        (sharpness / 1000.0) * 30 +  # الوضوح يساهم بنسبة 30%
        (1 - noise_ratio) * 20 +  # نقاء الصورة يساهم بنسبة 20%
        (1 - black_ratio) * 20  # نسبة المنطقة المفيدة تساهم بنسبة 20%
    )
    
    # التأكد من أن النتيجة بين 0 و 100
    quality_score = max(0, min(100, quality_score))
    
    return quality_score 