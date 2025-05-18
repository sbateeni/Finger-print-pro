import cv2
import numpy as np

def calculate_quality(image):
    """
    حساب جودة صورة البصمة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        float: درجة الجودة (0-100)
    """
    # حساب التباين
    contrast = calculate_contrast(image)
    
    # حساب السطوع
    brightness = calculate_brightness(image)
    
    # حساب الوضوح
    sharpness = calculate_sharpness(image)
    
    # حساب درجة الجودة النهائية
    quality_score = (contrast * 0.4 + brightness * 0.3 + sharpness * 0.3) * 100
    
    return quality_score

def calculate_contrast(image):
    """
    حساب تباين الصورة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        float: درجة التباين (0-1)
    """
    # حساب الانحراف المعياري للقيم الرمادية
    std_dev = np.std(image)
    
    # تحويل القيمة إلى نطاق 0-1
    contrast = min(std_dev / 128.0, 1.0)
    
    return contrast

def calculate_brightness(image):
    """
    حساب سطوع الصورة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        float: درجة السطوع (0-1)
    """
    # حساب متوسط القيم الرمادية
    mean_value = np.mean(image)
    
    # تحويل القيمة إلى نطاق 0-1
    brightness = 1.0 - abs(mean_value - 128) / 128.0
    
    return brightness

def calculate_sharpness(image):
    """
    حساب وضوح الصورة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        float: درجة الوضوح (0-1)
    """
    # تطبيق عامل لابلاس
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # حساب التباين في صورة لابلاس
    sharpness = np.var(laplacian)
    
    # تحويل القيمة إلى نطاق 0-1
    sharpness = min(sharpness / 1000.0, 1.0)
    
    return sharpness

def assess_image_quality(image):
    """
    تقييم جودة الصورة وإرجاع تقرير مفصل
    
    Args:
        image: صورة OpenCV
        
    Returns:
        dict: قاموس يحتوي على تقرير الجودة
    """
    quality = calculate_quality(image)
    contrast = calculate_contrast(image)
    brightness = calculate_brightness(image)
    sharpness = calculate_sharpness(image)
    
    # تقييم الجودة
    if quality >= 0.8:
        quality_level = "ممتازة"
    elif quality >= 0.6:
        quality_level = "جيدة"
    elif quality >= 0.4:
        quality_level = "متوسطة"
    else:
        quality_level = "ضعيفة"
    
    return {
        'overall_quality': quality,
        'quality_level': quality_level,
        'contrast': contrast,
        'brightness': brightness,
        'sharpness': sharpness
    } 