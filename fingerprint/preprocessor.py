import cv2
import numpy as np

def preprocess_image(image):
    """
    معالجة الصورة المسبقة للبصمة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        numpy.ndarray: الصورة المعالجة
    """
    # تحويل الصورة إلى تدرج الرمادي
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف
    edges = cv2.Canny(denoised, 100, 200)
    
    return denoised

def enhance_image_quality(image):
    """
    تحسين جودة الصورة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        dict: قاموس يحتوي على الصور في مراحل المعالجة المختلفة
    """
    # تحويل إلى تدرج الرمادي
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف
    edges = cv2.Canny(denoised, 100, 200)
    
    # تصحيح الإضاءة
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        final = denoised
    
    return {
        'original': image,
        'gray': gray,
        'enhanced': enhanced,
        'denoised': denoised,
        'edges': edges,
        'final': final
    }

def check_image_quality(image):
    """
    فحص جودة الصورة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        dict: قاموس يحتوي على مقاييس جودة الصورة
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    quality_metrics = {
        'resolution': f"{image.shape[1]}x{image.shape[0]}",
        'clarity': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        'noise_ratio': cv2.meanStdDev(gray)[1][0][0] / cv2.meanStdDev(gray)[0][0][0]
    }
    
    return quality_metrics 