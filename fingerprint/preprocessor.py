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
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء باستخدام Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف باستخدام Canny
    edges = cv2.Canny(denoised, 100, 200)
    
    # تطبيق مرشح Gaussian لإزالة الضوضاء المتبقية
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # تحسين التباين مرة أخرى
    enhanced_final = clahe.apply(blurred)
    
    return enhanced_final

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
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء باستخدام Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف باستخدام Canny
    edges = cv2.Canny(denoised, 100, 200)
    
    # تطبيق مرشح Gaussian لإزالة الضوضاء المتبقية
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # تحسين التباين مرة أخرى
    enhanced_final = clahe.apply(blurred)
    
    # تصحيح الإضاءة للصور الملونة
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        final = enhanced_final
    
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
    
    # حساب مقاييس الجودة
    quality_metrics = {
        'resolution': f"{image.shape[1]}x{image.shape[0]}",
        'clarity': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        'noise_ratio': cv2.meanStdDev(gray)[1][0][0] / cv2.meanStdDev(gray)[0][0][0],
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'entropy': -np.sum(np.histogram(gray, bins=256)[0] * np.log2(np.histogram(gray, bins=256)[0] + 1e-10))
    }
    
    return quality_metrics 