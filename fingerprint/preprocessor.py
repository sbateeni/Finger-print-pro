import cv2
import numpy as np
import platform
import psutil
import torch
import gc

def get_device_info():
    """Get system information for optimization"""
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory': psutil.virtual_memory().total,
        'platform': platform.system(),
        'cuda_available': False
    }
    
    try:
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_device'] = torch.cuda.get_device_name(0)
            info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory
    except Exception as e:
        print(f"Warning: CUDA check failed: {str(e)}")
    
    return info

def preprocess_image(image):
    """
    معالجة الصورة المسبقة للبصمة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        numpy.ndarray: الصورة المعالجة
    """
    device_info = get_device_info()
    
    # تحويل الصورة إلى تدرج الرمادي
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تغيير حجم الصورة إذا كانت كبيرة جداً
    if gray.shape[0] > device_info['max_image_size'][0] or gray.shape[1] > device_info['max_image_size'][1]:
        scale = min(device_info['max_image_size'][0] / gray.shape[0],
                   device_info['max_image_size'][1] / gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # تحسين التباين باستخدام CLAHE مع معلمات محسنة
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء باستخدام Bilateral Filter للحفاظ على الحواف
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # تحسين الحواف باستخدام Canny مع معلمات محسنة
    edges = cv2.Canny(denoised, 50, 150)
    
    # تطبيق مرشح Gaussian لإزالة الضوضاء المتبقية
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # تحسين التباين مرة أخرى باستخدام CLAHE
    enhanced_final = clahe.apply(blurred)
    
    # تطبيق عمليات مورفولوجية لتحسين جودة البصمة
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(enhanced_final, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # تطبيق Adaptive Thresholding
    binary = cv2.adaptiveThreshold(eroded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # تنظيف الذاكرة
    gc.collect()
    
    return binary

def enhance_image_quality(image):
    """
    تحسين جودة الصورة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        dict: قاموس يحتوي على الصور في مراحل المعالجة المختلفة
    """
    device_info = get_device_info()
    
    # تحويل إلى تدرج الرمادي
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تغيير حجم الصورة إذا كانت كبيرة جداً
    if gray.shape[0] > device_info['max_image_size'][0] or gray.shape[1] > device_info['max_image_size'][1]:
        scale = min(device_info['max_image_size'][0] / gray.shape[0],
                   device_info['max_image_size'][1] / gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء باستخدام Bilateral Filter
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # تحسين الحواف باستخدام Canny
    edges = cv2.Canny(denoised, 50, 150)
    
    # تطبيق مرشح Gaussian
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
    
    # تنظيف الذاكرة
    gc.collect()
    
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
    فحص جودة الصورة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        dict: قاموس يحتوي على مقاييس جودة الصورة
    """
    device_info = get_device_info()
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # تغيير حجم الصورة إذا كانت كبيرة جداً
    if gray.shape[0] > device_info['max_image_size'][0] or gray.shape[1] > device_info['max_image_size'][1]:
        scale = min(device_info['max_image_size'][0] / gray.shape[0],
                   device_info['max_image_size'][1] / gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # حساب مقاييس الجودة
    quality_metrics = {
        'resolution': f"{image.shape[1]}x{image.shape[0]}",
        'clarity': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        'noise_ratio': cv2.meanStdDev(gray)[1][0][0] / cv2.meanStdDev(gray)[0][0][0],
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'entropy': -np.sum(np.histogram(gray, bins=256)[0] * np.log2(np.histogram(gray, bins=256)[0] + 1e-10)),
        'device_info': device_info
    }
    
    # تنظيف الذاكرة
    gc.collect()
    
    return quality_metrics

def preprocess_fingerprint(image):
    """
    معالجة صورة البصمة وتحسين جودتها مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        
    Returns:
        numpy.ndarray: صورة البصمة المعالجة
    """
    device_info = get_device_info()
    
    # تحويل الصورة إلى تدرج رمادي
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # تغيير حجم الصورة إذا كانت كبيرة جداً
    if gray.shape[0] > device_info['max_image_size'][0] or gray.shape[1] > device_info['max_image_size'][1]:
        scale = min(device_info['max_image_size'][0] / gray.shape[0],
                   device_info['max_image_size'][1] / gray.shape[1])
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # تحسين التباين باستخدام CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # تطبيق فلتر Bilateral
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # تحسين الحواف باستخدام Canny
    edges = cv2.Canny(denoised, 50, 150)
    
    # تطبيق عمليات مورفولوجية
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # تطبيق Adaptive Thresholding
    binary = cv2.adaptiveThreshold(eroded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # تنظيف الذاكرة
    gc.collect()
    
    return binary 