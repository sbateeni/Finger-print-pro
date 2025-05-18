import cv2
import numpy as np
from scipy.spatial.distance import cosine
from .preprocessor import preprocess_image

def detect_minutiae(image):
    """اكتشاف النقاط المميزة في البصمة"""
    try:
        # تحسين الصورة
        enhanced = cv2.equalizeHist(image)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # اكتشاف الحواف
        edges = cv2.Canny(enhanced, 50, 150)
        
        # تطبيق عمليات مورفولوجية
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # العثور على الكنتورات
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        minutiae = {
            'ridge_endings': [],  # نهايات النتوءات
            'bifurcations': [],   # التفرعات
            'islands': [],        # الجزر
            'dots': [],           # النقاط
            'cores': [],          # النوى
            'deltas': []          # الدلتا
        }
        
        # تحليل الكنتورات
        for contour in contours:
            try:
                # حساب خصائص الكنتور
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # تجاهل الكنتورات الصغيرة جداً
                if area < 5:
                    continue
                    
                # تقريب الكنتور
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # تصنيف النقاط المميزة
                if len(approx) == 2:  # نهاية نتوء
                    minutiae['ridge_endings'].append(contour)
                elif len(approx) == 3:  # تفرع
                    minutiae['bifurcations'].append(contour)
                elif area < 50:  # نقطة
                    minutiae['dots'].append(contour)
                elif area < 200:  # جزيرة
                    minutiae['islands'].append(contour)
                elif len(approx) > 5:  # نواة أو دلتا
                    # تحليل الشكل لتحديد ما إذا كانت نواة أو دلتا
                    if cv2.isContourConvex(approx):
                        minutiae['cores'].append(contour)
                    else:
                        minutiae['deltas'].append(contour)
            except:
                continue
        
        return minutiae
        
    except Exception as e:
        print(f"Error in minutiae detection: {str(e)}")
        return None

def extract_features(image):
    """
    استخراج السمات من صورة البصمة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        dict: قاموس يحتوي على السمات المستخرجة
    """
    # معالجة الصورة
    processed = preprocess_image(image)
    
    # استخراج النقاط المميزة
    minutiae = detect_minutiae(processed['denoised'])
    
    return {
        'minutiae': minutiae,
        'processed_image': processed['denoised']
    }

def classify_minutiae(image, keypoints):
    """
    تصنيف النقاط المميزة في البصمة مع تحسينات الأداء
    
    Args:
        image (numpy.ndarray): صورة البصمة
        keypoints (list): قائمة النقاط المميزة
        
    Returns:
        list: قائمة تحتوي على أنواع النقاط المميزة
    """
    minutiae_types = []
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size)
        
        # استخراج المنطقة المحيطة بالنقطة
        roi = image[max(0, y-radius):min(image.shape[0], y+radius),
                   max(0, x-radius):min(image.shape[1], x+radius)]
        
        if roi.size == 0:
            minutiae_types.append('unknown')
            continue
        
        # حساب التدرج
        gradient_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # حساب التباين المحلي
        local_contrast = np.std(roi)
        
        # تحديد نوع النقطة المميزة
        if np.mean(gradient_magnitude) > 50 and local_contrast > 30:
            if np.std(gradient_direction) > 1.0:
                minutiae_types.append('bifurcation')
            else:
                minutiae_types.append('ridge_ending')
        else:
            minutiae_types.append('unknown')
    
    return minutiae_types

def match_features(features1, features2):
    """
    مقارنة سمات بصمتين
    
    Args:
        features1: سمات البصمة الأولى
        features2: سمات البصمة الثانية
        
    Returns:
        tuple: (نسبة التطابق، قائمة النقاط المتطابقة)
    """
    matches = []
    total_minutiae = 0
    
    # مقارنة كل نوع من النقاط المميزة
    for type_name in features1['minutiae'].keys():
        total_minutiae += len(features1['minutiae'][type_name])
        
        for contour1 in features1['minutiae'][type_name]:
            M1 = cv2.moments(contour1)
            if M1["m00"] != 0:
                cX1 = int(M1["m10"] / M1["m00"])
                cY1 = int(M1["m01"] / M1["m00"])
                
                # البحث عن أقرب نقطة في البصمة الثانية
                min_dist = float('inf')
                best_match = None
                
                for contour2 in features2['minutiae'][type_name]:
                    M2 = cv2.moments(contour2)
                    if M2["m00"] != 0:
                        cX2 = int(M2["m10"] / M2["m00"])
                        cY2 = int(M2["m01"] / M2["m00"])
                        
                        dist = np.sqrt((cX1 - cX2)**2 + (cY1 - cY2)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = (cX2, cY2)
                
                # إذا كانت المسافة أقل من عتبة معينة، نعتبر النقطتين متطابقتين
                if min_dist < 20:  # عتبة المسافة
                    matches.append(((cX1, cY1), best_match))
    
    # حساب نسبة التطابق
    match_score = (len(matches) / total_minutiae * 100) if total_minutiae > 0 else 0
    
    return match_score, matches 