import cv2
import numpy as np
from scipy.spatial.distance import cosine

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
    """استخراج المميزات من صورة البصمة"""
    try:
        # تحسين الصورة
        enhanced = cv2.equalizeHist(image)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # اكتشاف النقاط المميزة
        minutiae = detect_minutiae(enhanced)
        
        # استخراج المميزات باستخدام SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(enhanced, None)
        
        # تصنيف النقاط المميزة
        if keypoints is not None:
            minutiae_types = classify_minutiae(enhanced, keypoints)
        else:
            minutiae_types = []
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'minutiae': minutiae,
            'minutiae_types': minutiae_types
        }
        
    except Exception as e:
        print(f"خطأ في استخراج المميزات: {str(e)}")
        return None

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
    """مقارنة مميزات بصمتين"""
    try:
        if features1 is None or features2 is None:
            return 0.0, []
            
        # التحقق من وجود المميزات
        if not features1['keypoints'] or not features2['keypoints']:
            return 0.0, []
            
        # إنشاء FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # مطابقة المميزات
        matches = flann.knnMatch(features1['descriptors'], features2['descriptors'], k=2)
        
        # تطبيق نسبة Lowe
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # حساب نسبة التطابق
        if len(good_matches) > 0:
            match_score = (len(good_matches) / min(len(features1['keypoints']), len(features2['keypoints']))) * 100
        else:
            match_score = 0.0
            
        # تحضير التطابقات للعرض
        matches_for_display = []
        for match in good_matches:
            pt1 = features1['keypoints'][match.queryIdx].pt
            pt2 = features2['keypoints'][match.trainIdx].pt
            matches_for_display.append((pt1, pt2))
            
        return match_score, matches_for_display
        
    except Exception as e:
        print(f"خطأ في مقارنة المميزات: {str(e)}")
        return 0.0, [] 