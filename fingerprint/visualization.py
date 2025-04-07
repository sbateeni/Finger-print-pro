import cv2
import numpy as np

def draw_minutiae_points(image, minutiae_points, radius=3):
    """
    رسم النقاط المميزة على الصورة
    
    Args:
        image: صورة البصمة
        minutiae_points: قائمة النقاط المميزة (كل نقطة تحتوي على x, y, نوع النقطة, الزاوية)
        radius: نصف قطر الدائرة المرسومة
    
    Returns:
        صورة مع النقاط المميزة مرسومة عليها
    """
    # تحويل الصورة إلى BGR إذا كانت رمادية
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # نسخة من الصورة للرسم عليها
    marked_image = image.copy()
    
    for point in minutiae_points:
        x, y = int(point[0]), int(point[1])
        point_type = point[2] if len(point) > 2 else 'ending'  # نوع النقطة (ending أو bifurcation)
        angle = point[3] if len(point) > 3 else 0  # زاوية النقطة
        
        # لون مختلف لكل نوع من النقاط
        color = (0, 0, 255) if point_type == 'ending' else (0, 255, 0)  # أحمر للنهايات، أخضر للتفرعات
        
        # رسم دائرة عند النقطة
        cv2.circle(marked_image, (x, y), radius, color, -1)
        
        # رسم خط يشير إلى اتجاه النقطة
        if angle != 0:
            end_x = int(x + radius * 2 * np.cos(angle))
            end_y = int(y + radius * 2 * np.sin(angle))
            cv2.line(marked_image, (x, y), (end_x, end_y), color, 1)
    
    return marked_image

def draw_matching_lines(image1, image2, matching_pairs, gap=20):
    """
    رسم خطوط التطابق بين صورتين للبصمات
    
    Args:
        image1: الصورة الأولى
        image2: الصورة الثانية
        matching_pairs: قائمة من أزواج النقاط المتطابقة
        gap: المسافة بين الصورتين
    
    Returns:
        صورة تظهر الصورتين جنباً إلى جنب مع خطوط التطابق
    """
    # تحويل الصور إلى BGR إذا كانت رمادية
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # إنشاء صورة مجمعة
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2 + gap
    combined = np.zeros((h, w, 3), dtype=np.uint8)
    
    # نسخ الصور إلى الصورة المجمعة
    combined[:h1, :w1] = image1
    combined[:h2, w1+gap:] = image2
    
    # ألوان مختلفة للخطوط
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    # رسم خطوط التطابق
    for i, (p1, p2) in enumerate(matching_pairs):
        color = colors[i % len(colors)]
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # تعديل إحداثيات النقطة الثانية
        x2 += w1 + gap
        
        # رسم دوائر عند نقاط التطابق
        cv2.circle(combined, (x1, y1), 3, color, -1)
        cv2.circle(combined, (x2, y2), 3, color, -1)
        
        # رسم خط يربط النقاط المتطابقة
        cv2.line(combined, (x1, y1), (x2, y2), color, 1)
        
        # إضافة رقم للنقاط المتطابقة
        cv2.putText(combined, str(i+1), (x1-10, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(combined, str(i+1), (x2-10, y2-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return combined 