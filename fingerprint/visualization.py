import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_features(image, features):
    """
    تصور النقاط المميزة على الصورة
    
    Args:
        image (numpy.ndarray): صورة البصمة
        features (dict): مميزات البصمة
        
    Returns:
        numpy.ndarray: الصورة مع النقاط المميزة
    """
    # نسخ الصورة
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # رسم النقاط المميزة
    for i, (kp, m_type) in enumerate(zip(features['keypoints'], features['minutiae_types'])):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # تحديد لون النقطة حسب نوعها
        if m_type == 'ridge_ending':
            color = (0, 0, 255)  # أحمر
        elif m_type == 'bifurcation':
            color = (0, 255, 0)  # أخضر
        else:
            color = (255, 0, 0)  # أزرق
        
        # رسم النقطة
        cv2.circle(vis_image, (x, y), 3, color, -1)
        
        # رسم دائرة حول النقطة
        cv2.circle(vis_image, (x, y), int(kp.size), color, 1)
        
        # رسم اتجاه النقطة
        angle = kp.angle * np.pi / 180.0
        end_x = int(x + kp.size * np.cos(angle))
        end_y = int(y + kp.size * np.sin(angle))
        cv2.line(vis_image, (x, y), (end_x, end_y), color, 1)
    
    return vis_image

def visualize_matching(img1, img2, features1, features2, match_result):
    """
    تصور نتائج المطابقة بين بصمتين
    
    Args:
        img1 (numpy.ndarray): صورة البصمة الأولى
        img2 (numpy.ndarray): صورة البصمة الثانية
        features1 (dict): مميزات البصمة الأولى
        features2 (dict): مميزات البصمة الثانية
        match_result (dict): نتائج المطابقة
        
    Returns:
        numpy.ndarray: صورة تجمع بين البصمتين مع خطوط التطابق
    """
    # نسخ الصور
    vis_img1 = img1.copy()
    vis_img2 = img2.copy()
    
    # تحويل الصور إلى BGR إذا كانت رمادية
    if len(vis_img1.shape) == 2:
        vis_img1 = cv2.cvtColor(vis_img1, cv2.COLOR_GRAY2BGR)
    if len(vis_img2.shape) == 2:
        vis_img2 = cv2.cvtColor(vis_img2, cv2.COLOR_GRAY2BGR)
    
    # إنشاء صورة تجمع بين البصمتين
    h1, w1 = vis_img1.shape[:2]
    h2, w2 = vis_img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = vis_img1
    vis[:h2, w1:w1+w2] = vis_img2
    
    # رسم خطوط التطابق
    for match in match_result['matches']:
        idx1, idx2 = match
        pt1 = (int(features1['keypoints'][idx1].pt[0]),
               int(features1['keypoints'][idx1].pt[1]))
        pt2 = (int(features2['keypoints'][idx2].pt[0]) + w1,
               int(features2['keypoints'][idx2].pt[1]))
        
        # رسم خط التطابق
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
    
    return vis

def plot_quality_metrics(metrics):
    """
    رسم مقاييس جودة الصورة
    
    Args:
        metrics (dict): مقاييس جودة الصورة
        
    Returns:
        matplotlib.figure.Figure: الرسم البياني
    """
    # إنشاء الرسم البياني
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # استخراج القيم
    labels = list(metrics.keys())
    values = [float(v) if isinstance(v, (int, float)) else 0 for v in metrics.values()]
    
    # رسم الأعمدة
    bars = ax.bar(labels, values)
    
    # تخصيص الرسم
    ax.set_title('مقاييس جودة الصورة')
    ax.set_ylabel('القيمة')
    plt.xticks(rotation=45)
    
    # إضافة القيم على الأعمدة
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig 