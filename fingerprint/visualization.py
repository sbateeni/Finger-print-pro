import cv2
import numpy as np
import plotly.graph_objects as go

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
        
        # إضافة رقم النقطة
        cv2.putText(vis_image, str(i), (x+5, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
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
    for i, match in enumerate(match_result['matches']):
        idx1, idx2 = match
        pt1 = (int(features1['keypoints'][idx1].pt[0]),
               int(features1['keypoints'][idx1].pt[1]))
        pt2 = (int(features2['keypoints'][idx2].pt[0]) + w1,
               int(features2['keypoints'][idx2].pt[1]))
        
        # تحديد لون الخط حسب نوع النقطة المميزة
        if features1['minutiae_types'][idx1] == features2['minutiae_types'][idx2]:
            color = (0, 255, 0)  # أخضر للتطابق
        else:
            color = (0, 0, 255)  # أحمر لعدم التطابق
        
        # رسم خط التطابق
        cv2.line(vis, pt1, pt2, color, 1)
        
        # إضافة رقم التطابق
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(vis, str(i), (mid_x, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis

def plot_quality_metrics(metrics):
    """
    رسم مقاييس جودة الصورة باستخدام Plotly
    
    Args:
        metrics (dict): مقاييس جودة الصورة
        
    Returns:
        plotly.graph_objects.Figure: الرسم البياني
    """
    # استخراج القيم
    labels = list(metrics.keys())
    values = [float(v) if isinstance(v, (int, float)) else 0 for v in metrics.values()]
    
    # إنشاء الرسم البياني
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        )
    ])
    
    # تخصيص الرسم
    fig.update_layout(
        title='مقاييس جودة الصورة',
        xaxis_title='المقياس',
        yaxis_title='القيمة',
        xaxis_tickangle=45,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=100)
    )
    
    return fig 