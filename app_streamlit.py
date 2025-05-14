import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features
from fingerprint.quality import calculate_quality
import gc
import pandas as pd

# تعيين الحد الأقصى لحجم الصورة (بالبايت)
MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 5MB

def draw_minutiae(image, features, matches=None, other_image=None):
    """رسم النقاط المميزة على الصورة مع خطوط التطابق"""
    if features is None or 'minutiae' not in features:
        return image
        
    # نسخ الصورة
    img_with_minutiae = image.copy()
    
    # ألوان النقاط المميزة
    colors = {
        'ridge_endings': (0, 255, 0),    # أخضر
        'bifurcations': (255, 0, 0),     # أزرق
        'islands': (0, 0, 255),          # أحمر
        'dots': (255, 255, 0),           # أصفر
        'cores': (255, 0, 255),          # وردي
        'deltas': (0, 255, 255)          # سماوي
    }
    
    # رسم النقاط المميزة
    for minutiae_type, contours in features['minutiae'].items():
        color = colors[minutiae_type]
        for contour in contours:
            try:
                # رسم الكنتور
                cv2.drawContours(img_with_minutiae, [contour], -1, color, 2)
                
                # رسم مركز الكتلة
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(img_with_minutiae, (cX, cY), 5, color, -1)
                    
                    # رسم خط اتجاه النقطة المميزة
                    if minutiae_type in ['ridge_endings', 'bifurcations']:
                        try:
                            # حساب اتجاه النقطة المميزة
                            if len(contour) >= 5:  # يجب أن يكون هناك 5 نقاط على الأقل لرسم القطع الناقص
                                ellipse = cv2.fitEllipse(contour)
                                angle = ellipse[2]
                                length = 20
                                endX = int(cX + length * np.cos(np.radians(angle)))
                                endY = int(cY + length * np.sin(np.radians(angle)))
                                cv2.arrowedLine(img_with_minutiae, (cX, cY), (endX, endY), color, 2)
                        except:
                            # في حالة فشل حساب الاتجاه، نرسم خطاً بسيطاً
                            cv2.line(img_with_minutiae, (cX-10, cY), (cX+10, cY), color, 2)
            except:
                continue
    
    # رسم خطوط التطابق إذا كانت هناك تطابقات
    if matches is not None and other_image is not None:
        for match in matches:
            try:
                # الحصول على إحداثيات النقاط المتطابقة
                pt1 = (int(match[0][0]), int(match[0][1]))
                pt2 = (int(match[1][0]), int(match[1][1]))
                
                # رسم دائرة حول النقطة في الصورة الأولى
                cv2.circle(img_with_minutiae, pt1, 8, (0, 255, 255), 2)
                
                # رسم دائرة حول النقطة في الصورة الثانية
                cv2.circle(other_image, pt2, 8, (0, 255, 255), 2)
                
                # رسم خط التطابق
                cv2.line(img_with_minutiae, pt1, pt2, (0, 255, 255), 2)
            except:
                continue
    
    return img_with_minutiae

def create_minutiae_table(features):
    """إنشاء جدول للنقاط المميزة"""
    if features is None or 'minutiae' not in features:
        return None
        
    data = []
    for minutiae_type, contours in features['minutiae'].items():
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # حساب التوجه
                angle = 0
                if minutiae_type in ['ridge_endings', 'bifurcations']:
                    angle = cv2.fitEllipse(contour)[2]
                
                data.append({
                    'النوع': minutiae_type,
                    'الموقع (X,Y)': f'({cX}, {cY})',
                    'المساحة': f'{area:.2f}',
                    'المحيط': f'{perimeter:.2f}',
                    'التوجه': f'{angle:.2f}°'
                })
    
    return pd.DataFrame(data)

def show_minutiae_details(features):
    """عرض تفاصيل النقاط المميزة"""
    if features is None or 'minutiae' not in features:
        return
        
    # عرض إحصائيات النقاط المميزة
    st.markdown("#### إحصائيات النقاط المميزة")
    stats = {
        'نهاية نتوء': len(features['minutiae']['ridge_endings']),
        'تفرع': len(features['minutiae']['bifurcations']),
        'جزيرة': len(features['minutiae']['islands']),
        'نقطة': len(features['minutiae']['dots']),
        'نواة': len(features['minutiae']['cores']),
        'دلتا': len(features['minutiae']['deltas'])
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("نهايات النتوءات", stats['نهاية نتوء'])
        st.metric("التفرعات", stats['تفرع'])
    with col2:
        st.metric("الجزر", stats['جزيرة'])
        st.metric("النقاط", stats['نقطة'])
    with col3:
        st.metric("النوى", stats['نواة'])
        st.metric("الدلتا", stats['دلتا'])
    
    # عرض جدول النقاط المميزة
    st.markdown("#### تفاصيل النقاط المميزة")
    df = create_minutiae_table(features)
    if df is not None:
        st.markdown('<div class="minutiae-table">', unsafe_allow_html=True)
        st.table(df)
        st.markdown('</div>', unsafe_allow_html=True)

def process_image(image):
    """معالجة صورة البصمة"""
    try:
        # التحقق من حجم الصورة
        if len(image.getvalue()) > MAX_IMAGE_SIZE:
            raise ValueError("حجم الصورة كبير جداً. الحد الأقصى هو 5 ميجابايت")
            
        # تحويل الصورة إلى مصفوفة NumPy
        img_array = np.array(Image.open(image))
        
        # معالجة الصورة
        processed = preprocess_image(img_array)
        
        # استخراج المميزات
        features = extract_features(processed)
        
        # تنظيف الذاكرة
        gc.collect()
            
        return processed, features
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="نظام مقارنة البصمات",
        page_icon="👆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # تنسيق CSS مخصص
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .title-text {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-box {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .result-box {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
        }
        .quality-score {
            font-size: 1.2rem;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .match-score {
            font-size: 1.5rem;
            font-weight: bold;
            color: #27ae60;
            margin: 1rem 0;
        }
        .match-result {
            font-size: 1.2rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .match-result.match {
            background-color: #2ecc71;
            color: white;
        }
        .match-result.no-match {
            background-color: #e74c3c;
            color: white;
        }
        .stButton>button {
            width: 100%;
            background-color: #3498db;
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 10px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .minutiae-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 1rem 0;
        }
        .minutiae-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .minutiae-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
        }
        .minutiae-table {
            margin: 1rem 0;
            border-radius: 10px;
            overflow: hidden;
        }
        .minutiae-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .minutiae-table th {
            background-color: #3498db;
            color: white;
            padding: 0.5rem;
            text-align: right;
        }
        .minutiae-table td {
            padding: 0.5rem;
            border: 1px solid #ddd;
        }
        .minutiae-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .matching-lines {
            position: relative;
            margin: 2rem 0;
        }
        .matching-lines img {
            max-width: 100%;
            height: auto;
        }
        .matching-lines .line {
            position: absolute;
            background-color: rgba(255, 255, 0, 0.5);
            height: 2px;
            transform-origin: left center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="title-text">نظام مقارنة البصمات</h1>', unsafe_allow_html=True)
    
    # تحميل الصور
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### البصمة الأولى")
        fp1 = st.file_uploader("اختر البصمة الأولى", type=['png', 'jpg', 'jpeg'])
        if fp1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(Image.open(fp1), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### البصمة الثانية")
        fp2 = st.file_uploader("اختر البصمة الثانية", type=['png', 'jpg', 'jpeg'])
        if fp2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(Image.open(fp2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # زر المقارنة
    if st.button("مقارنة البصمات", key="compare"):
        if fp1 and fp2:
            # معالجة البصمات
            with st.spinner("جاري معالجة البصمات..."):
                processed1, features1 = process_image(fp1)
                processed2, features2 = process_image(fp2)
                
                if processed1 is not None and processed2 is not None:
                    # عرض النتائج
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### نتائج المقارنة")
                    
                    # رسم النقاط المميزة مع خطوط التطابق
                    img1_with_minutiae = draw_minutiae(processed1, features1)
                    img2_with_minutiae = draw_minutiae(processed2, features2)
                    
                    # الحصول على التطابقات
                    matches = match_features(features1, features2)
                    
                    # رسم خطوط التطابق
                    img1_with_matches = draw_minutiae(img1_with_minutiae, features1, matches, img2_with_minutiae)
                    img2_with_matches = draw_minutiae(img2_with_minutiae, features2, matches, img1_with_minutiae)
                    
                    # عرض النقاط المميزة مع خطوط التطابق
                    st.markdown("#### النقاط المميزة وخطوط التطابق")
                    st.markdown("""
                    <div class="minutiae-legend">
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(0, 255, 0);"></div>
                            <span>نهاية نتوء</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(255, 0, 0);"></div>
                            <span>تفرع</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(0, 0, 255);"></div>
                            <span>جزيرة</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(255, 255, 0);"></div>
                            <span>نقطة</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(255, 0, 255);"></div>
                            <span>نواة</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(0, 255, 255);"></div>
                            <span>دلتا</span>
                        </div>
                        <div class="minutiae-item">
                            <div class="minutiae-color" style="background-color: rgb(255, 255, 0);"></div>
                            <span>خطوط التطابق</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img1_with_matches, caption="البصمة الأولى مع النقاط المميزة وخطوط التطابق", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        quality1 = calculate_quality(processed1)
                        st.markdown(f'<div class="quality-score">جودة البصمة الأولى: {quality1:.2f}%</div>', unsafe_allow_html=True)
                        
                        # عرض تفاصيل النقاط المميزة للبصمة الأولى
                        show_minutiae_details(features1)
                        
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img2_with_matches, caption="البصمة الثانية مع النقاط المميزة وخطوط التطابق", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        quality2 = calculate_quality(processed2)
                        st.markdown(f'<div class="quality-score">جودة البصمة الثانية: {quality2:.2f}%</div>', unsafe_allow_html=True)
                        
                        # عرض تفاصيل النقاط المميزة للبصمة الثانية
                        show_minutiae_details(features2)
                    
                    # مقارنة البصمات
                    if features1 and features2:
                        match_score, matches = match_features(features1, features2)
                        st.markdown(f'<div class="match-score">نسبة التطابق: {match_score:.2f}%</div>', unsafe_allow_html=True)
                        
                        if match_score > 80:
                            st.markdown('<div class="match-result match">البصمتان متطابقتان</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="match-result no-match">البصمتان غير متطابقتين</div>', unsafe_allow_html=True)
                    else:
                        st.warning("لم يتم العثور على مميزات كافية في إحدى البصمتين")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("الرجاء اختيار كلا البصمتين للمقارنة")
    
    # معلومات النظام
    st.header("معلومات النظام")
    st.markdown("""
    - يدعم النظام مقارنة بصمات الأصابع باستخدام خوارزميات متقدمة
    - يمكن تحميل الصور بصيغ PNG, JPG, JPEG
    - يتم معالجة الصور تلقائياً لتحسين جودة المقارنة
    - يعرض النظام النقاط المميزة وخطوط التطابق
    - يحسب نسبة التطابق بين البصمتين
    """)

if __name__ == "__main__":
    main() 