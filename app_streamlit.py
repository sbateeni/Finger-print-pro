import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features, detect_minutiae
from fingerprint.quality import calculate_quality
import gc
import pandas as pd

# تعيين الحد الأقصى لحجم الصورة (بالبايت)
MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8MB

def process_image_stages(image_file):
    """معالجة الصورة عبر جميع المراحل"""
    stages = {}
    
    try:
        # تحويل ملف Streamlit إلى صورة OpenCV
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return stages
            
        # 🖼️ المرحلة 1: معالجة الصورة
        with st.spinner("جاري معالجة الصورة..."):
            # تحويل إلى تدرج رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # تحسين التباين
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # تخفيف الضوضاء
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # تحسين الحواف
            edges = cv2.Canny(denoised, 100, 200)
            
            stages['processed'] = denoised
            stages['edges'] = edges
        
        # 📍 المرحلة 2: استخراج السمات
        if 'processed' in stages:
            with st.spinner("جاري استخراج السمات..."):
                features = extract_features(stages['processed'])
                if features is not None:
                    stages['features'] = features
        
        # 📁 المرحلة 3: حفظ السمات
        if 'features' in stages:
            with st.spinner("جاري حفظ السمات..."):
                filename = f"features_{hash(str(image_file.name))}.json"
                save_features_to_json(stages['features'], filename)
                stages['saved_features'] = filename
        
        return stages
    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
        return stages

def save_features_to_json(features, filename):
    """حفظ السمات بتنسيق JSON"""
    # تحويل الكنتورات إلى قوائم
    minutiae_data = {}
    for type_name, contours in features['minutiae'].items():
        minutiae_data[type_name] = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                minutiae_data[type_name].append({
                    'x': cX,
                    'y': cY,
                    'type': type_name
                })
    
    # حفظ البيانات
    with open(filename, 'w') as f:
        json.dump(minutiae_data, f)

def draw_minutiae_with_matches(image, features, matches=None, other_image=None):
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
                    
                    # رسم دائرة حول النقطة المميزة
                    cv2.circle(img_with_minutiae, (cX, cY), 8, color, -1)
                    
                    # رسم رقم النقطة المميزة
                    cv2.putText(img_with_minutiae, str(len(features['minutiae'][minutiae_type])), 
                              (cX-10, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except:
                continue
    
    # رسم خطوط التطابق إذا كانت هناك تطابقات
    if matches is not None and other_image is not None:
        for i, match in enumerate(matches):
            try:
                # الحصول على إحداثيات النقاط المتطابقة
                pt1 = (int(match[0][0]), int(match[0][1]))
                pt2 = (int(match[1][0]), int(match[1][1]))
                
                # رسم دائرة حول النقطة في الصورة الأولى
                cv2.circle(img_with_minutiae, pt1, 10, (0, 255, 255), 2)
                
                # رسم دائرة حول النقطة في الصورة الثانية
                cv2.circle(other_image, pt2, 10, (0, 255, 255), 2)
                
                # رسم خط التطابق
                cv2.line(img_with_minutiae, pt1, pt2, (0, 255, 255), 2)
                
                # رسم رقم التطابق
                cv2.putText(img_with_minutiae, str(i+1), 
                          (pt1[0]-10, pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(other_image, str(i+1), 
                          (pt2[0]-10, pt2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            except:
                continue
    
    return img_with_minutiae

def show_minutiae_details(features):
    """عرض تفاصيل النقاط المميزة"""
    if features is None or 'minutiae' not in features:
        return
        
    # عرض إحصائيات النقاط المميزة
    st.markdown("#### إحصائيات النقاط المميزة")
    stats = {
        'نهاية نتوء': len(features['minutiae'].get('ridge_endings', [])),
        'تفرع': len(features['minutiae'].get('bifurcations', [])),
        'جزيرة': len(features['minutiae'].get('islands', [])),
        'نقطة': len(features['minutiae'].get('dots', [])),
        'نواة': len(features['minutiae'].get('cores', [])),
        'دلتا': len(features['minutiae'].get('deltas', []))
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
        .stage-container {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        .stage-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .stage-icon {
            font-size: 1.5rem;
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
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="title-text">نظام مقارنة البصمات</h1>', unsafe_allow_html=True)
    
    # تحميل الصور
    st.markdown("### تحميل البصمات")
    uploaded_files = st.file_uploader("اختر البصمات للمقارنة", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("بدء المعالجة"):
        if uploaded_files and len(uploaded_files) >= 2:
            # معالجة كل بصمة
            processed_stages = []
            for file in uploaded_files:
                # إعادة تعيين مؤشر الملف
                file.seek(0)
                stages = process_image_stages(file)
                processed_stages.append(stages)
            
            # عرض نتائج كل مرحلة
            for i, stages in enumerate(processed_stages):
                st.markdown(f'<div class="stage-container">', unsafe_allow_html=True)
                st.markdown(f'<h3>البصمة {i+1}</h3>', unsafe_allow_html=True)
                
                # 🖼️ عرض مرحلة معالجة الصورة
                if 'processed' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">🖼️</span> معالجة الصورة</div>', unsafe_allow_html=True)
                    st.image(stages['processed'], use_container_width=True)
                    
                    if 'edges' in stages:
                        st.markdown("#### حواف البصمة")
                        st.image(stages['edges'], use_container_width=True)
                
                # 📍 عرض مرحلة استخراج السمات
                if 'features' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">📍</span> استخراج السمات</div>', unsafe_allow_html=True)
                    img_with_minutiae = draw_minutiae_with_matches(stages['processed'], stages['features'])
                    st.image(img_with_minutiae, use_container_width=True)
                    
                    # عرض إحصائيات السمات
                    show_minutiae_details(stages['features'])
                
                # 📁 عرض مرحلة حفظ السمات
                if 'saved_features' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">📁</span> حفظ السمات</div>', unsafe_allow_html=True)
                    st.markdown(f"تم حفظ السمات في الملف: `{stages['saved_features']}`")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 🔍 عرض مرحلة مطابقة البصمات
            st.markdown('<div class="stage-container">', unsafe_allow_html=True)
            st.markdown('<div class="stage-title"><span class="stage-icon">🔍</span> مطابقة البصمات</div>', unsafe_allow_html=True)
            
            for i in range(len(processed_stages)):
                for j in range(i+1, len(processed_stages)):
                    if 'features' in processed_stages[i] and 'features' in processed_stages[j]:
                        st.markdown(f"#### مقارنة البصمة {i+1} مع البصمة {j+1}")
                        
                        # حساب التطابق
                        match_score, matches = match_features(
                            processed_stages[i]['features'],
                            processed_stages[j]['features']
                        )
                        
                        # عرض النتائج
                        st.markdown('<div class="match-info">', unsafe_allow_html=True)
                        st.markdown(f"نسبة التطابق: {match_score:.2f}%")
                        st.markdown(f"عدد النقاط المتطابقة: {len(matches)}")
                        
                        if match_score > 80:
                            st.markdown('<div class="match-result match">البصمتان متطابقتان</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="match-result no-match">البصمتان غير متطابقتين</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # عرض الصور مع خطوط التطابق
                        col1, col2 = st.columns(2)
                        with col1:
                            img1_with_matches = draw_minutiae_with_matches(
                                processed_stages[i]['processed'],
                                processed_stages[i]['features'],
                                matches,
                                processed_stages[j]['processed']
                            )
                            st.image(img1_with_matches, caption=f"البصمة {i+1}", use_container_width=True)
                        
                        with col2:
                            img2_with_matches = draw_minutiae_with_matches(
                                processed_stages[j]['processed'],
                                processed_stages[j]['features'],
                                matches,
                                processed_stages[i]['processed']
                            )
                            st.image(img2_with_matches, caption=f"البصمة {j+1}", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 📊 عرض مرحلة التحليل
            st.markdown('<div class="stage-container">', unsafe_allow_html=True)
            st.markdown('<div class="stage-title"><span class="stage-icon">📊</span> التحليل والإحصائيات</div>', unsafe_allow_html=True)
            
            # عرض إحصائيات عامة
            total_minutiae = sum(len(stages.get('features', {}).get('minutiae', {}).get(type_name, [])) 
                               for stages in processed_stages 
                               for type_name in ['ridge_endings', 'bifurcations', 'islands', 'dots', 'cores', 'deltas'])
            
            st.markdown(f"إجمالي عدد النقاط المميزة: {total_minutiae}")
            
            # عرض جودة كل بصمة
            for i, stages in enumerate(processed_stages):
                if 'processed' in stages:
                    quality = calculate_quality(stages['processed'])
                    st.markdown(f"جودة البصمة {i+1}: {quality:.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("الرجاء اختيار بصمتين على الأقل للمقارنة")
    
    # معلومات النظام
    st.header("معلومات النظام")
    st.markdown("""
    - يدعم النظام مقارنة بصمات الأصابع باستخدام خوارزميات متقدمة
    - يمكن تحميل الصور بصيغ PNG, JPG, JPEG
    - يتم معالجة الصور تلقائياً لتحسين جودة المقارنة
    - يعرض النظام النقاط المميزة وخطوط التطابق
    - يحسب نسبة التطابق بين البصمات
    - يدعم مقارنة أكثر من بصمتين في نفس الوقت
    - يحفظ السمات في ملفات JSON للرجوع إليها لاحقاً
    """)

if __name__ == "__main__":
    main() 