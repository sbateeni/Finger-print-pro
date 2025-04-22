import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import compare_fingerprints
from fingerprint.visualization import draw_minutiae_points, draw_matching_lines

# إعداد الصفحة
st.set_page_config(
    page_title="نظام مقارنة البصمات",
    page_icon="👆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS
st.markdown("""
<style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 1.2em;
    }
    .stProgress > div > div > div {
        background-color: #0d6efd;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .css-1d391kg:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.title("👆 نظام مقارنة البصمات")
st.markdown("---")

# إنشاء عمودين للصور
col1, col2 = st.columns(2)

# رفع الصور
with col1:
    st.subheader("البصمة الأولى")
    fingerprint1 = st.file_uploader("اختر صورة البصمة الأولى", type=['png', 'jpg', 'jpeg'], key="fp1")

with col2:
    st.subheader("البصمة الثانية")
    fingerprint2 = st.file_uploader("اختر صورة البصمة الثانية", type=['png', 'jpg', 'jpeg'], key="fp2")

# زر المقارنة
compare_button = st.button("🔍 مقارنة البصمات", type="primary")

# عرض الصور المرفوعة
if fingerprint1 and fingerprint2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(fingerprint1, caption="البصمة الأولى", use_container_width=True)
    with col2:
        st.image(fingerprint2, caption="البصمة الثانية", use_container_width=True)

# معالجة المقارنة
if compare_button and fingerprint1 and fingerprint2:
    # إنشاء شريط تقدم
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # تحويل الصور إلى numpy arrays
        img1 = np.array(Image.open(fingerprint1))
        img2 = np.array(Image.open(fingerprint2))
        
        # حفظ الصور مؤقتاً
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        img1_path = os.path.join(temp_dir, "fp1.jpg")
        img2_path = os.path.join(temp_dir, "fp2.jpg")
        
        cv2.imwrite(img1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        
        # معالجة الصور
        status_text.text("جاري معالجة الصور...")
        progress_bar.progress(20)
        
        processed_fp1 = preprocess_image(img1_path)
        processed_fp2 = preprocess_image(img2_path)
        
        # استخراج المميزات
        status_text.text("جاري استخراج المميزات...")
        progress_bar.progress(40)
        
        features1 = extract_features(processed_fp1)
        features2 = extract_features(processed_fp2)
        
        # رسم النقاط المميزة
        marked_fp1 = draw_minutiae_points(processed_fp1, features1)
        marked_fp2 = draw_minutiae_points(processed_fp2, features2)
        
        # مقارنة البصمات
        status_text.text("جاري مقارنة البصمات...")
        progress_bar.progress(60)
        
        match_score, matching_points = compare_fingerprints(features1, features2)
        
        # رسم خطوط التطابق
        status_text.text("جاري إنشاء التصور...")
        progress_bar.progress(80)
        
        matching_visualization = draw_matching_lines(marked_fp1, marked_fp2, list(zip(features1, features2)))
        
        # عرض النتائج
        status_text.text("اكتملت العملية!")
        progress_bar.progress(100)
        
        # إنشاء أعمدة لعرض النتائج
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(marked_fp1, caption="البصمة الأولى مع النقاط المميزة", use_container_width=True)
        
        with col2:
            st.image(marked_fp2, caption="البصمة الثانية مع النقاط المميزة", use_container_width=True)
        
        with col3:
            st.image(matching_visualization, caption="خطوط التطابق", use_container_width=True)
        
        # عرض إحصائيات المقارنة
        st.markdown("### 📊 نتائج المقارنة")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("نسبة التطابق", f"{match_score:.2%}")
        
        with col2:
            st.metric("عدد النقاط المميزة في البصمة الأولى", len(features1))
        
        with col3:
            st.metric("عدد النقاط المميزة في البصمة الثانية", len(features2))
        
        # عرض النقاط المتطابقة
        st.markdown("### 📍 النقاط المتطابقة")
        for i, match in enumerate(matching_points):
            point1 = match['point1']
            point2 = match['point2']
            st.write(f"نقطة تطابق {i+1}: ({point1['x']}, {point1['y']}) ↔ ({point2['x']}, {point2['y']})")
        
        # تنظيف الملفات المؤقتة
        os.remove(img1_path)
        os.remove(img2_path)
        
    except Exception as e:
        st.error(f"حدث خطأ: {str(e)}")
        progress_bar.progress(0)
        status_text.text("فشلت العملية!")

# معلومات إضافية في الشريط الجانبي
with st.sidebar:
    st.markdown("### ℹ️ معلومات النظام")
    st.markdown("""
    - نظام مقارنة البصمات باستخدام خوارزميات متقدمة
    - يدعم الصور بتنسيقات PNG, JPG, JPEG
    - يعرض النقاط المميزة وخطوط التطابق
    - يحسب نسبة التطابق بين البصمتين
    """)
    
    st.markdown("### 📝 تعليمات الاستخدام")
    st.markdown("""
    1. اختر صورتين للبصمات
    2. اضغط على زر المقارنة
    3. انتظر حتى اكتمال المعالجة
    4. راجع النتائج والإحصائيات
    """)
    
    st.markdown("### 🎨 رموز الألوان")
    st.markdown("""
    - 🔴 النقاط الحمراء: نهايات الخطوط
    - 🟢 النقاط الخضراء: نقاط التفرع
    - 🔵 الخطوط الزرقاء: خطوط التطابق
    """) 