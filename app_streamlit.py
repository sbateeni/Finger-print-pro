import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from fingerprint.preprocessor import preprocess_image, enhance_image_quality, check_image_quality
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import match_fingerprints
from fingerprint.visualization import visualize_features, visualize_matching, plot_quality_metrics

# تكوين الصفحة
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
    </style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown('<h1 class="title-text">نظام مقارنة البصمات</h1>', unsafe_allow_html=True)

# تقسيم الصفحة إلى أعمدة
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### البصمة الأولى")
    fp1_file = st.file_uploader("اختر صورة البصمة الأولى", type=['png', 'jpg', 'jpeg'], key="fp1")
    if fp1_file:
        fp1_image = Image.open(fp1_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(fp1_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown("### البصمة الثانية")
    fp2_file = st.file_uploader("اختر صورة البصمة الثانية", type=['png', 'jpg', 'jpeg'], key="fp2")
    if fp2_file:
        fp2_image = Image.open(fp2_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(fp2_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# زر المقارنة
if st.button("مقارنة البصمات", key="compare"):
    if fp1_file and fp2_file:
        with st.spinner("جاري معالجة البصمات..."):
            # تحويل الصور إلى مصفوفات NumPy
            fp1_array = np.array(Image.open(fp1_file))
            fp2_array = np.array(Image.open(fp2_file))
            
            # معالجة البصمات
            fp1_processed = preprocess_image(fp1_array)
            fp2_processed = preprocess_image(fp2_array)
            
            # استخراج المميزات
            fp1_features = extract_features(fp1_processed)
            fp2_features = extract_features(fp2_processed)
            
            # مقارنة البصمات
            match_result = match_fingerprints(fp1_features, fp2_features)
            
            # تصور النتائج
            fp1_vis = visualize_features(fp1_processed, fp1_features)
            fp2_vis = visualize_features(fp2_processed, fp2_features)
            matching_vis = visualize_matching(fp1_processed, fp2_processed, fp1_features, fp2_features, match_result)
            
            # عرض النتائج
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### نتائج المقارنة")
            
            # عرض الصور المعالجة
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(fp1_vis, caption="البصمة الأولى بعد المعالجة", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(fp2_vis, caption="البصمة الثانية بعد المعالجة", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # عرض صورة المقارنة
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(matching_vis, caption="تمثيل بصري للتطابق بين البصمتين", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # عرض نتيجة المقارنة
            st.markdown(f'<div class="match-score">نسبة التطابق: {match_result["score"]:.2f}%</div>', unsafe_allow_html=True)
            
            # عرض الحكم النهائي
            if match_result["is_match"]:
                st.markdown('<div class="match-result match">البصمتان متطابقتان</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="match-result no-match">البصمتان غير متطابقتين</div>', unsafe_allow_html=True)
            
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