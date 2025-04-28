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

# تعيين الحد الأقصى لحجم الصورة (بالبايت)
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB

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
        keypoints, descriptors = extract_features(processed)
        
        # تنظيف الذاكرة
        gc.collect()
            
        return processed, {'keypoints': keypoints, 'descriptors': descriptors}
        
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
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(processed1, caption="البصمة الأولى بعد المعالجة", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        quality1 = calculate_quality(processed1)
                        st.markdown(f'<div class="quality-score">جودة البصمة الأولى: {quality1:.2f}%</div>', unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(processed2, caption="البصمة الثانية بعد المعالجة", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        quality2 = calculate_quality(processed2)
                        st.markdown(f'<div class="quality-score">جودة البصمة الثانية: {quality2:.2f}%</div>', unsafe_allow_html=True)
                    
                    # مقارنة البصمات
                    if features1 and features2:
                        match_score = match_features(features1, features2)
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