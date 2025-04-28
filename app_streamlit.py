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
    
    st.title("نظام مقارنة البصمات")
    
    # تحميل الصور
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("البصمة الأولى")
        fp1 = st.file_uploader("اختر البصمة الأولى", type=['png', 'jpg', 'jpeg'])
        
    with col2:
        st.subheader("البصمة الثانية")
        fp2 = st.file_uploader("اختر البصمة الثانية", type=['png', 'jpg', 'jpeg'])
    
    if fp1 and fp2:
        # معالجة البصمات
        with st.spinner("جاري معالجة البصمات..."):
            processed1, features1 = process_image(fp1)
            processed2, features2 = process_image(fp2)
            
            if processed1 is not None and processed2 is not None:
                # عرض النتائج
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(processed1, caption="البصمة الأولى بعد المعالجة")
                    quality1 = calculate_quality(processed1)
                    st.write(f"جودة البصمة الأولى: {quality1:.2f}%")
                    
                with col2:
                    st.image(processed2, caption="البصمة الثانية بعد المعالجة")
                    quality2 = calculate_quality(processed2)
                    st.write(f"جودة البصمة الثانية: {quality2:.2f}%")
                
                # مقارنة البصمات
                if features1 and features2:
                    match_score = match_features(features1, features2)
                    st.write(f"نسبة التطابق: {match_score:.2f}%")
                    
                    if match_score > 80:
                        st.success("البصمتان متطابقتان")
                    else:
                        st.error("البصمتان غير متطابقتين")
                else:
                    st.warning("لم يتم العثور على مميزات كافية في إحدى البصمتين")

if __name__ == "__main__":
    main() 