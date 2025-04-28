import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from fingerprint.preprocessor import preprocess_image, enhance_image_quality, check_image_quality
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import match_fingerprints
from fingerprint.visualization import visualize_features, visualize_matching, plot_quality_metrics

# إعداد الصفحة
st.set_page_config(
    page_title="نظام مطابقة البصمات",
    page_icon="👆",
    layout="wide"
)

# تخصيص التصميم
st.markdown("""
    <style>
        .stApp {
            direction: rtl;
        }
        .upload-section {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .result-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.title("👆 نظام مطابقة البصمات")

# قسم تحميل البصمات
st.header("تحميل البصمات")
col1, col2 = st.columns(2)

with col1:
    st.subheader("البصمة الأولى")
    fp1_file = st.file_uploader("اختر البصمة الأولى", type=['png', 'jpg', 'jpeg'], key="fp1")
    if fp1_file:
        try:
            fp1_image = Image.open(fp1_file)
            st.image(fp1_image, caption="البصمة الأولى", use_container_width=True)
        except Exception as e:
            st.error(f"خطأ في تحميل الصورة: {str(e)}")

with col2:
    st.subheader("البصمة الثانية")
    fp2_file = st.file_uploader("اختر البصمة الثانية", type=['png', 'jpg', 'jpeg'], key="fp2")
    if fp2_file:
        try:
            fp2_image = Image.open(fp2_file)
            st.image(fp2_image, caption="البصمة الثانية", use_container_width=True)
        except Exception as e:
            st.error(f"خطأ في تحميل الصورة: {str(e)}")

# زر بدء المعالجة
if st.button("بدء المعالجة والمقارنة"):
    if fp1_file and fp2_file:
        try:
            # تحويل الصور إلى مصفوفات NumPy
            fp1_array = np.array(fp1_image)
            fp2_array = np.array(fp2_image)
            
            # إنشاء شريط تقدم
            progress_bar = st.progress(0)
            
            # معالجة البصمات
            with st.spinner("جاري معالجة البصمات..."):
                # تحسين جودة الصور
                fp1_enhanced = enhance_image_quality(fp1_array)
                fp2_enhanced = enhance_image_quality(fp2_array)
                progress_bar.progress(20)
                
                # فحص جودة الصور
                fp1_quality = check_image_quality(fp1_enhanced['final'])
                fp2_quality = check_image_quality(fp2_enhanced['final'])
                progress_bar.progress(40)
                
                # استخراج المميزات
                fp1_features = extract_features(fp1_enhanced['final'])
                fp2_features = extract_features(fp2_enhanced['final'])
                progress_bar.progress(60)
                
                # مقارنة البصمات
                match_result = match_fingerprints(fp1_features, fp2_features)
                progress_bar.progress(80)
                
                # تصور النتائج
                fp1_vis = visualize_features(fp1_enhanced['final'], fp1_features)
                fp2_vis = visualize_features(fp2_enhanced['final'], fp2_features)
                matching_vis = visualize_matching(fp1_enhanced['final'], fp2_enhanced['final'],
                                               fp1_features, fp2_features, match_result)
                progress_bar.progress(100)
            
            # عرض النتائج
            st.header("نتائج المعالجة")
            
            # عرض مقاييس الجودة
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("جودة البصمة الأولى")
                st.plotly_chart(plot_quality_metrics(fp1_quality), key="quality_chart1")
            with col2:
                st.subheader("جودة البصمة الثانية")
                st.plotly_chart(plot_quality_metrics(fp2_quality), key="quality_chart2")
            
            # عرض النقاط المميزة
            st.subheader("النقاط المميزة")
            col1, col2 = st.columns(2)
            with col1:
                st.image(fp1_vis, caption="النقاط المميزة - البصمة الأولى", use_container_width=True)
            with col2:
                st.image(fp2_vis, caption="النقاط المميزة - البصمة الثانية", use_container_width=True)
            
            # عرض نتيجة المطابقة
            st.subheader("نتيجة المطابقة")
            st.image(matching_vis, caption="تمثيل بصري للتطابق بين البصمتين", use_container_width=True)
            
            # عرض إحصائيات المطابقة
            st.markdown(f"""
            ### إحصائيات المطابقة
            - نسبة التطابق: {match_result['score']:.2%}
            - عدد النقاط المتطابقة: {match_result['match_count']}
            - النتيجة: {'تطابق' if match_result['is_match'] else 'عدم تطابق'}
            """)
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة البصمات: {str(e)}")
    else:
        st.error("الرجاء تحميل كلا البصمتين للمقارنة")

# معلومات النظام
st.header("معلومات النظام")
st.markdown("""
- يدعم النظام مقارنة بصمات الأصابع باستخدام خوارزميات متقدمة
- يمكن تحميل الصور بصيغ PNG, JPG, JPEG
- يتم معالجة الصور تلقائياً لتحسين جودة المقارنة
- يعرض النظام النقاط المميزة وخطوط التطابق
- يحسب نسبة التطابق بين البصمتين
""") 