import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import time
from fingerprint.preprocessor import Preprocessor
from fingerprint.feature_extractor import FeatureExtractor
from fingerprint.matcher import FingerprintMatcher
from fingerprint.visualization import Visualizer

# إنشاء كائنات المعالجة
preprocessor = Preprocessor()
feature_extractor = FeatureExtractor()
matcher = FingerprintMatcher()
visualizer = Visualizer()

def check_image_quality(image):
    """فحص جودة الصورة"""
    quality_metrics = {
        'resolution': f"{image.shape[1]}x{image.shape[0]}",
        'clarity': cv2.Laplacian(image, cv2.CV_64F).var(),
        'brightness': np.mean(image),
        'contrast': np.std(image),
        'noise_ratio': cv2.meanStdDev(image)[1][0][0] / cv2.meanStdDev(image)[0][0][0]
    }
    return quality_metrics

def enhance_image(image):
    """تحسين جودة الصورة"""
    # تحويل إلى تدرج الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف
    edges = cv2.Canny(denoised, 100, 200)
    
    # تصحيح الإضاءة
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return {
        'original': image,
        'gray': gray,
        'enhanced': enhanced,
        'denoised': denoised,
        'edges': edges,
        'final': final
    }

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
    .quality-metric {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        background-color: #e9ecef;
    }
    .preprocessing-step {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        
        # فحص جودة الصور
        status_text.text("جاري فحص جودة الصور...")
        progress_bar.progress(10)
        
        quality1 = check_image_quality(img1)
        quality2 = check_image_quality(img2)
        
        # عرض نتائج فحص الجودة
        st.markdown("### 📊 جودة الصور")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### البصمة الأولى")
            for metric, value in quality1.items():
                if metric == 'resolution':
                    st.markdown(f"""
                    <div class="quality-metric">
                        <strong>{metric}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="quality-metric">
                        <strong>{metric}:</strong> {float(value):.2f}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### البصمة الثانية")
            for metric, value in quality2.items():
                if metric == 'resolution':
                    st.markdown(f"""
                    <div class="quality-metric">
                        <strong>{metric}:</strong> {value}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="quality-metric">
                        <strong>{metric}:</strong> {float(value):.2f}
                    </div>
                    """, unsafe_allow_html=True)
        
        # تحسين الصور
        status_text.text("جاري تحسين الصور...")
        progress_bar.progress(20)
        
        enhanced1 = enhance_image(img1)
        enhanced2 = enhance_image(img2)
        
        # عرض خطوات المعالجة المسبقة
        st.markdown("### 🔄 خطوات المعالجة المسبقة")
        
        # البصمة الأولى
        st.markdown("#### البصمة الأولى")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### الصورة الأصلية")
            st.image(enhanced1['original'], use_container_width=True)
            st.markdown("##### تحسين التباين")
            st.image(enhanced1['enhanced'], use_container_width=True)
        
        with col2:
            st.markdown("##### تدرج الرمادي")
            st.image(enhanced1['gray'], use_container_width=True)
            st.markdown("##### إزالة الضوضاء")
            st.image(enhanced1['denoised'], use_container_width=True)
        
        with col3:
            st.markdown("##### الحواف")
            st.image(enhanced1['edges'], use_container_width=True)
            st.markdown("##### النتيجة النهائية")
            st.image(enhanced1['final'], use_container_width=True)
        
        # البصمة الثانية
        st.markdown("#### البصمة الثانية")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### الصورة الأصلية")
            st.image(enhanced2['original'], use_container_width=True)
            st.markdown("##### تحسين التباين")
            st.image(enhanced2['enhanced'], use_container_width=True)
        
        with col2:
            st.markdown("##### تدرج الرمادي")
            st.image(enhanced2['gray'], use_container_width=True)
            st.markdown("##### إزالة الضوضاء")
            st.image(enhanced2['denoised'], use_container_width=True)
        
        with col3:
            st.markdown("##### الحواف")
            st.image(enhanced2['edges'], use_container_width=True)
            st.markdown("##### النتيجة النهائية")
            st.image(enhanced2['final'], use_container_width=True)
        
        # حفظ الصور المؤقتة
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        img1_path = os.path.join(temp_dir, "fp1.jpg")
        img2_path = os.path.join(temp_dir, "fp2.jpg")
        
        cv2.imwrite(img1_path, enhanced1['final'])
        cv2.imwrite(img2_path, enhanced2['final'])
        
        # معالجة الصور
        status_text.text("جاري معالجة الصور...")
        progress_bar.progress(30)
        
        # تحويل الصور إلى numpy arrays
        processed_fp1 = preprocessor.preprocess_image(img1_path)
        processed_fp2 = preprocessor.preprocess_image(img2_path)
        
        # التأكد من أن الصور هي numpy arrays
        if not isinstance(processed_fp1, np.ndarray):
            processed_fp1 = np.array(processed_fp1)
        if not isinstance(processed_fp2, np.ndarray):
            processed_fp2 = np.array(processed_fp2)
        
        # استخراج المميزات
        status_text.text("جاري استخراج المميزات...")
        progress_bar.progress(40)
        
        features1 = feature_extractor.extract_features(processed_fp1)
        features2 = feature_extractor.extract_features(processed_fp2)
        
        # رسم النقاط المميزة
        marked_fp1 = visualizer.visualize_features(processed_fp1, features1)
        marked_fp2 = visualizer.visualize_features(processed_fp2, features2)
        
        # مقارنة البصمات
        status_text.text("جاري مقارنة البصمات...")
        progress_bar.progress(60)
        
        match_result = matcher.match_features(features1, features2)
        
        # رسم خطوط التطابق
        status_text.text("جاري إنشاء التصور...")
        progress_bar.progress(80)
        
        matching_visualization = visualizer.visualize_matching(marked_fp1, marked_fp2, features1, features2, match_result)
        
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
            st.metric("نسبة التطابق", f"{float(match_result['score']):.2%}")
        
        with col2:
            st.metric("عدد النقاط المميزة في البصمة الأولى", features1['count'])
        
        with col3:
            st.metric("عدد النقاط المميزة في البصمة الثانية", features2['count'])
        
        # عرض النقاط المتطابقة
        st.markdown("### 📍 النقاط المتطابقة")
        for i, match in enumerate(match_result['matches']):
            point1 = features1['minutiae'][match[0]]
            point2 = features2['minutiae'][match[1]]
            st.write(f"نقطة تطابق {i+1}: ({point1[0]}, {point1[1]}) ↔ ({point2[0]}, {point2[1]})")
        
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