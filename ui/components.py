import streamlit as st
from fingerprint.preprocessor import preprocess_fingerprint
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import compare_fingerprints
import cv2
import numpy as np
from PIL import Image
import io

def display_title():
    st.markdown('<h1 class="title-text">نظام مقارنة بصمات الأصابع</h1>', unsafe_allow_html=True)

def upload_fingerprints():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown('<h3>البصمة الأولى</h3>', unsafe_allow_html=True)
        fp1 = st.file_uploader("اختر البصمة الأولى", type=['jpg', 'jpeg', 'png'])
        if fp1:
            image = Image.open(fp1)
            st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown('<h3>البصمة الثانية</h3>', unsafe_allow_html=True)
        fp2 = st.file_uploader("اختر البصمة الثانية", type=['jpg', 'jpeg', 'png'])
        if fp2:
            image = Image.open(fp2)
            st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return fp1, fp2

def process_fingerprints(fp1, fp2):
    if fp1 and fp2:
        if st.button("مقارنة البصمات"):
            with st.spinner("جاري معالجة البصمات..."):
                # تحويل الملفات المرفوعة إلى مصفوفات NumPy
                img1 = Image.open(fp1)
                img2 = Image.open(fp2)
                img1_array = np.array(img1)
                img2_array = np.array(img2)
                
                # معالجة البصمات
                processed1 = preprocess_fingerprint(img1_array)
                processed2 = preprocess_fingerprint(img2_array)
                
                # استخراج المميزات
                features1 = extract_features(processed1)
                features2 = extract_features(processed2)
                
                # مقارنة البصمات
                match_score = compare_fingerprints(features1, features2)
                
                # عرض النتائج
                display_results(processed1, processed2, match_score)

def display_results(processed1, processed2, match_score):
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    # عرض الصور المعالجة
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3>البصمة الأولى المعالجة</h3>', unsafe_allow_html=True)
        st.image(processed1, use_container_width=True)
    
    with col2:
        st.markdown('<h3>البصمة الثانية المعالجة</h3>', unsafe_allow_html=True)
        st.image(processed2, use_container_width=True)
    
    # عرض نتيجة المقارنة
    st.markdown(f'<div class="match-score">نسبة التطابق: {match_score:.2f}%</div>', unsafe_allow_html=True)
    
    # تحديد النتيجة النهائية
    if match_score >= 80:
        st.markdown('<div class="match-result match">البصمتان متطابقتان</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="match-result no-match">البصمتان غير متطابقتين</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) 