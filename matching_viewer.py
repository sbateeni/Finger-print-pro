import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features
from fingerprint.quality import calculate_quality

def create_matching_image(image1, image2, matches):
    """إنشاء صورة واحدة تجمع بين البصمتين مع خطوط التطابق"""
    # تحضير الصور
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # إنشاء صورة جديدة تجمع بين البصمتين
    max_h = max(h1, h2)
    total_w = w1 + w2
    matching_image = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # وضع البصمتين في الصورة الجديدة
    matching_image[:h1, :w1] = image1
    matching_image[:h2, w1:] = image2
    
    # رسم خطوط التطابق
    for match in matches:
        try:
            # إحداثيات النقاط المتطابقة
            pt1 = (int(match[0][0]), int(match[0][1]))
            pt2 = (int(match[1][0]) + w1, int(match[1][1]))
            
            # رسم دائرة حول النقطة في البصمة الأولى
            cv2.circle(matching_image, pt1, 8, (0, 255, 255), 2)
            
            # رسم دائرة حول النقطة في البصمة الثانية
            cv2.circle(matching_image, pt2, 8, (0, 255, 255), 2)
            
            # رسم خط التطابق
            cv2.line(matching_image, pt1, pt2, (0, 255, 255), 2)
        except:
            continue
    
    return matching_image

def main():
    st.set_page_config(
        page_title="عرض التطابقات بين البصمات",
        page_icon="👆",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .matching-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 2rem 0;
        }
        .matching-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .match-info {
            margin: 1rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">عرض التطابقات بين البصمات</h1>', unsafe_allow_html=True)
    
    # تحميل الصور
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### البصمة الأولى")
        fp1 = st.file_uploader("اختر البصمة الأولى", type=['png', 'jpg', 'jpeg'], key="fp1")
        
    with col2:
        st.markdown("### البصمة الثانية")
        fp2 = st.file_uploader("اختر البصمة الثانية", type=['png', 'jpg', 'jpeg'], key="fp2")
    
    if st.button("عرض التطابقات"):
        if fp1 and fp2:
            with st.spinner("جاري معالجة البصمات..."):
                # معالجة البصمات
                processed1, features1 = process_image(fp1)
                processed2, features2 = process_image(fp2)
                
                if processed1 is not None and processed2 is not None:
                    # الحصول على التطابقات
                    match_score, matches = match_features(features1, features2)
                    
                    # إنشاء صورة التطابقات
                    matching_image = create_matching_image(processed1, processed2, matches)
                    
                    # عرض النتائج
                    st.markdown('<div class="matching-container">', unsafe_allow_html=True)
                    
                    # عرض صورة التطابقات
                    st.markdown('<div class="match-info">', unsafe_allow_html=True)
                    st.markdown(f"### نسبة التطابق: {match_score:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.image(matching_image, caption="البصمتان مع خطوط التطابق", use_container_width=True)
                    
                    # عرض تفاصيل التطابقات
                    st.markdown("### تفاصيل التطابقات")
                    st.markdown(f"عدد النقاط المتطابقة: {len(matches)}")
                    
                    # عرض جدول التطابقات
                    matches_data = []
                    for i, match in enumerate(matches):
                        matches_data.append({
                            "رقم التطابق": i+1,
                            "إحداثيات البصمة الأولى": f"({int(match[0][0])}, {int(match[0][1])})",
                            "إحداثيات البصمة الثانية": f"({int(match[1][0])}, {int(match[1][1])})"
                        })
                    
                    st.table(matches_data)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("حدث خطأ أثناء معالجة البصمات")
        else:
            st.error("الرجاء اختيار كلا البصمتين")

if __name__ == "__main__":
    main() 