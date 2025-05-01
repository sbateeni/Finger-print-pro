import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features, detect_minutiae
from fingerprint.quality import calculate_quality

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
                    cv2.circle(img_with_minutiae, (cX, cY), 5, color, -1)
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
                    
                    # رسم النقاط المميزة مع خطوط التطابق
                    img1_with_minutiae = draw_minutiae_with_matches(processed1, features1, matches, processed2)
                    img2_with_minutiae = draw_minutiae_with_matches(processed2, features2, matches, processed1)
                    
                    # إنشاء صورة التطابقات
                    matching_image = create_matching_image(processed1, processed2, matches)
                    
                    # عرض النتائج
                    st.markdown('<div class="matching-container">', unsafe_allow_html=True)
                    
                    # عرض صورة التطابقات
                    st.markdown('<div class="match-info">', unsafe_allow_html=True)
                    st.markdown(f"### نسبة التطابق: {match_score:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
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
                    
                    # عرض الصور
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(img1_with_minutiae, caption="البصمة الأولى مع النقاط المميزة وخطوط التطابق", use_container_width=True)
                        
                    with col2:
                        st.image(img2_with_minutiae, caption="البصمة الثانية مع النقاط المميزة وخطوط التطابق", use_container_width=True)
                    
                    # عرض صورة التطابقات
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