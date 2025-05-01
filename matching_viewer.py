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
    for i, match in enumerate(matches):
        try:
            # إحداثيات النقاط المتطابقة
            pt1 = (int(match[0][0]), int(match[0][1]))
            pt2 = (int(match[1][0]) + w1, int(match[1][1]))
            
            # رسم دائرة حول النقطة في البصمة الأولى
            cv2.circle(matching_image, pt1, 10, (0, 255, 255), 2)
            
            # رسم دائرة حول النقطة في البصمة الثانية
            cv2.circle(matching_image, pt2, 10, (0, 255, 255), 2)
            
            # رسم خط التطابق
            cv2.line(matching_image, pt1, pt2, (0, 255, 255), 2)
            
            # رسم رقم التطابق
            cv2.putText(matching_image, str(i+1), 
                       (pt1[0]-10, pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(matching_image, str(i+1), 
                       (pt2[0]-10, pt2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
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
        .fingerprint-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .fingerprint-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">عرض التطابقات بين البصمات</h1>', unsafe_allow_html=True)
    
    # تحميل الصور
    st.markdown("### تحميل البصمات")
    uploaded_files = st.file_uploader("اختر البصمات للمقارنة", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("عرض التطابقات"):
        if uploaded_files and len(uploaded_files) >= 2:
            with st.spinner("جاري معالجة البصمات..."):
                # معالجة البصمات
                processed_images = []
                features_list = []
                
                for file in uploaded_files:
                    processed, features = process_image(file)
                    if processed is not None and features is not None:
                        processed_images.append(processed)
                        features_list.append(features)
                
                if len(processed_images) >= 2:
                    # عرض النتائج
                    st.markdown('<div class="matching-container">', unsafe_allow_html=True)
                    
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
                    
                    # عرض البصمات مع النقاط المميزة
                    st.markdown("### البصمات مع النقاط المميزة")
                    st.markdown('<div class="fingerprint-grid">', unsafe_allow_html=True)
                    
                    for i, (img, features) in enumerate(zip(processed_images, features_list)):
                        with st.container():
                            st.markdown(f'<div class="fingerprint-item">', unsafe_allow_html=True)
                            st.markdown(f"#### البصمة {i+1}")
                            
                            # رسم النقاط المميزة
                            img_with_minutiae = draw_minutiae_with_matches(img, features)
                            st.image(img_with_minutiae, use_container_width=True)
                            
                            # عرض جودة البصمة
                            quality = calculate_quality(img)
                            st.markdown(f'<div class="quality-score">جودة البصمة: {quality:.2f}%</div>', unsafe_allow_html=True)
                            
                            # عرض إحصائيات النقاط المميزة
                            show_minutiae_details(features)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # عرض التطابقات بين كل زوج من البصمات
                    st.markdown("### التطابقات بين البصمات")
                    
                    for i in range(len(processed_images)):
                        for j in range(i+1, len(processed_images)):
                            st.markdown(f"#### مقارنة البصمة {i+1} مع البصمة {j+1}")
                            
                            # الحصول على التطابقات
                            match_score, matches = match_features(features_list[i], features_list[j])
                            
                            # إنشاء صورة التطابقات
                            matching_image = create_matching_image(processed_images[i], processed_images[j], matches)
                            
                            # عرض صورة التطابقات
                            st.markdown('<div class="match-info">', unsafe_allow_html=True)
                            st.markdown(f"نسبة التطابق: {match_score:.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.image(matching_image, use_container_width=True)
                            
                            # عرض تفاصيل التطابقات
                            st.markdown(f"عدد النقاط المتطابقة: {len(matches)}")
                            
                            # عرض جدول التطابقات
                            matches_data = []
                            for k, match in enumerate(matches):
                                matches_data.append({
                                    "رقم التطابق": k+1,
                                    f"إحداثيات البصمة {i+1}": f"({int(match[0][0])}, {int(match[0][1])})",
                                    f"إحداثيات البصمة {j+1}": f"({int(match[1][0])}, {int(match[1][1])})"
                                })
                            
                            st.table(matches_data)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("حدث خطأ أثناء معالجة البصمات")
        else:
            st.error("الرجاء اختيار بصمتين على الأقل للمقارنة")

if __name__ == "__main__":
    main() 