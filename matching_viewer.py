import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import os
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

def save_features_to_json(features, filename):
    """حفظ السمات بتنسيق JSON"""
    # تحويل الكنتورات إلى قوائم
    minutiae_data = {}
    for type_name, contours in features['minutiae'].items():
        minutiae_data[type_name] = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                minutiae_data[type_name].append({
                    'x': cX,
                    'y': cY,
                    'type': type_name
                })
    
    # حفظ البيانات
    with open(filename, 'w') as f:
        json.dump(minutiae_data, f)

def process_image_stages(image_file):
    """معالجة الصورة عبر جميع المراحل"""
    stages = {}
    
    # 🖼️ المرحلة 1: معالجة الصورة
    with st.spinner("جاري معالجة الصورة..."):
        processed = preprocess_image(image_file)
        if processed is not None:
            stages['processed'] = processed
    
    # 📍 المرحلة 2: استخراج السمات
    if 'processed' in stages:
        with st.spinner("جاري استخراج السمات..."):
            features = extract_features(stages['processed'])
            if features is not None:
                stages['features'] = features
    
    # 📁 المرحلة 3: حفظ السمات
    if 'features' in stages:
        with st.spinner("جاري حفظ السمات..."):
            filename = f"features_{hash(str(image_file))}.json"
            save_features_to_json(stages['features'], filename)
            stages['saved_features'] = filename
    
    return stages

def show_minutiae_details(features):
    """عرض تفاصيل النقاط المميزة"""
    if features is None or 'minutiae' not in features:
        return
        
    # إنشاء جدول للإحصائيات
    stats = {
        'نهاية نتوء': len(features['minutiae'].get('ridge_endings', [])),
        'تفرع': len(features['minutiae'].get('bifurcations', [])),
        'جزيرة': len(features['minutiae'].get('islands', [])),
        'نقطة': len(features['minutiae'].get('dots', [])),
        'نواة': len(features['minutiae'].get('cores', [])),
        'دلتا': len(features['minutiae'].get('deltas', []))
    }
    
    # عرض الإحصائيات
    st.markdown("#### إحصائيات النقاط المميزة")
    for type_name, count in stats.items():
        st.markdown(f"- {type_name}: {count}")

def main():
    st.set_page_config(
        page_title="نظام مقارنة البصمات",
        page_icon="👆",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stage-container {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        .stage-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .stage-icon {
            font-size: 1.5rem;
        }
        .stage-content {
            margin-left: 2rem;
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
        .match-info {
            margin: 1rem 0;
            padding: 1rem;
            background-color: #e9ecef;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="text-align: center;">نظام مقارنة البصمات</h1>', unsafe_allow_html=True)
    
    # تحميل الصور
    st.markdown("### تحميل البصمات")
    uploaded_files = st.file_uploader("اختر البصمات للمقارنة", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if st.button("بدء المعالجة"):
        if uploaded_files and len(uploaded_files) >= 2:
            # معالجة كل بصمة
            processed_stages = []
            for file in uploaded_files:
                stages = process_image_stages(file)
                processed_stages.append(stages)
            
            # عرض نتائج كل مرحلة
            for i, stages in enumerate(processed_stages):
                st.markdown(f'<div class="stage-container">', unsafe_allow_html=True)
                st.markdown(f'<h3>البصمة {i+1}</h3>', unsafe_allow_html=True)
                
                # 🖼️ عرض مرحلة معالجة الصورة
                if 'processed' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">🖼️</span> معالجة الصورة</div>', unsafe_allow_html=True)
                    st.image(stages['processed'], use_container_width=True)
                
                # 📍 عرض مرحلة استخراج السمات
                if 'features' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">📍</span> استخراج السمات</div>', unsafe_allow_html=True)
                    img_with_minutiae = draw_minutiae_with_matches(stages['processed'], stages['features'])
                    st.image(img_with_minutiae, use_container_width=True)
                    
                    # عرض إحصائيات السمات
                    show_minutiae_details(stages['features'])
                
                # 📁 عرض مرحلة حفظ السمات
                if 'saved_features' in stages:
                    st.markdown('<div class="stage-title"><span class="stage-icon">📁</span> حفظ السمات</div>', unsafe_allow_html=True)
                    st.markdown(f"تم حفظ السمات في الملف: `{stages['saved_features']}`")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 🔍 عرض مرحلة مطابقة البصمات
            st.markdown('<div class="stage-container">', unsafe_allow_html=True)
            st.markdown('<div class="stage-title"><span class="stage-icon">🔍</span> مطابقة البصمات</div>', unsafe_allow_html=True)
            
            for i in range(len(processed_stages)):
                for j in range(i+1, len(processed_stages)):
                    if 'features' in processed_stages[i] and 'features' in processed_stages[j]:
                        st.markdown(f"#### مقارنة البصمة {i+1} مع البصمة {j+1}")
                        
                        # حساب التطابق
                        match_score, matches = match_features(
                            processed_stages[i]['features'],
                            processed_stages[j]['features']
                        )
                        
                        # إنشاء صورة التطابق
                        matching_image = create_matching_image(
                            processed_stages[i]['processed'],
                            processed_stages[j]['processed'],
                            matches
                        )
                        
                        # عرض النتائج
                        st.markdown('<div class="match-info">', unsafe_allow_html=True)
                        st.markdown(f"نسبة التطابق: {match_score:.2f}%")
                        st.markdown(f"عدد النقاط المتطابقة: {len(matches)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.image(matching_image, use_container_width=True)
                        
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
            
            # 📊 عرض مرحلة التحليل
            st.markdown('<div class="stage-container">', unsafe_allow_html=True)
            st.markdown('<div class="stage-title"><span class="stage-icon">📊</span> التحليل والإحصائيات</div>', unsafe_allow_html=True)
            
            # عرض إحصائيات عامة
            total_minutiae = sum(len(stages.get('features', {}).get('minutiae', {}).get(type_name, [])) 
                               for stages in processed_stages 
                               for type_name in ['ridge_endings', 'bifurcations', 'islands', 'dots', 'cores', 'deltas'])
            
            st.markdown(f"إجمالي عدد النقاط المميزة: {total_minutiae}")
            
            # عرض جودة كل بصمة
            for i, stages in enumerate(processed_stages):
                if 'processed' in stages:
                    quality = calculate_quality(stages['processed'])
                    st.markdown(f"جودة البصمة {i+1}: {quality:.2f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("الرجاء اختيار بصمتين على الأقل للمقارنة")

if __name__ == "__main__":
    main() 