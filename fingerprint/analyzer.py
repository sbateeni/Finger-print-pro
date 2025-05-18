import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from .quality import assess_image_quality

def analyze_fingerprint_details(image, features):
    """
    تحليل تفصيلي للبصمة
    
    Args:
        image: صورة OpenCV
        features: سمات البصمة
        
    Returns:
        dict: قاموس يحتوي على نتائج التحليل
    """
    # تحليل التردد
    frequency_analysis = analyze_frequency(image)
    
    # تحليل الاتجاهات
    direction_analysis = analyze_directions(features)
    
    # إحصائيات النقاط المميزة
    minutiae_stats = calculate_minutiae_statistics(features)
    
    # تقييم جودة الصورة
    quality_metrics = assess_image_quality(image)
    
    return {
        'frequency': frequency_analysis,
        'directions': direction_analysis,
        'minutiae_stats': minutiae_stats,
        'quality': quality_metrics
    }

def analyze_frequency(image):
    """
    تحليل التردد في البصمة
    
    Args:
        image: صورة OpenCV
        
    Returns:
        dict: قاموس يحتوي على نتائج تحليل التردد
    """
    # تطبيق تحويل فورييه
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # حساب التوزيع الترددي
    freq_distribution = np.histogram(magnitude_spectrum.flatten(), bins=50)[0]
    
    return {
        'magnitude_spectrum': magnitude_spectrum,
        'distribution': freq_distribution
    }

def analyze_directions(features):
    """
    تحليل اتجاهات النقاط المميزة
    
    Args:
        features: سمات البصمة
        
    Returns:
        dict: قاموس يحتوي على نتائج تحليل الاتجاهات
    """
    directions = []
    
    # حساب اتجاهات النقاط المميزة
    for type_name, contours in features['minutiae'].items():
        for contour in contours:
            if len(contour) >= 5:
                # حساب الإهليلج المناسب
                (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                directions.append(angle)
    
    # حساب التوزيع الاتجاهي
    if directions:
        direction_hist = np.histogram(directions, bins=36, range=(0, 360))[0]
    else:
        direction_hist = np.zeros(36)
    
    return {
        'directions': directions,
        'histogram': direction_hist
    }

def calculate_minutiae_statistics(features):
    """
    حساب إحصائيات النقاط المميزة
    
    Args:
        features: سمات البصمة
        
    Returns:
        dict: قاموس يحتوي على الإحصائيات
    """
    stats = {
        'total': 0,
        'by_type': {},
        'areas': [],
        'perimeters': []
    }
    
    # حساب الإحصائيات لكل نوع من النقاط المميزة
    for type_name, contours in features['minutiae'].items():
        stats['by_type'][type_name] = len(contours)
        stats['total'] += len(contours)
        
        # حساب المساحات والمحيطات
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            stats['areas'].append(area)
            stats['perimeters'].append(perimeter)
    
    # حساب المتوسطات
    if stats['areas']:
        stats['avg_area'] = np.mean(stats['areas'])
        stats['avg_perimeter'] = np.mean(stats['perimeters'])
    else:
        stats['avg_area'] = 0
        stats['avg_perimeter'] = 0
    
    return stats

def show_advanced_analysis(stages):
    """
    عرض نتائج التحليل المتقدم
    
    Args:
        stages: مراحل معالجة البصمة
    """
    if 'features' not in stages:
        return
    
    # عرض إحصائيات النقاط المميزة
    st.markdown("#### إحصائيات النقاط المميزة")
    minutiae_stats = calculate_minutiae_statistics(stages['features'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("إجمالي النقاط المميزة:", minutiae_stats['total'])
        st.write("متوسط المساحة:", f"{minutiae_stats['avg_area']:.2f}")
        st.write("متوسط المحيط:", f"{minutiae_stats['avg_perimeter']:.2f}")
    
    with col2:
        st.write("التوزيع حسب النوع:")
        for type_name, count in minutiae_stats['by_type'].items():
            st.write(f"- {type_name}: {count}")
    
    # عرض تحليل التردد
    if 'processed' in stages:
        st.markdown("#### تحليل التردد")
        frequency_analysis = analyze_frequency(stages['processed'])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frequency_analysis['distribution'])
        ax.set_title("توزيع التردد")
        st.pyplot(fig)
    
    # عرض تحليل الاتجاهات
    st.markdown("#### تحليل الاتجاهات")
    direction_analysis = analyze_directions(stages['features'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(36), direction_analysis['histogram'])
    ax.set_title("توزيع الاتجاهات")
    ax.set_xlabel("الاتجاه (درجة)")
    ax.set_ylabel("العدد")
    st.pyplot(fig)
    
    # عرض مقاييس الجودة
    if 'processed' in stages:
        st.markdown("#### مقاييس الجودة")
        quality_metrics = assess_image_quality(stages['processed'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("الجودة العامة", f"{quality_metrics['overall_quality']:.1f}%")
        with col2:
            st.metric("التباين", f"{quality_metrics['contrast']:.2f}")
        with col3:
            st.metric("الوضوح", f"{quality_metrics['sharpness']:.2f}")
        
        st.write("مستوى الجودة:", quality_metrics['quality_level']) 