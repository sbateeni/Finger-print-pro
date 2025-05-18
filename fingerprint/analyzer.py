import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

def analyze_fingerprint_details(image, features):
    """تحليل تفصيلي للبصمة"""
    details = {}
    
    try:
        # تحليل الترددات
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
        
        # تحويل القيم إلى نطاق [0, 1]
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
        details['frequency_analysis'] = magnitude_spectrum
        
        # تحليل الاتجاهات
        if 'minutiae' in features:
            directions = []
            for type_name, contours in features['minutiae'].items():
                for contour in contours:
                    try:
                        if len(contour) >= 5:
                            ellipse = cv2.fitEllipse(contour)
                            directions.append(ellipse[2])
                    except:
                        continue
            details['directions'] = directions
        
        # تحليل النقاط المميزة
        minutiae_stats = {}
        for type_name, contours in features.get('minutiae', {}).items():
            minutiae_stats[type_name] = {
                'count': len(contours),
                'areas': [cv2.contourArea(c) for c in contours],
                'perimeters': [cv2.arcLength(c, True) for c in contours]
            }
        details['minutiae_stats'] = minutiae_stats
        
        # تحليل جودة الصورة
        details['quality_metrics'] = {
            'contrast': np.std(image),
            'brightness': np.mean(image),
            'sharpness': cv2.Laplacian(image, cv2.CV_64F).var()
        }
        
        return details
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحليل البصمة: {str(e)}")
        return details

def show_advanced_analysis(stages):
    """عرض التحليل المتقدم للبصمة"""
    if 'processed' in stages and 'features' in stages:
        # تحليل البصمة
        details = analyze_fingerprint_details(stages['processed'], stages['features'])
        
        # عرض نتائج التحليل
        st.markdown("### التحليل المتقدم للبصمة")
        
        # عرض إحصائيات النقاط المميزة
        st.markdown("#### إحصائيات النقاط المميزة")
        minutiae_stats = details.get('minutiae_stats', {})
        for type_name, stats in minutiae_stats.items():
            st.markdown(f"**{type_name}:**")
            st.markdown(f"- العدد: {stats['count']}")
            if stats['areas']:
                st.markdown(f"- متوسط المساحة: {np.mean(stats['areas']):.2f}")
                st.markdown(f"- متوسط المحيط: {np.mean(stats['perimeters']):.2f}")
        
        # عرض مقاييس جودة الصورة
        st.markdown("#### مقاييس جودة الصورة")
        quality_metrics = details.get('quality_metrics', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("التباين", f"{quality_metrics.get('contrast', 0):.2f}")
        with col2:
            st.metric("السطوع", f"{quality_metrics.get('brightness', 0):.2f}")
        with col3:
            st.metric("الوضوح", f"{quality_metrics.get('sharpness', 0):.2f}")
        
        # عرض تحليل الترددات
        if 'frequency_analysis' in details:
            st.markdown("#### تحليل الترددات")
            try:
                # تحويل الصورة إلى تنسيق مناسب للعرض
                freq_image = (details['frequency_analysis'] * 255).astype(np.uint8)
                st.image(freq_image, use_container_width=True)
            except Exception as e:
                st.error(f"حدث خطأ في عرض تحليل الترددات: {str(e)}")
        
        # عرض تحليل الاتجاهات
        if 'directions' in details and details['directions']:
            st.markdown("#### تحليل الاتجاهات")
            try:
                fig = plt.figure(figsize=(10, 4))
                plt.hist(details['directions'], bins=36, range=(0, 360))
                plt.title("توزيع اتجاهات النقاط المميزة")
                plt.xlabel("الزاوية (درجة)")
                plt.ylabel("العدد")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"حدث خطأ في عرض تحليل الاتجاهات: {str(e)}") 