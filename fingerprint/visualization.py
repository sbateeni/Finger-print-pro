import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import Any
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# تعريف الألوان المستخدمة
COLORS = {
    'endpoint': (0, 255, 0),  # أخضر
    'bifurcation': (0, 0, 255),  # أزرق
    'matching': (255, 0, 0),  # أحمر
    'background': (255, 255, 255)  # أبيض
}

def draw_minutiae_points(image: np.ndarray, features: List[Dict]) -> np.ndarray:
    """
    رسم النقاط المميزة على الصورة
    """
    try:
        # نسخ الصورة
        marked_image = image.copy()

        # رسم كل نقطة مميزة
        for feature in features:
            x, y = int(feature['x']), int(feature['y'])
            feature_type = feature['type']
            
            # تحديد لون النقطة حسب نوعها
            if feature_type == 'endpoint':
                color = COLORS['endpoint']
                radius = 3
            else:  # bifurcation
                color = COLORS['bifurcation']
                radius = 4
            
            # رسم النقطة
            cv2.circle(marked_image, (x, y), radius, color, -1)
            
            # رسم اتجاه النقطة
            angle = feature.get('angle', 0)
            length = 10
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            cv2.line(marked_image, (x, y), (end_x, end_y), color, 1)

        return marked_image

    except Exception as e:
        logger.error(f'خطأ في رسم النقاط المميزة: {str(e)}')
        raise

def draw_matching_lines(image1: np.ndarray, image2: np.ndarray,
                      matching_points: List[Tuple[Dict, Dict]]) -> np.ndarray:
    """
    رسم خطوط التطابق بين الصورتين
    """
    try:
        # دمج الصورتين
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        # تحويل الصور إلى RGB إذا كانت رمادية
        if len(image1.shape) == 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        if len(image2.shape) == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        # وضع الصور في الصورة المجمعة
        combined[:h1, :w1] = image1
        combined[:h2, w1:w1+w2] = image2

        # رسم خطوط التطابق
        for point1, point2 in matching_points:
            x1, y1 = int(point1['x']), int(point1['y'])
            x2, y2 = int(point2['x']), int(point2['y'])
            
            # رسم خط التطابق
            cv2.line(combined, (x1, y1), (x2 + w1, y2), (0, 255, 0), 2)
            
            # رسم نقاط النهاية
            cv2.circle(combined, (x1, y1), 3, (0, 255, 0), -1)
            cv2.circle(combined, (x2 + w1, y2), 3, (0, 255, 0), -1)

        return combined

    except Exception as e:
        logger.error(f'خطأ في رسم خطوط التطابق: {str(e)}')
        raise

class FingerprintVisualizer:
    def __init__(self):
        self.marker_size = 5
        self.line_thickness = 2
        self.heatmap_cmap = 'viridis'
        self.output_dir = 'visualizations'
        self._setup_output_dir()

    def _setup_output_dir(self):
        """
        إعداد مجلد المخرجات
        """
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        except Exception as e:
            logger.error(f'خطأ في إعداد مجلد المخرجات: {str(e)}')
            raise

    def create_heatmap(self, image: np.ndarray, features: List[Dict]) -> np.ndarray:
        """
        إنشاء خريطة حرارية لكثافة النقاط المميزة
        """
        try:
            # إنشاء مصفوفة فارغة
            heatmap = np.zeros(image.shape[:2], dtype=np.float32)

            # إضافة النقاط المميزة
            for feature in features:
                x, y = int(feature['x']), int(feature['y'])
                heatmap[y, x] += 1

            # تطبيق Gaussian blur
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

            # تطبيع القيم
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            # تطبيق خريطة الألوان
            cmap = plt.get_cmap(self.heatmap_cmap)
            colored_heatmap = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

            # دمج مع الصورة الأصلية
            alpha = 0.7
            blended = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)

            return blended

        except Exception as e:
            logger.error(f'خطأ في إنشاء الخريطة الحرارية: {str(e)}')
            raise

    def plot_feature_distribution(self, features: List[Dict]) -> plt.Figure:
        """
        رسم توزيع المميزات
        """
        try:
            # تجميع البيانات
            types = [f['type'] for f in features]
            angles = [f['angle'] for f in features]
            qualities = [f.get('quality', 0) for f in features]

            # إنشاء الرسم البياني
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # رسم توزيع الأنواع
            sns.countplot(x=types, ax=ax1)
            ax1.set_title('توزيع أنواع النقاط المميزة')

            # رسم توزيع الزوايا
            sns.histplot(angles, bins=36, ax=ax2)
            ax2.set_title('توزيع زوايا النقاط المميزة')

            # رسم توزيع الجودة
            sns.histplot(qualities, bins=20, ax=ax3)
            ax3.set_title('توزيع جودة النقاط المميزة')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f'خطأ في رسم توزيع المميزات: {str(e)}')
            raise

    def create_3d_visualization(self, features: List[Dict]) -> plt.Figure:
        """
        إنشاء تصور ثلاثي الأبعاد للمميزات
        """
        try:
            # تجميع البيانات
            x = [f['x'] for f in features]
            y = [f['y'] for f in features]
            z = [f.get('quality', 0) for f in features]
            types = [f['type'] for f in features]
            colors = [COLORS[t] for t in types]

            # إنشاء الرسم البياني
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # رسم النقاط
            scatter = ax.scatter(x, y, z, c=colors, s=50)

            # إضافة تسميات
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('الجودة')
            ax.set_title('تصور ثلاثي الأبعاد للنقاط المميزة')

            return fig

        except Exception as e:
            logger.error(f'خطأ في إنشاء التصور ثلاثي الأبعاد: {str(e)}')
            raise

    def save_visualization(self, image: np.ndarray, filename: str) -> str:
        """
        حفظ التصور
        """
        try:
            # إنشاء اسم ملف فريد
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{filename}_{timestamp}.png"
            output_path = os.path.join(self.output_dir, output_filename)

            # حفظ الصورة
            cv2.imwrite(output_path, image)

            return output_path

        except Exception as e:
            logger.error(f'خطأ في حفظ التصور: {str(e)}')
            raise

    def create_comparison_report(self, image1: np.ndarray, image2: np.ndarray,
                               features1: List[Dict], features2: List[Dict],
                               matching_points: List[Tuple[Dict, Dict]],
                               match_score: float) -> Dict:
        """
        إنشاء تقرير مقارنة
        """
        try:
            # إنشاء التصورات
            marked_image1 = draw_minutiae_points(image1, features1)
            marked_image2 = draw_minutiae_points(image2, features2)
            matching_image = draw_matching_lines(image1, image2, matching_points)
            heatmap1 = self.create_heatmap(image1, features1)
            heatmap2 = self.create_heatmap(image2, features2)
            distribution_fig = self.plot_feature_distribution(features1 + features2)
            visualization_3d = self.create_3d_visualization(features1 + features2)

            # حفظ التصورات
            marked1_path = self.save_visualization(marked_image1, 'marked1')
            marked2_path = self.save_visualization(marked_image2, 'marked2')
            matching_path = self.save_visualization(matching_image, 'matching')
            heatmap1_path = self.save_visualization(heatmap1, 'heatmap1')
            heatmap2_path = self.save_visualization(heatmap2, 'heatmap2')
            distribution_path = self.save_visualization(
                np.array(distribution_fig.canvas.renderer.buffer_rgba()),
                'distribution'
            )
            visualization_3d_path = self.save_visualization(
                np.array(visualization_3d.canvas.renderer.buffer_rgba()),
                '3d_visualization'
            )

            # إنشاء التقرير
            report = {
                'timestamp': datetime.now().isoformat(),
                'match_score': match_score,
                'num_features1': len(features1),
                'num_features2': len(features2),
                'num_matching_points': len(matching_points),
                'visualizations': {
                    'marked1': marked1_path,
                    'marked2': marked2_path,
                    'matching': matching_path,
                    'heatmap1': heatmap1_path,
                    'heatmap2': heatmap2_path,
                    'distribution': distribution_path,
                    '3d_visualization': visualization_3d_path
                },
                'statistics': {
                    'feature_types1': {
                        'endpoint': len([f for f in features1 if f['type'] == 'endpoint']),
                        'bifurcation': len([f for f in features1 if f['type'] == 'bifurcation'])
                    },
                    'feature_types2': {
                        'endpoint': len([f for f in features2 if f['type'] == 'endpoint']),
                        'bifurcation': len([f for f in features2 if f['type'] == 'bifurcation'])
                    },
                    'average_quality1': np.mean([f.get('quality', 0) for f in features1]),
                    'average_quality2': np.mean([f.get('quality', 0) for f in features2])
                }
            }

            # حفظ التقرير
            report_path = os.path.join(self.output_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

            return report

        except Exception as e:
            logger.error(f'خطأ في إنشاء تقرير المقارنة: {str(e)}')
            raise

    def draw_minutiae_points(self, image, minutiae_points):
        """رسم النقاط المميزة على الصورة"""
        try:
            # نسخ الصورة لتجنب تعديل الأصل
            marked_image = image.copy()
            
            # تحويل الصورة إلى RGB إذا كانت رمادية
            if len(marked_image.shape) == 2:
                marked_image = cv2.cvtColor(marked_image, cv2.COLOR_GRAY2BGR)
            
            # رسم كل نقطة مميزة
            for point in minutiae_points:
                x = int(point['x'])
                y = int(point['y'])
                angle = point.get('angle', 0)
                quality = point.get('quality', 1.0)
                feature_type = point.get('type', 'endpoint')
                
                # تحديد لون النقطة بناءً على نوعها
                if feature_type == 'endpoint':
                    color = (0, 255, 0)  # أخضر للنقاط النهائية
                else:
                    color = (255, 0, 0)  # أحمر لنقاط التفرع
                
                # رسم دائرة حول النقطة
                cv2.circle(marked_image, (x, y), 3, color, -1)
                
                # رسم خط يشير إلى الاتجاه
                if angle is not None:
                    end_x = int(x + 15 * np.cos(np.radians(angle)))
                    end_y = int(y + 15 * np.sin(np.radians(angle)))
                    cv2.line(marked_image, (x, y), (end_x, end_y), color, 1)
            
            return marked_image
        except Exception as e:
            logger.error(f'خطأ في رسم النقاط المميزة: {str(e)}')
            return image

    def draw_matching_lines(self, image1, image2, matching_points):
        """رسم خطوط التطابق بين الصورتين"""
        try:
            # دمج الصورتين
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            
            # تحويل الصور إلى RGB إذا كانت رمادية
            if len(image1.shape) == 2:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            if len(image2.shape) == 2:
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            
            # وضع الصور في الصورة المجمعة
            combined[:h1, :w1] = image1
            combined[:h2, w1:w1+w2] = image2

            # رسم خطوط التطابق
            for point1, point2 in matching_points:
                x1, y1 = int(point1['x']), int(point1['y'])
                x2, y2 = int(point2['x']), int(point2['y'])
                
                # رسم خط التطابق
                cv2.line(combined, (x1, y1), (x2 + w1, y2), (0, 255, 0), 2)
                
                # رسم نقاط النهاية
                cv2.circle(combined, (x1, y1), 3, (0, 255, 0), -1)
                cv2.circle(combined, (x2 + w1, y2), 3, (0, 255, 0), -1)

            return combined
        except Exception as e:
            logger.error(f'خطأ في رسم خطوط التطابق: {str(e)}')
            return image1

class Visualizer:
    """فئة لعرض نتائج معالجة البصمات"""
    
    def __init__(self):
        """تهيئة الفئة"""
        self.colors = {
            'endpoint': (0, 255, 0),  # أخضر
            'branch': (0, 0, 255),    # أزرق
            'match': (255, 0, 0),     # أحمر
            'text': (255, 255, 255)   # أبيض
        }
        
    def visualize_features(self, image: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """عرض المميزات المستخرجة على الصورة"""
        try:
            # نسخ الصورة
            vis_image = image.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
                
            # رسم النقاط المميزة
            for point in features['minutiae']:
                x, y, type_, angle = point
                color = self.colors[type_]
                
                # رسم النقطة
                cv2.circle(vis_image, (x, y), 3, color, -1)
                
                # رسم اتجاه النقطة
                end_x = int(x + 10 * np.cos(np.radians(angle)))
                end_y = int(y + 10 * np.sin(np.radians(angle)))
                cv2.line(vis_image, (x, y), (end_x, end_y), color, 1)
                
            # إضافة معلومات
            info = f"عدد النقاط: {features['count']}"
            cv2.putText(vis_image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, self.colors['text'], 2)
            
            return vis_image
            
        except Exception as e:
            logger.error(f'خطأ في عرض المميزات: {str(e)}')
            raise
            
    def visualize_matching(self, image1: np.ndarray, image2: np.ndarray, 
                          features1: Dict[str, Any], features2: Dict[str, Any],
                          matches: Dict[str, Any]) -> np.ndarray:
        """عرض نتائج المطابقة"""
        try:
            # إنشاء صورة مركبة
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            vis_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            
            # نسخ الصور
            if len(image1.shape) == 2:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            if len(image2.shape) == 2:
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            vis_image[:h1, :w1] = image1
            vis_image[:h2, w1:w1+w2] = image2
            
            # رسم خطوط المطابقة
            for i, j in matches['matches']:
                pt1 = features1['minutiae'][i][:2]
                pt2 = (features2['minutiae'][j][0] + w1, features2['minutiae'][j][1])
                cv2.line(vis_image, pt1, pt2, self.colors['match'], 1)
                
            # إضافة معلومات
            info = f"درجة التطابق: {matches['score']:.2f}%"
            cv2.putText(vis_image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, self.colors['text'], 2)
            
            return vis_image
            
        except Exception as e:
            logger.error(f'خطأ في عرض نتائج المطابقة: {str(e)}')
            raise 