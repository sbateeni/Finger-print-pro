import cv2
import numpy as np
from fingerprint.feature_extractor import FeatureExtractor
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extraction(image_path):
    try:
        # قراءة الصورة
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"لا يمكن قراءة الصورة من المسار: {image_path}")

        # إنشاء كائن FeatureExtractor
        extractor = FeatureExtractor()

        # استخراج المميزات
        features = extractor.extract_features(image)

        # طباعة عدد المميزات المستخرجة
        logger.info(f"تم استخراج {len(features)} نقطة مميزة")

        # طباعة تفاصيل كل نقطة مميزة
        for i, feature in enumerate(features):
            logger.info(f"النقطة {i+1}:")
            logger.info(f"  الموقع: ({feature['x']}, {feature['y']})")
            logger.info(f"  النوع: {feature['type']}")
            logger.info(f"  الزاوية: {feature['angle']:.2f}")
            logger.info(f"  الجودة: {feature['quality']:.2f}")

        return features

    except Exception as e:
        logger.error(f"حدث خطأ: {str(e)}")
        raise

if __name__ == "__main__":
    # مسار الصورة المراد اختبارها
    image_path = "test_fingerprint.jpg"  # قم بتغيير هذا المسار إلى مسار الصورة الخاصة بك
    
    try:
        features = test_feature_extraction(image_path)
        logger.info("تم استخراج المميزات بنجاح!")
    except Exception as e:
        logger.error(f"فشل في استخراج المميزات: {str(e)}") 