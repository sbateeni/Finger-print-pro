import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from fingerprint.preprocessor import Preprocessor
from fingerprint.feature_extractor import FeatureExtractor
from fingerprint.matcher import FingerprintMatcher, draw_matching_lines
from fingerprint.visualization import FingerprintVisualizer
from fingerprint.performance_monitor import PerformanceMonitor
import cv2
import time
import shutil
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import math
import concurrent.futures
import threading
from functools import lru_cache
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
PREPROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'preprocessed')
FEATURES_FOLDER = os.path.join(UPLOAD_FOLDER, 'features')
MATCHING_FOLDER = os.path.join(UPLOAD_FOLDER, 'matching')
CACHE_FOLDER = os.path.join(UPLOAD_FOLDER, 'cache')

# Ensure upload folders exist and are clean
for folder in [UPLOAD_FOLDER, ORIGINAL_FOLDER, PROCESSED_FOLDER, PREPROCESSED_FOLDER, FEATURES_FOLDER, MATCHING_FOLDER, CACHE_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    # تنظيف الملفات القديمة
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f'خطأ في حذف الملف {file_path}: {str(e)}')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# إنشاء كائنات المعالجة
preprocessor = Preprocessor()
feature_extractor = FeatureExtractor()
matcher = FingerprintMatcher()
visualizer = FingerprintVisualizer()
performance_monitor = PerformanceMonitor()

# إنشاء كائن للتحكم في الذاكرة
memory_manager = threading.Lock()

# إنشاء كائن للتحكم في التخزين المؤقت
cache_manager = threading.Lock()

# إنشاء كائن للتحكم في المعالجة المتوازية
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=100)
def validate_image(file):
    """التحقق من صحة الصورة مع التخزين المؤقت"""
    try:
        # التحقق من نوع الملف
        if not allowed_file(file.filename):
            return False, 'نوع الملف غير مدعوم. الأنواع المدعومة هي: PNG, JPG, JPEG, GIF'
        
        # التحقق من حجم الملف
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return False, f'حجم الملف كبير جداً. الحد الأقصى هو {MAX_FILE_SIZE/1024/1024}MB'
        
        # التحقق من أن الملف صورة صالحة
        img_bytes = file.read()
        file.seek(0)  # إعادة المؤشر إلى بداية الملف
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False, 'الملف ليس صورة صالحة'
        
        return True, img  # إرجاع الصورة مع حالة التحقق
    except Exception as e:
        return False, f'خطأ في التحقق من الصورة: {str(e)}'

def enhance_image(image):
    """تحسين جودة الصورة"""
    try:
        # تحويل إلى تدرج الرمادي
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # تحسين التباين
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # إزالة الضوضاء
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # تحسين الحواف
        edges = cv2.Canny(denoised, 100, 200)
        
        # تصحيح الإضاءة
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            final = enhanced
        
        return {
            'original': image,
            'gray': gray,
            'enhanced': enhanced,
            'denoised': denoised,
            'edges': edges,
            'final': final
        }
    except Exception as e:
        logger.error(f'خطأ في تحسين الصورة: {str(e)}')
        raise

def save_preprocessing_steps(image, filename, folder):
    """حفظ خطوات المعالجة المسبقة"""
    try:
        # معالجة الصورة
        processed = enhance_image(image)
        steps = {}
        
        # حفظ كل خطوة من خطوات المعالجة
        for step_name, step_image in processed.items():
            if isinstance(step_image, np.ndarray):  # التأكد من أن الصورة صالحة
                step_filename = f"{step_name}_{filename}"
                step_path = os.path.join(folder, step_filename)
                cv2.imwrite(step_path, step_image)
                steps[step_name] = f'uploads/preprocessed/{step_filename}'
        
        return steps
    except Exception as e:
        logger.error(f'خطأ في حفظ خطوات المعالجة المسبقة: {str(e)}')
        raise

def extract_enhanced_features(image):
    """استخراج المميزات المحسنة"""
    try:
        return feature_extractor.extract_features(image)
    except Exception as e:
        logger.error(f'فشل في استخراج المميزات: {str(e)}')
        raise

def calculate_angle(skeleton, x, y):
    """حساب زاوية النقطة المميزة"""
    try:
        # استخدام خوارزمية Sobel لحساب التدرج
        sobelx = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(skeleton.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # حساب الزاوية
        angle = np.arctan2(sobely[y, x], sobelx[y, x]) * 180 / np.pi
        return float(angle)
    except Exception as e:
        logger.error(f'خطأ في حساب الزاوية: {str(e)}')
        return 0.0

def calculate_point_quality(skeleton, x, y):
    """حساب جودة النقطة المميزة"""
    try:
        # حساب عدد الجيران في دائرة نصف قطرها 5 بيكسل
        radius = 5
        quality = 0
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if 0 <= y + i < skeleton.shape[0] and 0 <= x + j < skeleton.shape[1]:
                    if skeleton[y + i, x + j]:
                        quality += 1
        
        # تطبيع الجودة
        max_quality = (2 * radius + 1) ** 2
        normalized_quality = quality / max_quality
        
        return float(normalized_quality)
    except Exception as e:
        logger.error(f'خطأ في حساب جودة النقطة: {str(e)}')
        return 0.0

def save_feature_visualization(image, features, filename, folder):
    """حفظ تصور المميزات"""
    # رسم النقاط المميزة
    marked_image = visualizer.draw_minutiae_points(image, features)
    
    # حفظ الصورة
    feature_filename = f"features_{filename}"
    feature_path = os.path.join(folder, feature_filename)
    cv2.imwrite(feature_path, marked_image)
    
    return url_for('static', filename=f'uploads/features/{feature_filename}')

@lru_cache(maxsize=100)
def calculate_feature_distance(feature1, feature2):
    """حساب المسافة بين مميزتين مع التخزين المؤقت"""
    # حساب المسافة الإقليدية بين النقاط
    point_distance = math.sqrt((feature1['x'] - feature2['x'])**2 + (feature1['y'] - feature2['y'])**2)
    
    # حساب الفرق في الزوايا
    angle_diff = min(abs(feature1['angle'] - feature2['angle']), 360 - abs(feature1['angle'] - feature2['angle']))
    
    # حساب المسافة الكلية (مرجحة)
    total_distance = 0.7 * point_distance + 0.3 * angle_diff
    
    return total_distance

def match_fingerprints(features1, features2):
    """مقارنة البصمات باستخدام خوارزمية متقدمة"""
    try:
        # إنشاء مصفوفة المسافات
        distance_matrix = np.zeros((len(features1), len(features2)))
        
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                # حساب المسافة الإقليدية بين النقاط
                point_distance = math.sqrt((f1['x'] - f2['x'])**2 + (f1['y'] - f2['y'])**2)
                
                # حساب الفرق في الزوايا
                angle_diff = min(abs(f1['angle'] - f2['angle']), 360 - abs(f1['angle'] - f2['angle']))
                
                # حساب المسافة الكلية (مرجحة)
                total_distance = 0.7 * point_distance + 0.3 * angle_diff
                
                # تطبيق عامل الجودة
                quality_factor = (f1['quality'] + f2['quality']) / 2
                distance_matrix[i, j] = total_distance * (1 - quality_factor)
        
        # استخدام خوارزمية Hungarian للعثور على أفضل تطابق
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # حساب درجة التطابق
        total_distance = distance_matrix[row_ind, col_ind].sum()
        max_possible_distance = len(row_ind) * 100  # قيمة افتراضية للحد الأقصى للمسافة
        
        match_score = 1 - (total_distance / max_possible_distance)
        
        # تجميع النقاط المتطابقة
        matching_points = []
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] < 50:  # حد المسافة المقبول
                matching_points.append((features1[i], features2[j]))
        
        return float(match_score), matching_points
    except Exception as e:
        logger.error(f'فشل في مقارنة البصمات: {str(e)}')
        raise

def save_matching_visualization(image1, image2, matching_points, filename, folder):
    """حفظ تصور التطابق"""
    try:
        # رسم خطوط التطابق
        matching_image = draw_matching_lines(image1, image2, matching_points)
        
        # حفظ الصورة
        matching_filename = f"matching_{filename}"
        matching_path = os.path.join(folder, matching_filename)
        cv2.imwrite(matching_path, matching_image)
        
        return f'uploads/matching/{matching_filename}'
    except Exception as e:
        logger.error(f'خطأ في حفظ تصور التطابق: {str(e)}')
        raise

def optimize_memory():
    """تحسين استخدام الذاكرة"""
    with memory_manager:
        # تنظيف الذاكرة المؤقتة
        gc.collect()
        
        # تقليل استخدام الذاكرة
        process = psutil.Process()
        if process.memory_percent() > 80:
            logger.warning('استخدام الذاكرة مرتفع، جاري تنظيف الذاكرة...')
            gc.collect()
            if process.memory_percent() > 80:
                logger.warning('استخدام الذاكرة لا يزال مرتفعاً، جاري إعادة تشغيل الخادم...')
                os._exit(1)

@app.route('/')
def index():
    logger.info('تم تحميل الصفحة الرئيسية')
    return render_template('index.html')

@app.route('/status')
def get_status():
    """الحصول على حالة المعالجة الحالية"""
    status = performance_monitor.get_current_status()
    if status:
        return jsonify(status)
    return jsonify({'error': 'لا توجد عملية جارية'}), 404

@app.route('/upload', methods=['POST'])
def upload_files():
    """معالجة رفع ملفات البصمات"""
    try:
        start_time = time.time()  # تسجيل وقت البدء
        logger.info("بدء معالجة الملفات المرفوعة...")
        
        # التحقق من وجود الملفات
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            logger.error("لم يتم العثور على ملفات البصمات في الطلب")
            return jsonify({'error': 'يرجى رفع صورتين للبصمات'}), 400

        fp1 = request.files['fingerprint1']
        fp2 = request.files['fingerprint2']

        # التحقق من صحة الملفات
        if fp1.filename == '' or fp2.filename == '':
            logger.error("تم رفع ملفات فارغة")
            return jsonify({'error': 'لم يتم اختيار ملفات'}), 400

        logger.info(f"تم استلام الملفات: {fp1.filename} و {fp2.filename}")

        # قراءة الصور
        logger.info("جاري قراءة الصور...")
        img1 = cv2.imdecode(np.frombuffer(fp1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(fp2.read(), np.uint8), cv2.IMREAD_COLOR)

        # التحقق من صحة الصور
        if img1 is None or img2 is None:
            logger.error("فشل في قراءة الصور")
            return jsonify({'error': 'فشل في قراءة الصور'}), 400

        logger.info("تم قراءة الصور بنجاح")

        # معالجة الصور
        logger.info("جاري معالجة الصور...")
        processed1 = enhance_image(img1)
        processed2 = enhance_image(img2)

        # التحقق من أن الصور المعالجة صالحة
        if not isinstance(processed1.get('final'), np.ndarray) or not isinstance(processed2.get('final'), np.ndarray):
            logger.error("فشل في معالجة الصور")
            return jsonify({'error': 'فشل في معالجة الصور'}), 400

        logger.info("تمت معالجة الصور بنجاح")

        # استخراج المميزات
        logger.info("جاري استخراج المميزات...")
        try:
            features1 = feature_extractor.extract_features(processed1['final'])
            features2 = feature_extractor.extract_features(processed2['final'])
            logger.info(f"تم استخراج {len(features1)} و {len(features2)} نقطة مميزة")
        except Exception as e:
            logger.error(f'خطأ في استخراج المميزات: {str(e)}')
            return jsonify({'error': f'فشل في استخراج المميزات: {str(e)}'}), 500

        # مقارنة البصمات
        logger.info("جاري مقارنة البصمات...")
        try:
            match_score, matching_points = matcher.match_fingerprints(features1, features2)
            logger.info(f"تم حساب درجة التطابق: {match_score:.2f}")
        except Exception as e:
            logger.error(f'خطأ في مقارنة البصمات: {str(e)}')
            return jsonify({'error': f'فشل في مقارنة البصمات: {str(e)}'}), 500

        # حفظ الصور المعالجة
        logger.info("جاري حفظ النتائج...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fp1_filename = f"fp1_{timestamp}.jpg"
        fp2_filename = f"fp2_{timestamp}.jpg"

        # حفظ الصور المعالجة
        cv2.imwrite(os.path.join(PROCESSED_FOLDER, fp1_filename), processed1['final'])
        cv2.imwrite(os.path.join(PROCESSED_FOLDER, fp2_filename), processed2['final'])

        # حفظ خطوات المعالجة
        preprocessing_steps1 = save_preprocessing_steps(img1, fp1_filename, PREPROCESSED_FOLDER)
        preprocessing_steps2 = save_preprocessing_steps(img2, fp2_filename, PREPROCESSED_FOLDER)

        # حفظ تصور المميزات
        features_viz1 = save_feature_visualization(processed1['final'], features1, fp1_filename, FEATURES_FOLDER)
        features_viz2 = save_feature_visualization(processed2['final'], features2, fp2_filename, FEATURES_FOLDER)

        # حفظ تصور التطابق
        matching_viz = save_matching_visualization(processed1['final'], processed2['final'], matching_points, f"match_{timestamp}.jpg", MATCHING_FOLDER)

        logger.info("تم حفظ جميع النتائج بنجاح")

        # إرجاع النتائج
        return jsonify({
            'match_score': float(match_score),
            'matching_points': matching_points,
            'processed_image1': url_for('static', filename=f'uploads/processed/{fp1_filename}'),
            'processed_image2': url_for('static', filename=f'uploads/processed/{fp2_filename}'),
            'matching_visualization': url_for('static', filename=f'uploads/matching/match_{timestamp}.jpg'),
            'performance_stats': {
                'totalTime': time.time() - start_time,
                'memoryUsage': psutil.Process().memory_info().rss,
                'cpuUsage': psutil.Process().cpu_percent()
            }
        })

    except Exception as e:
        logger.error(f'خطأ في معالجة الملفات: {str(e)}')
        return jsonify({'error': str(e)}), 500

def cleanup_old_files():
    """تنظيف الملفات القديمة من مجلدات التحميل"""
    try:
        for folder in [ORIGINAL_FOLDER, PROCESSED_FOLDER, PREPROCESSED_FOLDER, FEATURES_FOLDER, MATCHING_FOLDER, CACHE_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.remove(file_path)
    except Exception as e:
        logger.error(f'فشل في تنظيف الملفات القديمة: {str(e)}')

# تنظيف الملفات المؤقتة عند بدء التشغيل
cleanup_old_files()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 