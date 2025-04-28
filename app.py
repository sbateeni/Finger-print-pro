import os
import logging
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from fingerprint.preprocessor import Preprocessor
from fingerprint.feature_extractor import FeatureExtractor
from fingerprint.matcher import FingerprintMatcher
from fingerprint.visualization import Visualizer
import cv2
import numpy as np
from PIL import Image
import time
import json
import shutil
import concurrent.futures
import threading
from functools import lru_cache
import psutil
import gc
import math
from scipy.optimize import linear_sum_assignment

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
visualizer = Visualizer()

# إنشاء كائن للتحكم في الذاكرة
memory_manager = threading.Lock()

# إنشاء كائن للتحكم في التخزين المؤقت
cache_manager = threading.Lock()

# إنشاء كائن للتحكم في المعالجة المتوازية
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# إنشاء كائن لمراقبة الأداء
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.current_status = {}
        self.lock = threading.Lock()
    
    def update_status(self, status):
        with self.lock:
            self.current_status = status
    
    def get_current_status(self):
        with self.lock:
            return self.current_status

performance_monitor = PerformanceMonitor()

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

def check_image_quality(image):
    """فحص جودة الصورة"""
    quality_metrics = {
        'resolution': f"{image.shape[1]}x{image.shape[0]}",
        'clarity': float(cv2.Laplacian(image, cv2.CV_64F).var()),
        'brightness': float(np.mean(image)),
        'contrast': float(np.std(image)),
        'noise_ratio': float(cv2.meanStdDev(image)[1][0][0] / cv2.meanStdDev(image)[0][0][0])
    }
    return quality_metrics

def enhance_image(image):
    """تحسين جودة الصورة"""
    # تحويل إلى تدرج الرمادي
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تحسين التباين
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # إزالة الضوضاء
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # تحسين الحواف
    edges = cv2.Canny(denoised, 100, 200)
    
    # تصحيح الإضاءة
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return {
        'original': image,
        'gray': gray,
        'enhanced': enhanced,
        'denoised': denoised,
        'edges': edges,
        'final': final
    }

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

def draw_matching_lines(image1, image2, matching_points):
    """رسم خطوط التطابق بين البصمتين"""
    try:
        # إنشاء صورة جديدة لعرض التطابق
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # إنشاء صورة فارغة
        matching_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        # نسخ الصورتين
        matching_image[:h1, :w1] = image1
        matching_image[:h2, w1:w1+w2] = image2
        
        # رسم خطوط التطابق
        for point1, point2 in matching_points:
            pt1 = (int(point1['x']), int(point1['y']))
            pt2 = (int(point2['x']) + w1, int(point2['y']))
            cv2.line(matching_image, pt1, pt2, (0, 255, 0), 1)
        
        return matching_image
    except Exception as e:
        logger.error(f'خطأ في رسم خطوط التطابق: {str(e)}')
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

def save_processed_image(image, filename, folder):
    """حفظ الصورة المعالجة"""
    try:
        # التأكد من وجود المجلد
        os.makedirs(folder, exist_ok=True)
        
        # حفظ الصورة
        file_path = os.path.join(folder, filename)
        cv2.imwrite(file_path, image)
        
        # إرجاع المسار النسبي للصورة
        return f'uploads/{os.path.basename(folder)}/{filename}'
    except Exception as e:
        logger.error(f'خطأ في حفظ الصورة: {str(e)}')
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
    try:
        if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
            return jsonify({'error': 'يرجى رفع صورتين للبصمات'}), 400

        fp1 = request.files['fingerprint1']
        fp2 = request.files['fingerprint2']

        if fp1.filename == '' or fp2.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملفات'}), 400

        # قراءة الصور
        img1 = np.array(Image.open(fp1))
        img2 = np.array(Image.open(fp2))

        # فحص جودة الصور
        quality1 = check_image_quality(img1)
        quality2 = check_image_quality(img2)

        # تحسين الصور
        enhanced1 = enhance_image(img1)
        enhanced2 = enhance_image(img2)

        # حفظ الصور المعالجة
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        processed_images = {
            'fp1': {
                'original': save_processed_image(enhanced1['original'], f'fp1_original_{timestamp}.jpg', PROCESSED_FOLDER),
                'enhanced': save_processed_image(enhanced1['enhanced'], f'fp1_enhanced_{timestamp}.jpg', PROCESSED_FOLDER),
                'gray': save_processed_image(enhanced1['gray'], f'fp1_gray_{timestamp}.jpg', PROCESSED_FOLDER),
                'denoised': save_processed_image(enhanced1['denoised'], f'fp1_denoised_{timestamp}.jpg', PROCESSED_FOLDER),
                'edges': save_processed_image(enhanced1['edges'], f'fp1_edges_{timestamp}.jpg', PROCESSED_FOLDER),
                'final': save_processed_image(enhanced1['final'], f'fp1_final_{timestamp}.jpg', PROCESSED_FOLDER)
            },
            'fp2': {
                'original': save_processed_image(enhanced2['original'], f'fp2_original_{timestamp}.jpg', PROCESSED_FOLDER),
                'enhanced': save_processed_image(enhanced2['enhanced'], f'fp2_enhanced_{timestamp}.jpg', PROCESSED_FOLDER),
                'gray': save_processed_image(enhanced2['gray'], f'fp2_gray_{timestamp}.jpg', PROCESSED_FOLDER),
                'denoised': save_processed_image(enhanced2['denoised'], f'fp2_denoised_{timestamp}.jpg', PROCESSED_FOLDER),
                'edges': save_processed_image(enhanced2['edges'], f'fp2_edges_{timestamp}.jpg', PROCESSED_FOLDER),
                'final': save_processed_image(enhanced2['final'], f'fp2_final_{timestamp}.jpg', PROCESSED_FOLDER)
            }
        }

        # معالجة الصور
        processed_fp1 = preprocessor.preprocess_image(enhanced1['final'])
        processed_fp2 = preprocessor.preprocess_image(enhanced2['final'])

        # التأكد من أن الصور هي numpy arrays
        if not isinstance(processed_fp1, np.ndarray):
            processed_fp1 = np.array(processed_fp1)
        if not isinstance(processed_fp2, np.ndarray):
            processed_fp2 = np.array(processed_fp2)

        # استخراج المميزات
        features1 = feature_extractor.extract_features(processed_fp1)
        features2 = feature_extractor.extract_features(processed_fp2)

        # رسم النقاط المميزة
        marked_fp1 = visualizer.visualize_features(processed_fp1, features1)
        marked_fp2 = visualizer.visualize_features(processed_fp2, features2)

        # حفظ صور النقاط المميزة
        features_images = {
            'fp1': save_processed_image(marked_fp1, f'fp1_features_{timestamp}.jpg', FEATURES_FOLDER),
            'fp2': save_processed_image(marked_fp2, f'fp2_features_{timestamp}.jpg', FEATURES_FOLDER)
        }

        # مقارنة البصمات
        match_result = matcher.match_features(features1, features2)

        # رسم خطوط التطابق
        matching_visualization = visualizer.visualize_matching(marked_fp1, marked_fp2, features1, features2, match_result)
        matching_image = save_processed_image(matching_visualization, f'matching_{timestamp}.jpg', MATCHING_FOLDER)

        return jsonify({
            'success': True,
            'quality1': quality1,
            'quality2': quality2,
            'processed_images': processed_images,
            'features_images': features_images,
            'matching_image': matching_image,
            'match_score': float(match_result['score']),
            'features_count1': features1['count'],
            'features_count2': features2['count'],
            'matching_points': match_result['matches']
        })

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
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