import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import compare_fingerprints
from fingerprint.visualization import draw_minutiae_points, draw_matching_lines
from fingerprint.performance_monitor import PerformanceMonitor
import cv2
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

# Ensure upload folders exist
for folder in [UPLOAD_FOLDER, ORIGINAL_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# إنشاء كائن مراقب الأداء
performance_monitor = PerformanceMonitor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    logger.info('بدء عملية رفع الملفات')
    
    # بدء مراقبة الأداء
    performance_monitor.start_monitoring(total_steps=5)  # 5 خطوات رئيسية
    
    if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
        logger.error('لم يتم تحديد الملفات المطلوبة')
        performance_monitor.add_step('رفع الملفات', 'failed')
        return jsonify({'error': 'يجب رفع صورتين للبصمات'}), 400
    
    fp1 = request.files['fingerprint1']
    fp2 = request.files['fingerprint2']
    
    if fp1.filename == '' or fp2.filename == '':
        logger.error('لم يتم اختيار الملفات')
        performance_monitor.add_step('رفع الملفات', 'failed')
        return jsonify({'error': 'لم يتم اختيار الملفات'}), 400
    
    if not (allowed_file(fp1.filename) and allowed_file(fp2.filename)):
        logger.error('نوع الملف غير مدعوم')
        performance_monitor.add_step('رفع الملفات', 'failed')
        return jsonify({'error': 'نوع الملف غير مدعوم. الأنواع المدعومة هي: PNG, JPG, JPEG, GIF'}), 400
    
    try:
        # حفظ الصور الأصلية
        performance_monitor.add_step('رفع الملفات')
        fp1_filename = secure_filename(fp1.filename)
        fp2_filename = secure_filename(fp2.filename)
        
        fp1_path = os.path.join(ORIGINAL_FOLDER, fp1_filename)
        fp2_path = os.path.join(ORIGINAL_FOLDER, fp2_filename)
        
        logger.info('حفظ الصور الأصلية')
        fp1.save(fp1_path)
        fp2.save(fp2_path)
        logger.info(f'تم حفظ الصور في: {fp1_path} و {fp2_path}')
        performance_monitor.update_step('رفع الملفات')
        
        # معالجة الصور
        performance_monitor.add_step('معالجة الصور')
        logger.info('بدء معالجة الصور')
        try:
            processed_fp1 = preprocess_image(fp1_path)
            processed_fp2 = preprocess_image(fp2_path)
            logger.info('تم معالجة الصور بنجاح')
            performance_monitor.update_step('معالجة الصور')
        except Exception as e:
            logger.error(f'فشل في معالجة الصور: {str(e)}')
            performance_monitor.update_step('معالجة الصور', 'failed')
            return jsonify({'error': 'فشل في معالجة الصور'}), 500
        
        # استخراج المميزات
        performance_monitor.add_step('استخراج المميزات')
        logger.info('بدء استخراج المميزات')
        try:
            features1 = extract_features(processed_fp1)
            features2 = extract_features(processed_fp2)
            if not features1 or not features2:
                raise Exception('فشل في استخراج المميزات من الصور')
            logger.info(f'تم استخراج {len(features1)} نقطة مميزة من الصورة الأولى و {len(features2)} من الصورة الثانية')
            
            # رسم النقاط المميزة على الصور
            marked_fp1 = draw_minutiae_points(processed_fp1, features1)
            marked_fp2 = draw_minutiae_points(processed_fp2, features2)
            
            # حفظ الصور مع النقاط المميزة
            marked_fp1_filename = f'marked_{fp1_filename}'
            marked_fp2_filename = f'marked_{fp2_filename}'
            marked_fp1_path = os.path.join(PROCESSED_FOLDER, marked_fp1_filename)
            marked_fp2_path = os.path.join(PROCESSED_FOLDER, marked_fp2_filename)
            cv2.imwrite(marked_fp1_path, marked_fp1)
            cv2.imwrite(marked_fp2_path, marked_fp2)
            performance_monitor.update_step('استخراج المميزات')
            
        except Exception as e:
            logger.error(f'فشل في استخراج المميزات: {str(e)}')
            performance_monitor.update_step('استخراج المميزات', 'failed')
            return jsonify({'error': 'فشل في استخراج المميزات من الصور'}), 500
        
        # مقارنة البصمات
        performance_monitor.add_step('مقارنة البصمات')
        logger.info('بدء مقارنة البصمات')
        try:
            match_score, matching_points = compare_fingerprints(features1, features2)
            logger.info(f'نتيجة المقارنة: {match_score:.2%}, عدد النقاط المتطابقة: {len(matching_points)}')
            
            # رسم خطوط التطابق بين البصمتين
            matching_lines = draw_matching_lines(marked_fp1, marked_fp2, list(zip(features1, features2)))
            matching_filename = f'matching_{fp1_filename}'
            matching_path = os.path.join(PROCESSED_FOLDER, matching_filename)
            cv2.imwrite(matching_path, matching_lines)
            performance_monitor.update_step('مقارنة البصمات')
            
        except Exception as e:
            logger.error(f'فشل في مقارنة البصمات: {str(e)}')
            performance_monitor.update_step('مقارنة البصمات', 'failed')
            return jsonify({'error': 'فشل في مقارنة البصمات'}), 500
        
        # إنشاء روابط للصور
        performance_monitor.add_step('إنشاء النتائج')
        marked_fp1_url = url_for('static', filename=f'uploads/processed/{marked_fp1_filename}')
        marked_fp2_url = url_for('static', filename=f'uploads/processed/{marked_fp2_filename}')
        matching_url = url_for('static', filename=f'uploads/processed/{matching_filename}')
        performance_monitor.update_step('إنشاء النتائج')
        
        # إيقاف المراقبة
        performance_stats = performance_monitor.stop_monitoring()
        
        return jsonify({
            'success': True,
            'match_score': match_score,
            'matching_points': matching_points,
            'marked1': marked_fp1_url,
            'marked2': marked_fp2_url,
            'matching_visualization': matching_url,
            'num_features1': len(features1),
            'num_features2': len(features2),
            'performance_stats': performance_stats
        })
        
    except Exception as e:
        logger.error(f'حدث خطأ: {str(e)}')
        performance_monitor.stop_monitoring()
        return jsonify({'error': str(e)}), 500

# Clean up uploaded files periodically
def cleanup_old_files():
    """
    تنظيف الملفات القديمة من مجلدات التحميل
    """
    try:
        for folder in [ORIGINAL_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                # حذف الملفات الأقدم من ساعة
                if os.path.getmtime(file_path) < time.time() - 3600:
                    os.remove(file_path)
    except Exception as e:
        logger.error(f'فشل في تنظيف الملفات القديمة: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 