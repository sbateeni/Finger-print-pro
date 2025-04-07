import os
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features
from fingerprint.matcher import compare_fingerprints
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ORIGINAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'original')
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directories if they don't exist
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info('تم تحميل الصفحة الرئيسية')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    logger.info('بدء عملية رفع الملفات')
    
    if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
        logger.error('لم يتم تحديد الملفات المطلوبة')
        return jsonify({'error': 'يجب رفع صورتين للبصمات'}), 400
    
    fp1 = request.files['fingerprint1']
    fp2 = request.files['fingerprint2']
    
    if fp1.filename == '' or fp2.filename == '':
        logger.error('لم يتم اختيار الملفات')
        return jsonify({'error': 'لم يتم اختيار الملفات'}), 400
    
    if not (allowed_file(fp1.filename) and allowed_file(fp2.filename)):
        logger.error('نوع الملف غير مدعوم')
        return jsonify({'error': 'نوع الملف غير مدعوم. الأنواع المدعومة هي: PNG, JPG, JPEG, GIF'}), 400
    
    try:
        # Save original images with secure filenames
        fp1_filename = secure_filename(fp1.filename)
        fp2_filename = secure_filename(fp2.filename)
        
        fp1_path = os.path.join(ORIGINAL_FOLDER, fp1_filename)
        fp2_path = os.path.join(ORIGINAL_FOLDER, fp2_filename)
        
        logger.info('حفظ الصور الأصلية')
        fp1.save(fp1_path)
        fp2.save(fp2_path)
        logger.info(f'تم حفظ الصور في: {fp1_path} و {fp2_path}')
        
        # Process images
        logger.info('بدء معالجة الصور')
        try:
            processed_fp1 = preprocess_image(fp1_path)
            processed_fp2 = preprocess_image(fp2_path)
            logger.info('تم معالجة الصور بنجاح')
        except Exception as e:
            logger.error(f'فشل في معالجة الصور: {str(e)}')
            return jsonify({'error': 'فشل في معالجة الصور'}), 500
        
        # Save processed images
        processed_fp1_path = os.path.join(PROCESSED_FOLDER, f'processed_{fp1_filename}')
        processed_fp2_path = os.path.join(PROCESSED_FOLDER, f'processed_{fp2_filename}')
        cv2.imwrite(processed_fp1_path, processed_fp1)
        cv2.imwrite(processed_fp2_path, processed_fp2)
        logger.info('تم حفظ الصور المعالجة')
        
        # Extract features
        logger.info('بدء استخراج المميزات')
        try:
            features1 = extract_features(processed_fp1)
            features2 = extract_features(processed_fp2)
            if not features1 or not features2:
                raise Exception('فشل في استخراج المميزات من الصور')
            logger.info(f'تم استخراج {len(features1)} نقطة مميزة من الصورة الأولى و {len(features2)} من الصورة الثانية')
        except Exception as e:
            logger.error(f'فشل في استخراج المميزات: {str(e)}')
            return jsonify({'error': 'فشل في استخراج المميزات من الصور'}), 500
        
        # Compare fingerprints
        logger.info('بدء مقارنة البصمات')
        try:
            match_score, matching_points = compare_fingerprints(features1, features2)
            logger.info(f'نتيجة المقارنة: {match_score:.2%}, عدد النقاط المتطابقة: {len(matching_points)}')
        except Exception as e:
            logger.error(f'فشل في مقارنة البصمات: {str(e)}')
            return jsonify({'error': 'فشل في مقارنة البصمات'}), 500
        
        # Generate URLs for the images
        fp1_url = url_for('static', filename=f'uploads/original/{fp1_filename}')
        fp2_url = url_for('static', filename=f'uploads/original/{fp2_filename}')
        processed_fp1_url = url_for('static', filename=f'uploads/processed/processed_{fp1_filename}')
        processed_fp2_url = url_for('static', filename=f'uploads/processed/processed_{fp2_filename}')
        
        return jsonify({
            'success': True,
            'match_score': match_score,
            'matching_points': matching_points,
            'original1': fp1_url,
            'original2': fp2_url,
            'processed1': processed_fp1_url,
            'processed2': processed_fp2_url,
            'num_features1': len(features1),
            'num_features2': len(features2)
        })
        
    except Exception as e:
        logger.error(f'حدث خطأ: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info('بدء تشغيل التطبيق')
    app.run(debug=True) 