from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
import os
import json
from fingerprint.preprocessor import preprocess_image
from fingerprint.feature_extractor import extract_features, match_features
from fingerprint.quality import calculate_quality
import io
import base64

app = Flask(__name__)

# تعيين المسارات
DATA_DIR = os.getenv('DATA_DIR', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# تعيين الحد الأقصى لحجم الصورة (بالبايت)
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', 8 * 1024 * 1024))  # 8MB

# تكوين التطبيق
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE
app.config['UPLOAD_FOLDER'] = DATA_DIR

def process_image(image_data):
    """معالجة الصورة واستخراج السمات"""
    try:
        # تحويل البيانات إلى صورة OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None, "فشل في قراءة الصورة"
        
        # معالجة الصورة
        processed = preprocess_image(img)
        if processed is None:
            return None, "فشل في معالجة الصورة"
        
        # استخراج السمات
        features = extract_features(processed['denoised'])
        if features is None:
            return None, "فشل في استخراج السمات"
        
        return {
            'processed_image': processed['denoised'],
            'features': features
        }, None
    except Exception as e:
        return None, str(e)

def save_features_to_json(features, filename):
    """حفظ السمات بتنسيق JSON"""
    try:
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
        
        return True
    except Exception as e:
        print(f"خطأ في حفظ السمات: {str(e)}")
        return False

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """معالجة الصورة واستخراج السمات"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم إرسال صورة'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        # قراءة الصورة
        image_data = file.read()
        if len(image_data) > MAX_IMAGE_SIZE:
            return jsonify({'error': 'حجم الصورة كبير جداً'}), 400
        
        # معالجة الصورة
        result, error = process_image(image_data)
        if error:
            return jsonify({'error': error}), 400
        
        # حفظ السمات
        filename = os.path.join(DATA_DIR, f"features_{hash(file.filename)}.json")
        if save_features_to_json(result['features'], filename):
            # تحويل الصورة المعالجة إلى base64
            _, buffer = cv2.imencode('.png', result['processed_image'])
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'processed_image': processed_image_b64,
                'features_file': filename
            })
        else:
            return jsonify({'error': 'فشل في حفظ السمات'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """مقارنة بصمتين"""
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'يجب إرسال بصمتين للمقارنة'}), 400
        
        # معالجة البصمة الأولى
        file1 = request.files['image1']
        result1, error1 = process_image(file1.read())
        if error1:
            return jsonify({'error': f'خطأ في البصمة الأولى: {error1}'}), 400
        
        # معالجة البصمة الثانية
        file2 = request.files['image2']
        result2, error2 = process_image(file2.read())
        if error2:
            return jsonify({'error': f'خطأ في البصمة الثانية: {error2}'}), 400
        
        # مقارنة البصمتين
        match_score, matches = match_features(result1['features'], result2['features'])
        
        # إنشاء صورة المطابقة
        matching_image = create_matching_image(
            result1['processed_image'],
            result2['processed_image'],
            result1['features'],
            result2['features'],
            matches
        )
        
        # تحويل صورة المطابقة إلى base64
        _, buffer = cv2.imencode('.png', matching_image)
        matching_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'match_score': match_score,
            'matches_count': len(matches),
            'matching_image': matching_image_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_matching_image(image1, image2, features1, features2, matches):
    """إنشاء صورة المطابقة"""
    # إنشاء صورة تجمع بين البصمتين
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    max_h = max(h1, h2)
    total_w = w1 + w2
    matching_image = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    
    # وضع البصمتين في الصورة
    matching_image[:h1, :w1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    matching_image[:h2, w1:] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # رسم النقاط المميزة
    colors = {
        'ridge_endings': (0, 255, 0),    # أخضر
        'bifurcations': (255, 0, 0),     # أزرق
        'islands': (0, 0, 255),          # أحمر
        'dots': (255, 255, 0),           # أصفر
        'cores': (255, 0, 255),          # وردي
        'deltas': (0, 255, 255)          # سماوي
    }
    
    # رسم النقاط المميزة للبصمة الأولى
    for type_name, contours in features1.get('minutiae', {}).items():
        color = colors[type_name]
        for contour in contours:
            try:
                cv2.drawContours(matching_image, [contour], -1, color, 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(matching_image, (cX, cY), 5, color, -1)
            except:
                continue
    
    # رسم النقاط المميزة للبصمة الثانية
    for type_name, contours in features2.get('minutiae', {}).items():
        color = colors[type_name]
        for contour in contours:
            try:
                contour_shifted = contour + np.array([w1, 0])
                cv2.drawContours(matching_image, [contour_shifted], -1, color, 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) + w1
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(matching_image, (cX, cY), 5, color, -1)
            except:
                continue
    
    # رسم خطوط التطابق
    for i, match in enumerate(matches):
        try:
            pt1 = (int(match[0][0]), int(match[0][1]))
            pt2 = (int(match[1][0]) + w1, int(match[1][1]))
            
            # رسم دائرة حول النقاط المتطابقة
            cv2.circle(matching_image, pt1, 8, (0, 255, 255), 2)
            cv2.circle(matching_image, pt2, 8, (0, 255, 255), 2)
            
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

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 