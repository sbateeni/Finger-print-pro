import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from utils.preprocess import preprocess_fingerprint
from utils.extract_features import extract_features
from utils.match_fingerprint import match_fingerprint

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))

# تعديل مسارات المجلدات لتكون مطلقة
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving uploaded file {filename}: {str(e)}")
        return "File not found", 404

@app.route('/results/<filename>')
def result_file(filename):
    try:
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving result file {filename}: {str(e)}")
        return "File not found", 404

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fingerprint1' not in request.files or 'fingerprint2' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file1 = request.files['fingerprint1']
    file2 = request.files['fingerprint2']
    
    if file1.filename == '' or file2.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
        try:
            # Generate unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            
            filename1 = f"{timestamp}_{unique_id}_1_{secure_filename(file1.filename)}"
            filename2 = f"{timestamp}_{unique_id}_2_{secure_filename(file2.filename)}"
            
            # Save original files
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file1.save(file1_path)
            file2.save(file2_path)
            
            # Match fingerprints (returns: score, kp1_count, kp2_count, good_matches_count, match_filename, minutiae1_filename, minutiae2_filename, sourceafis_score)
            match_score, kp1_count, kp2_count, good_matches_count, match_filename, minutiae1_filename, minutiae2_filename, sourceafis_score = match_fingerprint(file1_path, file2_path, app.config['RESULTS_FOLDER'])
            
            if match_filename is None or minutiae1_filename is None or minutiae2_filename is None:
                flash('Error processing images')
                return redirect(url_for('index'))
            
            # Determine result
            if match_score >= 80:
                result_text = "Match Found!"
                result_type = "success"
            elif match_score >= 50:
                result_text = "Possible Match"
                result_type = "warning"
            else:
                result_text = "No Match"
                result_type = "danger"
            
            return render_template('result.html',
                                 score=match_score,
                                 result_text=result_text,
                                 result_type=result_type,
                                 image1=filename1,
                                 image2=filename2,
                                 match_image=match_filename,
                                 minutiae1_image=minutiae1_filename,
                                 minutiae2_image=minutiae2_filename,
                                 kp1_count=kp1_count,
                                 kp2_count=kp2_count,
                                 good_matches_count=good_matches_count,
                                 sourceafis_score=sourceafis_score)
            
        except Exception as e:
            flash(f'Error processing fingerprints: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False) 