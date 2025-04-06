from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import cv2
import numpy as np
from preprocessing.segmentation import preprocess_fingerprint
from biometric_features.extract_core_delta import extract_core_delta
from matcher import match_fingerprints
import logging
from utils import setup_logging, save_image, load_image, calculate_metrics

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fingerprints.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Setup logging
setup_logging()

# Add CSP headers
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: blob:;"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    fingerprints = db.relationship('Fingerprint', backref='user', lazy=True)

class Fingerprint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    processed_path = db.Column(db.String(255))
    quality_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    minutiae = db.relationship('Minutiae', backref='fingerprint', lazy=True)
    matches = db.relationship('Match', backref='fingerprint', lazy=True)

class Minutiae(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fingerprint_id = db.Column(db.Integer, db.ForeignKey('fingerprint.id'), nullable=False)
    x = db.Column(db.Integer, nullable=False)
    y = db.Column(db.Integer, nullable=False)
    angle = db.Column(db.Float)
    type = db.Column(db.String(20))  # 'ending' or 'bifurcation'

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fingerprint_id = db.Column(db.Integer, db.ForeignKey('fingerprint.id'), nullable=False)
    matched_id = db.Column(db.Integer, db.ForeignKey('fingerprint.id'), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            email=email
        )
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'User registered successfully'})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return jsonify({'message': 'Login successful'})
        
        return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/upload', methods=['POST'])
@login_required
def upload_fingerprint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = f"{current_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process fingerprint
        processed_img = preprocess_fingerprint(filepath)
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{filename}")
        save_image(processed_img, processed_path)
        
        # Extract features
        core, delta = extract_core_delta(processed_img)
        
        # Save to database
        fingerprint = Fingerprint(
            user_id=current_user.id,
            image_path=filepath,
            processed_path=processed_path,
            quality_score=float(np.mean(processed_img))  # Simple quality metric
        )
        db.session.add(fingerprint)
        db.session.commit()
        
        return jsonify({
            'message': 'Fingerprint uploaded successfully',
            'fingerprint_id': fingerprint.id
        })
    
    except Exception as e:
        logging.error(f"Error processing fingerprint: {str(e)}")
        return jsonify({'error': 'Error processing fingerprint'}), 500

@app.route('/match', methods=['POST'])
@login_required
def match_fingerprint():
    if 'file' not in request.files:
        logging.error("No file provided in match request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("Empty filename in match request")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Saved match file to: {filepath}")
        
        # Process and match fingerprint
        matches = match_fingerprints(filepath)
        logging.info(f"Found {len(matches)} matches")
        
        return jsonify({
            'matches': matches,
            'message': 'Matching completed successfully'
        })
    
    except Exception as e:
        logging.error(f"Error in match_fingerprint: {str(e)}")
        return jsonify({'error': f'Error matching fingerprint: {str(e)}'}), 500

@app.route('/fingerprints')
@login_required
def list_fingerprints():
    fingerprints = Fingerprint.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        'id': f.id,
        'image_path': f.image_path,
        'processed_path': f.processed_path,
        'quality_score': f.quality_score,
        'created_at': f.created_at.isoformat()
    } for f in fingerprints])

@app.route('/status')
@login_required
def check_status():
    """Check system status and database contents"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Count fingerprints
        fingerprint_count = Fingerprint.query.count()
        user_count = User.query.count()
        
        return jsonify({
            'status': 'ok',
            'fingerprints_in_database': fingerprint_count,
            'users_in_database': user_count,
            'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
            'processed_folder': os.path.exists(app.config['PROCESSED_FOLDER'])
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000) 