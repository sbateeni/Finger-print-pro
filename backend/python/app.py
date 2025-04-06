from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Import custom modules
from preprocessing.quality_assessment import assess_quality
from preprocessing.segmentation import segment_fingerprint
from biometric_features.extract_core_delta import extract_core_delta
from matcher import match_fingerprints
from database.models import db, Fingerprint

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_fingerprint():
    if 'fingerprint' not in request.files:
        return jsonify({"error": "No fingerprint image provided"}), 400
    
    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join('temp', file.filename)
        file.save(temp_path)
        
        # Process the fingerprint
        quality_score = assess_quality(temp_path)
        if quality_score < 0.7:  # Threshold for acceptable quality
            return jsonify({"error": "Low quality fingerprint image"}), 400
        
        segmented_image = segment_fingerprint(temp_path)
        core, delta = extract_core_delta(segmented_image)
        
        # Store in database
        fingerprint = Fingerprint(
            image_path=temp_path,
            quality_score=quality_score,
            core_x=core[0] if core else None,
            core_y=core[1] if core else None,
            delta_x=delta[0] if delta else None,
            delta_y=delta[1] if delta else None
        )
        db.session.add(fingerprint)
        db.session.commit()
        
        return jsonify({
            "message": "Fingerprint processed successfully",
            "quality_score": quality_score,
            "core": core,
            "delta": delta
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/match', methods=['POST'])
def match_fingerprint():
    if 'fingerprint' not in request.files:
        return jsonify({"error": "No fingerprint image provided"}), 400
    
    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join('temp', file.filename)
        file.save(temp_path)
        
        # Match against database
        matches = match_fingerprints(temp_path)
        
        return jsonify({
            "matches": matches
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True) 