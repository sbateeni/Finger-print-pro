from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Backend API URL
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'fingerprint' not in request.files:
        return jsonify({"error": "No fingerprint image provided"}), 400
    
    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Forward the file to the backend
        files = {'fingerprint': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{BACKEND_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": "Backend processing failed"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match():
    if 'fingerprint' not in request.files:
        return jsonify({"error": "No fingerprint image provided"}), 400
    
    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Forward the file to the backend
        files = {'fingerprint': (file.filename, file.stream, file.content_type)}
        response = requests.post(f"{BACKEND_URL}/api/match", files=files)
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": "Backend matching failed"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 