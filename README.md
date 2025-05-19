# Fingerprint Matching Application

A web application for comparing fingerprint images using OpenCV and SIFT features.

## Features

- Upload and compare two fingerprint images
- Real-time image preview
- Feature point visualization
- Match score calculation
- Clean and responsive UI

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fingerprint-matching.git
cd fingerprint-matching
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to:
```
http://localhost:5000
```

## Deployment on Render

1. Fork this repository to your GitHub account

2. Create a new Web Service on Render:
   - Connect your GitHub repository
   - Select Python as the runtime
   - Use the following settings:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
   - Add environment variables:
     - `PYTHON_VERSION`: 3.9.0
     - `SECRET_KEY`: (auto-generated)

3. Click "Create Web Service"

## Project Structure

```
fingerprint-matching/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render configuration
├── gunicorn.conf.py      # Gunicorn configuration
├── templates/            # HTML templates
│   ├── index.html       # Upload page
│   └── result.html      # Results page
├── static/              # Static files
│   └── style.css        # CSS styles
└── utils/               # Utility functions
    ├── preprocess.py    # Image preprocessing
    ├── extract_features.py  # Feature extraction
    └── match_fingerprint.py # Fingerprint matching
```

## Dependencies

- Flask==2.3.3
- opencv-python==4.8.0.76
- numpy==1.24.3
- scikit-image==0.21.0
- gunicorn==21.2.0
- Pillow==10.0.0
- python-dotenv==1.0.1

## License

MIT License 