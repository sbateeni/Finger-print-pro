# Fingerprint Comparison System

A web-based system for comparing and matching fingerprint images using computer vision and pattern recognition techniques.

## Features

- Upload and compare two fingerprint images
- Automatic image preprocessing and enhancement
- Feature extraction and minutiae point detection
- Accurate fingerprint matching algorithm
- Modern and user-friendly web interface
- Real-time comparison results
- Support for drag-and-drop file upload

## Requirements

- Python 3.7+
- Flask
- OpenCV
- NumPy
- Pillow
- scikit-image
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fingerprint-comparison-system.git
cd fingerprint-comparison-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload two fingerprint images using the web interface
4. Click the "Compare Fingerprints" button to see the results

## Project Structure

```
Fingerprint-Comparison-System/
│
├── static/
│   ├── css/
│   │   └── styles.css          # UI styling
│   ├── js/
│   │   └── main.js            # Frontend functionality
│   └── uploads/               # Upload directory
│       ├── original/          # Original fingerprints
│       └── processed/         # Processed fingerprints
│
├── templates/
│   └── index.html            # Main page template
│
├── fingerprint/
│   ├── preprocessor.py       # Image preprocessing
│   ├── feature_extractor.py  # Feature extraction
│   └── matcher.py           # Fingerprint matching
│
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## Technical Details

### Image Preprocessing
- Grayscale conversion
- Noise reduction
- Contrast enhancement
- Ridge skeletonization

### Feature Extraction
- Minutiae point detection
- Ridge ending detection
- Bifurcation detection
- Angle calculation

### Matching Algorithm
- Spatial distance calculation
- Angle difference computation
- Match score calculation
- False match filtering

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for image processing tools
- scikit-image for advanced image processing algorithms
- Flask framework for web application development 