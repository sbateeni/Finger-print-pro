# نظام مقارنة البصمات

نظام ويب لمقارنة البصمات وتحديد نسبة التطابق بينها باستخدام تقنيات معالجة الصور والذكاء الاصطناعي.

## المميزات

- معالجة أولية للصور وتحسين جودتها
- استخراج النقاط المميزة من البصمات
- مقارنة البصمات وحساب نسبة التطابق
- عرض مرئي للنقاط المتطابقة
- واجهة مستخدم سهلة الاستخدام باللغة العربية

## المتطلبات

- Python 3.9+
- Flask
- OpenCV
- NumPy
- SciPy
- scikit-image

## التثبيت

1. نسخ المستودع:
```bash
git clone https://github.com/yourusername/fingerprint-comparison.git
cd fingerprint-comparison
```

2. إنشاء بيئة افتراضية:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

## التشغيل

1. تشغيل التطبيق:
```bash
python app.py
```

2. فتح المتصفح على العنوان:
```
http://localhost:5000
```

## الاستخدام

1. رفع صورتي البصمات المراد مقارنتهما
2. النقر على زر "مقارنة البصمات"
3. انتظار النتائج التي تشمل:
   - نسبة التطابق
   - عدد النقاط المميزة في كل بصمة
   - عرض مرئي للنقاط المتطابقة

## النشر على Render

1. إنشاء حساب على [Render](https://render.com)
2. ربط المستودع بـ Render
3. إنشاء خدمة ويب جديدة
4. تحديد المتغيرات البيئية المطلوبة
5. النشر تلقائياً

## الترخيص

هذا المشروع مرخص تحت [MIT License](LICENSE).

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

## Acknowledgments

- OpenCV community for image processing tools
- scikit-image for advanced image processing algorithms
- Flask framework for web application development 