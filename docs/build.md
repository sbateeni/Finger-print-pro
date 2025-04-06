# Building the Fingerprint Recognition System

This document provides instructions for building and setting up the fingerprint recognition system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- SQLite
- CUDA-capable GPU (optional, for faster deep learning processing)

## Installation Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd fingerprint-pro
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install backend dependencies:
```bash
cd backend/python
pip install -r requirements.txt
```

4. Install frontend dependencies:
```bash
cd ../../frontend
pip install -r requirements.txt
```

5. Initialize the database:
```bash
cd ../backend/python/database
python init_db.py
```

## Building the Deep Learning Model

1. Prepare the training data:
```bash
cd ../../datasets
python prepare_data.py
```

2. Train the model:
```bash
cd ../python
python train_model.py
```

3. The trained model will be saved in `models/fingerprint_cnn.pth`

## Configuration

The system can be configured by modifying the following files:

- `backend/python/.env`: Environment variables for the backend
- `frontend/.env`: Environment variables for the frontend
- `backend/python/config.py`: Configuration settings for the fingerprint processing

## Building for Production

1. Set up a production environment:
```bash
export FLASK_ENV=production
```

2. Configure the web server (e.g., Nginx, Apache) to serve the frontend and proxy requests to the backend.

3. Set up SSL certificates for secure communication.

4. Configure the database for production use (e.g., PostgreSQL instead of SQLite).

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all required packages are installed
   - Check the requirements.txt files for version compatibility

2. **Database Issues**
   - Verify SQLite is installed and accessible
   - Check database permissions
   - Ensure the database directory exists and is writable

3. **Model Training Issues**
   - Verify CUDA is properly installed if using GPU
   - Check training data format and structure
   - Ensure sufficient disk space for model storage

### Getting Help

- Check the project's issue tracker for known issues
- Consult the documentation in the `docs/` directory
- Contact the development team for support

## Contributing

See `docs/contribute.md` for guidelines on contributing to the project. 