# Fingerprint Recognition System

A comprehensive fingerprint recognition system that combines traditional biometric algorithms with deep learning approaches for accurate fingerprint matching.

## Project Structure

The project is organized into several main components:

- `backend/`: Contains the core fingerprint processing and matching algorithms
- `frontend/`: Web interface for user interaction
- `datasets/`: Storage for fingerprint datasets
- `tests/`: Unit and integration tests
- `docs/`: Project documentation
- `api/`: API definitions and implementations

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js (for frontend development)
- SQLite (for development database)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd fingerprint-pro
```

2. Set up the backend:
```bash
cd backend/python
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
pip install -r requirements.txt
```

4. Initialize the database:
```bash
cd backend/python/database
python init_db.py
```

### Running the Application

1. Start the backend server:
```bash
cd backend/python
python app.py
```

2. Start the frontend server:
```bash
cd frontend
python app.py
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- `build.md`: Build and compilation instructions
- `run.md`: Running and deployment instructions
- `contribute.md`: Contribution guidelines
- `preprocessing.md`: Preprocessing module documentation
- `architecture.md`: System architecture overview
- `api_reference.md`: API documentation
- `testing.md`: Testing guidelines

## Contributing

Please read `docs/contribute.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 