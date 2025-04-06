from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
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