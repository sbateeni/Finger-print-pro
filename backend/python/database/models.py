from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Fingerprint(Base):
    __tablename__ = 'fingerprints'
    
    id = Column(Integer, primary_key=True)
    image_path = Column(String(255), nullable=False)
    quality_score = Column(Float, nullable=False)
    core_x = Column(Float)
    core_y = Column(Float)
    delta_x = Column(Float)
    delta_y = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    minutiae = relationship("Minutiae", back_populates="fingerprint")
    matches = relationship("Match", back_populates="fingerprint")

class Minutiae(Base):
    __tablename__ = 'minutiae'
    
    id = Column(Integer, primary_key=True)
    fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'), nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    angle = Column(Float, nullable=False)
    type = Column(String(20), nullable=False)  # 'ridge_ending' or 'bifurcation'
    quality = Column(Float, nullable=False)
    
    # Relationships
    fingerprint = relationship("Fingerprint", back_populates="minutiae")

class Match(Base):
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'), nullable=False)
    matched_fingerprint_id = Column(Integer, ForeignKey('fingerprints.id'), nullable=False)
    score = Column(Float, nullable=False)
    algorithm = Column(String(50), nullable=False)  # 'minutiae', 'deep_learning', or 'hybrid'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    fingerprint = relationship("Fingerprint", foreign_keys=[fingerprint_id], back_populates="matches")
    matched_fingerprint = relationship("Fingerprint", foreign_keys=[matched_fingerprint_id]) 