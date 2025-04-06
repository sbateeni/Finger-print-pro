from models import db, Fingerprint, Minutiae, Match
import os

def init_db():
    """
    Initialize the database and create necessary tables.
    """
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname('database.db'), exist_ok=True)
        
        # Create all tables
        db.create_all()
        
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

if __name__ == '__main__':
    init_db() 