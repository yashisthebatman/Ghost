import sys
import os

# This is the crucial change. It adds the container's WORKDIR ('/app')
# to the Python path, so it can find the 'app' module.
# The 'backend' directory on the host is mounted as '/app' in the container.
sys.path.insert(0, '/app')

from app.database import engine
from app.models import Base

def initialize_database():
    print("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"An error occurred while creating tables: {e}")

if __name__ == "__main__":
    initialize_database()