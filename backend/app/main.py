import os
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import get_db
from models import Sessions, LapSummaries

app = FastAPI(title="Ghost Engineer API")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Resolves to /app inside the Docker container
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data/processed/ghost_laps"

# --- UTILITY FUNCTIONS ---
def load_telemetry_file(filename):
    """
    Loads a parquet file from the data directory.
    Sanitizes data (NaNs, Infs) and ensures a time index exists.
    """
    path = DATA_DIR / filename
    
    # Robustness: Try replacing underscores with spaces if file not found initially
    if not path.exists():
        alt_name = filename.replace("_", " ")
        path = DATA_DIR / alt_name
    
    if path.exists():
        try:
            df = pd.read_parquet(path)
            
            # 1. Sanitize Numerical Errors
            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            
            # 2. Ensure Time Column exists (Critical for Frontend Charts)
            if 'time' not in df.columns:
                # Create synthetic 100Hz time index if missing
                df['time'] = np.arange(len(df)) * 0.01

            # 3. Return as list of dicts (JSON compatible)
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    return None

# --- ENDPOINTS ---

@app.get("/")
def health():
    """Health check to ensure backend is online."""
    return {"status": "online", "mode": "strict_db", "version": "2.0.0"}

@app.get("/session/context")
def get_context(db: Session = Depends(get_db)):
    """
    Returns the metadata for the latest session (Vehicle ID, Weather).
    Dependent on database population.
    """
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    
    if not session:
        # Return neutral default if DB is empty
        return {
            "vehicle_id": "WAITING_FOR_DATA",
            "session_name": "No Session Loaded",
            "weather": {"track_temp": 0, "condition": "N/A"}
        }
        
    return {
        "vehicle_id": session.vehicle_id,
        "session_name": session.session_name,
        "weather": session.weather_data or {}
    }

@app.get("/session/laps")
def get_laps_list(db: Session = Depends(get_db)):
    """
    Returns the list of available human laps for the sidebar.
    Renames them to 'Human Lap X' format.
    """
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    if not session: 
        return []
    
    laps = db.query(LapSummaries)\
             .filter(LapSummaries.session_id == session.id)\
             .order_by(LapSummaries.lap_number)\
             .all()
    
    return [{
        "id": l.lap_number,                  # Used for API lookups
        "name": f"Human Lap {l.lap_number}", # Display Name
        "lap_time": l.lap_time,
        "s1": l.s1_time,
        "s2": l.s2_time,
        "s3": l.s3_time,
        "status": "PB" if l.lap_number == 3 else "Attempt" # Simplified logic
    } for l in laps]

@app.get("/laps/optimal")
def get_optimal():
    """
    Returns the AI Generated Optimal Lap (Ghost).
    Filename: ghost_lap_final.parquet
    """
    data = load_telemetry_file("ghost_lap_final.parquet")
    if not data: 
        raise HTTPException(status_code=404, detail="Optimal lap generation pending. Run synthesize_lap.py.")
    return data

@app.get("/laps/human/{lap_id}")
def get_human_lap(lap_id: int):
    """
    Returns the telemetry for a specific human attempt.
    Filename: real_lap_{id}.parquet
    """
    filename = f"real_lap_{lap_id}.parquet"
    
    data = load_telemetry_file(filename)
    
    # Fallback: Sometimes the 'best' lap is named 'real_lap.parquet'
    # If specific ID fails and it is lap 3 (usually the PB in our scripts), try the generic name
    if not data and lap_id == 3:
        data = load_telemetry_file("real_lap.parquet")

    if not data:
         raise HTTPException(status_code=404, detail=f"Human Lap {lap_id} not found on disk.")
    return data