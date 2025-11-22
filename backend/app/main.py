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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data/processed/ghost_laps"

# --- UTILS ---
def load_telemetry_file(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        if 'time' not in df.columns:
            df['time'] = np.arange(len(df)) * 0.01
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# --- ENDPOINTS ---

@app.get("/")
def health():
    return {"status": "online", "mode": "strict_db"}

@app.get("/session/context")
def get_context(db: Session = Depends(get_db)):
    # Strictly fetch from DB
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    if not session:
        # Return empty structure if no session exists
        return {
            "vehicle_id": "WAITING_FOR_DATA",
            "session_name": "--",
            "weather": {"track_temp": 0, "condition": "--"}
        }
    return {
        "vehicle_id": session.vehicle_id,
        "session_name": session.session_name,
        "weather": session.weather_data
    }

@app.get("/session/laps")
def get_laps_list(db: Session = Depends(get_db)):
    """Returns ONLY the laps present in the database."""
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    if not session: 
        return []
    
    laps = db.query(LapSummaries)\
             .filter(LapSummaries.session_id == session.id)\
             .order_by(LapSummaries.lap_number)\
             .all()
    
    return [{
        "lap_number": l.lap_number,
        "lap_time": l.lap_time,
        "s1": l.s1_time,
        "s2": l.s2_time,
        "s3": l.s3_time,
        "status": "PB" if l.lap_number == 3 else "Attempt" # You can refine logic here
    } for l in laps]

@app.get("/laps/optimal")
def get_optimal():
    """Renamed from /laps/ghost to be specific"""
    data = load_telemetry_file("ghost_lap_final.parquet")
    if not data: 
        raise HTTPException(404, "Optimal lap generation pending.")
    return data

@app.get("/laps/human/{lap_id}")
def get_human_lap(lap_id: int):
    """Fetches a specific human attempt"""
    filename = f"real_lap_{lap_id}.parquet"
    # Handle PB naming convention if your script saves the best as 'real_lap.parquet'
    # For this logic, we assume register_laps_db synced filenames correctly.
    data = load_telemetry_file(filename)
    
    # Fallback for the "best" lap if file naming is inconsistent
    if not data and lap_id == 3:
        data = load_telemetry_file("real_lap.parquet")

    if not data:
         raise HTTPException(404, f"Lap {lap_id} not found.")
    return data