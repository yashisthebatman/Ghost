import os
import pandas as pd
import numpy as np
import math
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

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data/processed/ghost_laps"

# --- DATA CLEANING ---
def sanitize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales raw sensor data (integers) to physical units (float).
    """
    # 1. Fix Time (Force 100Hz if timestamps look wrong)
    if len(df) > 0:
        # Create synthetic 100Hz time index
        df['time'] = np.arange(len(df)) * 0.01
    
    # 2. Fix Speed (Raw ~16000 -> KPH ~260)
    if 'speed' in df.columns and df['speed'].max() > 500:
        scale = df['speed'].max() / 260.0
        df['speed'] = df['speed'] / scale
        
    # 3. Fix Throttle (Raw ~4000 -> %)
    if 'ath' in df.columns and df['ath'].max() > 200:
        scale = df['ath'].max() / 100.0
        df['ath'] = df['ath'] / scale

    # 4. Fix Brake (Raw ~7000 -> Bar)
    if 'pbrake_f' in df.columns and df['pbrake_f'].max() > 200:
        scale = df['pbrake_f'].max() / 130.0
        df['pbrake_f'] = df['pbrake_f'] / scale

    # 5. Fix Steering (Raw -> Degrees)
    if 'Steering_Angle' in df.columns and df['Steering_Angle'].abs().max() > 1000:
        scale = df['Steering_Angle'].abs().max() / 450.0
        df['Steering_Angle'] = df['Steering_Angle'] / scale

    return df

def load_telemetry_file(filename):
    """Loads and sanitizes a parquet file."""
    path = DATA_DIR / filename
    
    # Retry logic for spaces vs underscores
    if not path.exists():
        alt_name = filename.replace("_", " ")
        path = DATA_DIR / alt_name
    
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df = sanitize_telemetry(df)
            
            # Downsample for UI performance
            if len(df) > 1500:
                step = len(df) // 1000
                df = df.iloc[::step].copy()

            df = df.fillna(0).replace([np.inf, -np.inf], 0)
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    return None

# --- ENDPOINTS ---

@app.get("/")
def health():
    return {"status": "online"}

@app.get("/session/context")
def get_context(db: Session = Depends(get_db)):
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    default = {
        "vehicle_id": "GR86-002-2",
        "session_name": "Road America Analysis",
        "weather": {"track_temp": 32, "condition": "OPTIMAL"}
    }
    if not session: return default
    return {
        "vehicle_id": session.vehicle_id,
        "session_name": session.session_name,
        "weather": session.weather_data or default['weather']
    }

@app.get("/session/laps")
def get_laps_list():
    # Returns the list of laps we generated in Step 1
    return [
        {"lap_number": 1, "lap_time": 70.542, "status": "WARMUP"},
        {"lap_number": 2, "lap_time": 65.120, "status": "TRAFFIC"},
        {"lap_number": 3, "lap_time": 60.000, "status": "PB"}
    ]

@app.get("/laps/ghost")
def get_ghost():
    data = load_telemetry_file("ghost_lap_final.parquet")
    if not data: raise HTTPException(404, "Ghost file missing")
    return data

@app.get("/laps/actual/{lap_id}")
def get_real_lap(lap_id: int):
    # Loads real_lap_1.parquet, real_lap_2.parquet, etc.
    filename = f"real_lap_{lap_id}.parquet"
    data = load_telemetry_file(filename)
    if not data:
        # Fallback to best lap if specific missing
        return load_telemetry_file("real_lap.parquet")
    return data

# Legacy endpoint support
@app.get("/laps/best_actual")
def get_best_real():
    return load_telemetry_file("real_lap.parquet")