import os
import pandas as pd
import numpy as np
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db
from models import Sessions, LapSummaries, MicroSectors, GeneratedLaps

app = FastAPI(title="Ghost in the Machine API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATHS ---
if os.path.exists("/app/data/processed"):
    DATA_DIR = Path("/app/data/processed/ghost_laps") 
else:
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data/processed/ghost_laps"

STATIC_DIR = Path("backend/app/static")
if not STATIC_DIR.exists():
    STATIC_DIR = Path("/app/backend/app/static")
    if not STATIC_DIR.exists():
        STATIC_DIR = Path("static_assets")
        STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- CRITICAL FIX: NAN SANITIZER ---
def clean_float(val):
    """Converts NaN/Inf to 0.0 or None to ensure JSON compliance."""
    if val is None:
        return 0.0
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return 0.0
        return f_val
    except (ValueError, TypeError):
        return 0.0

def load_telemetry_file(filename: str):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        # Fallbacks
        if "ghost" in filename: file_path = DATA_DIR / "ghost_lap_final.parquet"
        if "real" in filename: file_path = DATA_DIR / "real_lap.parquet"
        
    if not file_path.exists(): return None
        
    try:
        df = pd.read_parquet(file_path)
        # Downsample
        df_ui = df.iloc[::10].copy()
        
        # REPLACE ALL NaNs/Infs GLOBALLY IN DATAFRAME
        df_ui = df_ui.replace([np.inf, -np.inf], 0)
        df_ui = df_ui.fillna(0)
        
        # Round
        df_ui = df_ui.round(4)
        
        return df_ui.to_dict(orient="records")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Online"}

@app.get("/laps/best_actual")
def get_best_actual_lap():
    data = load_telemetry_file("real_lap.parquet")
    if not data: raise HTTPException(404, "Real lap not found")
    return data

@app.get("/laps/ghost")
def get_ghost_lap():
    data = load_telemetry_file("ghost_lap.parquet")
    if not data: raise HTTPException(404, "Ghost lap not found")
    return data

@app.get("/session/context")
def get_session_context(db: Session = Depends(get_db)):
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    if not session:
        return {"vehicle_id": "Unknown", "session_name": "No Data"}
    
    # Sanitize Weather Data JSON
    w = session.weather_data if session.weather_data else {}
    safe_weather = {k: (clean_float(v) if isinstance(v, (int, float)) else v) for k, v in w.items()}

    return {
        "vehicle_id": session.vehicle_id,
        "session_name": session.session_name,
        "weather": safe_weather
    }

@app.get("/session/laps")
def get_lap_list(db: Session = Depends(get_db)):
    session = db.query(Sessions).order_by(Sessions.processed_at.desc()).first()
    if not session: return []
    
    laps = db.query(LapSummaries).filter(LapSummaries.session_id == session.id).order_by(LapSummaries.lap_number).all()
    
    # --- MANUALLY CONSTRUCT & CLEAN RESPONSE ---
    # This prevents the SQLAlchemy object -> JSON conversion from hitting a NaN
    clean_laps = []
    for l in laps:
        clean_laps.append({
            "lap_number": l.lap_number,
            "lap_time": clean_float(l.lap_time),
            "s1_time": clean_float(l.s1_time),
            "s2_time": clean_float(l.s2_time),
            "top_speed_kph": clean_float(l.top_speed_kph)
        })
        
    return clean_laps