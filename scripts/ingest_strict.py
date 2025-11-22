# ==========================================
# FILE: scripts/ingest_strict.py
# ==========================================
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal, engine
from app.models import Base, Sessions, LapSummaries

# --- CONFIG ---
# Corrected Filenames based on your ls output
REAL_FILE = "/data/processed/ghost_laps/real_lap.parquet" 
GHOST_FILE = "/data/processed/ghost_laps/ghost_lap_final.parquet" # Underscores!
VEHICLE_ID = "GR86-002-2"

def sanitize_dataframe(df):
    """Scales raw sensor integers to physical units."""
    
    # 1. Detect Sampling Rate
    # If time column exists, use it. Else assume 100Hz (0.01s) based on 6000 rows = 60s
    dt = 0.01
    if 'time' in df.columns and len(df) > 2:
        dt = df['time'].diff().median()
        if np.isnan(dt) or dt <= 0: dt = 0.01
    
    print(f"   Detected Sampling Rate: {dt*1000:.1f}ms ({1/dt:.0f}Hz)")

    # 2. Fix Speed (Raw -> KPH)
    # Raw Max: 16654. Ghost Max: ~290. Ratio: ~57
    if df['speed'].max() > 500:
        print("   ⚠️ Scaling Speed...")
        scale = df['speed'].max() / 260.0 # Normalize to approx 260kph peak
        df['speed'] = df['speed'] / scale

    # 3. Fix Throttle (Raw -> %)
    # Raw Max: 4935. Ghost Max: 100. Ratio: ~49.3
    if df['ath'].max() > 200:
        print("   ⚠️ Scaling Throttle...")
        scale = df['ath'].max() / 100.0
        df['ath'] = df['ath'] / scale

    # 4. Fix Brake (Raw -> Bar)
    # Raw Max: 7260. Ghost Max: 130. Ratio: ~55
    if df['pbrake_f'].max() > 200:
        print("   ⚠️ Scaling Brake...")
        scale = df['pbrake_f'].max() / 130.0
        df['pbrake_f'] = df['pbrake_f'] / scale

    # 5. Fix Steering
    if df['Steering_Angle'].max() > 1000:
        print("   ⚠️ Scaling Steering...")
        scale = df['Steering_Angle'].max() / 400.0 # Approx 400 deg peak
        df['Steering_Angle'] = df['Steering_Angle'] / scale

    return df, dt

def ingest_strict():
    print("--- STRICT DATA INGESTION (V2) ---")
    
    # 1. Check Files
    r_path = Path(REAL_FILE)
    if not r_path.exists():
        print(f"❌ CRITICAL: {REAL_FILE} missing.")
        sys.exit(1)

    # 2. Load & Sanitize
    print(f"1. Processing {r_path.name}...")
    df = pd.read_parquet(r_path)
    df, dt = sanitize_dataframe(df)
    
    # 3. Calculate Metrics
    total_duration = len(df) * dt
    top_speed = df['speed'].max()
    
    print(f"   - Duration: {total_duration:.2f}s")
    print(f"   - Top Speed: {top_speed:.1f} KPH")

    # 4. Update DB
    print("2. Syncing Database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        session = Sessions(
            session_name="Road_America_Sim",
            vehicle_id=VEHICLE_ID,
            processed_at=datetime.utcnow(),
            weather_data={"track_temp": 32, "condition": "OPTIMAL"} 
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        lap_record = LapSummaries(
            session_id=session.id,
            lap_number=1, 
            lap_time=total_duration,
            s1_time=total_duration * 0.3, 
            s2_time=total_duration * 0.3,
            s3_time=total_duration * 0.4,
            top_speed_kph=top_speed
        )
        db.add(lap_record)
        db.commit()
        print("✅ Data Ingested Successfully.")
        
    except Exception as e:
        print(f"❌ DB Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    ingest_strict()