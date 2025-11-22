import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import Sessions, LapSummaries

DATA_DIR = Path("/data/processed/ghost_laps")

def register_real_data():
    print("--- 📡 SYNCING DATABASE WITH GENERATED FILES ---")
    db = SessionLocal()
    session = db.query(Sessions).first()
    
    if not session:
        print("❌ No session found. Run ingestion first.")
        return

    # 1. Clear old dummy data
    db.query(LapSummaries).filter(LapSummaries.session_id == session.id).delete()
    
    # 2. Scan for actual files
    files = {
        "ghost": DATA_DIR / "ghost_lap_final.parquet",
        "lap_1": DATA_DIR / "real_lap_1.parquet",
        "lap_2": DATA_DIR / "real_lap_2.parquet",
        "lap_3": DATA_DIR / "real_lap.parquet" # Best/PB
    }

    laps_to_insert = []

    # Process Real Laps
    for lap_num, (key, path) in enumerate([("lap_1", files['lap_1']), ("lap_2", files['lap_2']), ("lap_3", files['lap_3'])], 1):
        if path.exists():
            df = pd.read_parquet(path)
            
            # DYNAMIC CALCULATION & TYPE CASTING
            # Cast numpy types to native python types for Postgres
            duration = float(df['time'].iloc[-1] - df['time'].iloc[0])
            top_speed = float(df['speed'].max())
            
            # Calculate sector splits (Approximation)
            s1 = float(duration * 0.305)
            s2 = float(duration * 0.345)
            s3 = float(duration - s1 - s2)

            print(f"   📍 Registered Lap {lap_num}: {duration:.3f}s (Top Speed: {top_speed:.1f} kph)")
            
            laps_to_insert.append(LapSummaries(
                session_id=int(session.id),
                lap_number=int(lap_num),
                lap_time=duration,
                s1_time=round(s1, 3),
                s2_time=round(s2, 3),
                s3_time=round(s3, 3),
                top_speed_kph=round(top_speed, 1)
            ))
        else:
            print(f"   ⚠️ Missing file: {path}")

    # 3. Check Ghost File validity
    if files['ghost'].exists():
        df_ghost = pd.read_parquet(files['ghost'])
        ghost_time = float(df_ghost['time'].iloc[-1])
        print(f"   👻 GHOST LAP VERIFIED: {ghost_time:.3f}s")
    
    db.add_all(laps_to_insert)
    db.commit()
    print("✅ Database updated with PHYSICAL lap times.")
    db.close()

if __name__ == "__main__":
    register_real_data()