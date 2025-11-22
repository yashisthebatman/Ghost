import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sqlalchemy import func

sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import MicroSectors, Sessions, LapSummaries

# --- CONFIG ---
DATA_DIR = Path("/data/processed")
OUTPUT_DIR = DATA_DIR / "ghost_laps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH = DATA_DIR / "scaler.joblib"

def main():
    print("\n📸 PROCESSING REAL LAP & SESSION HISTORY...")
    
    if not SCALER_PATH.exists():
        print("❌ Scaler not found.")
        return
    scaler = joblib.load(SCALER_PATH)
    db = SessionLocal()

    # 1. Find the Reference Lap (Must match synthesize_lap logic)
    best_lap = db.query(MicroSectors.lap_number)\
                 .group_by(MicroSectors.lap_number)\
                 .order_by(func.count(MicroSectors.id).desc())\
                 .first()
    
    if not best_lap: return
    target_lap_num = best_lap[0]

    # 2. Extract & Stitch
    sectors = db.query(MicroSectors).filter(MicroSectors.lap_number == target_lap_num).order_by(MicroSectors.id).all()
    
    real_data_list = []
    for sector in sectors:
        path = Path(sector.snippet_path)
        if path.exists():
            raw = np.load(path)
            # Force 100 length alignment
            if len(raw) != 100:
                if len(raw) > 100: d = raw[:100]
                else: d = np.pad(raw, ((0, 100-len(raw)), (0,0)), 'edge')
                real_data_list.append(d)
            else:
                real_data_list.append(raw)

    if not real_data_list: return

    # 3. Save "Best Actual" Lap
    real_stitched = np.concatenate(real_data_list, axis=0)
    real_phys = scaler.inverse_transform(real_stitched)
    
    cols = ['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']
    df_best = pd.DataFrame(real_phys, columns=cols)
    df_best['time'] = np.arange(len(df_best)) * 0.01
    
    df_best.to_parquet(OUTPUT_DIR / "real_lap.parquet")
    print(f"✅ Saved Reference Lap: {len(df_best)/100:.2f}s")

    # 4. Generate Session History (Warmup, Traffic)
    # This creates the files needed for the UI list
    laps_config = [
        {"n": 1, "type": "WARMUP", "factor": 0.85},
        {"n": 2, "type": "TRAFFIC", "factor": 0.92},
        {"n": 3, "type": "PB",      "factor": 1.00}
    ]
    
    # Update DB with these new laps
    session = db.query(Sessions).first()
    db.query(LapSummaries).filter(LapSummaries.session_id == session.id).delete()

    for l in laps_config:
        df_new = df_best.copy()
        
        # Apply Physics
        df_new['speed'] = df_new['speed'] * l['factor']
        # Time Dilation
        df_new['time'] = df_new['time'] / l['factor']
        final_time = df_new['time'].iloc[-1]
        
        # Save File
        df_new.to_parquet(OUTPUT_DIR / f"real_lap_{l['n']}.parquet")
        
        # Add to DB
        db.add(LapSummaries(
            session_id=session.id,
            lap_number=l['n'],
            lap_time=final_time,
            s1_time=final_time * 0.3,
            s2_time=final_time * 0.3,
            s3_time=final_time * 0.4,
            top_speed_kph=df_new['speed'].max()
        ))
        print(f"   -> Generated Lap {l['n']} ({final_time:.2f}s)")

    db.commit()
    db.close()
    print("🎉 Session History Complete.")

if __name__ == "__main__":
    main()