# ==========================================
# FILE: scripts/generate_session_laps.py
# ==========================================
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Correct Docker Path
DATA_DIR = Path("/data/processed/ghost_laps")
SOURCE_FILE = DATA_DIR / "real_lap.parquet"

def generate_laps():
    print("--- GENERATING PHYSICAL SESSION FILES ---")
    
    # 1. Load Source
    if not SOURCE_FILE.exists():
        # Try fallback
        SOURCE_FILE_ALT = DATA_DIR / "real lap.parquet"
        if SOURCE_FILE_ALT.exists():
            df_best = pd.read_parquet(SOURCE_FILE_ALT)
        else:
            print(f"❌ Source file missing at {SOURCE_FILE}")
            sys.exit(1)
    else:
        df_best = pd.read_parquet(SOURCE_FILE)

    print(f"   Loaded Source: {len(df_best)} rows")

    # 2. Define Session Laps
    # We mathematically alter the best lap to create previous attempts
    laps_config = [
        {"id": 1, "type": "WARMUP", "speed_factor": 0.85, "noise": 500}, # High noise (unstable)
        {"id": 2, "type": "TRAFFIC", "speed_factor": 0.92, "noise": 200}, # Med noise
        {"id": 3, "type": "PB",      "speed_factor": 1.00, "noise": 0},   # Pure data
    ]

    for lap in laps_config:
        print(f"   Generating Lap {lap['id']} ({lap['type']})...")
        
        df_new = df_best.copy()
        
        # Apply Physics (Working with Raw Integers 16000 range)
        # 1. Scale Speed
        df_new['speed'] = df_new['speed'] * lap['speed_factor']
        
        # 2. Add Noise
        if lap['noise'] > 0:
            noise = np.random.normal(0, lap['noise'], size=len(df_new))
            df_new['speed'] = df_new['speed'] + noise
            df_new['speed'] = df_new['speed'].clip(lower=0)

        # 3. Time Dilation (Slower lap = more time)
        # Since we can't easily add rows, we just slow down the timestamp
        if 'time' in df_new.columns:
             df_new['time'] = df_new['time'] * (1 / lap['speed_factor'])

        # Save
        filename = f"real_lap_{lap['id']}.parquet"
        save_path = DATA_DIR / filename
        df_new.to_parquet(save_path)
        print(f"      -> Saved: {filename}")

    print("✅ Session files generated.")

if __name__ == "__main__":
    generate_laps()