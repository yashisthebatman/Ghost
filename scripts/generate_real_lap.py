import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("/data/processed/ghost_laps")
SOURCE_FILE = DATA_DIR / "ghost_lap_final.parquet" # Use the corrected ghost as the base

def main():
    print("\n📸 GENERATING SESSION HISTORY (PHYSICS CORRECTED)...")
    
    if not SOURCE_FILE.exists():
        print(f"❌ Source {SOURCE_FILE} not found. Run synthesize_lap.py first.")
        return

    # 1. Load the NOW CORRECTED Ghost Lap
    df_base = pd.read_parquet(SOURCE_FILE)
    base_time = df_base['time'].iloc[-1]
    print(f"   Base Lap Time: {base_time:.2f}s")

    # 2. Generate Variations (Reverse logic: Real driver is usually SLOWER than AI)
    laps_config = [
        {"n": 1, "type": "WARMUP",  "speed_factor": 0.85, "noise": 2.0},
        {"n": 2, "type": "TRAFFIC", "speed_factor": 0.92, "noise": 1.5},
        {"n": 3, "type": "PB",      "speed_factor": 0.98, "noise": 0.5} # PB is slightly slower than optimal AI
    ]
    
    # Save the "Best Real" file
    df_base.to_parquet(DATA_DIR / "real_lap.parquet")

    for l in laps_config:
        df_new = df_base.copy()
        
        # Apply Physics
        # Slower speed = Higher lap time
        df_new['speed'] = df_new['speed'] * l['speed_factor']
        
        # Recalculate time based on speed drop (approximate)
        # If speed is 0.9x, time is roughly 1/0.9x
        time_factor = 1 / l['speed_factor']
        df_new['time'] = df_new['time'] * time_factor
        
        # Add realistic noise
        if l['noise'] > 0:
            noise = np.random.normal(0, l['noise'], size=len(df_new))
            df_new['speed'] = (df_new['speed'] + noise).clip(lower=0)

        filename = f"real_lap_{l['n']}.parquet"
        df_new.to_parquet(DATA_DIR / filename)
        
        final_t = df_new['time'].iloc[-1]
        print(f"   -> Generated Lap {l['n']}: {final_t:.2f}s")

    print("✅ Session Files Updated.")

if __name__ == "__main__":
    main()