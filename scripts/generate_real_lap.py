# ==============================================================================
# SCRIPT: GENERATE REAL LAP COMPARISON (FIXED)
# Usage: python scripts/generate_real_lap.py
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Navigate to project root (assuming script is in /scripts)
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data/processed"
SNIPPETS_DIR = DATA_DIR / "snippets"
OUTPUT_DIR = DATA_DIR / "ghost_laps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = DATA_DIR / "scaler.joblib"

def main():
    print(f"📂 Project Root: {BASE_DIR}")
    
    # 1. Intelligent Metadata Detection
    meta_path_1 = DATA_DIR / "snippets_metadata.csv"
    meta_path_2 = DATA_DIR / "metadata.csv"
    
    final_meta_path = None
    
    if meta_path_1.exists():
        final_meta_path = meta_path_1
        print(f"✅ Found metadata: {final_meta_path.name}")
    elif meta_path_2.exists():
        final_meta_path = meta_path_2
        print(f"✅ Found metadata (fallback): {final_meta_path.name}")
    else:
        print(f"❌ Error: Metadata not found in {DATA_DIR}")
        print("   Expected 'snippets_metadata.csv' or 'metadata.csv'")
        return

    if not SCALER_PATH.exists():
        print(f"❌ Error: scaler.joblib not found in {DATA_DIR}")
        return

    print("   Loading Metadata & Scaler...")
    metadata = pd.read_csv(final_meta_path)
    scaler = joblib.load(SCALER_PATH)

    # 2. Sequence Selection
    # CRITICAL: This must match the sequence used by the AI Synthesis Engine
    start_idx = 25
    end_idx = start_idx + 60
    
    if len(metadata) < end_idx:
        print(f"⚠️ Metadata shorter than expected ({len(metadata)} rows). Using full length.")
        canonical_sequence = metadata
    else:
        canonical_sequence = metadata.iloc[start_idx:end_idx]

    print(f"🧵 Stitching {len(canonical_sequence)} snippets...")
    real_lap_data = []

    # 3. Stitching Loop
    for idx, row in canonical_sequence.iterrows():
        # Robust filename handling (fixes Windows/Linux path separator issues)
        raw_path_str = row['snippet_path']
        # If the path string contains slashes, split it; otherwise just take it
        if "/" in str(raw_path_str):
            filename = raw_path_str.split("/")[-1]
        elif "\\" in str(raw_path_str):
            filename = raw_path_str.split("\\")[-1]
        else:
            filename = raw_path_str
            
        path = SNIPPETS_DIR / filename
        
        try:
            if not path.exists():
                print(f"   ⚠️ Warning: Snippet {filename} missing. Filling with zeros.")
                real_lap_data.append(np.zeros((100, 7)))
                continue
                
            raw = np.load(path)
            
            # Architecture Constraint: Must be 100 steps
            if len(raw) > 100: 
                d = raw[:100]
            elif len(raw) < 100:
                # Edge padding prevents jumps at seams
                pad_width = 100 - len(raw)
                d = np.pad(raw, ((0, pad_width), (0,0)), 'edge')
            else:
                d = raw
                
            real_lap_data.append(d)
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            real_lap_data.append(np.zeros((100, 7)))

    if not real_lap_data:
        print("❌ Fatal: No data loaded.")
        return

    # 4. Final Assembly
    real_stitched = np.concatenate(real_lap_data, axis=0)
    
    print("   Inverse Scaling to Real Units...")
    real_phys = scaler.inverse_transform(real_stitched)

    cols = ['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']
    df_real = pd.DataFrame(real_phys, columns=cols)
    df_real['time'] = np.arange(len(df_real)) * 0.01
    
    # Physics Cleanup
    df_real['speed'] = df_real['speed'].clip(lower=0)
    
    # 5. Save
    save_path = OUTPUT_DIR / "real_lap.parquet"
    df_real.to_parquet(save_path)
    print(f"✅ SUCCESS: Real Lap saved to {save_path}")

if __name__ == "__main__":
    main()