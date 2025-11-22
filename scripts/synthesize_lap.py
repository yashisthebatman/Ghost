import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sqlalchemy import func
from scipy.ndimage import zoom

# Boilerplate setup
sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import MicroSectors

# --- CONFIG ---
DATA_DIR = Path("/data/processed")
MODEL_DIR = Path("/data/models") 
OUTPUT_DIR = DATA_DIR / "ghost_laps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH = DATA_DIR / "scaler.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL ARCHITECTURE ---
class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim, n_features, n_conditions, hidden_dim, n_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.cond_embedding = nn.Sequential(nn.Linear(n_conditions, latent_dim), nn.LeakyReLU(0.2))
        self.lstm = nn.LSTM(input_size=latent_dim * 2, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, n_features), nn.Tanh())

    def forward(self, z, conditions):
        z_exp = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        c_exp = self.cond_embedding(conditions).unsqueeze(1).repeat(1, self.seq_len, 1)
        return self.linear(self.lstm(torch.cat([z_exp, c_exp], dim=2))[0])

def load_model(name):
    path = MODEL_DIR / f"{name}_generator_final.pth"
    if not path.exists(): return None
    model = LSTMGenerator(64, 7, 2, 256, 2, 100).to(DEVICE)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    except: return None

def synthesize():
    print("\n🏁 STARTING PHYSICS-CORRECTED SYNTHESIS...")
    
    if not SCALER_PATH.exists():
        print("❌ Scaler missing.")
        return
    scaler = joblib.load(SCALER_PATH)

    # 1. Load Models
    models = {
        "Turn": load_model("Turn"),
        "Straight": load_model("Straight"),
        "Braking": load_model("Braking"),
        "Acceleration": load_model("Acceleration"),
    }
    default_model = next((m for m in models.values() if m), None)
    if not default_model:
        print("❌ No models found.")
        return

    # 2. Query Track Data
    db = SessionLocal()
    best_lap = db.query(MicroSectors.lap_number)\
                 .group_by(MicroSectors.lap_number)\
                 .order_by(func.count(MicroSectors.id).desc())\
                 .first()
    
    if not best_lap: return
    target_lap = best_lap[0]
    
    # ORDER BY ID IS CRITICAL FOR TRACK SHAPE
    sectors = db.query(MicroSectors)\
                .filter(MicroSectors.lap_number == target_lap)\
                .order_by(MicroSectors.id)\
                .all()
    
    print(f"✅ Processing {len(sectors)} sectors for Lap {target_lap}")
    
    # 3. Generate & Resample
    ghost_data = []
    
    with torch.no_grad():
        for sector in sectors:
            active_model = models.get(sector.sector_type, default_model)
            
            # Physics Conditioning
            target_time = float(sector.time_delta)
            entry_speed = float(sector.entry_speed)
            
            cond = torch.tensor([[target_time, entry_speed]], dtype=torch.float32).to(DEVICE)
            z = torch.randn(1, 64).to(DEVICE)
            
            # 1. Raw Output (Always 100 points / 1.0 sec duration in latent space)
            fake_raw = active_model(z, cond).cpu().numpy()[0] # Shape: (100, 7)
            
            # 2. Time Resampling (CRITICAL FIX)
            # We need to squash/stretch 100 points to match 'target_time'
            # Assuming 100Hz target rate:
            target_points = max(int(target_time * 100), 5) # Min 5 points to prevent collapse
            
            # Calculate zoom factor (e.g., 0.5 to shrink 100 -> 50)
            zoom_factor = target_points / 100.0
            
            # Resample along axis 0 (time), keep axis 1 (features)
            # We use order=1 (linear interpolation) for smoothness
            fake_resampled = zoom(fake_raw, (zoom_factor, 1), order=1)
            
            ghost_data.append(fake_resampled)

    # 4. Stitch & Physics Correction
    if ghost_data:
        full_lap_norm = np.concatenate(ghost_data, axis=0)
        full_lap_phys = scaler.inverse_transform(full_lap_norm)
        
        cols = ['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']
        df = pd.DataFrame(full_lap_phys, columns=cols)
        
        # --- FIX 1: Speed Units (Raw -> KPH) ---
        max_speed = df['speed'].max()
        if max_speed > 500:
            print(f"   ⚠️ Detected Raw Speed Units (Max: {max_speed:.0f}). Scaling to KPH...")
            scale_factor = max_speed / 270.0 # Assume GT3/Cup car max ~270kph
            df['speed'] = df['speed'] / scale_factor
        
        # --- FIX 2: Throttle/Brake Units ---
        if df['ath'].max() > 200:
             df['ath'] = df['ath'] / (df['ath'].max() / 100.0)
        
        if df['pbrake_f'].max() > 200:
             df['pbrake_f'] = df['pbrake_f'] / (df['pbrake_f'].max() / 100.0) # Normalize to ~100 bar/percent

        # --- FIX 3: Cleanup ---
        df['speed'] = df['speed'].clip(lower=0)
        df['ath'] = df['ath'].clip(0, 100)
        df['pbrake_f'] = df['pbrake_f'].clip(lower=0)
        
        # Time Index (100Hz)
        df['time'] = np.arange(len(df)) * 0.01
        
        save_path = OUTPUT_DIR / "ghost_lap_final.parquet"
        df.to_parquet(save_path)
        
        final_duration = df['time'].iloc[-1]
        print(f"🎉 GHOST LAP SAVED")
        print(f"   ⏱️ Actual Duration: {final_duration:.2f}s")
        print(f"   🚀 Top Speed: {df['speed'].max():.1f} KPH")
        print(f"   📂 {save_path}")
    
    db.close()

if __name__ == "__main__":
    synthesize()