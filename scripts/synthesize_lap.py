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
from scipy.signal import savgol_filter

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

# --- PHYSICS CONSTANTS FOR NORMALIZATION ---
# These must match the approximate range seen during training
MAX_SPEED_PHYSICAL = 265.0 # KPH
MAX_TIME_DELTA = 10.0      # Seconds

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
    print("\n🏁 STARTING AI SYNTHESIS (ORGANIC DRIVING MODE)...")
    
    if not SCALER_PATH.exists(): 
        print("❌ Scaler missing.")
        return
    scaler = joblib.load(SCALER_PATH)

    # 1. LOAD MODELS
    models = {
        "Turn": load_model("Turn"),
        "Braking": load_model("Braking"),
        "Straight": load_model("Straight"),
        "Acceleration": load_model("Acceleration")
    }
    default_model = models["Turn"]
    if not default_model: 
        print("❌ Models not found.")
        return

    db = SessionLocal()
    best_lap = db.query(MicroSectors.lap_number).group_by(MicroSectors.lap_number).order_by(func.count(MicroSectors.id).desc()).first()
    
    if not best_lap: return
    
    # 2. GET SECTORS
    sectors = db.query(MicroSectors).filter(MicroSectors.lap_number == best_lap[0]).order_by(MicroSectors.id).all()
    print(f"✅ Processing {len(sectors)} sectors.")
    
    ghost_data = []
    
    # Track the *Physical* exit speed to chain to the next sector
    # Start at a realistic rolling speed (e.g., crossing start line at 160)
    current_phys_speed = 160.0 

    with torch.no_grad():
        for i, sector in enumerate(sectors):
            
            # Select Model
            active_model = models.get(sector.sector_type)
            if not active_model:
                if sector.sector_type == "Acceleration": active_model = models.get("Straight", default_model)
                else: active_model = default_model

            # 3. PREPARE CONDITIONS (NORMALIZED)
            # CRITICAL FIX: We must normalize inputs to [-1, 1] or [0, 1] range
            # otherwise the Tanh activation saturates at 1.0 (Max Speed) instantly.
            
            target_time = float(sector.time_delta)
            
            # Normalize Speed: Map 0-265 KPH -> 0.0-1.0
            norm_entry_speed = np.clip(current_phys_speed / MAX_SPEED_PHYSICAL, 0, 1)
            
            # Normalize Time: Map 0-10s -> 0.0-1.0
            norm_time = np.clip(target_time / MAX_TIME_DELTA, 0, 1)

            # Create Tensor
            cond = torch.tensor([[norm_time, norm_entry_speed]], dtype=torch.float32).to(DEVICE)
            z = torch.randn(1, 64).to(DEVICE) # Random noise for variance
            
            # 4. GENERATE RAW (Normalized -1 to 1)
            fake_norm = active_model(z, cond).cpu().numpy()[0]

            # 5. RESAMPLE TIME
            target_points = max(int(target_time * 100), 5)
            fake_resampled = zoom(fake_raw_norm := fake_norm, (target_points/100.0, 1), order=1)
            
            # 6. INVERSE TRANSFORM THIS CHUNK IMMEDIATELEY
            # We need to know the physical exit speed for the next loop
            chunk_phys = scaler.inverse_transform(fake_resampled)
            
            # 7. UNIT CORRECTION (On the fly)
            # If the scaler outputs raw integers (0-16000), scale down to KPH
            # We check the first value. If it's huge, we scale the whole chunk.
            if np.max(chunk_phys[:, 0]) > 500:
                chunk_phys[:, 0] = chunk_phys[:, 0] / (16500.0 / 265.0) # Approx scaling
                chunk_phys[:, 1] = chunk_phys[:, 1] / (4000.0 / 100.0)  # Throttle
                chunk_phys[:, 2] = chunk_phys[:, 2] / (7000.0 / 100.0)  # Brake

            # Update Chaining Variable for next loop
            # We blend the previous exit with current predicted entry to smooth the transition
            # But primarily we trust the output of this sector's end.
            current_phys_speed = chunk_phys[-1, 0] # Last speed value

            ghost_data.append(chunk_phys)

    # 8. STITCH & POST-PROCESS
    if ghost_data:
        full_lap = np.concatenate(ghost_data, axis=0)
        
        # DataFrame Construction
        cols = ['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']
        df = pd.DataFrame(full_lap, columns=cols)
        
        # 9. ORGANIC SMOOTHING (The "Human" Touch)
        # Raw GAN output can be jittery. We apply a Savitzky-Golay filter
        # Window length 15 (0.15s), Polyorder 3 preserves peaks but kills jitter.
        try:
            df['speed'] = savgol_filter(df['speed'], window_length=15, polyorder=3)
            df['ath'] = savgol_filter(df['ath'], window_length=11, polyorder=3)
            df['pbrake_f'] = savgol_filter(df['pbrake_f'], window_length=11, polyorder=3)
        except:
            pass # Skip if lap is too short for filter

        # 10. CLAMPING (Physics Limits)
        df['speed'] = df['speed'].clip(lower=0)
        df['ath'] = df['ath'].clip(0, 100)
        df['pbrake_f'] = df['pbrake_f'].clip(lower=0)
        
        # Time Index
        df['time'] = np.arange(len(df)) * 0.01
        
        save_path = OUTPUT_DIR / "ghost_lap_final.parquet"
        df.to_parquet(save_path)
        
        print(f"🎉 ORGANIC GHOST LAP GENERATED: {df['time'].iloc[-1]:.2f}s")
        print(f"   -> Top Speed: {df['speed'].max():.1f} KPH")
        print(f"   -> Average Speed: {df['speed'].mean():.1f} KPH")

    db.close()

if __name__ == "__main__":
    synthesize()