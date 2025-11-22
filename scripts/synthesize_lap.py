import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sqlalchemy import func

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
    print("\n🏁 STARTING FULL LAP SYNTHESIS (2:04 TARGET)...")
    
    if not SCALER_PATH.exists():
        print("❌ Scaler missing. Run process_telemetry.py first.")
        return
    scaler = joblib.load(SCALER_PATH)

    # 1. Load Models (Context Switching)
    models = {
        "Turn": load_model("Turn"),
        "Straight": load_model("Straight"),
        "Braking": load_model("Braking"),
        "Acceleration": load_model("Acceleration"),
    }
    default_model = next((m for m in models.values() if m), None)
    if not default_model:
        print("❌ No models found in /data/models.")
        return

    # 2. Find the Reference Lap (The Longest One)
    db = SessionLocal()
    # Query finds the lap_number with the MAXIMUM count of sectors
    best_lap = db.query(MicroSectors.lap_number)\
                 .group_by(MicroSectors.lap_number)\
                 .order_by(func.count(MicroSectors.id).desc())\
                 .first()
    
    if not best_lap:
        print("❌ No sector data in DB.")
        return
    
    target_lap = best_lap[0]
    
    # 3. Fetch ALL sectors for that lap
    sectors = db.query(MicroSectors).filter(MicroSectors.lap_number == target_lap).order_by(MicroSectors.id).all()
    print(f"✅ Selected Reference Lap: {target_lap} ({len(sectors)} sectors)")
    
    # 4. Generate Sequence
    ghost_data = []
    with torch.no_grad():
        for sector in sectors:
            # Intelligent Model Selection
            active_model = models.get(sector.sector_type, default_model)
            
            # Condition: Optimize time by 1%
            target_time = sector.time_delta * 0.99
            
            cond = torch.tensor([[target_time, sector.entry_speed]], dtype=torch.float32).to(DEVICE)
            z = torch.randn(1, 64).to(DEVICE)
            
            fake = active_model(z, cond).cpu().numpy()[0]
            ghost_data.append(fake)

    # 5. Stitch & Save
    if ghost_data:
        full_lap_norm = np.concatenate(ghost_data, axis=0)
        full_lap_phys = scaler.inverse_transform(full_lap_norm)
        
        cols = ['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']
        df = pd.DataFrame(full_lap_phys, columns=cols)
        
        # Clean Physics
        df['speed'] = df['speed'].clip(lower=0)
        df['ath'] = df['ath'].clip(0, 100)
        
        # Time Index (100Hz)
        df['time'] = np.arange(len(df)) * 0.01
        
        save_path = OUTPUT_DIR / "ghost_lap_final.parquet"
        df.to_parquet(save_path)
        
        print(f"🎉 FULL GHOST LAP SAVED: {len(df)/100:.2f}s duration")
    
    db.close()

if __name__ == "__main__":
    synthesize()