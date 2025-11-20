# ==============================================================================
# SOTA LSTM-GAN TRAINING (LOCAL SHOWCASE VERSION)
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

# --- LOCAL CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CONFIG = {
    "training_params": {
        "batch_size": 64,
        "epochs": 10, # Reduced for demo purposes
        "lr_g": 0.001,
        "lr_c": 0.001,
        "augmentation_factor": 2, # Reduced for demo
    },
    "model_params": {
        "seq_len": 100,
        "n_features": 7,
        "n_conditions": 2,
        "latent_dim": 64,
        "hidden_dim": 256,
        "n_layers": 2,
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Local Training on: {DEVICE}")

# --- MODEL DEFINITIONS (LSTM) ---
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

class LSTMCritic(nn.Module):
    def __init__(self, n_features, n_conditions, hidden_dim, n_layers, seq_len, **kwargs):
        super().__init__()
        self.cond_embedding = nn.Sequential(nn.Linear(n_conditions, 64), nn.LeakyReLU(0.2))
        self.lstm = nn.LSTM(input_size=n_features + 64, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.LayerNorm(128), nn.LeakyReLU(0.2), nn.Linear(128, 1))

    def forward(self, x, conditions):
        c_exp = self.cond_embedding(conditions).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat([x, c_exp], dim=2))
        pooled, _ = torch.max(out, dim=1)
        return self.head(pooled)

# --- DATA PIPELINE ---
def prepare_dataset(metadata, snippets_dir, augment=False):
    data_list, cond_list = [], []
    for idx in range(len(metadata)):
        meta = metadata.iloc[idx]
        path = snippets_dir / Path(meta['snippet_path']).name
        try:
            raw = np.load(path)
            if len(raw) > 100: d = raw[:100]
            else: d = np.pad(raw, ((0, 100 - len(raw)), (0, 0)), 'constant')
            data_list.append(d)
            cond_list.append([meta['time_delta'], meta['entry_speed']])
        except: continue
            
    X = torch.tensor(np.array(data_list), dtype=torch.float32)
    C = torch.tensor(np.array(cond_list), dtype=torch.float32)
    return TensorDataset(X.to(DEVICE), C.to(DEVICE))

# --- TRAINING LOOP ---
def train_local():
    meta_path = DATA_DIR / "snippets_metadata.csv"
    if not meta_path.exists():
        print("❌ Data not found. Skipping training.")
        return

    metadata = pd.read_csv(meta_path)
    sector_type = 'Braking' # Demo on Braking sector
    print(f"🏁 Starting Demo Training for: {sector_type}")
    
    sector_meta = metadata[metadata['sector_type'] == sector_type]
    ds = prepare_dataset(sector_meta, DATA_DIR / "snippets")
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    gen = LSTMGenerator(**CONFIG["model_params"]).to(DEVICE)
    crit = LSTMCritic(**CONFIG["model_params"]).to(DEVICE)
    opt_g = torch.optim.Adam(gen.parameters(), lr=0.001)
    opt_c = torch.optim.Adam(crit.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(5): # Short demo
        for real, cond in loader:
            # Train Critic
            opt_c.zero_grad()
            valid = torch.ones(real.size(0), 1, device=DEVICE)
            fake_label = torch.zeros(real.size(0), 1, device=DEVICE)
            
            real_loss = criterion(crit(real, cond), valid)
            z = torch.randn(real.size(0), 64, device=DEVICE)
            fake_loss = criterion(crit(gen(z, cond).detach(), cond), fake_label)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_c.step()

            # Train Generator
            opt_g.zero_grad()
            g_loss = criterion(crit(gen(z, cond), cond), valid)
            g_loss.backward()
            opt_g.step()
            
        print(f"Epoch {epoch+1}: G_Loss={g_loss.item():.4f} D_Loss={d_loss.item():.4f}")

    torch.save(gen.state_dict(), MODEL_DIR / f"{sector_type}_demo.pth")
    print("✅ Demo Training Complete.")

if __name__ == "__main__":
    train_local()