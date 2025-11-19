import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from pathlib import Path
from tqdm import tqdm

# Setup paths and database access
sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import MicroSectors

# ==============================================================================
# SOTA MODEL CONFIGURATION
# ==============================================================================
CONFIG = {
    "env": {
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    },
    "paths": {
        "processed_data": Path("/data/processed"),
        "snippets": Path("/data/processed/snippets"),
    },
    "training_params": {
        "batch_size": 64,
        "epochs": 100, # Increased for more stable WGAN training
        "lr": 1e-4,
        "critic_iterations": 5, # Train critic 5 times per generator train
        "lambda_gp": 10, # Gradient penalty coefficient
    },
    "model_params": {
        "seq_len": 100, # All snippets will be padded/truncated to this length
        "n_features": len(['speed', 'ath', 'pbrake_f', 'Steering_Angle', 'Steering_Angle_roc', 'ath_roc', 'pbrake_f_roc']),
        "n_conditions": 2, # time_delta, entry_speed
        "latent_dim": 64, # Noise vector size
        "embed_dim": 128, # Main dimension for the Transformer
        "n_heads": 4,     # Number of attention heads
        "n_layers": 3,    # Number of Transformer layers
        "dropout": 0.1,
    }
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# ==============================================================================

# --- Data Loading ---
class SnippetDataset(Dataset):
    def __init__(self, snippets_metadata, snippets_dir, seq_len):
        self.metadata = snippets_metadata
        self.snippets_dir = snippets_dir
        self.seq_len = seq_len

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        snippet_path = meta['snippet_path']
        
        data = np.load(snippet_path)
        
        # Pad or truncate to fixed sequence length
        if len(data) > self.seq_len:
            data = data[:self.seq_len, :]
        elif len(data) < self.seq_len:
            padding = np.zeros((self.seq_len - len(data), data.shape[1]))
            data = np.vstack((data, padding))
            
        conditions = torch.tensor([meta['time_delta'], meta['entry_speed']], dtype=torch.float32)
        return torch.tensor(data, dtype=torch.float32), conditions

# --- Transformer Model Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Generator (Transformer Decoder) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, n_features, n_conditions, embed_dim, n_heads, n_layers, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.cond_embed = nn.Linear(n_conditions, embed_dim)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_layer = nn.Linear(embed_dim, n_features)
        self.tanh = nn.Tanh()

    def forward(self, z, conditions):
        cond_embedding = self.cond_embed(conditions).unsqueeze(1)
        latent_embedding = self.latent_embed(z).unsqueeze(1)
        
        # Memory for the decoder is the combined condition and latent embeddings
        memory = torch.cat([cond_embedding, latent_embedding], dim=1)
        
        # Decoder input starts as a learnable start-of-sequence token
        tgt = torch.zeros(z.size(0), self.seq_len, memory.size(2)).to(z.device)
        tgt = self.pos_encoder(tgt)

        output = self.transformer_decoder(tgt, memory)
        output = self.output_layer(output)
        return self.tanh(output)

# --- Critic (Transformer Encoder) ---
class Critic(nn.Module):
    def __init__(self, n_features, n_conditions, embed_dim, n_heads, n_layers, seq_len, dropout):
        super().__init__()
        self.cond_embed = nn.Linear(n_conditions, embed_dim)
        self.data_embed = nn.Linear(n_features, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * (seq_len + 1), 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, conditions):
        cond_embedding = self.cond_embed(conditions).unsqueeze(1)
        data_embedding = self.data_embed(x)
        
        full_seq = torch.cat([cond_embedding, data_embedding], dim=1)
        full_seq = self.pos_encoder(full_seq)
        
        output = self.transformer_encoder(full_seq)
        output = output.view(output.size(0), -1) # Flatten
        return self.output_layer(output)

# --- WGAN-GP Gradient Penalty ---
def compute_gradient_penalty(critic, real_samples, fake_samples, conditions):
    alpha = torch.randn(real_samples.size(0), 1, 1, device=DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, conditions)
    
    fake = torch.ones(real_samples.size(0), 1, device=DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- Training Orchestration ---
def train_wgan_gp(sector_type: str, metadata: list):
    p = CONFIG["training_params"]
    mp = CONFIG["model_params"]

    dataset = SnippetDataset(metadata, CONFIG["paths"]["snippets"], mp["seq_len"])
    dataloader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=True, drop_last=True)
    
    generator = Generator(mp["latent_dim"], mp["n_features"], mp["n_conditions"], mp["embed_dim"], mp["n_heads"], mp["n_layers"], mp["seq_len"], mp["dropout"]).to(DEVICE)
    critic = Critic(mp["n_features"], mp["n_conditions"], mp["embed_dim"], mp["n_heads"], mp["n_layers"], mp["seq_len"], mp["dropout"]).to(DEVICE)
    
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=p["lr"], betas=(0.5, 0.9))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=p["lr"], betas=(0.5, 0.9))

    mlflow.set_experiment("WGAN-GP Transformer Training")
    with mlflow.start_run(run_name=f"Train_{sector_type}") as run:
        mlflow.log_params(p)
        mlflow.log_params(mp)
        mlflow.log_param("sector_type", sector_type)
        
        print(f"\n--- Training model for sector: {sector_type} ---")
        for epoch in range(p["epochs"]):
            for i, (real_snippets, conditions) in enumerate(dataloader):
                real_snippets = real_snippets.to(DEVICE)
                conditions = conditions.to(DEVICE)
                
                # --- Train Critic ---
                optimizer_c.zero_grad()
                
                noise = torch.randn(p["batch_size"], mp["latent_dim"], device=DEVICE)
                fake_snippets = generator(noise, conditions)
                
                real_validity = critic(real_snippets, conditions)
                fake_validity = critic(fake_snippets, conditions)
                
                gradient_penalty = compute_gradient_penalty(critic, real_snippets.data, fake_snippets.data, conditions.data)
                
                w_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                c_loss = w_loss + p["lambda_gp"] * gradient_penalty
                
                c_loss.backward()
                optimizer_c.step()

                # --- Train Generator ---
                if i % p["critic_iterations"] == 0:
                    optimizer_g.zero_grad()
                    
                    noise = torch.randn(p["batch_size"], mp["latent_dim"], device=DEVICE)
                    gen_snippets = generator(noise, conditions)
                    
                    g_loss = -torch.mean(critic(gen_snippets, conditions))
                    
                    g_loss.backward()
                    optimizer_g.step()

            print(f"[Epoch {epoch+1}/{p['epochs']}] [C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}] [W dist: {-w_loss.item():.4f}]")
            mlflow.log_metric("critic_loss", c_loss.item(), step=epoch)
            mlflow.log_metric("generator_loss", g_loss.item(), step=epoch)
            mlflow.log_metric("wasserstein_distance", -w_loss.item(), step=epoch)

        # Log and register the final model
        mlflow.pytorch.log_model(generator, "generator", registered_model_name=f"{sector_type}_Generator")
        mlflow.pytorch.log_model(critic, "critic", registered_model_name=f"{sector_type}_Critic")
        print(f"--- Finished training for {sector_type}. Models registered in MLflow. ---")


def main():
    mlflow.set_tracking_uri(CONFIG["env"]["mlflow_tracking_uri"])
    db = SessionLocal()
    
    try:
        # Get all distinct sector types from the database
        sector_types = db.query(MicroSectors.sector_type).distinct().all()
        sector_types = [s[0] for s in sector_types]
        
        for sector_type in sector_types:
            print(f"\nProcessing sector type: {sector_type}")
            # Query metadata for the current sector type
            snippets_metadata = db.query(MicroSectors).filter(MicroSectors.sector_type == sector_type).all()
            metadata_dicts = [
                {"snippet_path": s.snippet_path, "time_delta": s.time_delta, "entry_speed": s.entry_speed}
                for s in snippets_metadata
            ]
            
            if len(metadata_dicts) < CONFIG["training_params"]["batch_size"]:
                print(f"Skipping '{sector_type}': only {len(metadata_dicts)} snippets, not enough for a batch.")
                continue

            train_wgan_gp(sector_type, metadata_dicts)
            
    finally:
        db.close()

if __name__ == "__main__":
    main()