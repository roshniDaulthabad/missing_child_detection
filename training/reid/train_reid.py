import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.ai.hardware import HardwareProfile
from app.ai.reid.feature_extractor import ReIDBackbone

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        loss = F_relu = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

def train_reid(epochs: int = 3, batch_size: int = 4, lr: float = 0.001):
    HardwareProfile.print_cpu_training_banner("Person Re-ID Feature Extractor Training (CPU)")
    
    device = torch.device("cpu")
    print(f"\n[CONFIG] Training Person Re-ID Backbone on CPU for {epochs} epochs (batch={batch_size}, lr={lr})...")
    
    model = ReIDBackbone(feature_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin=0.3)
    
    start_time = time.time()
    
    # Simulate calibrated pedestrian batches (Anchor, Positive, Negative)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 10
        model.train()
        for b in range(num_batches):
            # Synthetic feature triplets representing camera-to-camera domain variations
            anchor = torch.randn(batch_size, 3, 256, 128)
            # Positive has similar features with noise
            positive = anchor + 0.1 * torch.randn_like(anchor)
            # Negative has distinct features
            negative = torch.randn(batch_size, 3, 256, 128)
            
            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch [{epoch}/{epochs}] - Triplet Loss: {avg_loss:.4f}")
        
    duration = time.time() - start_time
    save_path = BASE_DIR / "models" / "reid" / "osnet_reid.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    
    print("\n" + "=" * 65)
    print("  RE-ID TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Duration:         {round(duration, 2)} seconds")
    print(f"  Saved Checkpoint: {save_path}")
    print("=" * 65)
    return str(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    train_reid(epochs=args.epochs, batch_size=args.batch)
