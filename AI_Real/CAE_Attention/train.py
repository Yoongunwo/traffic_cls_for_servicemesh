import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**10
EPOCHS = 20

# === Dataset ===
class PacketDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])[0]
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        if x.shape[0] < self.HW:
            x = np.pad(x, (0, self.HW - x.shape[0]))
        x = x[:self.HW].reshape(1, H, W)
        return torch.tensor(x, dtype=torch.float32)

def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = "attack/*/*.npy" if is_attack else "benign/*.npy"
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

# === Attention CAE ===
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        attn = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn

class CAEAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.attn = SpatialAttention()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
            nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.attn(enc)
        return self.decoder(enc)

# === Training ===
def train_model(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x in dataloader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")
    return model

# === Evaluation ===
def evaluate_model(model, benign_paths, attack_paths):
    model.eval()
    def recon_error(paths):
        errs = []
        with torch.no_grad():
            for f in paths:
                x = np.load(f)[0]
                x = np.nan_to_num(x)
                x = np.clip(x, 0, 255).astype(np.float32) / 255.0
                if x.shape[0] < H * W:
                    x = np.pad(x, (0, H * W - x.shape[0]))
                x = x[:H * W].reshape(1, H, W)
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                recon = model(x)
                err = F.mse_loss(recon, x, reduction='none').mean().item()
                errs.append(err)
        return np.array(errs)

    benign_scores = recon_error(benign_paths)
    attack_scores = recon_error(attack_paths)

    y_true = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
    y_score = np.concatenate([benign_scores, attack_scores])
    y_pred = (y_score > np.percentile(benign_scores, 95)).astype(int)

    print("\n=== Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_score):.4f}")

# === Run (in smaller scale) ===
benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
benign_test = load_paths(DATA_DIR, is_attack=False, limit=5000)
attack_test = load_paths(DATA_DIR, is_attack=True, limit=5000)

train_loader = DataLoader(PacketDataset(benign_train), batch_size=BATCH_SIZE, shuffle=True)
model = CAEAttention().to(DEVICE)
model = train_model(model, train_loader)

torch.save(model.state_dict(), f"./AI_Real/CAE_Attention/Model/CAE_Attention_Model_{EPOCHS}.pth")

evaluate_model(model, benign_test, attack_test)
