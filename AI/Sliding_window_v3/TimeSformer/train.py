import os, re
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config
DATA_DIR = './Data_CIC/Session_Windows_15'
WINDOW = 5
H, W = 34, 44
BATCH = 2**4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 파싱 함수 ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

# === Dataset ===
class SequenceDataset(Dataset):
    def __init__(self, data_dir, is_attack=False, per_class=10000, window=5):
        self.samples, self.labels = [], []
        label = int(is_attack)
        files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
        session_map = defaultdict(list)
        for f in files:
            sid, idx = parse_name(f)
            session_map[sid].append((idx, f))
        for fs in session_map.values():
            fs.sort()
            paths = [f for _, f in fs]
            for i in range(len(paths) - window + 1):
                self.samples.append(paths[i:i+window])
                self.labels.append(label)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        imgs = []
        for path in paths:
            x = np.load(path)  # [15, 1479]
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] < H * W:
                x = np.pad(x, ((0, 0), (0, H * W - x.shape[1])))
            elif x.shape[1] > H * W:
                x = x[:, :H * W]
            x = x.reshape(x.shape[0], 1, H, W)  # [15, 1, H, W]
            imgs.append(torch.tensor(x, dtype=torch.float32))  # [15, 1, H, W]
        stacked = torch.stack(imgs, dim=0)  # [T=5, 15, 1, H, W]
        return stacked, self.labels[idx]

    def __len__(self):
        return len(self.samples)

# === TimeSformer 기반 모델 ===
class TimeSformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TimeSformerAE(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # ← 이걸로 [B*T*F, D, 1, 1]
        )
        self.encoder = nn.Sequential(*[TimeSformerBlock(embed_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
            nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
        )

    def forward(self, x):  # x: [B, T, F, 1, H, W]
        B, T, F, C, H_, W_ = x.shape
        x = x.view(B * T * F, C, H_, W_)  # [B*T*F, 1, H, W]
        
        # Patch Embedding with AvgPool to get [B*T*F, D, 1, 1] → squeeze to [B*T*F, D]
        x = self.patch_embed(x).squeeze(-1).squeeze(-1)  # [B*T*F, embed_dim]
        
        x = x.view(B, T * F, -1)  # [B, T*F, embed_dim]
        x = self.encoder(x)       # TimeSformer Encoding
        x = x.view(B * T * F, -1) # Flatten for decoder
        
        recon = self.decoder(x).view(B, T, F, 1, H_, W_)  # [B, T, F, 1, H, W]
        return recon

# === Train ===
def train(model, loader, optimizer):
    model.train()
    total = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = F.mse_loss(recon, x, reduction='none').mean(dim=(2,3,4,5)).mean(dim=1)
            scores.extend(loss.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(scores), np.array(labels)

# === Run ===
train_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=50000, window=WINDOW)
test_b_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=5000, window=WINDOW)
test_a_dataset = SequenceDataset(DATA_DIR, is_attack=True, per_class=5000, window=WINDOW)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_b_dataset + test_a_dataset, batch_size=1)

model = TimeSformerAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer)
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

torch.save(model.state_dict(), './AI/Sliding_window_v3/TimeSformer/Model/timesformer_ae.pth')

scores, labels = evaluate(model, test_loader)
threshold = np.percentile(scores[:5000], 95)
preds = (scores > threshold).astype(int)

print(classification_report(labels, preds, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(labels, scores):.4f}")
