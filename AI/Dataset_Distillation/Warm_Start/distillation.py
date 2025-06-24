# warmstart_dualloss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from glob import glob
import os
import numpy as np


DATA_DIR = './Data_CIC/Session_Windows_15'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H, W = 34, 44
BATCH_SIZE = 2**10

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim)
        )
    def forward(self, x): return self.net(x)

# Dummy Benign Data
class DummyBenignDataset(Dataset):
    def __init__(self, n=1000):
        self.data = torch.rand(n, 1, H, W)
        self.labels = torch.zeros(n, dtype=torch.long)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]
    def __len__(self): return len(self.data)

# Distillation
def distill(encoder, loader, n=20, steps=500, lr=0.1, lam_input=1.0):
    encoder.eval()
    feats, imgs = [], []
    for x, _ in loader:
        x = x.to(DEVICE)
        with torch.no_grad(): f = encoder(x)
        feats.append(f); imgs.append(x)
    real_feats = torch.cat(feats); real_imgs = torch.cat(imgs)
    real_mean_img = real_imgs.mean(0, keepdim=True)

    idx = torch.randperm(real_imgs.size(0))[:n]
    syn = deepcopy(real_imgs[idx]) + 0.05 * torch.randn_like(real_imgs[idx])
    syn = syn.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([syn], lr=lr)

    for step in range(steps):
        f_syn = encoder(syn)
        dists = torch.cdist(f_syn, real_feats)
        loss_feat = torch.min(dists, dim=1)[0].mean()
        loss_input = F.mse_loss(syn, real_mean_img.expand_as(syn))
        loss = loss_feat + lam_input * loss_input
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"[{step}] feat={loss_feat.item():.4f}, input={loss_input.item():.4f}")

    return syn.detach().cpu()

def load_image_paths(data_dir, per_class=1000):
    files = sorted(glob(os.path.join(data_dir, 'benign/*.npy')))[:per_class]
    return files

class CICBenignDataset(Dataset):
    def __init__(self, file_paths):
        self.frames = []
        for path in file_paths:
            x = np.load(path)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            for t in range(x.shape[0]):
                vec = x[t]
                if vec.shape[0] < H * W:
                    vec = np.pad(vec, (0, H * W - vec.shape[0]))
                else:
                    vec = vec[:H * W]
                img = vec.reshape(1, H, W)
                self.frames.append(img)
        self.frames = torch.tensor(np.array(self.frames), dtype=torch.float32)
        self.labels = torch.zeros(len(self.frames), dtype=torch.long)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]

# Run
encoder = CNNEncoder(out_dim=128).to(DEVICE)

print("Loading benign data...")
benign_paths = load_image_paths(DATA_DIR, per_class=5000)
benign_dataset = CICBenignDataset(benign_paths)
benign_loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

syn_images = distill(encoder, benign_loader)
torch.save(syn_images, "./AI/Dataset_Distillation/Warm_Start/distilled_images_warmstart_dual.pt")
print("✅ Saved to distilled_images_warmstart_dual.pt")
