import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from copy import deepcopy
from torchvision import transforms
from glob import glob

# === Config ===
DATA_DIR = './Data_CIC/Session_Windows_15'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H, W = 34, 44
BATCH_SIZE = 2**10

# === Simple CNN ===
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        return self.classifier(self.encoder(x))

# === Simple Noise-based Augmentation ===
def augment(x):
    noise = torch.randn_like(x) * 0.05
    return torch.clamp(x + noise, 0, 1)

# === Gradient Matching with Unrolled Steps ===
def compute_unrolled_loss(syn_img, syn_lbl, real_loader, model_fn, unroll_steps=3, lr_inner=0.1):
    model = model_fn().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_inner)

    for _ in range(unroll_steps):
        model.train()
        pred = model(augment(syn_img))
        loss = F.cross_entropy(pred, syn_lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    real_x, real_y = next(iter(real_loader))
    real_x, real_y = real_x.to(DEVICE), real_y.to(DEVICE)
    loss_real = F.cross_entropy(model(real_x), real_y)
    return loss_real

# === DSA-style Dataset Distillation ===
def run_dsa_distillation(real_loader, model_fn, image_shape=(1, H, W), n_syn=20, steps=500, lr_outer=0.1):
    syn_img = torch.randn((n_syn,) + image_shape, device=DEVICE, requires_grad=True)
    syn_lbl = torch.zeros(n_syn, dtype=torch.long, device=DEVICE)  # all benign
    optimizer = torch.optim.SGD([syn_img], lr=lr_outer)

    for step in range(steps):
        optimizer.zero_grad()
        loss = compute_unrolled_loss(syn_img, syn_lbl, real_loader, model_fn, unroll_steps=3)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

    return syn_img.detach().cpu()

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

# === 실행 ===
if __name__ == '__main__':
    print("Loading benign data...")
    benign_paths = load_image_paths(DATA_DIR, per_class=500)
    benign_dataset = CICBenignDataset(benign_paths)
    benign_loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

    syn_imgs = run_dsa_distillation(
        benign_loader,
        model_fn=lambda: SimpleCNN(out_dim=2),
        image_shape=(1, H, W),
        n_syn=20,
        steps=500,
        lr_outer=0.1
    )

    torch.save(syn_imgs, "./AI/Dataset_Distillation/distilled_images_dsa.pt")
    print("✅ distilled_images_dsa.pt 저장 완료")
