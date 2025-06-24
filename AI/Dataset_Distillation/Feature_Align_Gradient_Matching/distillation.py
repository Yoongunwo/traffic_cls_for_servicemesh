import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import numpy as np
from glob import glob
import os

# === Config ===
DATA_DIR = './Data_CIC/Session_Windows_15'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H, W = 34, 44
BATCH_SIZE = 2**10

# === Simple CNN Model with Encoder Access ===
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

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        if return_feat:
            return feat
        return self.classifier(feat)

# === Dummy Benign Dataset (real benign data placeholder) ===
class DummyBenignDataset(Dataset):
    def __init__(self, num_samples=1000, H=34, W=44):
        self.data = torch.rand((num_samples, 1, H, W))
        self.labels = torch.zeros(num_samples, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# === Simple Augmentation ===
def augment(x):
    noise = torch.randn_like(x) * 0.05
    return torch.clamp(x + noise, 0, 1)

# === Distillation Step with Feature Alignment ===
def compute_feature_aligned_loss(syn_img, syn_lbl, real_loader, model_fn, steps=3, inner_lr=0.1, lambda_feat=1.0, device='cuda'):
    model = model_fn().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)

    for _ in range(steps):
        model.train()
        pred_syn = model(augment(syn_img))
        loss_syn = F.cross_entropy(pred_syn, syn_lbl)
        optimizer.zero_grad()
        loss_syn.backward()
        optimizer.step()

    model.eval()
    real_x, real_y = next(iter(real_loader))
    real_x, real_y = real_x.to(device), real_y.to(device)
    pred_real = model(real_x)
    loss_real = F.cross_entropy(pred_real, real_y)

    # Feature alignment
    syn_feat = model(augment(syn_img), return_feat=True)
    real_feat = model(real_x, return_feat=True)
    loss_feat = F.mse_loss(syn_feat.mean(dim=0), real_feat.mean(dim=0))

    return loss_real + lambda_feat * loss_feat

# === Dataset Distillation ===
def dataset_distillation_with_feat_alignment(real_loader, model_fn, image_shape=(1, 34, 44), num_syn_images=20, steps=500, lr=0.1, lambda_feat=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    syn_images = torch.randn((num_syn_images,) + image_shape, requires_grad=True, device=device)
    syn_labels = torch.zeros(num_syn_images, dtype=torch.long, device=device)

    syn_optimizer = torch.optim.SGD([syn_images], lr=lr)

    for step in range(steps):
        syn_optimizer.zero_grad()
        loss = compute_feature_aligned_loss(
            syn_images, syn_labels, real_loader, model_fn,
            steps=3, inner_lr=0.1, lambda_feat=lambda_feat, device=device
        )
        loss.backward()
        syn_optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Total Loss (Real + FeatAlign): {loss.item():.4f}")

    return syn_images.detach().cpu()

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

# === Run with Dummy Benign Data ===
print("Loading benign data...")
benign_paths = load_image_paths(DATA_DIR, per_class=1000)
benign_dataset = CICBenignDataset(benign_paths)
benign_loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

syn_data = dataset_distillation_with_feat_alignment(
    real_loader=benign_loader,
    model_fn=lambda: SimpleCNN(out_dim=2),
    image_shape=(1, 34, 44),
    num_syn_images=20,
    steps=500,
    lr=0.1,
    lambda_feat=5.0  # Increase to enforce stronger alignment
)

torch.save(syn_data, "./AI/Dataset_Distillation/Feature_Align_Gradient_Matching/distilled_images_feat_align.pt")
print("✅ Saved distilled images with feature alignment loss.")

