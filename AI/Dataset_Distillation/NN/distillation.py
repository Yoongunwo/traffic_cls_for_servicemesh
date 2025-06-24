import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import os


DATA_DIR = './Data_CIC/Session_Windows_15'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H, W = 34, 44
BATCH_SIZE = 2**10

# === Fixed CNN Encoder ===
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
    def forward(self, x):
        return self.net(x)

# === Dummy Benign Dataset ===
class DummyBenignDataset(Dataset):
    def __init__(self, num_samples=1000, H=34, W=44):
        self.data = torch.rand((num_samples, 1, H, W))
        self.labels = torch.zeros(num_samples, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# === Feature Extractor for All Real Data ===
def extract_all_features(dataloader, encoder, device='cuda'):
    encoder.eval()
    all_feats = []
    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            f = encoder(x)
        all_feats.append(f)
    return torch.cat(all_feats, dim=0)  # shape [N, D]

# === Distillation via Nearest-Neighbor Feature Matching ===
def distill_nearest_neighbor(real_loader, encoder, image_shape=(1, 34, 44), num_syn=20, steps=500, lr=0.1):
    device = next(encoder.parameters()).device
    encoder.eval()

    # Extract all real features once
    real_feats = extract_all_features(real_loader, encoder, device=device)

    syn_images = torch.randn((num_syn,) + image_shape, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([syn_images], lr=lr)

    for step in range(steps):
        f_syn = encoder(syn_images)  # [num_syn, D]
        dists = torch.cdist(f_syn, real_feats, p=2)  # [num_syn, num_real]
        loss = torch.mean(torch.min(dists, dim=1)[0])  # Nearest neighbor distance per syn sample

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] NN Matching Loss: {loss.item():.4f}")

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

# === Run Example ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = CNNEncoder(out_dim=128).to(device)

print("Loading benign data...")
benign_paths = load_image_paths(DATA_DIR, per_class=5000)
benign_dataset = CICBenignDataset(benign_paths)
benign_loader = DataLoader(benign_dataset, batch_size=BATCH_SIZE, shuffle=True)

syn_images = distill_nearest_neighbor(
    real_loader=benign_loader,
    encoder=encoder,
    image_shape=(1, 34, 44),
    num_syn=20,
    steps=500,
    lr=0.1
)

torch.save(syn_images, "./AI/Dataset_Distillation/NN/distilled_images_nnmatch.pt")
print("✅ Saved distilled images using Nearest-Neighbor Feature Matching.")

