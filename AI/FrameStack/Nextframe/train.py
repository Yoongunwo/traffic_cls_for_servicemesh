# nextframe_transformer.py
import os
import re
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset [T, 1, H, W] frame stack
class FrameSequenceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        stack = torch.stack(imgs[:-1], dim=0)  # [T-1, 1, H, W]
        target = imgs[-1]                     # [1, H, W]
        label = self.labels[idx + self.window_size - 1]
        return stack, target, label

# Next-Frame Prediction Model
class NextFrameTTransformer(nn.Module):
    def __init__(self, T=3, h=16, w=16, d_model=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64*4*4, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 4x4 → 8x8
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # 8x8 → 16x16
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Sigmoid()  # Final output: [B, 1, 16, 16]
        )

    def forward(self, x):  # x: [B, T-1, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encoder(x).view(B, T, -1)  # [B, T-1, D]
        z = self.temporal(z)               # [B, T-1, D]
        z_last = z[:, -1]                  # [B, D]
        out = self.decoder(z_last)         # [B, 1, H, W]
        return out

# Evaluate with ROC, F1, etc.
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    scores, labels = [], []
    for x, target, y in test_loader:
        x, target = x.to(device), target.to(device)
        out = model(x)
        err = ((out - target) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        scores.extend(err)
        labels.extend(y.numpy())

    threshold = np.percentile(scores, 95)
    preds = (np.array(scores) > threshold).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))

# Helpers

def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

# Main training pipeline
def run():
    ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
    WIN = 4
    BATCH = 1024 * 8
    EPOCHS = 10

    print(f"[NextFrameTTransformer] Window size: {WIN}, Data: {ROOT}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)
    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    attack_p, attack_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    benign_ds = FrameSequenceDataset(benign_p, benign_l, transform, WIN)
    attack_ds = FrameSequenceDataset(attack_p, attack_l, transform, WIN)

    train_idx, val_idx = train_test_split(list(range(len(benign_ds))), test_size=5000, random_state=42)
    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)
    test_ds = ConcatDataset([val_ds, attack_ds])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = NextFrameTTransformer(T=WIN-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, target, _ in train_loader:
            x, target = x.to(device), target.to(device)
            recon = model(x)
            loss = criterion(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'./AI/FrameStack/Nextframe/Model/nextframe_transformer_{WIN}.pth')
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    run()
