import os
import re
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Dataset with FFT-based 2-channel features (amplitude + phase)
class FFTStatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img).squeeze(0)  # [H, W]
            imgs.append(img)
        stack = torch.stack(imgs, dim=0).numpy()  # [T, H, W]

        fft_vals = np.fft.fft(stack, axis=0)
        amplitude = np.abs(fft_vals).mean(axis=0)
        phase = np.angle(fft_vals).mean(axis=0)

        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-6)
        phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-6)

        result = torch.tensor(np.stack([amplitude, phase], axis=0), dtype=torch.float32)  # [2, H, W]
        return result, self.labels[idx + self.window_size - 1]

# ✅ Simple Autoencoder for 2-channel FFT input
class FFTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 8 → 4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ✅ Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        recon = model(x)
        loss = ((x - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        all_scores.extend(loss)
        all_labels.extend(y.numpy())

    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))

# ✅ Utilities
def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

# ✅ Main
ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
WIN = 4
BATCH_SIZE = 1024 * 32
EPOCHS = 10

if __name__ == '__main__':
    print(f"FFT-AE Window size: {WIN}, Data: {ROOT}")
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p, benign_l = benign_p[:55000 + WIN - 1], benign_l[:55000 + WIN - 1]
    atk_p, atk_l = attack_p[:5000 + WIN - 1], attack_l[:5000 + WIN - 1]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    benign_ds = FFTStatDataset(benign_p, benign_l, transform, WIN)
    atk_ds = FFTStatDataset(atk_p, atk_l, transform, WIN)

    train_idx, val_idx = train_test_split(list(range(len(benign_ds))), test_size=5000, random_state=42)
    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)

    test_ds = ConcatDataset([val_ds, atk_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFTAutoencoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'./AI/FFT/AE/Model/fft_ae_{WIN}.pth')
    evaluate(model, test_loader, device)
