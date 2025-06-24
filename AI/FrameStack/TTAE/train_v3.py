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

# ✅ Diff 기반 시퀀스 데이터셋
class DiffFrameStackDataset(Dataset):
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
        stack = torch.stack(imgs, dim=0)  # [T, 1, H, W]
        diffs = stack[1:] - stack[:-1]    # [T-1, 1, H, W]
        target = stack[-1]               # [1, H, W]
        return diffs, target, self.labels[idx + self.window_size]

# ✅ TTAE 모델
class DiffTTAE(nn.Module):
    def __init__(self, T=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64*4*4, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.Sequential(
            nn.Linear(64*4*4, 64*4*4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T-1, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encoder(x).view(B, T, -1)
        z = self.temporal(z)
        z_last = z[:, -1]
        recon = self.decoder(z_last)
        return recon

# ✅ 평가 함수
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y, label in loader:
        x, y = x.to(device), y.to(device)
        recon = model(x)
        err = ((recon - y) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        scores.extend(err)
        labels.extend(label.numpy())

    threshold = np.percentile(scores, 95)
    preds = (np.array(scores) > threshold).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))

# ✅ 경로 및 설정
ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
WIN = 9
BATCH = 1024
EPOCHS = 25

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

def run():
    print(f"[Diff TTAE] Window size: {WIN}, Data: {ROOT}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p, benign_l = benign_p[:55000+WIN], benign_l[:55000+WIN]
    attack_p, attack_l = attack_p[:5000+WIN], attack_l[:5000+WIN]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_ds = DiffFrameStackDataset(benign_p, benign_l, transform, WIN)
    atk_ds = DiffFrameStackDataset(attack_p, attack_l, transform, WIN)

    train_idx, val_idx = train_test_split(list(range(len(benign_ds))), test_size=5000, random_state=42)
    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)
    test_ds = ConcatDataset([val_ds, atk_ds])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = DiffTTAE(T=WIN-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            recon = model(x)
            loss = criterion(recon, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'./AI/FrameStack/TTAE/Model/ttaediff_{WIN}_v3.pth')
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    run()
