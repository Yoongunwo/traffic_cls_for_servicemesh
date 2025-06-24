import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import re

# ✅ 차분 기반 데이터셋
class DiffChannelDataset(Dataset):
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
                img = self.transform(img)
            imgs.append(img)

        diffs = [imgs[i+1] - imgs[i] for i in range(self.window_size - 1)]
        stack = torch.stack(diffs, dim=0).squeeze(1)  # [T-1, H, W]
        diff_mean = stack.mean(dim=0)
        diff_std = stack.std(dim=0)
        stat = torch.stack([diff_mean, diff_std], dim=0)  # [2, H, W]

        return stat, self.labels[idx + self.window_size - 1]

# ✅ 간단한 AE
class DiffAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ✅ 학습 및 평가 루프
def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            scores = ((x - out) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.numpy())

    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(f"\n[Threshold = {threshold:.6f}]")
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))

# ✅ 유틸

def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label, stack_folder=False):
    image_paths = []
    labels = []
    if stack_folder:
        for subfolder in sorted(os.listdir(path)):
            subfolder_path = os.path.join(path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            for f in sorted(files, key=natural_key):
                image_paths.append(os.path.join(subfolder_path, f))
                labels.append(label)
    else:
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
        image_paths = [os.path.join(path, f) for f in files]
        labels = [label] * len(image_paths)
    return image_paths, labels

# ✅ 실행
if __name__ == "__main__":
    PREPROCESSING_TYPE = 'hilbert'
    ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq'
    BATCH_SIZE = 1024 * 32
    WIN = 9
    EPOCHS = 20
    random_state = 42

    print(f"Diff AE Window size: {WIN}, Data: {ROOT}")

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    benign_ds = DiffChannelDataset(benign_p, benign_l, transform, WIN)
    atk_ds = DiffChannelDataset(atk_p, atk_l, transform, WIN)

    total_len = len(benign_ds)
    indices = list(range(total_len))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)

    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)
    test_ds = ConcatDataset([val_ds, atk_ds])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffAutoEncoder()

    train_and_evaluate(model, train_loader, test_loader, device, epochs=EPOCHS)

    torch.save(model.state_dict(), f'./AI/Diff/AE/Model/diff_ae_{PREPROCESSING_TYPE}_{WIN}.pth')
