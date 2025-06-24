import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import random
import re
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms

class FrameStackDataset(Dataset):
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
                img = self.transform(img)  # [1, H, W]
            imgs.append(img)
        stack = torch.stack(imgs, dim=1)  # [1, T, H, W]
        label = self.labels[idx + self.window_size - 1]
        return stack, label

class FrameStackAE(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(T, 3, 3), stride=1, padding=(0, 1, 1)), nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=1), nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=(T, 3, 3), stride=1, padding=(0, 1, 1)), nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, 1, T, H, W]
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def natural_key(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

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
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon = model(x)
            loss = ((x - recon) ** 2).mean(dim=(1,2,3,4)).cpu().numpy()
            scores.extend(loss)
            labels.extend(y.numpy())

    threshold = np.percentile(scores, 95)
    preds = (np.array(scores) > threshold).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))

ROOT = './Data_CIC/Fri_hilbert_32/'
WIN = 16
BATCH_SIZE = 1024 * 32
EPOCHS = 10

if __name__ == '__main__':
    print(f"FrameStackAE Window size: {WIN}, Data: {ROOT}, Epochs: {EPOCHS}")
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack/PortScan'), 1)

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_ds = FrameStackDataset(benign_p, benign_l, transform, WIN)
    atk_ds = FrameStackDataset(atk_p, atk_l, transform, WIN)

    total_len = len(benign_ds)
    indices = list(range(total_len))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=42, shuffle=True)

    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)

    test_ds = ConcatDataset([val_ds, atk_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FrameStackAE(T=WIN)
    train_and_evaluate(model, train_loader, test_loader, device, epochs=EPOCHS)
    torch.save(model.state_dict(), f'./AI/FrameStack/AE/Model/frame_stack_ae_{WIN}.pth')