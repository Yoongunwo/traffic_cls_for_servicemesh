import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

import re

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            ResBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ResBlock(32, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            ResBlock(128, 64),
            nn.Upsample(scale_factor=2),
            ResBlock(64, 32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class DiffusionAutoencoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        recon = model(x)
        score = ((x - recon) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
        all_scores.extend(score)
        all_labels.extend(y.numpy())
    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))

def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
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
    evaluate(model, test_loader, device)

class StatisticalChannelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = [self.transform(Image.open(self.image_paths[idx + i]).convert('L')) for i in range(self.window_size)]
        stack = torch.stack(imgs, dim=0).squeeze(1)  # [W, H, W]
        mean = stack.mean(0, keepdim=True)
        std = stack.std(0, keepdim=True)
        min_ = stack.min(0, keepdim=True).values
        max_ = stack.max(0, keepdim=True).values
        stat = torch.cat([mean, std, min_, max_], dim=0)  # [4, H, W]
        return stat, self.labels[idx + self.window_size - 1]

def natural_key(string):
    # 문자열 내 숫자를 기준으로 정렬 (예: packet_2 < packet_10)
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

PREPROCESSING_TYPE = 'hilbert'
ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq'
BATCH_SIZE = 1024 * 16
WIN = 9
random_state = 42

def start_train():
    print(f"Window size: {WIN}, Data: {ROOT}")
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, WIN)
    atk_ds = StatisticalChannelDataset(atk_p, atk_l, transform, WIN)

    total_len = len(benign_ds)
    indices = list(range(total_len))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)

    train_ds = torch.utils.data.Subset(benign_ds, train_idx)
    val_ds = torch.utils.data.Subset(benign_ds, val_idx)
    
    test_ds = ConcatDataset([val_ds, atk_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = DiffusionAutoencoder(in_channels=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_and_evaluate(model, train_loader, test_loader, device, epochs=10)

    torch.save(model.state_dict(), f'./AI/Statistic/Diffsion_AE/Model/diffusion_ae_{WIN}.pth')

if __name__ == "__main__":
    start_train()
    # To evaluate, you can create a similar function that loads the model and calls evaluate() on a test dataset.
    # For example:
    # def start_eval():
    #     ...
    #     evaluate(model, test_loader, device)
    # start_eval()