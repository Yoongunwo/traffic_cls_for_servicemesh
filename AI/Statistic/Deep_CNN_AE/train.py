import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

import re

from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

# ✅ 통계 기반 채널 생성 Dataset
class StatisticalChannelDataset(Dataset):
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
        stack = torch.stack(imgs, dim=0)  # [T, C, H, W] = [4, 1, 16, 16]
        stack = stack.squeeze(1)  # [4, 16, 16]

        mean_img = stack.mean(dim=0, keepdim=True)  # [1, 16, 16]
        std_img = stack.std(dim=0, keepdim=True)    # [1, 16, 16]
        min_img = stack.min(dim=0, keepdim=True).values
        max_img = stack.max(dim=0, keepdim=True).values

        stat_stack = torch.cat([mean_img, std_img, min_img, max_img], dim=0)  # [4, 16, 16]
        label = self.labels[idx + self.window_size - 1]
        return stat_stack, label

# ✅ Deep CNN Autoencoder
class DeepStatCNN_AE(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → [B, 128, 4, 4]
            nn.Flatten(),  # [B, 2048]
            nn.Linear(128 * 4 * 4, 256), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 4 * 4), nn.ReLU(),  # ✅ 복원
            nn.Unflatten(1, (128, 4, 4)),  # ✅ [B, 128, 4, 4]
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 4x4 → 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8x8 → 16x16
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),  # 16x16 유지
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class DeepStatCNN_AE32(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 → 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 → 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 8x8 → 4x4
            nn.Flatten(),  # [B, 256 * 4 * 4] = [B, 4096]
            nn.Linear(256 * 4 * 4, 512), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 256 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # [B, 256, 4, 4]

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 4x4 → 8x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),   # 8x8 → 16x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),    # 16x16 → 32x32

            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),  # 유지
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ✅ Train + Evaluate
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        recon = model(x)
        score = ((x - recon)**2).mean(dim=[1,2,3]).cpu().numpy()
        all_scores.extend(score)
        all_labels.extend(y.numpy())

    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))


def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = loss_fn(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total/len(train_loader):.4f}")

    evaluate(model, test_loader, device)
    return model

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
WIN = 16
random_state = 42
EPOCHS = 10

def start_train():
    print(f"Statistic Deep CNN AE Window size: {WIN}, Data: {ROOT}")
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, WIN)
    atk_ds = StatisticalChannelDataset(atk_p, atk_l, transform, WIN)

    total_len = len(benign_ds)
    indices = list(range(total_len))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)

    train_ds = torch.utils.data.Subset(benign_ds, train_idx)
    val_ds = torch.utils.data.Subset(benign_ds, val_idx)
    # print(train_ds[0])
    
    test_ds = ConcatDataset([val_ds, atk_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepStatCNN_AE()
    model = train_and_evaluate(model, train_loader, test_loader, device, epochs=EPOCHS)

    torch.save(model.state_dict(), f'./AI/Statistic/Deep_CNN_AE/Model/deep_stat_cnn_ae_{WIN}_{EPOCHS}.pth')

# ✅ Main
if __name__ == '__main__':
    start_train()