import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

import re

# ✅ Dataset for statistical feature maps
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
        stack = torch.stack(imgs, dim=0).squeeze(1)
        mean_img = stack.mean(dim=0, keepdim=True)
        std_img = stack.std(dim=0, keepdim=True)
        min_img = stack.min(dim=0, keepdim=True).values
        max_img = stack.max(dim=0, keepdim=True).values
        stat_stack = torch.cat([mean_img, std_img, min_img, max_img], dim=0)
        label = self.labels[idx + self.window_size - 1]
        return stat_stack, label

# ✅ CNN + Transformer Autoencoder
class CNNTransformerAE(nn.Module):
    def __init__(self, in_channels=4, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # (B, 128, 4, 4)
        )
        self.flatten = nn.Flatten(start_dim=2)  # → (B, 128, 16)
        self.positional_encoding = nn.Parameter(torch.randn(1, 16, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.project = nn.Linear(128, d_model)
        self.unproject = nn.Linear(d_model, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 4x4 → 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 3, stride=2, padding=1, output_padding=1),  # 8x8 → 16x16
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size(0)
        x = self.encoder_cnn(x)
        x = self.flatten(x).permute(0, 2, 1)  # (B, 16, 128)
        x = self.project(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = self.unproject(x).permute(0, 2, 1).reshape(b, 128, 4, 4)
        x = self.decoder(x)
        return x

# ✅ Train + Eval
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for x, y in test_loader:
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

# ✅ Config
PREPROCESSING_TYPE = 'hilbert'
ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq'
BATCH_SIZE = 1024 * 16
win = 16

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

def run_training():
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    print(benign_p[:5])

    train_p, train_l = benign_p[:50000], benign_l[:50000]
    val_p, val_l = benign_p[50000:55000], benign_l[50000:55000]
    atk_p, atk_l = attack_p[:5000], attack_l[:5000]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    train_ds = StatisticalChannelDataset(train_p, train_l, transform, window_size=win)
    val_ds_b = StatisticalChannelDataset(val_p, val_l, transform, window_size=win)
    val_ds_a = StatisticalChannelDataset(atk_p, atk_l, transform, window_size=win)

    test_ds = ConcatDataset([val_ds_b, val_ds_a])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformerAE(in_channels=4)
    model = train_and_evaluate(model, train_loader, test_loader, device, epochs=10)
    torch.save(model.state_dict(), f'./AI/Statistic/CNN_Transformer_AE/Model/stat_cnn_transformer_ae_{win}.pth')

run_training()
