# ✅ OC-SVM 및 Deep SVDD 기반 이상 탐지 구현

import os
import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from collections import Counter

import joblib

import os
import sys

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

# ✅ 간단한 피처 추출 CNN
class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128)
        )

    def forward(self, x):
        return self.encoder(x)

# ✅ 특징 추출 함수
@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        f = model(x).cpu().numpy()
        feats.extend(f)
        labels.extend(y.numpy())
    return np.array(feats), np.array(labels)

# ✅ 메인 함수

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_train = cnn_train.PacketImageDataset('./Data/byte_16/front_image/train', transform, is_flat_structure=True, label=0)
    normal_test = cnn_train.PacketImageDataset('./Data/byte_16/front_image/test', transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset('./Data/attack_to_byte_16', transform, is_flat_structure=False, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    train_loader = DataLoader(normal_train, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = FeatureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ✅ 간단한 pretrain
    model.train()

    epochs = 50

    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            f = model(x)
            loss = criterion(f, f.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Pretrain Loss: {total_loss / len(train_loader):.4f}")

    # ✅ 특징 추출
    feats_train, _ = extract_features(model, train_loader, device)
    feats_test, labels_test = extract_features(model, test_loader, device)

    # ✅ Deep SVDD (중심 거리 기반)
    print("\n🔹 Deep SVDD")
    center = feats_train.mean(axis=0)
    dists = np.linalg.norm(feats_test - center, axis=1)
    threshold = np.percentile(dists, 95)
    preds = (dists > threshold).astype(int)
    print(classification_report(labels_test, preds, digits=4))

    os.makedirs("./AI/Model/Deep_SVDD/Model", exist_ok=True)
    torch.save(model.state_dict(), "./AI/Model/Deep_SVDD/front_deep_svdd_model.pth")
    np.save("./AI/Model/Deep_SVDD/front_deep_svdd_center.npy", center)
    np.save("./AI/Model/Deep_SVDD/front_deep_svdd_threshold.npy", threshold)

if __name__ == '__main__':
    main()
