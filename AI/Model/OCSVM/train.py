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
    
class DeepFeatureCNN(nn.Module):
    def __init__(self):
        super(DeepFeatureCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 8, 8]

            nn.Conv2d(64, 128, 3, padding=1),  # [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 4, 4]

            nn.Conv2d(128, 256, 3, padding=1),  # [B, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))  # [B, 256, 2, 2]
        )
        self.fc = nn.Linear(256 * 2 * 2, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

def train_model(device, train_loader, epoches, model_dir, cnn_path, ocsvm_path):
    model = DeepFeatureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ✅ 간단한 pretrain
    model.train()

    for epoch in range(epoches):
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

    feats_train, _ = extract_features(model, train_loader, device)

    print("\n🔹 One-Class SVM")
    clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    clf.fit(feats_train)

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, cnn_path))
    joblib.dump(clf, os.path.join(model_dir, ocsvm_path))

# ✅ 메인 함수

TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/OCSVM/Model'
CNN_MODEL_PATH = './AI/Model/OCSVM/Model/cic_ocsvm_deep_cnn_epoch50.pth'
OCSVM_MODEL_PATH = './AI/Model/OCSVM/Model/cic_deep_ocsvm.pkl'

BATCH_SIZE = 4096*8

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])


    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    # model = FeatureCNN().to(device)
    model = DeepFeatureCNN().to(device)
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

    feats_train, _ = extract_features(model, train_loader, device)

    print("\n🔹 One-Class SVM")
    clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    clf.fit(feats_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), CNN_MODEL_PATH)
    joblib.dump(clf, OCSVM_MODEL_PATH)

    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    feats_test, labels_test = extract_features(model, test_loader, device)

    preds = clf.predict(feats_test)
    preds = [0 if p == 1 else 1 for p in preds]  # 이상이면 1
    print(classification_report(labels_test, preds, digits=4))


if __name__ == '__main__':
    main()
