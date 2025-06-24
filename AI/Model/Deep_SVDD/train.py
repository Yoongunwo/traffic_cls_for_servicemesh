import os
import sys
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import joblib

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

class FeatureCNN32(nn.Module):
    def __init__(self):
        super(FeatureCNN32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128)
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

def train_model(device, train_loader, epoches, model_dir, model_path, center_path, threshold_path):
    model = FeatureCNN32().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    feats_init, _ = extract_features(model, train_loader, device)
    center = feats_init.mean(axis=0)
    center_tensor = torch.tensor(center, dtype=torch.float32).to(device)

    model.train()
    for epoch in range(epoches):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            f = model(x)
            loss = torch.mean((f - center_tensor) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Deep SVDD Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, model_path))
    np.save(os.path.join(model_dir, center_path), center)

    feats_test, labels_test = extract_features(model, train_loader, device)
    dists = np.linalg.norm(feats_test - center, axis=1)
    threshold = np.percentile(dists, 95)
    
    np.save(os.path.join(model_dir, threshold_path), threshold)
    

# ✅ 메인 함수
TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/Deep_SVDD/Model'
CNN_MODEL_PATH = './AI/Model/Deep_SVDD/cic_deep_svdd_model_epoch50.pth'
CENTER_PATH = './AI/Model/Deep_SVDD/cic_deep_svdd_center.npy'
THRESHOLD_PATH = './AI/Model/Deep_SVDD/cic_deep_svdd_threshold.npy'

BATCH_SIZE = 4096 * 16

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)
    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    model = FeatureCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ✅ 특징 추출 후 중심(center) 계산
    print("Initializing center from normal data")
    feats_init, _ = extract_features(model, train_loader, device)
    center = feats_init.mean(axis=0)
    center_tensor = torch.tensor(center, dtype=torch.float32).to(device)

    # ✅ Deep SVDD training (f(x) - c)^2
    print("Training Deep SVDD")
    model.train()
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            f = model(x)
            loss = torch.mean((f - center_tensor) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Deep SVDD Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), CNN_MODEL_PATH)
    np.save(CENTER_PATH, center)


    # ✅ 특징 추출 (테스트)
    print("\n🔹 Evaluating on test data")
    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    feats_test, labels_test = extract_features(model, test_loader, device)
    dists = np.linalg.norm(feats_test - center, axis=1)
    threshold = np.percentile(dists, 95)
    preds = (dists > threshold).astype(int)
    print(classification_report(labels_test, preds, digits=4))

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(THRESHOLD_PATH, threshold)

if __name__ == '__main__':
    main()
