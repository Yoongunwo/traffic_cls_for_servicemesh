import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

# === 설정 ===
DATA_DIR = './Data_CIC/Session_Windows_15'
MAX_SAMPLES = 50000  # 총 학습/테스트용 데이터 수 제한
WINDOW_SIZE = 15
BATCH_SIZE = 1024 * 32
EPOCHS = 10
LR = 1e-3
THRESHOLD_PERCENTILE = 95  # 이상 판단 기준

# === Dataset ===
class UnlabeledDataset(Dataset):
    def __init__(self, file_paths):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        return x

class LabeledDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        y = torch.tensor(self.labels[idx])
        return x, y

# === CAE 모델 ===
class CAE(nn.Module):
    def __init__(self, input_shape):
        super(CAE, self).__init__()
        T, F = input_shape
        self.encoder = nn.Sequential(
            nn.Linear(F, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, F)
        )

    def forward(self, x):
        B, T, F = x.shape
        x = x.view(B * T, F)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(B, T, -1)

# === 학습 루프 ===
def train(model, loader, optimizer, criterion, device):
    model.train()
    for x in loader:
        x = x.to(device)
        output = model(x)
        loss = criterion(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.6f}")  # 진행 상황 출력

# === 평가 ===
def compute_reconstruction_errors(model, loader, device):
    model.eval()
    errors, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            loss = torch.mean((x - out) ** 2, dim=[1, 2])  # [B]
            errors.extend(loss.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(errors), np.array(labels)

# === 데이터 로딩 ===
def load_datasets(max_samples=50000):
    benign = glob(os.path.join(DATA_DIR, 'benign', '*.npy'))
    attack = [f for d in glob(os.path.join(DATA_DIR, 'attack', '*')) for f in glob(os.path.join(d, '*.npy'))]
    print(f"🔍 Found {len(benign)} benign and {len(attack)} attack samples")
    benign = benign[:max_samples // 2]
    attack = attack[:max_samples // 2]
    all_files = benign + attack
    labels = [0] * len(benign) + [1] * len(attack)
    return benign, attack, all_files, labels

# === 실행 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
benign, attack, all_files, labels = load_datasets(MAX_SAMPLES)
input_shape = np.load(benign[0]).shape

# 훈련은 정상 세션만으로
unlabeled_ds = UnlabeledDataset(benign)
train_size = int(len(unlabeled_ds) * 0.9)
train_ds, val_ds = random_split(unlabeled_ds, [train_size, len(unlabeled_ds) - train_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# 모델 정의
model = CAE(input_shape).to(device)
# 모델을 불러오려면 다음 줄의 주석을 해제하세요
# model.load_state_dict(torch.load(f'./AI/Sliding_window/CAE/Model/cae_model_{EPOCHS}.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 학습
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion, device)
    print(f"✅ Epoch {epoch+1} complete")

torch.save(model.state_dict(), f'./AI/Sliding_window/CAE/Model/cae_model_{EPOCHS}.pth')

# 평가
test_loader = DataLoader(LabeledDataset(all_files, labels), batch_size=4096*4)
errors, y_true = compute_reconstruction_errors(model, test_loader, device)
threshold = np.percentile(errors[y_true == 0], THRESHOLD_PERCENTILE)
y_pred = (errors > threshold).astype(int)
print(f"🔎 Threshold: {threshold:.6f}")
print(classification_report(y_true, y_pred))
print(f"ROC AUC: {roc_auc_score(y_true, errors):.4f}")