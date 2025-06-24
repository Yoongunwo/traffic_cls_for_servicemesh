import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# === 설정 ===
DATA_DIR = './Data_CIC/Session_Windows_15'
MAX_SAMPLES = 50000
WINDOW_SIZE = 15
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
THRESHOLD_PERCENTILE = 95

# === Dataset 정의 ===
class SequenceDataset(Dataset):
    def __init__(self, files, labels=None):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        if data.shape != (15, 1479):
            raise ValueError(f"Invalid shape {data.shape} in {self.files[idx]}")
        data_tensor = torch.tensor(data, dtype=torch.float32)
        if self.labels is not None:
            return data_tensor, self.labels[idx]
        else:
            return data_tensor

# === 입력 리쉐이프 ===
def reshape_input(data):  # data: [B, 15, 1479]
    B, T, F = data.shape
    H, W = 33, 45  # 33x45 = 1485, trim 6
    if F < H * W:
        pad = H * W - F
        data = torch.nn.functional.pad(data, (0, pad), mode='constant', value=0)
    elif F > H * W:
        data = data[:, :, :H * W]
    data = data.reshape(B, 1, T, H, W)
    return data

# === 모델 정의 ===
class CNN_AE(nn.Module):
    def __init__(self):
        super(CNN_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),       # [B, 8, 15, 33, 45]
            nn.ReLU(),
            nn.MaxPool3d((1, 3, 3)),                         # [B, 8, 15, 11, 15]
            nn.Conv3d(8, 16, kernel_size=3, padding=1),      # [B, 16, 15, 11, 15]
            nn.ReLU(),
            nn.MaxPool3d((1, 1, 3)),                         # [B, 16, 15, 11, 5]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=(1, 1, 3), stride=(1, 1, 3)),  # [B, 8, 15, 11, 15]
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=(1, 3, 3), stride=(1, 3, 3)),   # [B, 1, 15, 33, 45]
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# === 데이터 로딩 ===
def load_unlabeled_data(data_dir, max_samples=None):
    benign_files = sorted(glob(os.path.join(data_dir, 'benign', '*.npy')))
    if max_samples:
        benign_files = benign_files[:max_samples]
    return SequenceDataset(benign_files)

def load_test_data(data_dir, limit_each=5000):
    benign = sorted(glob(os.path.join(data_dir, 'benign', '*.npy')))[:limit_each]
    attack = sorted(glob(os.path.join(data_dir, 'attack', '*', '*.npy')))[:limit_each]
    files = benign + attack
    labels = [0] * len(benign) + [1] * len(attack)
    return SequenceDataset(files, labels), labels

# === 학습 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = load_unlabeled_data(DATA_DIR, max_samples=MAX_SAMPLES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset, test_labels = load_test_data(DATA_DIR)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CNN_AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    for x in train_loader:
        x = x.to(device)
        x = reshape_input(x)
        out = model(x)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"__ Epoch {epoch + 1} complete")

# === 평가 ===
model.eval()
scores = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        x_reshaped = reshape_input(x)
        out = model(x_reshaped)
        loss = torch.mean((x_reshaped - out) ** 2, dim=[1, 2, 3, 4])
        scores.extend(loss.cpu().numpy())

scores = np.array(scores)
labels = np.array(test_labels)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
print(f"__ Threshold: {threshold:.6f}")
print(classification_report(labels, scores > threshold, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(labels, scores):.4f}")
