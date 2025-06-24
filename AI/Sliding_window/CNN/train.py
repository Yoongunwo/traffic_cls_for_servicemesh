import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# === 설정 ===
DATA_DIR = './Data_CIC/Session_Windows_15'
BATCH_SIZE = 1024 * 8
EPOCHS = 10
LEARNING_RATE = 0.001
WINDOW_SIZE = 15

# === Dataset 정의 ===
class TrafficDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])  # shape: [T, H, W]
        data = torch.tensor(data, dtype=torch.float32) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

# === 모델 정의 ===
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * WINDOW_SIZE * 8 * 91, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# === 데이터 로딩 ===
def load_dataset(data_dir):
    benign = glob(os.path.join(data_dir, 'benign', '*.npy'))
    attacks = [f for d in glob(os.path.join(data_dir, 'attack', '*')) for f in glob(os.path.join(d, '*.npy'))]
    files = benign + attacks
    labels = [0] * len(benign) + [1] * len(attacks)
    return train_test_split(files, labels, test_size=0.2, random_state=42)

# === 학습 루프 ===
def train_model(model, loader, optimizer, criterion, device):
    model.train()

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        data = data.unsqueeze(1)  # [B, 1, T, H, W]
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# === 평가 루프 ===
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            data = data.unsqueeze(1)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    print(classification_report(all_labels, all_preds))

# === 실행 ===
train_files, test_files, train_labels, test_labels = load_dataset(DATA_DIR)
train_loader = DataLoader(TrafficDataset(train_files, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TrafficDataset(test_files, test_labels), batch_size=BATCH_SIZE)

model = Simple3DCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    train_model(model, train_loader, optimizer, criterion, device)
    print(f"✅ Epoch {epoch+1} finished")

evaluate_model(model, test_loader, device)
