import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

# Dataset for SimCLR
class SingleFrameDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            x1 = self.transform(img)
            x2 = self.transform(img)
        return x1, x2

# Encoder model
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.encoder(x)

# Contrastive Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        N = z1.size(0)

        representations = torch.cat([z1, z2], dim=0)  # [2N, D]
        similarity_matrix = torch.matmul(representations, representations.T)  # [2N, 2N]
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z1.device)

        # mask out self-similarity
        mask = torch.eye(2*N, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        similarity_matrix /= self.temperature
        loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss

# Feature extraction
@torch.no_grad()
def extract_embeddings(paths, model, transform, device):
    model.eval()
    features = []
    for path in tqdm(paths):
        img = Image.open(path).convert('L')
        img = transform(img).unsqueeze(0).to(device)
        emb = model(img).cpu().numpy()
        features.append(emb[0])
    return np.array(features)

# Natural sort
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

# Image path loader
def get_image_paths(folder):
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')], key=natural_key)
    return paths

# Main process
ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
BENIGN_PATH = os.path.join(ROOT, 'benign_train')
ATTACK_PATH = os.path.join(ROOT, 'attack')
WIN = 4
BATCH = 256
EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
benign_paths = get_image_paths(BENIGN_PATH)[:55000]
attack_paths = get_image_paths(ATTACK_PATH)[:5000]

simclr_transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.RandomResizedCrop(16, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_ds = SingleFrameDataset(benign_paths, simclr_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# Pretrain encoder
model = SimCLR_Encoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = NTXentLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x1, x2 in train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1 = model(x1)
        z2 = model(x2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Feature extraction
val_transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
X_benign = extract_embeddings(benign_paths[50000:], model, val_transform, device)
X_attack = extract_embeddings(attack_paths, model, val_transform, device)
y_benign = np.zeros(len(X_benign))
y_attack = np.ones(len(X_attack))

X_test = np.vstack([X_benign, X_attack])
y_test = np.concatenate([y_benign, y_attack])

# One-Class SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(extract_embeddings(benign_paths[:50000], model, val_transform, device))
X_test_scaled = scaler.transform(X_test)

clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
clf.fit(X_train)
y_pred = clf.predict(X_test_scaled)
y_pred = (y_pred == -1).astype(int)

# Evaluation
report = classification_report(y_test, y_pred, output_dict=True)
report["roc_auc"] = {"f1-score": roc_auc_score(y_test, y_pred)}
result = pd.DataFrame(report).T

print(result)

