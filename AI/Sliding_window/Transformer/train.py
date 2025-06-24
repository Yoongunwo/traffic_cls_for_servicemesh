import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerAutoEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.reconstructor = nn.Linear(emb_dim, input_dim)

    def forward(self, x):
        # x: [B, T, F]
        x = self.embedding(x)  # [B, T, E]
        x = self.pos_encoder(x)  # [B, T, E]
        x = x.permute(1, 0, 2)  # → [T, B, E]

        memory = self.encoder(x)  # [T, B, E]
        output = self.decoder(x, memory)  # [T, B, E]
        output = self.reconstructor(output.permute(1, 0, 2))  # [B, T, F]
        return output

import os
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 하이퍼파라미터
DATA_DIR = './Data_CIC/Session_Windows_15'
BATCH_SIZE = 1024 * 4
EPOCHS = 10
LR = 1e-3
WINDOW_SIZE = 15

class SequenceDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # [T, F]
        return torch.tensor(data, dtype=torch.float32)

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
    return files, labels

def train(model, loader, optimizer, criterion, device):
    model.train()
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}", end='\r')

def evaluate(model, files, labels, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for path in files:
            x = torch.tensor(np.load(path), dtype=torch.float32).unsqueeze(0).to(device)
            x_hat = model(x)
            score = torch.mean((x - x_hat) ** 2).item()
            scores.append(score)
    auc = roc_auc_score(labels, scores)
    print(f"ROC AUC: {auc:.4f}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = load_unlabeled_data(DATA_DIR, max_samples=50000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

sample = np.load(glob(os.path.join(DATA_DIR, 'benign', '*.npy'))[0])
input_dim = sample.shape[1]

model = TransformerAutoEncoder(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, criterion, device)
    print(f"✅ Epoch {epoch+1} complete")

# 테스트
test_files, test_labels = load_test_data(DATA_DIR)
evaluate(model, test_files, test_labels, device)
