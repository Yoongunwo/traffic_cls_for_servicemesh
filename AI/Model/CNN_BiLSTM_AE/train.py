import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from torchvision import transforms
import numpy as np
import os
import sys

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

# ✅ 모델 정의
class CNN_BiLSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim=256, cnn_channels=32, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.encoder_lstm = nn.LSTM(cnn_channels, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTM(2 * lstm_hidden, cnn_channels, lstm_layers, batch_first=True)
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(cnn_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # (B, 1, 256) - input to Conv1d
        x = self.encoder_cnn(x)       # (B, C, L/2)
        x = x.permute(0, 2, 1)        # (B, L/2, C)
        x, _ = self.encoder_lstm(x)   # (B, L/2, 2H)
        x, _ = self.decoder_lstm(x)   # (B, L/2, C)
        x = x.permute(0, 2, 1)        # (B, C, L/2)
        x = self.decoder_cnn(x)       # (B, 1, L)
        return x.squeeze(1)           # (B, L)

# ✅ 학습 함수
def train(model, dataloader, device, epochs=50):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0
        for x, _ in dataloader:
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[{epoch+1}/{epochs}] Loss: {total / len(dataloader):.4f}")

# ✅ 이상 탐지 평가
def evaluate(model, dataloader, threshold, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            loss = F.mse_loss(out, x, reduction='none').view(x.size(0), -1).mean(dim=1)
            scores.extend(loss.cpu().numpy())
            labels.extend(y.numpy())
    preds = (np.array(scores) > threshold).astype(int)
    print("\n CNN-BiLSM-AE Classification Report:")
    print(classification_report(labels, preds, digits=4, zero_division=0))
    print(f"ROC AUC: {roc_auc_score(labels, scores):.4f}")

def train_model(device, train_loader, epoches, model_dir, model_path, threshold_path):
    # 모델 학습
    model = CNN_BiLSTM_Autoencoder(input_dim=256)
    train(model, train_loader, device, epoches)

    # threshold 계산 (3-sigma 기준)
    losses = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            l = F.mse_loss(out, x, reduction='none').view(x.size(0), -1).mean(dim=1)
            losses.extend(l.cpu().numpy())
    threshold = np.mean(losses) + 3 * np.std(losses)
    print(f"\n✅ Threshold: {threshold:.6f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_path))
    np.save(os.path.join(model_dir, threshold_path), np.array([threshold]))

TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/CNN_BiLSTM_AE/Model'
MODEL_PATH = './AI/Model/CNN_BiLSTM_AE/Model/cic_cnn_bilstm_ae_epoch50.pth'
THRESHOLD_PATH = './AI/Model/CNN_BiLSTM_AE/Model/cic_threshold.npy'

BATCH_SIZE = 4096

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # ⬅️ flatten: (1,16,16) → (256,)
    ])

    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 학습
    model = CNN_BiLSTM_Autoencoder(input_dim=256)
    train(model, train_loader, device)

    # threshold 계산 (3-sigma 기준)
    losses = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            l = F.mse_loss(out, x, reduction='none').view(x.size(0), -1).mean(dim=1)
            losses.extend(l.cpu().numpy())
    threshold = np.mean(losses) + 3 * np.std(losses)
    print(f"\n✅ Threshold: {threshold:.6f}")

    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # save
    # model save
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    np.save(THRESHOLD_PATH, np.array([threshold]))

    # 평가
    evaluate(model, test_loader, threshold, device)

if __name__ == '__main__':
    main()
