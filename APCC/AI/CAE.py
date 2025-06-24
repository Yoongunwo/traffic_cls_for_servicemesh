import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
BATCH_SIZE = 256
EPOCHS = 10

# === Dataset ===
class SingleFrameDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        vec = x[0]
        if vec.shape[0] < self.HW:
            vec = np.pad(vec, (0, self.HW - vec.shape[0]))
        img = vec[:self.HW].reshape(1, H, W)
        return torch.tensor(img, dtype=torch.float32)

# === CAE ===
class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),  # [1, 34, 44] -> [32, 34, 44]
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # -> [64, 17, 22]
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # -> [128, 9, 11]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),  # -> [64, 18, 22]
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # -> [32, 36, 44]
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),                   # -> [1, 36, 44]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = F.interpolate(x_hat, size=(34, 44), mode='bilinear', align_corners=False)
        return x_hat

# === Load paths ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

# === Reconstruction error ===
def reconstruction_errors(model, paths):
    model.eval()
    errors = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            vec = x[0]
            if vec.shape[0] < H * W:
                vec = np.pad(vec, (0, H * W - vec.shape[0]))
            img = vec[:H * W].reshape(1, H, W)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recon = model(img_tensor)
            loss = nn.functional.mse_loss(recon, img_tensor).item()
            errors.append(loss)
    return np.array(errors)

# === Train and Evaluate CAE ===
def train_and_evaluate_cae():
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    test_benign_paths = load_paths(DATA_DIR, is_attack=False, limit=1000)
    test_attack_paths = load_paths(DATA_DIR, is_attack=True, limit=1000)

    train_loader = DataLoader(SingleFrameDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    model = CAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")
    
    torch.save(model.state_dict(), './APCC/Model/cae_model.pth')

    # Inference
    benign_errors = reconstruction_errors(model, test_benign_paths)
    attack_errors = reconstruction_errors(model, test_attack_paths)

    all_errors = np.concatenate([benign_errors, attack_errors])
    labels = np.array([0]*len(benign_errors) + [1]*len(attack_errors))

    # === 최적 threshold 계산 (F1 기준)
    precisions, recalls, thresholds = precision_recall_curve(labels, all_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"Best threshold (F1-optimal): {best_threshold:.6f}")

    # === 예측 및 평가
    preds = (all_errors > best_threshold).astype(int)

    print("\n=== CAE Anomaly Detection Evaluation (Optimal Threshold) ===")
    print(classification_report(labels, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC: {roc_auc_score(labels, all_errors):.4f}")
    print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")

train_and_evaluate_cae()
