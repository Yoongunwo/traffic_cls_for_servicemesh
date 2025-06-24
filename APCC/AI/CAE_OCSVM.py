# Re-import required modules after kernel reset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H, W = 34, 44
BATCH_SIZE = 256
EPOCHS = 10
FEAT_DIM = 128

# === Dummy paths (replace with real ones during local run) ===
DATA_DIR = './Data_CIC/Session_Windows_15'

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
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.Upsample(size=(H, W), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# === Load paths ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

# === Train CAE ===
def train_cae():
    print("Training CAE on benign samples...")
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    train_loader = DataLoader(SingleFrameDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)

    model = CAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            _, recon = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), './APCC/Model/cae_ocsvm.pth')
    return model

# === Inference ===
def extract_features(paths, model):
    model.eval()
    feats = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            vec = x[0]
            if vec.shape[0] < H * W:
                vec = np.pad(vec, (0, H * W - vec.shape[0]))
            img = vec[:H * W].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z, _ = model(img)
            feats.append(z.squeeze(0).cpu().numpy())
    return np.array(feats)

# === Run CAE + One-Class SVM ===
def run():
    model = train_cae()

    benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
    benign_feats = extract_features(benign_train, model)

    ocsvm = OneClassSVM(kernel='rbf', gamma=10.0, nu=0.1)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, is_attack=False, limit=1000)
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=1000)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    test_feats = np.vstack([benign_feats, attack_feats])
    test_labels = np.array([0] * len(benign_feats) + [1] * len(attack_feats))
    scores = ocsvm.decision_function(test_feats)
    fpr, tpr, thresholds = roc_curve(test_labels, -scores)

    f1_scores = [f1_score(test_labels, (-scores >= thresh).astype(int)) for thresh in thresholds]
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    print(f"Best threshold (by F1): {best_thresh:.4f}, F1: {f1_scores[best_idx]:.4f}")

    # 새로운 예측 생성
    best_preds = (-scores >= best_thresh).astype(int)

    # 결과 출력
    print("\n=== One-Class SVM Evaluation with Optimal Threshold ===")
    print(classification_report(test_labels, best_preds, target_names=["Benign", "Attack"], digits=4))
    auc = roc_auc_score(test_labels, -scores)
    print(f"ROC AUC Score: {auc:.4f}")

run()
