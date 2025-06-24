import os, re
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Config
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 파싱 ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

# === Dataset Load ===
def load_image_sequence_paths(data_dir, is_attack=False, per_class=1000, window=5):
    label = int(is_attack)
    files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
    session_map = defaultdict(list)
    for f in files:
        sid, idx = parse_name(f)
        session_map[sid].append((idx, f))
    samples, labels = [], []
    for fs in session_map.values():
        fs.sort()
        paths = [f for _, f in fs]
        for i in range(len(paths) - window + 1):
            samples.append(paths[i:i+window])
            labels.append(label)
    return samples, labels

# === CNN Feature Extractor ===
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Feature Extractor ===
def extract_features(samples, cnn_model):
    cnn_model.eval()
    feats = []
    for seq in samples:
        imgs = []
        for path in seq:
            x = np.load(path)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] < H * W:
                x = np.pad(x, ((0, 0), (0, H * W - x.shape[1])))
            elif x.shape[1] > H * W:
                x = x[:, :H * W]
            x = x.reshape(x.shape[0], 1, H, W)
            imgs.append(torch.tensor(x, dtype=torch.float32))  # [15, 1, H, W]
        x_tensor = torch.stack(imgs).view(-1, 1, H, W).to(DEVICE)
        with torch.no_grad():
            feat = cnn_model(x_tensor).cpu().numpy()  # [T*15, D]
        feats.append(np.mean(feat, axis=0))  # mean over window
    return np.stack(feats)

# === Load Data ===
print("Loading benign (train)...")
benign_train_paths, _ = load_image_sequence_paths(DATA_DIR, is_attack=False, per_class=100000)

print("Loading benign (test)...")
benign_test_paths, benign_test_labels = load_image_sequence_paths(DATA_DIR, is_attack=False, per_class=10000)

print("Loading attack...")
attack_paths, attack_labels = load_image_sequence_paths(DATA_DIR, is_attack=True, per_class=10000)

# === Extract Features ===
cnn = CNNEncoder().to(DEVICE)
X_train = extract_features(benign_train_paths, cnn)           # Only benign for training
X_test = extract_features(benign_test_paths + attack_paths, cnn)
y_test = np.array(benign_test_labels + attack_labels)

# === One-Class SVM Training & Evaluation ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma=1.0)
ocsvm.fit(X_train_scaled)

y_pred = ocsvm.predict(X_test_scaled)
y_pred = (y_pred == -1).astype(int)  # -1 → anomaly → label 1

print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
