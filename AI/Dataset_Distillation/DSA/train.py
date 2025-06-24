import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from glob import glob
import re

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
BATCH_SIZE = 512

# === Data loader ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

def load_image_paths_single_window(data_dir, is_attack=False, per_class=1000):
    label = int(is_attack)
    files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
    samples = [[f] for f in files]
    labels = [label] * len(samples)
    return samples, labels

# === CNN Encoder ===
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
            for t in range(x.shape[0]):
                vec = x[t]
                if vec.shape[0] < H * W:
                    vec = np.pad(vec, (0, H * W - vec.shape[0]))
                else:
                    vec = vec[:H * W]
                img = vec.reshape(1, H, W)
                imgs.append(torch.tensor(img, dtype=torch.float32))
        x_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            feat = cnn_model(x_tensor).cpu().numpy()
        feats.append(np.mean(feat, axis=0))  # [T, F] → mean pool
    return np.stack(feats)

# === Load distilled synthetic data ===
print("Loading distilled synthetic images...")
syn_images = torch.load('./AI/Dataset_Distillation/DSA/distilled_images_dsa.pt')  # shape: [N, 1, H, W]
syn_images = syn_images.to(DEVICE)

# === Load benign/attack test data ===
print("Loading test samples...")
benign_test_paths, benign_test_labels = load_image_paths_single_window(DATA_DIR, is_attack=False, per_class=10000)
attack_paths, attack_labels = load_image_paths_single_window(DATA_DIR, is_attack=True, per_class=10000)

# === Initialize CNN encoder ===
print("Extracting features from synthetic (train) and real (test) samples...")
cnn = CNNEncoder(out_dim=128).to(DEVICE)

# === Train OCSVM on synthetic image features ===
with torch.no_grad():
    syn_feats = cnn(syn_images).cpu().numpy()  # [N, F]

X_train = syn_feats
X_test = extract_features(benign_test_paths + attack_paths, cnn)
y_test = np.array(benign_test_labels + attack_labels)

# === Normalize and fit OCSVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernel = 'rbf'  # RBF 커널 사용
nu = 0.1  # 비율 파라미터z
degree = 3  # 다항식 커널의 경우 차수
gamma = 1e-3  # RBF 커널의 경우 감마 파라미터
print(f"Using One-Class SVM with kernel={kernel}, nu={nu}, degree={degree}, gamma={gamma}")

ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, degree=degree)
ocsvm.fit(X_train_scaled)

y_pred = ocsvm.predict(X_test_scaled)
y_pred = (y_pred == -1).astype(int)

# === Report ===
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
