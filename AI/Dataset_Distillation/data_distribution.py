import os
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44

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

# === Util: 파일 경로 로딩 ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

def load_image_paths_single_window(data_dir, is_attack=False, per_class=1000):
    files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
    samples = [[f] for f in files]
    return samples

# === CNN Feature 추출 ===
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
        feats.append(np.mean(feat, axis=0))  # [T, F] → [F]
    return np.stack(feats)

# === 1. Load test samples ===
print("Loading test samples...")
benign_paths = load_image_paths_single_window(DATA_DIR, is_attack=False, per_class=1000)
attack_paths = load_image_paths_single_window(DATA_DIR, is_attack=True, per_class=1000)

# === 2. Load distilled synthetic images ===
print("Loading distilled synthetic images...")
syn_images = torch.load('./AI/Dataset_Distillation/Warm_Start/distilled_images_warmstart_dual.pt').to(DEVICE)

# === 3. Initialize encoder ===
print("Extracting features...")
cnn = CNNEncoder().to(DEVICE)

# === 4. Extract features ===
benign_feats = extract_features(benign_paths, cnn)
attack_feats = extract_features(attack_paths, cnn)
with torch.no_grad():
    syn_feats = cnn(syn_images).cpu().numpy()

# === 5. PCA for visualization ===
print("Performing PCA...")
X_all = np.vstack([benign_feats, attack_feats, syn_feats])
y_all = (
    [0] * len(benign_feats) +  # Benign
    [1] * len(attack_feats) +  # Attack
    [2] * len(syn_feats)       # Distilled
)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_all)

# === 6. Plot ===
colors = ['tab:blue', 'tab:red', 'tab:green']
labels = ['Benign', 'Attack', 'Distilled']

plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    mask = np.array(y_all) == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.6, label=label, color=colors[i])

plt.title("PCA of CNN Feature Space")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_feature_space_warm.png")
plt.show()
