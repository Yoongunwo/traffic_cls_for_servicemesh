import os
import numpy as np
import torch
from glob import glob
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
FEAT_DIM = 128
TEMPERATURE = 0.1

# === Encoder (must match the model used during training) ===
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# === Load data paths ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

# === Extract CNN features ===
def extract_features(paths, model):
    model.eval()
    feats = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)[0]
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < H * W:
                x = np.pad(x, (0, H * W - x.shape[0]))
            img = x[:H * W].reshape(1, H, W)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)

# === Load model and extract features ===
model = Encoder().to(DEVICE)
model.load_state_dict(torch.load(f'./Pre_train/Single/Model/cnn_contrastive_encoder_{TEMPERATURE}.pth'))

benign_paths = load_paths(DATA_DIR, is_attack=False, limit=1000)
attack_paths = load_paths(DATA_DIR, is_attack=True, limit=1000)
benign_feats = extract_features(benign_paths, model)
attack_feats = extract_features(attack_paths, model)

X = np.vstack([benign_feats, attack_feats])
y = np.array([0] * len(benign_feats) + [1] * len(attack_feats))

# # === PCA projection ===
# from sklearn.manifold import TSNE
# X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], s=10, alpha=0.5, label='Benign')
# plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], s=10, alpha=0.5, label='Attack', color='red')
# plt.title("t-SNE Visualization of CNN Features")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./Pre_train/Single/Map/pca_feature_space.png")
# plt.close()

# === Train OCSVM and score ===
ocsvm = OneClassSVM(kernel='rbf', gamma=1.0, nu=0.001)
ocsvm.fit(benign_feats)
scores = ocsvm.decision_function(X)

print(f"# Benign scores: {len(scores[y==0])}")
print(f"# Attack scores: {len(scores[y==1])}")
print(f"Attack score range: {scores[y==1].min()} to {scores[y==1].max()}")

print("Attack score stats:")
print("min:", np.min(scores[y==1]))
print("max:", np.max(scores[y==1]))
print("mean:", np.mean(scores[y==1]))

bins = np.linspace(scores.min(), scores.max(), 100)

plt.figure(figsize=(8, 6))
plt.hist(scores[y == 0], bins=bins, alpha=0.6, label='Benign', color='skyblue')
plt.hist(scores[y == 1], bins=bins, alpha=0.6, label='Attack', color='salmon')

plt.yscale("log")
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("OCSVM Score Distribution (Log Scale, Aligned Bins)")
plt.xlabel("OCSVM Score")
plt.ylabel("Frequency (log scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./AI_Real/CNN_OCSVM/Model/ocsvm_score_overlay_logscale_fixed_bins.png")
plt.close()

# === Plot OCSVM score density with log scale ===

plt.figure(figsize=(8, 6))
plt.hist(scores[y == 0], bins=100, alpha=0.6, label='Benign', color='skyblue', density=True)
plt.hist(scores[y == 1], bins=100, alpha=0.6, label='Attack', color='salmon', density=True)

plt.xlim(scores.min() - 0.0005, scores.max() + 0.0005)
plt.yscale("log")  # 밀도도 log scale로 보기
plt.axvline(x=0, color='gray', linestyle='--', label='OCSVM Decision Boundary')

plt.title("OCSVM Score Density (Benign vs Attack)")
plt.xlabel("OCSVM Score")
plt.ylabel("Density (log scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./AI_Real/CNN_OCSVM/Model/ocsvm_score_density_logscale.png")
plt.close()