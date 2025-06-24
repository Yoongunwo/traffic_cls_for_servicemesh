import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import OneClassSVM

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_k8s/Session_Windows_15'
H, W = 34, 44
BATCH_SIZE = 2**10
EPOCHS = 10
FEAT_DIM = 128

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

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        #     nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
        #     nn.Linear(256, out_dim)
        # )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Load paths ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'save_front/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

# === Deep SVDD Loss ===
def deep_svdd_loss(z, c):
    return torch.mean(torch.sum((z - c) ** 2, dim=1))
def soft_svdd_loss(z, c, R=1e-3, nu=0.1):
    dists = torch.sum((z - c) ** 2, dim=1)
    return R**2 + (1/nu) * torch.mean(F.relu(dists - R**2))


# === Initialize center from random benign batch ===
def initialize_center(dataloader, model):
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            z = model(x)
            return torch.mean(z, dim=0)

# === Train ===

def train_deep_svdd():
    print("Training Deep SVDD on benign samples...")
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    train_loader = DataLoader(SingleFrameDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder().to(DEVICE)
    center = initialize_center(train_loader, model).detach().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            z = model(x)
            loss = deep_svdd_loss(z, center)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), f'./AI_Real/DeepSVDD/Model/deep_svdd_encoder_k8s_{EPOCHS}.pth')
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
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)


def plot():
    # === PCA Plot ===
    print("Evaluating on benign and attack samples...")
    benign_test = load_paths(DATA_DIR, is_attack=False, limit=1000)
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=1000)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    X = np.vstack([benign_feats, attack_feats])
    y = [0]*len(benign_feats) + [1]*len(attack_feats)

    pca = PCA(n_components=2).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    umap_embed = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    # === Plotting
    def plot_scatter(X_embedded, title, filename):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_embedded[:len(benign_feats), 0], X_embedded[:len(benign_feats), 1],
                    label='Benign', alpha=0.6, s=10)
        plt.scatter(X_embedded[len(benign_feats):, 0], X_embedded[len(benign_feats):, 1],
                    label='Attack', alpha=0.6, s=10, color='red')
        plt.title(title)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_scatter(pca, "Deep SVDD Feature Space (PCA)", "./Pre_train/Single/Map/pca_deep_svdd_soft.png")
    plot_scatter(tsne, "Deep SVDD Feature Space (t-SNE)", "./Pre_train/Single/Map/tsne_deep_svdd_soft.png")
    plot_scatter(umap_embed, "Deep SVDD Feature Space (UMAP)", "./Pre_train/Single/Map/umap_deep_svdd_soft.png")


def main():
    model = train_deep_svdd()

    benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
    benign_feats = extract_features(benign_train, model)

    kernel = 'rbf'  # or 'linear', 'poly', etc.
    gamma = 10.0
    nu = 0.1

    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, is_attack=False, limit=1000)
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=1000)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    test_feats = np.vstack([benign_feats, attack_feats])
    test_labels = np.array([0] * len(benign_feats) + [1] * len(attack_feats))
    scores = ocsvm.decision_function(test_feats)
    preds = (scores < 0).astype(int)  # OCSVM: 음수면 이상 탐지

    print("\n=== One-Class SVM Evaluation ===")
    print(classification_report(test_labels, preds, target_names=["Benign", "Attack"]))
    auc = roc_auc_score(test_labels, -scores)

    precisions, recalls, thresholds = precision_recall_curve(test_labels, scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    preds_best = (-scores < best_threshold).astype(int)
    print(f"Best Threshold: {best_threshold:.6f}")
    print(classification_report(test_labels, preds_best, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC Score: {auc:.4f}")

if __name__ == "__main__":
    main()