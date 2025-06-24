import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import random

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
BATCH_SIZE = 2**9
EPOCHS = 20
FEAT_DIM = 128
TEMPERATURE = 0.1
H, W = 1479, 5  # Input shape

# === Contrastive Dataset ===
class ContrastiveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self): return len(self.paths)

    def augment(self, x):
        noise = np.random.normal(0, 0.01, size=x.shape)
        return np.clip(x + noise, 0, 1)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])  # <-- 수정: [0] 제거
        x = x.T
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        x = x[:, :5]  # (1479, 5)

        img1 = self.augment(x).reshape(1, H, W)
        img2 = self.augment(x).reshape(1, H, W)
        return torch.tensor(img1, dtype=torch.float32), torch.tensor(img2, dtype=torch.float32)

# === Encoder ===
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

# === NT-Xent Loss ===
def nt_xent(z1, z2, temperature=TEMPERATURE):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    N = z1.shape[0]

    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # remove self-similarity

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels])  # j = i + N and i = j - N

    return F.cross_entropy(sim, labels)

# === Utility ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'save_front/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

def extract_features(paths, model):
    model.eval()
    feats = []
    with torch.no_grad():
        for path in paths:
            x = np.load(path)  # (1479, 15)
            x = x.T
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            x = x[:, :5]  # (1479, 5)

            img = torch.tensor(x.reshape(1, H, W), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            z = model(img).squeeze(0).cpu().numpy()
            feats.append(z)
    return np.array(feats)

# === Training Contrastive Encoder ===
def train_encoder():
    print("Training contrastive encoder...")
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    dataset = ContrastiveDataset(train_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x1, x2 in dataloader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            z1, z2 = model(x1), model(x2)
            loss = nt_xent(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), f'./AI_Real_v2/Model/cnn_contrastive_encoder_k8s_{EPOCHS}.pth')
    return model

# === Main Pipeline ===
def main():
    model = train_encoder()

    benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
    benign_feats = extract_features(benign_train, model)

    gamma = 1e-5
    nu = 0.01
    kernel = 'rbf'
    print(f"Epochs: {EPOCHS}")
    print(f"Training OCSVM with gamma={gamma}, nu={nu}, kernel={kernel}")
    ocsvm = OneClassSVM(gamma=gamma, nu=nu, kernel=kernel)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, is_attack=False, limit=11214)
    benign_test = benign_test[-1214:]
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=1214)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    X = np.vstack([benign_feats, attack_feats])
    y = np.array([0]*len(benign_feats) + [1]*len(attack_feats))
    scores = ocsvm.decision_function(X)
    preds = (scores < 0).astype(int)  # 음수 = 이상 탐지

    print("\n=== OCSVM Evaluation ===")
    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold (F1): {best_thresh:.4f}")
    preds = (scores < -best_thresh).astype(int)
    print(classification_report(y, preds, target_names=["Benign", "Attack"]))
    print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y, -scores):.4f}")
    

if __name__ == "__main__":
    main()
