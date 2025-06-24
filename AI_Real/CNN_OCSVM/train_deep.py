import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import random

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**10
EPOCHS = 10
FEAT_DIM = 128
TEMPERATURE = 0.1

# === Dataset for Contrastive ===
class ContrastiveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

    def augment(self, vec):
        noise = np.random.normal(0, 0.01, size=vec.shape)
        return np.clip(vec + noise, 0, 1)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])[0]
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        if x.shape[0] < self.HW:
            x = np.pad(x, (0, self.HW - x.shape[0]))
        vec = x[:self.HW]
        img1 = self.augment(vec).reshape(1, H, W)
        img2 = self.augment(vec).reshape(1, H, W)
        return torch.tensor(img1, dtype=torch.float32), torch.tensor(img2, dtype=torch.float32)

# === Encoder ===
class Encoder_6x64(nn.Module):
    def __init__(self, out_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # H/2 x W/2

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # H/4 x W/4

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((2, 2)),  # 2x2

            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoderv2(nn.Module):
    def __init__(self, out_dim=128):  # FEAT_DIM과 동일하게 설정
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # ✅ 한 번만 사용 (34x44 → 17x22)

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            # ❌ MaxPool2 제거

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.AdaptiveAvgPool2d((4, 4)),  # ✅ 더 큰 feature map 보존
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoder_4x128(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 34x44 → 17x22

            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
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
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

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

# === Training Contrastive Encoder ===
def train_encoder():
    print("Training contrastive encoder...")
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    dataset = ContrastiveDataset(train_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Encoder_4x128().to(DEVICE)
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

    torch.save(model.state_dict(), f'./AI_Real/CNN_OCSVM/Model/cnn_deep_contrastive_encoder_{TEMPERATURE}_{EPOCHS}_4x128.pth')
    return model

# === Main Pipeline ===
def main():
    model = train_encoder()

    benign_train = load_paths(DATA_DIR, is_attack=False, limit=10000)
    benign_feats = extract_features(benign_train, model)

    gamma = 0.1
    nu = 0.01
    kernel = 'rbf'
    print(f"Epochs: {EPOCHS}")
    print(f"Training OCSVM with gamma={gamma}, nu={nu}, kernel={kernel}")
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    ocsvm.fit(benign_feats)

    benign_test = load_paths(DATA_DIR, is_attack=False, limit=12500)
    attack_test = load_paths(DATA_DIR, is_attack=True, limit=12500)
    benign_feats = extract_features(benign_test, model)
    attack_feats = extract_features(attack_test, model)

    X = np.vstack([benign_feats, attack_feats])
    y = np.array([0]*len(benign_feats) + [1]*len(attack_feats))
    scores = ocsvm.decision_function(X)

    print("\n=== OCSVM Evaluation ===")
    precision, recall, thresholds = precision_recall_curve(y, -scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold (F1): {best_thresh:.4f}")
    preds = (scores < -best_thresh).astype(int)
    print(classification_report(y, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC Score with Best Threshold: {roc_auc_score(y, -scores):.4f}")


if __name__ == "__main__":
    main()
