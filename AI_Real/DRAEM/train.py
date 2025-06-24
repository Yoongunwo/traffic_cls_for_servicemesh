import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import random

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
EPOCHS = 10
BATCH_SIZE = 2**11
NOISE_STD = 10.0

# === Dataset ===
class DRAEMDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.HW = H * W

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])[0]
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        if x.shape[0] < self.HW:
            x = np.pad(x, (0, self.HW - x.shape[0]))
        x = x[:self.HW].reshape(1, H, W)

        label = 0
        # if random.random() < 0.3:
        #     # simulate anomaly with block masking
        #     x_corrupt = x.copy()
        #     i = random.randint(0, H - 8)
        #     j = random.randint(0, W - 8)
        #     x_corrupt[:, i:i+8, j:j+8] = 0.0
        #     label = 1
        # else:
        #     x_corrupt = x + NOISE_STD * np.random.randn(*x.shape)
        if random.random() < 0.3:
            # 기존 block masking
            x_corrupt = x.copy()
            h, w = random.randint(6, 12), random.randint(6, 12)
            i, j = random.randint(0, H - h), random.randint(0, W - w)
            x_corrupt[:, i:i+h, j:j+w] = 0.0
            label = 1
        elif random.random() < 0.5:
            # 새로운 spike noise
            x_corrupt = x + (np.random.rand(*x.shape) < 0.01).astype(np.float32)
            label = 1
        else:
            # gaussian noise만
            std = np.random.uniform(0.01, 0.1)
            x_corrupt = x + std * np.random.randn(*x.shape)
            label = 0

        return torch.tensor(x, dtype=torch.float32), torch.tensor(x_corrupt, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

def load_paths(is_attack=False, limit=10000):
    pattern = "attack/*/*.npy" if is_attack else "benign/*.npy"
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === Model Components ===
class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out

class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(x)

# === Training ===
def train_draem(train_paths):
    loader = DataLoader(DRAEMDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    ae = DenoisingAE().to(DEVICE)
    clf = ResidualClassifier().to(DEVICE)

    opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        ae.train(); clf.train(); total_loss = 0
        for clean, corrupt, labels in loader:
            clean, corrupt, labels = clean.to(DEVICE), corrupt.to(DEVICE), labels.to(DEVICE)
            recon = ae(corrupt)
            # residual = torch.abs(clean - recon)
            residual = (clean - recon)
            residual = (residual - residual.mean()) / (residual.std() + 1e-8)  # normalize
            logits = clf(residual)

            loss_recon = F.mse_loss(recon, clean)
            loss_cls = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss_recon + 0.5 * loss_cls

            opt_ae.zero_grad(); opt_clf.zero_grad()
            loss.backward()
            opt_ae.step(); opt_clf.step()
            total_loss += loss.item()

        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader):.4f}")

    torch.save(ae.state_dict(), f"./AI_Real/DRAEM/Model/ae_cic_{EPOCHS}.pth")
    torch.save(clf.state_dict(), f"./AI_Real/DRAEM/Model/classifier_cic_{EPOCHS}.pth")
    return ae, clf

# === Inference ===
def eval_draem(ae, clf, benign_paths, attack_paths):
    def scores(paths):
        s = []
        ae.eval(); clf.eval()
        with torch.no_grad():
            for p in paths:
                x = np.load(p)[0]
                x = np.nan_to_num(x)
                x = np.clip(x, 0, 255).astype(np.float32) / 255.0
                if x.shape[0] < H*W:
                    x = np.pad(x, (0, H*W - x.shape[0]))
                x = torch.tensor(x[:H*W].reshape(1, H, W), dtype=torch.float32).to(DEVICE).unsqueeze(0)
                recon = ae(x)
                residual = torch.abs(x - recon)
                score = torch.sigmoid(clf(residual)).item()
                s.append(score)
        return np.array(s)

    b_scores = scores(benign_paths)
    a_scores = scores(attack_paths)
    y_true = np.array([0]*len(b_scores) + [1]*len(a_scores))
    y_score = np.concatenate([b_scores, a_scores])

    thresh = np.percentile(b_scores, 95)
    y_pred = (y_score > thresh).astype(int)

    print("\n=== DRAEM Evaluation ===")
    print(f"Fixed Threshold (95%% percentile): {thresh:.6f}")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    preds = (y_score > best_thresh).astype(int)

    print(f"\nBest Threshold (max F1): {best_thresh:.6f}")
    print(classification_report(y_true, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")

# === Main ===
if __name__ == "__main__":
    benign_train = load_paths(False, 10000)
    ae, clf = train_draem(benign_train)

    benign_test = load_paths(False, 1214)
    attack_test = load_paths(True, 1214)
    eval_draem(ae, clf, benign_test, attack_test)
