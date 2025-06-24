import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**10
EPOCHS = 20

# === Dataset ===
class CAEDataset(Dataset):
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
        return torch.tensor(x[:self.HW].reshape(1, H, W), dtype=torch.float32)

# === CAE Model ===
class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),  # [B, 32, 34, 44]
            nn.MaxPool2d(2),                            # [B, 32, 17, 22]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), # [B, 64, 17, 22]
            nn.MaxPool2d(2),                            # [B, 64, 8, 11]
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU() # [B, 128, 8, 11]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(), # → [B, 64, 16, 22]
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # → [B, 32, 32, 44]
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),                   # → [B, 1, 32, 44]
            nn.Upsample(size=(34, 44), mode='bilinear', align_corners=False)  # 최종 크기 맞추기
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# === Utility ===
def load_paths(is_attack, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

def compute_errors(model, paths):
    model.eval()
    errors = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = [CAEDataset([p])[0] for p in paths[i:i+BATCH_SIZE]]
        x = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            recon = model(x)
            loss = F.mse_loss(recon, x, reduction='none')
            errors.extend(loss.view(x.size(0), -1).mean(dim=1).cpu().numpy())
    return np.array(errors)

# === Training ===
train_paths = load_paths(False, limit=10000)
train_loader = DataLoader(CAEDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)

model = CAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x in train_loader:
        x = x.to(DEVICE)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), f"./AI_Real/CAE/Model/cae_model_{EPOCHS}.pth")

# === Evaluation ===
benign_test = load_paths(False, limit=5000)
attack_test = load_paths(True, limit=5000)

benign_errors = compute_errors(model, benign_test)
attack_errors = compute_errors(model, attack_test)

threshold = np.percentile(benign_errors, 95)
print(f"Threshold: {threshold:.6f}")

y_true = np.array([0]*len(benign_errors) + [1]*len(attack_errors))
y_score = np.concatenate([benign_errors, attack_errors])
y_pred = (y_score > threshold).astype(int)

print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
print("ROC AUC:", roc_auc_score(y_true, y_score))
