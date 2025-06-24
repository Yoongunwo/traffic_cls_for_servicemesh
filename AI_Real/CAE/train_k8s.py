import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**12
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
    pattern = 'attack/*/*.npy' if is_attack else 'save_front/*.npy'
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

# Perceptual Loss 모듈
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU()
        )
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        fx, fy = self.features(x), self.features(y)
        return F.mse_loss(fx, fy)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def gaussian_window(self, window_size, sigma=1.5):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size):
        _1D_window = self.gaussian_window(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def forward(self, img1, img2):
        if img1.is_cuda:
            window = self.window.cuda(img1.device).type_as(img1)
        else:
            window = self.window.type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()  # SSIM은 1에 가까울수록 유사 → Loss는 1-SSIM



def total_loss_fn(x, recon, alpha=0.7):
    mse = F.mse_loss(recon, x)
    perceptual = percep_loss(recon, x)
    return alpha * mse + (1 - alpha) * perceptual


# === Training ===
train_paths = load_paths(False, limit=10000)
train_loader = DataLoader(CAEDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)

model = CAE().to(DEVICE)

# 기존 MSE + Perceptual Loss 조합
# percep_loss = PerceptualLoss().to(DEVICE)
ssim_loss = SSIMLoss(window_size=11).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x in train_loader:
        x = x.to(DEVICE)
        recon = model(x)
        loss = ssim_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{epoch+1}/{EPOCHS}] SSIM Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), f"./AI_Real/CAE/Model/cae_model_k8s_{EPOCHS}.pth")

# === Evaluation ===
benign_test = load_paths(False, limit=1214)
attack_test = load_paths(True, limit=1214)

benign_errors = compute_errors(model, benign_test)
attack_errors = compute_errors(model, attack_test)

threshold = np.percentile(benign_errors, 95)
print(f"Threshold: {threshold:.6f}")

y_true = np.array([0]*len(benign_errors) + [1]*len(attack_errors))
y_score = np.concatenate([benign_errors, attack_errors])
y_pred = (y_score > threshold).astype(int)

print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
print("ROC AUC:", roc_auc_score(y_true, y_score))

precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]

preds = (y_score > best_threshold).astype(int)
print(f"Best Threshold: {best_threshold:.6f}")
print(classification_report(y_true, preds, target_names=["Benign", "Attack"], digits=4))
print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")
