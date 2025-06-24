import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from tqdm import tqdm

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**9
EPOCHS = 20
TIMESTEPS = 50

# === Dataset ===
class PacketDataset(Dataset):
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

def load_paths(is_attack, limit=10000):
    pattern = "attack/*/*.npy" if is_attack else "save_front/*.npy"
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === UNet-like Denoiser ===
class Denoiser(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels+1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1)
        )

    def forward(self, x, t):
        B = x.shape[0]
        t_embed = (t / TIMESTEPS).view(B, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_embed], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# === Diffusion Schedule ===
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x_start, t, noise):
    sqrt_alpha_cumprod = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[t]) ** 0.5
    return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

# === Training Function ===
def train(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            x = x.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            x_noisy = torch.stack([q_sample(x[i], t[i], noise[i]) for i in range(x.size(0))])
            pred = model(x_noisy, t.float().view(-1, 1, 1, 1))
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")
    return model

# === Evaluation ===
def compute_scores(model, paths):
    model.eval()
    scores = []
    with torch.no_grad():
        for path in paths:
            x = np.load(path)[0]
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            if x.shape[0] < H * W:
                x = np.pad(x, (0, H * W - x.shape[0]))
            x = torch.tensor(x[:H * W].reshape(1, H, W), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            noise = torch.randn_like(x)
            t = torch.randint(0, TIMESTEPS, (1,), device=DEVICE)
            x_noisy = q_sample(x[0], t[0], noise[0]).unsqueeze(0)
            pred_noise = model(x_noisy, t.float().view(-1, 1, 1, 1))
            score = F.mse_loss(pred_noise, noise).item()
            scores.append(score)
    return np.array(scores)

# === Main ===
train_paths = load_paths(False, limit=10000)
train_loader = DataLoader(PacketDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
model = Denoiser().to(DEVICE)
model = train(model, train_loader)
torch.save(model.state_dict(), f"./AI_Real/Diffusion_AE/Model/diffusion_ae_k8s_{EPOCHS}.pth")

benign_test = load_paths(False, limit=1214)
attack_test = load_paths(True, limit=1214)
benign_scores = compute_scores(model, benign_test)
attack_scores = compute_scores(model, attack_test)

threshold = np.percentile(benign_scores, 95)
y_true = np.array([0]*len(benign_scores) + [1]*len(attack_scores))
y_score = np.concatenate([benign_scores, attack_scores])
y_pred = (y_score > threshold).astype(int)

print("\n=== Evaluation ===")
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