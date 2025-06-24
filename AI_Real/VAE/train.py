import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_k8s/Session_Windows_15"
H, W = 34, 44
LATENT_DIM = 32
EPOCHS = 20
BATCH_SIZE = 2**14

# === Dataset ===
class VAEDataset(Dataset):
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
        return torch.tensor(x, dtype=torch.float32)

def load_paths(is_attack=False, limit=10000):
    pattern = "attack/*/*.npy" if is_attack else "save_front/*.npy"
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === VAE Model ===
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # → [B, 32, 17, 22]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # → [B, 64, 9, 11]
            nn.ReLU()
        )
        self.flatten_dim = 64 * 9 * 11
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, ≈18, ≈22]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, ≈36, ≈44]
            nn.Upsample(size=(34, 44), mode='bilinear', align_corners=False),  
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 9, 11)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# === Train ===
def train_vae(paths):
    loader = DataLoader(VAEDataset(paths), batch_size=BATCH_SIZE, shuffle=True)
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for x in loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader.dataset):.4f}")

    torch.save(model.state_dict(), f"./AI_Real/VAE/Model/vae_k8s_{EPOCHS}.pth")
    return model

# === Evaluate ===
def eval_vae(model, benign_paths, attack_paths):
    model.eval()
    def scores(paths):
        s = []
        with torch.no_grad():
            for p in paths:
                x = np.load(p)[0]
                x = np.nan_to_num(x)
                x = np.clip(x, 0, 255).astype(np.float32) / 255.0
                if x.shape[0] < H * W:
                    x = np.pad(x, (0, H * W - x.shape[0]))
                x = torch.tensor(x[:H * W].reshape(1, H, W), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                recon, mu, logvar = model(x)
                score = F.mse_loss(recon, x, reduction='mean').item()
                s.append(score)
        return np.array(s)

    b_scores = scores(benign_paths)
    a_scores = scores(attack_paths)
    y_true = np.array([0] * len(b_scores) + [1] * len(a_scores))
    y_score = np.concatenate([b_scores, a_scores])

    thresh = np.percentile(b_scores, 95)
    y_pred = (y_score > thresh).astype(int)
    print(f"\n=== VAE Eval ===\n95% Threshold: {thresh:.6f}")
    print(classification_report(y_true, y_pred))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    preds = (y_score > best_thresh).astype(int)
    print(f"\nBest F1 Threshold: {best_thresh:.6f}")
    print(classification_report(y_true, preds, digits=4))
    print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")

# === Main ===
if __name__ == "__main__":
    benign_train = load_paths(False, 10000)
    model = train_vae(benign_train)
    benign_test = load_paths(False, 1214)
    attack_test = load_paths(True, 1214)
    eval_vae(model, benign_test, attack_test)
