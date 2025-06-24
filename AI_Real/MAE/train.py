import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
PATCH_SIZE = 5
EPOCHS = 20
BATCH_SIZE = 2**12
MASK_RATIO = 0.75

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
        x = x[:self.HW].reshape(1, H, W)
        return torch.tensor(x, dtype=torch.float32)

def load_paths(is_attack=False, limit=10000):
    pattern = "attack/*/*.npy" if is_attack else "benign/*.npy"
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === MAE Model ===
class PatchEmbed(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(1, 64, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)  # [B, 64, H', W']

class MAE(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed = PatchEmbed(patch_size)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4), num_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(64, patch_size * patch_size),
            nn.ReLU(),
            nn.Linear(patch_size * patch_size, patch_size * patch_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        N, D = patches.shape[1], patches.shape[2]
        len_mask = int(N * self.mask_ratio)

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, len_mask:]

        patches_keep = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        enc_out = self.encoder(patches_keep)

        dec_tokens = torch.zeros(B, N, D, device=x.device)
        dec_tokens.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), enc_out)
        rec_patches = self.decoder(dec_tokens).view(B, N, 1, self.patch_size, self.patch_size)

        # Fold back
        H_out = H // self.patch_size
        W_out = W // self.patch_size
        rec_img = torch.zeros(B, 1, H, W, device=x.device)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                rec_img[:, :, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size] = rec_patches[:, idx]
                idx += 1
        return rec_img

# === Training & Evaluation ===
def train_mae(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCHS):
        total = 0
        for x in loader:
            x = x.to(DEVICE)
            rec = model(x)
            loss = F.mse_loss(rec, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total / len(loader):.4f}")
    return model

def eval_mae(model, benign_paths, attack_paths):
    def recon_err(paths):
        errs = []
        model.eval()
        with torch.no_grad():
            for p in paths:
                x = np.load(p)[0]
                x = np.nan_to_num(x)
                x = np.clip(x, 0, 255).astype(np.float32) / 255.0
                if x.shape[0] < H * W:
                    x = np.pad(x, (0, H * W - x.shape[0]))
                x = torch.tensor(x[:H*W].reshape(1, H, W), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                rec = model(x)
                err = F.mse_loss(rec, x).item()
                errs.append(err)
        return np.array(errs)

    b_scores = recon_err(benign_paths)
    a_scores = recon_err(attack_paths)
    y_true = np.array([0]*len(b_scores) + [1]*len(a_scores))
    y_score = np.concatenate([b_scores, a_scores])
    thresh = np.percentile(b_scores, 95)
    y_pred = (y_score > thresh).astype(int)

    print("\n=== MAE Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    preds = (y_score > best_threshold).astype(int)
    print(f"Best Threshold: {best_threshold:.6f}")
    print(classification_report(y_true, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")

# === Run ===
if __name__ == "__main__":
    benign_train = load_paths(False, 10000)

    train_loader = DataLoader(PacketDataset(benign_train), batch_size=BATCH_SIZE, shuffle=True)
    model = MAE().to(DEVICE)
    model = train_mae(model, train_loader)
    torch.save(model.state_dict(), f"./AI_Real/MAE/Model/mae_model_{EPOCHS}.pth")

    benign_test = load_paths(False, 5000)
    attack_test = load_paths(True, 5000)
    eval_mae(model, benign_test, attack_test)
