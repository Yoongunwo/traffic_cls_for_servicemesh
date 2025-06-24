import os, numpy as np, torch, torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
PATCH_SIZE = 4
BATCH_SIZE = 2**10
EPOCHS = 15

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

# === Transformer AutoEncoder ===
class TransformerAutoEncoder(nn.Module):
    def __init__(self, img_size=(H, W), patch_size=PATCH_SIZE, emb_dim=128, depth=4, heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        self.output_proj = nn.Linear(emb_dim, patch_size * patch_size)
        self.unpatchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.repatch = nn.Fold(output_size=img_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        patches += self.pos_embed
        encoded = self.encoder(patches)
        decoded = self.decoder(patches, encoded)
        out = self.output_proj(decoded)  # [B, N, P*P]
        out = out.transpose(1, 2).reshape(B, -1, self.n_patches)
        img = self.repatch(out)
        return img

# === Helper ===
def load_paths(data_dir, is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(data_dir, pattern)))[:limit]

def reconstruction_errors(model, paths):
    model.eval()
    errors = []
    with torch.no_grad():
        for f in paths:
            x = np.load(f)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            vec = x[0]
            if vec.shape[0] < H * W:
                vec = np.pad(vec, (0, H * W - vec.shape[0]))
            img = vec[:H * W].reshape(1, H, W)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            recon = model(img_tensor)
            loss = nn.functional.mse_loss(recon, img_tensor).item()
            errors.append(loss)
    return np.array(errors)

# === Train and Evaluate ===
def train_and_evaluate():
    train_paths = load_paths(DATA_DIR, is_attack=False, limit=10000)
    test_benign_paths = load_paths(DATA_DIR, is_attack=False, limit=1000)
    test_attack_paths = load_paths(DATA_DIR, is_attack=True, limit=1000)

    train_loader = DataLoader(SingleFrameDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    model = TransformerAutoEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x in train_loader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total_loss / len(train_loader):.4f}")
    
    torch.save(model.state_dict(), './APCC/Model/transformer_ae.pth')

    # Evaluation
    benign_errors = reconstruction_errors(model, test_benign_paths)
    attack_errors = reconstruction_errors(model, test_attack_paths)

    all_errors = np.concatenate([benign_errors, attack_errors])
    labels = np.array([0]*len(benign_errors) + [1]*len(attack_errors))

    # Optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(labels, all_errors)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    preds = (all_errors > best_threshold).astype(int)

    print(f"\n=== Transformer AE Anomaly Detection ===")
    print(f"Best Threshold: {best_threshold:.6f}")
    print(classification_report(labels, preds, target_names=["Benign", "Attack"], digits=4))
    print(f"ROC AUC: {roc_auc_score(labels, all_errors):.4f}")
    print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
