import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
H, W = 34, 44
BATCH_SIZE = 2**4
EPOCHS = 20

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

def load_paths(is_attack=False, limit=10000):
    pattern = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    return sorted(glob(os.path.join(DATA_DIR, pattern)))[:limit]

# === Transformer AutoEncoder ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAE(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.HW = H * W
        self.in_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.HW)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.HW, 1)
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)  # [seq, batch, dim]
        memory = self.encoder(x)
        out = self.decoder(x, memory)
        out = self.out_proj(out).permute(1, 0, 2).view(B, 1, H, W)
        return out

# === Training ===
def train(model, loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        total = 0
        for x in loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[{epoch+1}/{EPOCHS}] Loss: {total / len(loader):.4f}")
    return model

# === Evaluation ===
def compute_errors(model, paths):
    model.eval()
    errs = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = [PacketDataset([p])[0] for p in paths[i:i+BATCH_SIZE]]
        x = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            recon = model(x)
            loss = F.mse_loss(recon, x, reduction='none')
            errs.extend(loss.view(x.size(0), -1).mean(dim=1).cpu().numpy())
    return np.array(errs)

# === Run ===
train_paths = load_paths(False, 10000)

train_loader = DataLoader(PacketDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
model = TransformerAE().to(DEVICE)
model = train(model, train_loader)
torch.save(model.state_dict(), f"./AI_Real/Transformer_AE/Model/transformer_ae_{EPOCHS}.pth")

test_benign = load_paths(False, 5000)
test_attack = load_paths(True, 5000)
benign_errors = compute_errors(model, test_benign)
attack_errors = compute_errors(model, test_attack)

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