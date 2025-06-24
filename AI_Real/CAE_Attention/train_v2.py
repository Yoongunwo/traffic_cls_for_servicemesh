import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class CAEAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32), nn.MaxPool2d(2),
            SelfAttention(32),
            ResidualBlock(32, 64), nn.MaxPool2d(2),
            ResidualBlock(64, 128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            ResidualBlock(32, 32),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
            nn.Upsample(size=(34, 44), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
    def forward(self, x): return self.block(x)


import os, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./Data_CIC/Session_Windows_15"
BATCH_SIZE, EPOCHS = 512, 20
H, W = 34, 44

class CAEDataset(Dataset):
    def __init__(self, paths):
        self.paths, self.HW = paths, H * W
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        x = np.load(self.paths[idx])[0]
        x = np.nan_to_num(x)
        x = np.clip(x, 0, 255).astype(np.float32) / 255.0
        x = np.pad(x, (0, self.HW - len(x)))[:self.HW]
        return torch.tensor(x.reshape(1, H, W), dtype=torch.float32)

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

model = CAEAttention().to(DEVICE)
feature_net = FeatureExtractor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(CAEDataset(load_paths(False)), batch_size=BATCH_SIZE, shuffle=True)

# Training
for epoch in range(EPOCHS):
    model.train()
    total = 0
    for x in train_loader:
        x = x.to(DEVICE)
        recon = model(x)
        recon_feat = feature_net(recon)
        orig_feat = feature_net(x)
        loss_recon = F.mse_loss(recon, x)
        loss_percep = F.mse_loss(recon_feat, orig_feat)
        loss = loss_recon + 0.2 * loss_percep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"[{epoch+1}/{EPOCHS}] Total Loss: {total / len(train_loader):.4f}")

torch.save(model.state_dict(), f"./AI_Real/CAE/Model/cae_model_v2_{EPOCHS}.pth")

# Evaluation
benign_test = load_paths(False, 5000)
attack_test = load_paths(True, 5000)

benign_err = compute_errors(model, benign_test)
attack_err = compute_errors(model, attack_test)
thresh = np.percentile(benign_err, 95)

y_true = np.array([0]*len(benign_err) + [1]*len(attack_err))
y_score = np.concatenate([benign_err, attack_err])
y_pred = (y_score > thresh).astype(int)

print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
print("ROC AUC:", roc_auc_score(y_true, y_score))
