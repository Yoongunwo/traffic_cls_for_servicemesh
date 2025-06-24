import os, re
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config
DATA_DIR = './Data_CIC/Session_Windows_15'
WINDOW = 5
H, W = 34, 44
BATCH = 2**6
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 파싱 함수 ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

# === Dataset ===
class SequenceDataset(Dataset):
    def __init__(self, data_dir, is_attack=False, per_class=10000, window=5):
        self.samples, self.labels = [], []
        label = int(is_attack)
        files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
        session_map = defaultdict(list)
        for f in files:
            sid, idx = parse_name(f)
            session_map[sid].append((idx, f))
        for fs in session_map.values():
            fs.sort()
            paths = [f for _, f in fs]
            for i in range(len(paths) - window + 1):
                self.samples.append(paths[i:i+window])
                self.labels.append(label)
        self.window = window

    def __getitem__(self, idx):
        paths = self.samples[idx]
        imgs = []
        for path in paths:
            x = np.load(path)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            x = x.reshape(x.shape[0], -1)
            if x.shape[1] < H * W:
                x = np.pad(x, ((0, 0), (0, H * W - x.shape[1])))
            elif x.shape[1] > H * W:
                x = x[:, :H * W]
            x = x.reshape(x.shape[0], 1, H, W)
            imgs.append(torch.tensor(x, dtype=torch.float32))
        return torch.stack(imgs), self.labels[idx]

    def __len__(self):
        return len(self.samples)

# Model
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.gru = nn.GRU(32 * 4 * 4, 128, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):  # x: [B, T, 15, 1, H, W]
        B, T, P, C, H, W_ = x.shape
        x = x.view(B * T * P, C, H, W_)  # [B*T*15, 1, H, W]
        feat = self.cnn(x)              # [B*T*15, 32, 4, 4]
        feat = feat.view(B, T, P, -1)   # [B, T, 15, F]
        feat = feat.view(B, T * P, -1)  # [B, T*15, F]
        _, h = self.gru(feat)           # [1, B, 128]
        return self.cls(h[-1])          # [B, 2]


# Train
def train(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

# Eval
def evaluate(model, loader):
    model.eval()
    preds, scores, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            scores.extend(prob)
            labels.extend(y.numpy())
    return np.array(labels), np.array(preds), np.array(scores)

# Main

train_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=50000, window=WINDOW)
test_b_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=5000, window=WINDOW)
test_a_dataset = SequenceDataset(DATA_DIR, is_attack=True, per_class=5000, window=WINDOW)

# print("First training sample file paths:")
# for file_list in train_dataset.samples[:3]:
#     for fpath in file_list:
#         print(fpath)

# print("\nFirst attack sample file paths:")
# for file_list in test_a_dataset.samples[:3]:
#     for fpath in file_list:
#         print(fpath)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_b_dataset + test_a_dataset, batch_size=1)

model = GRUModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer, criterion)
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

torch.save(model.state_dict(), './AI/Sliding_window_v3/CNN_GRU/Model/cnn_gru.pth')

y_true, y_pred, y_score = evaluate(model, test_loader)
print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))
print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")