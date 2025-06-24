import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# === 설정 ===
DATA_DIR = './Data_CIC/Session_Windows_15'
MAX_SESSIONS = 50000
WINDOWS_PER_SESSION = 10
WINDOW_SIZE = 15
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
THRESHOLD_PERCENTILE = 95
H, W = 33, 45

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(33, 45), patch_size=(11, 15), emb_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)  # [B, D, N]
        self.transpose = lambda x: x.transpose(1, 2)  # [B, N, D]

    def forward(self, x):  # x: [B*T, 1, H, W]
        x = self.proj(x)   # [B*T, D, H/P, W/P]
        x = self.flatten(x)
        x = self.transpose(x)
        return x  # [B*T, N, D]

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=128, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, emb_dim)
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class TTAE(nn.Module):
    def __init__(self, emb_dim=128, patch_size=(11, 15), n_blocks=4, seq_len=10, patch_num=9):
        super().__init__()
        self.patch_embed = PatchEmbed(emb_dim=emb_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len * patch_num, emb_dim))
        self.transformer = nn.Sequential(*[TransformerBlock(emb_dim) for _ in range(n_blocks)])
        self.reconstruct = nn.Linear(emb_dim, patch_size[0] * patch_size[1])
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.patch_num = patch_num

    def forward(self, x):  # [B, T, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # [B*T, N, D]
        x = x.reshape(B, T * self.patch_num, -1)
        x = x + self.pos_embed[:, :x.shape[1]]
        z = self.transformer(x)  # [B, T*N, D]
        recon = self.reconstruct(z).reshape(B, T, self.patch_num, *self.patch_size)
        recon = recon.reshape(B, T, 1, H, W)
        return recon

# === Dataset 로딩 ===
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = [np.load(f) for f in self.sequences[idx]]  # list of [15, 1479]
        x = np.stack(seq_data, axis=0)  # [T, 15, 1479]
        x = x.reshape(WINDOWS_PER_SESSION, -1)

        # Normalize input to [0, 1]
        x = x.astype(np.float32) / 255.0

        # Padding or trimming
        if x.shape[1] < H * W:
            x = np.pad(x, ((0, 0), (0, H * W - x.shape[1])), mode='constant')
        elif x.shape[1] > H * W:
            x = x[:, :H * W]

        x = x.reshape(WINDOWS_PER_SESSION, 1, H, W)
        return torch.tensor(x, dtype=torch.float32), self.labels[idx]

def group_session_windows(data_dir, max_sessions=None, windows_per_session=10):
    files = sorted(glob(os.path.join(data_dir, 'benign', '*.npy')))
    session_dict = defaultdict(list)
    for f in files:
        try:
            name = os.path.basename(f)
            parts = name.split('_')
            if len(parts) < 3:
                continue
            session_id = f"{parts[1]}"
            window_idx = int(parts[2].replace('.npy', ''))
            session_dict[session_id].append((window_idx, f))
        except:
            continue

    sequences = []
    for session_id, items in session_dict.items():
        sorted_files = [f for _, f in sorted(items)]
        for i in range(0, len(sorted_files) - windows_per_session + 1):
            seq_files = sorted_files[i:i + windows_per_session]
            sequences.append(seq_files)
        if max_sessions and len(sequences) >= max_sessions:
            break
    return sequences

def group_session_windows_test(data_dir, is_attack=False, max_per_class=100):
    label = 1 if is_attack else 0
    if is_attack:
        files = sorted(glob(os.path.join(data_dir, 'attack', '*', '*.npy')))
    else:
        files = sorted(glob(os.path.join(data_dir, 'benign', '*.npy')))
    session_dict = defaultdict(list)
    for f in files:
        parts = os.path.basename(f).split('_')
        if len(parts) < 3:
            continue
        session_id = parts[1]
        idx = int(parts[2].replace('.npy', ''))
        session_dict[session_id].append((idx, f))
    sequences, labels = [], []
    for sid, items in session_dict.items():
        sorted_files = [f for _, f in sorted(items)]
        for i in range(0, len(sorted_files) - WINDOWS_PER_SESSION + 1):
            seq = sorted_files[i:i + WINDOWS_PER_SESSION]
            sequences.append(seq)
            labels.append(label)
            if len(sequences) >= max_per_class:
                return sequences, labels
    return sequences, labels

# === 테스트 데이터 로딩 및 평가 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequences = group_session_windows(DATA_DIR, max_sessions=MAX_SESSIONS, windows_per_session=WINDOWS_PER_SESSION)
dataset = SequenceDataset(sequences, labels=[0] * len(sequences))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TTAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    for x in loader:
        x, _ = x  # x is a tuple (data, labels)
        x = x.to(device)  # [B, T, 1, H, W]
        out = model(x)
        loss = criterion(out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")


benign_seq, benign_lbl = group_session_windows_test(DATA_DIR, is_attack=False, max_per_class=5000)
attack_seq, attack_lbl = group_session_windows_test(DATA_DIR, is_attack=True, max_per_class=5000)

all_seq = benign_seq + attack_seq
all_lbl = benign_lbl + attack_lbl
test_dataset = SequenceDataset(all_seq, all_lbl)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()

scores, gt = [], []
with torch.no_grad():
    for x, label in test_loader:
        x = x.to(device)  # [B, T, 1, H, W]
        out = model(x)
        loss = F.mse_loss(out, x, reduction='none')
        loss = loss.view(loss.size(0), -1).mean(dim=1)
        scores.extend(loss.cpu().numpy())
        gt.extend(label.numpy())

scores = np.array(scores)
gt = np.array(gt)
threshold = np.percentile(scores, THRESHOLD_PERCENTILE)
print(f"__ Threshold: {threshold:.6f}")
print(classification_report(gt, scores > threshold, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(gt, scores):.4f}")
