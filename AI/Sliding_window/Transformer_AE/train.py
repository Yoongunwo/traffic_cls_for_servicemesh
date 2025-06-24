import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 설정 ===
DATA_DIR = './Data_CIC/StatSession'  # path to [T, F] numpy arrays
WINDOWS_PER_SESSION = 10
FEAT_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === SimCLR 네트워크 정의 ===
class StatEncoder(nn.Module):
    def __init__(self, input_dim, proj_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, proj_dim),
        )

    def forward(self, x):  # x: [B, T, F]
        x = x.mean(dim=1)  # [B, F]
        z = self.net(x)
        z = self.projector(z)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    N = z1.size(0)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    sim = sim / temperature
    mask = ~torch.eye(2*N, dtype=bool).to(z.device)
    sim = sim.masked_select(mask).reshape(2*N, -1)

    positives = F.cosine_similarity(z1, z2) / temperature
    positives = torch.cat([positives, positives], dim=0)

    loss = F.cross_entropy(sim, labels)
    return loss

# === Dataset ===
class StatSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = np.stack([np.load(f) for f in self.sequences[idx]], axis=0)  # [T, F]
        x = torch.tensor(x, dtype=torch.float32)
        return x, self.labels[idx]

def load_stat_sessions(data_dir, is_attack=False, max_per_class=5000):
    subdir = 'attack' if is_attack else 'benign'
    files = sorted(glob(os.path.join(data_dir, subdir, '*', '*.npy')))
    session_dict = defaultdict(list)
    for f in files:
        session_id = os.path.basename(f).split('_')[1]
        window_idx = int(os.path.basename(f).split('_')[2].replace('.npy', ''))
        session_dict[session_id].append((window_idx, f))
    
    sequences, labels = [], []
    for _, items in session_dict.items():
        sorted_files = [f for _, f in sorted(items)]
        for i in range(len(sorted_files) - WINDOWS_PER_SESSION + 1):
            seq = sorted_files[i:i + WINDOWS_PER_SESSION]
            sequences.append(seq)
            labels.append(1 if is_attack else 0)
            if len(sequences) >= max_per_class:
                break
        if len(sequences) >= max_per_class:
            break
    return sequences, labels

# === 학습 데이터 로딩 및 SimCLR pretrain ===
benign_seq, benign_lbl = load_stat_sessions(DATA_DIR, is_attack=False, max_per_class=5000)
train_dataset = StatSequenceDataset(benign_seq, benign_lbl)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

feat_dim = np.load(benign_seq[0][0]).shape[-1]
model = StatEncoder(input_dim=feat_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    for x, _ in train_loader:
        x1 = x + 0.01 * torch.randn_like(x)  # gaussian jitter
        x2 = x + 0.01 * torch.randn_like(x)
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] SimCLR Loss: {loss.item():.4f}")

# === SVM 학습용 feature 추출 ===
model.eval()
train_feats, train_labels = [], []
with torch.no_grad():
    for x, y in train_loader:
        x = x.to(DEVICE)
        z = model(x).cpu().numpy()
        train_feats.extend(z)
        train_labels.extend(y.numpy())

clf = SVC(kernel='linear', probability=True)
clf.fit(train_feats, train_labels)

# === 테스트 ===
test_benign, test_lbl_b = load_stat_sessions(DATA_DIR, is_attack=False, max_per_class=5000)
test_attack, test_lbl_a = load_stat_sessions(DATA_DIR, is_attack=True, max_per_class=5000)
all_seq = test_benign + test_attack
all_lbl = test_lbl_b + test_lbl_a

test_dataset = StatSequenceDataset(all_seq, all_lbl)
test_loader = DataLoader(test_dataset, batch_size=1)

session_preds, session_probs, session_gt = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.squeeze(0).to(DEVICE)  # [T, F]
        feats = model(x.unsqueeze(0).repeat(WINDOWS_PER_SESSION, 1, 1))  # [T, F] → [T, D]
        feats_np = feats.cpu().numpy()
        votes = clf.predict(feats_np)
        prob = clf.predict_proba(feats_np)[:, 1].mean()
        pred = 1 if votes.sum() > (WINDOWS_PER_SESSION // 2) else 0
        session_preds.append(pred)
        session_probs.append(prob)
        session_gt.append(y.item())

# === 평가 ===
print(classification_report(session_gt, session_preds, target_names=["Benign", "Attack"]))
print(f"ROC AUC: {roc_auc_score(session_gt, session_probs):.4f}")
