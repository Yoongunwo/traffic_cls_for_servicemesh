import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

DATA_DIR = './Data_CIC/Session_Windows_15'
MAX_SESSIONS = 50000
WINDOWS_PER_SESSION = 10
WINDOW_SIZE = 15
BATCH_SIZE = 2**10
EPOCHS = 10
LR = 1e-3
THRESHOLD_PERCENTILE = 95
H, W = 33, 45

class TemporalCNNEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x):  # [B, T, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)                    # → [B*T, 1, H, W]
        f = self.frame_encoder(x).view(B, T, -1)      # → [B, T, 64]

        f = f.mean(dim=1)  # temporal mean pooling → [B, 64]
        z = self.projector(f)                         # → [B, feat_dim]
        return F.normalize(z, dim=1)

def simclr_sequence_augment(x):  # x: [B, T, 1, H, W]
    x_aug = []
    for seq in x:
        seq_aug = []
        for frame in seq:
            img = transforms.ToPILImage()(frame.squeeze(0))
            img = simclr_aug(img)
            seq_aug.append(img)
        seq_tensor = torch.stack(seq_aug).unsqueeze(1)  # [T, 1, H, W]
        x_aug.append(seq_tensor)
    return torch.stack(x_aug)  # [B, T, 1, H, W]

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

from torchvision import transforms

simclr_aug = transforms.Compose([
    transforms.RandomResizedCrop((33, 45), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequences = group_session_windows(DATA_DIR, max_sessions=MAX_SESSIONS, windows_per_session=WINDOWS_PER_SESSION)
dataset = SequenceDataset(sequences, labels=[0] * len(sequences))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Pretraining with only benign data
model = TemporalCNNEncoder().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    for x, _ in loader:  # x: [B, T, 1, H, W]
        x1 = simclr_sequence_augment(x)
        x2 = simclr_sequence_augment(x)
        x1, x2 = x1.cuda(), x2.cuda()

        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] SimCLR Loss: {loss.item():.4f}")

torch.save(model.state_dict(), './AI/Sliding_window/SimCLR/Model/temporal_cnn_encoder_v2.pth')

# Feature extraction (test)
model.eval()

benign_seq, benign_lbl = group_session_windows_test(DATA_DIR, is_attack=False, max_per_class=5000)
attack_seq, attack_lbl = group_session_windows_test(DATA_DIR, is_attack=True, max_per_class=5000)

all_seq = benign_seq + attack_seq
all_lbl = benign_lbl + attack_lbl
test_dataset = SequenceDataset(all_seq, all_lbl)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

features, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        B, T, C, H_, W_ = x.shape
        x = x.view(B * T, C, H_, W_).cuda()  # → [B*T, 1, H, W]
        feats = model.frame_encoder(x).view(B, T, -1)  # [B, T, 64]
        feats = feats.cpu().numpy()
        features.extend(feats.mean(axis=1))  # session-level mean
        labels.extend(y.numpy())


from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

clf = SVC(kernel='linear', probability=True)
clf.fit(features[:5000], labels[:5000])  # benign+attack mixture or synthetic labels

pred = clf.predict(features)
prob = clf.predict_proba(features)[:, 1]

print(classification_report(labels, pred, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(labels, prob):.4f}")
