import os
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# === 설정 ===
DATA_DIR = './Data_CIC/Session_Windows_15'
WINDOWS_PER_SESSION = 10
H, W = 33, 45
BATCH_SIZE = 2**10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 모델 정의 ===
class SimpleCNN(nn.Module):
    def __init__(self, feat_dim=128):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim)
        )

    def forward(self, x):  # x: [B, 1, H, W]
        f = self.encoder(x)
        return F.normalize(self.projector(f), dim=1)

# === SimCLR loss ===
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    N = z1.size(0)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    sim = sim / temperature
    mask = ~torch.eye(2*N, dtype=bool).to(z.device)
    sim = sim.masked_select(mask).reshape(2*N, -1)

    loss = F.cross_entropy(sim, labels)
    return loss

# === Dataset 정의 ===
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = [np.load(f) for f in self.sequences[idx]]
        x = np.stack(seq_data, axis=0)
        x = x.reshape(WINDOWS_PER_SESSION, -1).astype(np.float32) / 255.0
        if x.shape[1] < H * W:
            x = np.pad(x, ((0, 0), (0, H * W - x.shape[1])), mode='constant')
        elif x.shape[1] > H * W:
            x = x[:, :H * W]
        x = x.reshape(WINDOWS_PER_SESSION, 1, H, W)
        return torch.tensor(x, dtype=torch.float32), self.labels[idx]

# === 데이터 그룹핑 ===
def group_session_windows(data_dir, is_attack=False, max_per_class=100):
    label = 1 if is_attack else 0
    path = 'attack/*/*.npy' if is_attack else 'benign/*.npy'
    files = sorted(glob(os.path.join(data_dir, path)))
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
        if len(sorted_files) < WINDOWS_PER_SESSION:
            continue
        for i in range(len(sorted_files) - WINDOWS_PER_SESSION + 1):
            sequences.append(sorted_files[i:i + WINDOWS_PER_SESSION])
            labels.append(label)
            if len(sequences) >= max_per_class:
                break
        if len(sequences) >= max_per_class:
            break
    return sequences, labels

# === SimCLR augmentation ===
simclr_aug = transforms.Compose([
    transforms.RandomResizedCrop((33, 45), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

# === 모델 및 pretraining ===
model = SimpleCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Pretraining with only benign data
train_seq, train_lbl = group_session_windows(DATA_DIR, is_attack=False, max_per_class=5000)
train_dataset = SequenceDataset(train_seq, train_lbl)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(10):
    model.train()
    for x, _ in train_loader:
        x = x[:, 0]  # [B, T, 1, H, W] → [B, 1, H, W]
        x1 = torch.stack([simclr_aug(to_pil_image(img.squeeze(0))) for img in x])
        x2 = torch.stack([simclr_aug(to_pil_image(img.squeeze(0))) for img in x])
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[Epoch {epoch+1}] SimCLR Loss: {loss.item():.4f}")

# === SVM 학습용 feature 추출 ===
svm_train_seq, svm_train_lbl = group_session_windows(DATA_DIR, is_attack=False, max_per_class=2500)
svm_attack_seq, svm_attack_lbl = group_session_windows(DATA_DIR, is_attack=True, max_per_class=2500)
all_svm_seq = svm_train_seq + svm_attack_seq
all_svm_lbl = svm_train_lbl + svm_attack_lbl
svm_dataset = SequenceDataset(all_svm_seq, all_svm_lbl)
svm_loader = DataLoader(svm_dataset, batch_size=1, shuffle=False)

model.eval()
train_feats, train_labels = [], []
with torch.no_grad():
    for x, y in svm_loader:
        x = x.squeeze(0).to(DEVICE)
        feats = model(x).cpu().numpy()
        train_feats.extend(feats)
        train_labels.extend([y.item()] * feats.shape[0])

clf = SVC(kernel='linear', probability=True)
clf.fit(train_feats, train_labels)

# === 평가 ===
test_benign_seq, test_benign_lbl = group_session_windows(DATA_DIR, is_attack=False, max_per_class=5000)
test_attack_seq, test_attack_lbl = group_session_windows(DATA_DIR, is_attack=True, max_per_class=5000)
test_all_seq = test_benign_seq + test_attack_seq
test_all_lbl = test_benign_lbl + test_attack_lbl
test_dataset = SequenceDataset(test_all_seq, test_all_lbl)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

session_preds, session_gt, session_probs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.squeeze(0).to(DEVICE)
        feats = model(x).cpu().numpy()
        votes = clf.predict(feats)
        pred = 1 if votes.sum() > (WINDOWS_PER_SESSION // 2) else 0
        prob = clf.predict_proba(feats)[:, 1].mean()
        session_preds.append(pred)
        session_gt.append(y.item())
        session_probs.append(prob)

print(classification_report(session_gt, session_preds, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(session_gt, session_probs):.4f}")
