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
BATCH_SIZE = 2^11
EPOCHS = 10
LR = 1e-3
THRESHOLD_PERCENTILE = 95
H, W = 33, 45

# === ConvLSTM 기반 Autoencoder 정의 ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, input_seq):
        b, t, c, h, w = input_seq.size()
        h_cur = [torch.zeros(b, self.hidden_dim, h, w, device=input_seq.device) for _ in range(self.num_layers)]
        c_cur = [torch.zeros(b, self.hidden_dim, h, w, device=input_seq.device) for _ in range(self.num_layers)]
        outputs = []
        for time_step in range(t):
            x = input_seq[:, time_step]
            for i, cell in enumerate(self.cells):
                h_cur[i], c_cur[i] = cell(x, h_cur[i], c_cur[i])
                x = h_cur[i]
            outputs.append(h_cur[-1])
        return torch.stack(outputs, dim=1), (h_cur, c_cur)

class ConvLSTM_AE(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], kernel_size=3):
        super(ConvLSTM_AE, self).__init__()
        self.encoder = ConvLSTM(input_dim, hidden_dims[0], kernel_size, num_layers=1)
        self.encoder2 = ConvLSTM(hidden_dims[0], hidden_dims[1], kernel_size, num_layers=1)
        self.decoder1 = ConvLSTM(hidden_dims[1], hidden_dims[0], kernel_size, num_layers=1)
        self.decoder2 = ConvLSTM(hidden_dims[0], input_dim, kernel_size, num_layers=1)
        self.conv_out = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)

    def forward(self, x):  # x: [B, T, 1, H, W]
        # Encoder
        z1, _ = self.encoder(x)               # [B, T, 32, H, W]
        z2, _ = self.encoder2(z1)             # [B, T, 64, H, W]

        # Decoder
        d1, _ = self.decoder1(z2)             # [B, T, 32, H, W]
        d2, _ = self.decoder2(d1)             # [B, T, 1, H, W]

        # Output refinement
        B, T, C, H, W = d2.shape
        d2 = d2.reshape(B * T, C, H, W)
        out = self.conv_out(d2)               # [B*T, 1, H, W]
        out = out.reshape(B, T, C, H, W)
        return out

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

model = ConvLSTM_AE().to(device)
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
