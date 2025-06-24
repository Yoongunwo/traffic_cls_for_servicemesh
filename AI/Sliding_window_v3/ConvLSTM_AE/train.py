import os
import re
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# === Config ===
DATA_DIR = './Data_CIC/Session_Windows_15'
WINDOW = 5
H, W = 34, 44
BATCH = 64
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Parsing Function ===
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
            imgs.append(torch.tensor(x[0], dtype=torch.float32))
        stacked = torch.stack(imgs, dim=0)  # [T, 1, H, W]
        return stacked, self.labels[idx]

    def __len__(self):
        return len(self.samples)

# === ConvLSTM Modules ===
class ConvLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList([
            ConvLSTMBlock(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        b, t, _, h, w = x.size()
        h_t, c_t = [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)], \
                   [torch.zeros(b, self.hidden_dim, h, w, device=x.device) for _ in range(self.num_layers)]
        for time in range(t):
            input_ = x[:, time]
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(input_, h_t[i], c_t[i])
                input_ = h_t[i]
        return h_t[-1]

class ConvLSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, kernel_size, num_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMBlock(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)

    def forward(self, context):
        b, _, h, w = context.size()
        h_t, c_t = [context for _ in range(self.num_layers)], [torch.zeros_like(context) for _ in range(self.num_layers)]
        outputs = []
        input_ = context
        for _ in range(self.seq_len):
            for i, cell in enumerate(self.cells):
                h_t[i], c_t[i] = cell(input_, h_t[i], c_t[i])
                input_ = h_t[i]
            outputs.append(self.conv_out(h_t[-1]).unsqueeze(1))
        return torch.cat(outputs, dim=1)

class ConvLSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=2, seq_len=5):
        super().__init__()
        self.encoder = ConvLSTMEncoder(input_dim, hidden_dim, kernel_size, num_layers)
        self.decoder = ConvLSTMDecoder(hidden_dim, input_dim, kernel_size, num_layers, seq_len)

    def forward(self, x):  # x: [B, T, 1, H, W]
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

# === Train and Evaluate ===
def train(model, loader, optimizer):
    model.train()
    total = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        recon = model(x)
        loss = F.mse_loss(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = F.mse_loss(recon, x, reduction='none').mean(dim=(1,2,3,4))
            scores.extend(loss.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(scores), np.array(labels)

# === Run ===
train_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=50000, window=WINDOW)
test_b_dataset = SequenceDataset(DATA_DIR, is_attack=False, per_class=5000, window=WINDOW)
test_a_dataset = SequenceDataset(DATA_DIR, is_attack=True, per_class=5000, window=WINDOW)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_b_dataset + test_a_dataset, batch_size=1)

model = ConvLSTMAutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer)
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f}")

torch.save(model.state_dict(), './AI/Sliding_window_v3/ConvLSTM_AE/Model/conv_lstm_ae.pth')

scores, labels = evaluate(model, test_loader)
threshold = np.percentile(scores[:5000], 95)
preds = (scores > threshold).astype(int)

print(classification_report(labels, preds, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(labels, scores):.4f}")
