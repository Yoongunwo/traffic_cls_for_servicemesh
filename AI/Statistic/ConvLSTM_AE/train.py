import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve
import re

# ✅ Dataset
class StatisticalChannelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img.squeeze(0))  # shape: [H, W]
        stack = torch.stack(imgs, dim=0)  # shape: [T, H, W]
        stat = torch.stack([
            stack.mean(dim=0),
            stack.std(dim=0),
            stack.min(dim=0).values,
            stack.max(dim=0).values
        ], dim=0)  # shape: [C, H, W]
        return stat, self.labels[idx + self.window_size - 1]

# ✅ ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# ✅ ConvLSTM Autoencoder
class ConvLSTM_AE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, kernel_size=3):
        super().__init__()
        self.encoder_conv = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.lstm = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        self.decoder_conv = nn.Conv2d(hidden_dim, input_dim, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.encoder_conv(x)
        h_cur = torch.zeros_like(x)
        c_cur = torch.zeros_like(x)
        h_next, c_next = self.lstm(x, h_cur, c_cur)
        recon = self.decoder_conv(h_next)
        return recon

# ✅ 평가
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device)
        recon = model(x)
        err = ((x - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        scores.extend(err)
        labels.extend(y.numpy())
    scores, labels = np.array(scores), np.array(labels)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]

    print(f"\n[Threshold = {best_thresh:.6f}]")
    preds = (scores > best_thresh).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))

# ✅ 학습
def train(model, train_loader, test_loader, device, epochs=10):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0
        for x, _ in train_loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss: {total / len(train_loader):.4f}")
    evaluate(model, test_loader, device)

# ✅ 유틸
def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
BATCH_SIZE = 1024 * 16
WIN = 9
EPOCHS = 10
random_state = 42

# ✅ 실행
def main():
    print(f"ConvLSTM+AE Window size: {WIN}, Data: {ROOT}")

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    attack_p, attack_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, WIN)
    attack_ds = StatisticalChannelDataset(attack_p, attack_l, transform, WIN)

    indices = list(range(len(benign_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)
    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)

    test_ds = ConcatDataset([val_ds, attack_ds])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = ConvLSTM_AE()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, test_loader, device, epochs=EPOCHS)

    #  save
    torch.save(model.state_dict(), f'./AI/Statistic/ConvLSTM_AE/Model/conv_lstm_ae_{WIN}.pth')

if __name__ == "__main__":
    main()
