import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Dataset: Sliding window to sequence of 16x16 grayscale images
class SequencePacketDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, seq_len=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.image_paths) - self.seq_len + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.seq_len):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        x_seq = torch.stack(imgs, dim=0)  # [T, C, H, W]
        y = self.labels[idx + self.seq_len - 1]
        return x_seq, y

# ✅ ConvLSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# ✅ ConvLSTM Autoencoder
class ConvLSTMAE(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, kernel_size=3):
        super().__init__()
        self.encoder_cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.decoder_cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)  # ✅ input_dim=1
        self.out_conv = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        h, c = torch.zeros(B, self.encoder_cell.hidden_dim, H, W).to(x_seq.device), \
               torch.zeros(B, self.encoder_cell.hidden_dim, H, W).to(x_seq.device)

        # 🔹 Encode
        for t in range(T):
            h, c = self.encoder_cell(x_seq[:, t], h, c)

        # 🔹 Decode
        dec_input = torch.zeros(B, C, H, W).to(x_seq.device)  # 👈 Important: (B, 1, H, W)
        dec_outs = []
        for _ in range(T):
            h, c = self.decoder_cell(dec_input, h, c)
            dec_frame = self.out_conv(h)
            dec_outs.append(dec_frame)
            dec_input = dec_frame  # feed previous output

        return torch.stack(dec_outs, dim=1)  # [B, T, C, H, W]

# ✅ Training + Evaluation
def train_and_eval(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            scores = ((out - x) ** 2).mean(dim=[1, 2, 3, 4]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.numpy())

    # Threshold
    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)

    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))
    return model

# ✅ 데이터셋 경로 설정
def get_paths_and_labels(root, label):
    paths, labels = [], []
    for file in sorted(os.listdir(root)):
        if file.endswith('.png'):
            paths.append(os.path.join(root, file))
            labels.append(label)
    return paths, labels

PREPROCESSING_TYPE = 'hilbert'

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    seq_len = 16
    batch_size = 1024

    benign_paths, benign_labels = get_paths_and_labels(f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train', 0)
    attack_paths, attack_labels = get_paths_and_labels(f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/attack', 1)

    train_ds = SequencePacketDataset(benign_paths[:50000], benign_labels[:50000], transform, seq_len)
    test_benign = SequencePacketDataset(benign_paths[50000:55000], benign_labels[50000:55000], transform, seq_len)
    test_attack = SequencePacketDataset(attack_paths[:5000], attack_labels[:5000], transform, seq_len)

    test_ds = torch.utils.data.ConcatDataset([test_benign, test_attack])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = ConvLSTMAE()
    model = train_and_eval(model, train_loader, test_loader, device, epochs=10)

    # Save model
    torch.save(model.state_dict(), f'./AI/Model/ConvLSTM_AE/Model/convlstm_ae_anomaly_detector_{seq_len}.pth')

