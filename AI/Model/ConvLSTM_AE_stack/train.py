import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Dataset: 4채널로 스택된 패킷 이미지 [C=4, H=16, W=16]
class StackedChannelDataset(Dataset):
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
            imgs.append(img)
        stacked = torch.cat(imgs, dim=0)  # [4, 16, 16]
        label = self.labels[idx + self.window_size - 1]
        return stacked, label

# ✅ ConvLSTM 셀 정의
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# ✅ ConvLSTM 기반 AE
class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.encoder = ConvLSTMCell(input_dim, hidden_dim)
        self.decoder = ConvLSTMCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        h = torch.zeros(B, 32, H, W, device=x.device)
        c = torch.zeros(B, 32, H, W, device=x.device)

        h, c = self.encoder(x, h, c)

        # 👇 수정된 decoder 입력
        dec_input = torch.zeros(B, 32, H, W, device=x.device)
        h, c = self.decoder(dec_input, h, c)

        out = self.output_layer(h)
        return out

# ✅ 학습 + 평가
def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            scores = ((x - out) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.numpy())

    threshold = np.percentile(all_scores, 95)
    preds = (np.array(all_scores) > threshold).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(all_labels, all_scores))

    return model

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

    win_size = 9
    batch_size = 1024 * 4

    benign_paths, benign_labels = get_paths_and_labels(f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train', 0)
    attack_paths, attack_labels = get_paths_and_labels(f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/attack', 1)

    train_ds = StackedChannelDataset(benign_paths[:50000], benign_labels[:50000], transform, win_size)
    test_benign = StackedChannelDataset(benign_paths[50000:55000], benign_labels[50000:55000], transform, win_size)
    test_attack = StackedChannelDataset(attack_paths[:5000], attack_labels[:5000], transform, win_size)

    test_ds = torch.utils.data.ConcatDataset([test_benign, test_attack])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = ConvLSTMAutoencoder(input_dim=win_size, hidden_dim=32)
    model = train_and_evaluate(model, train_loader, test_loader, device, epochs=10)

    # Save model
    torch.save(model.state_dict(), f'./AI/Model/ConvLSTM_AE_stack/Model/convlstm_ae_anomaly_detector_{win_size}.pth')
