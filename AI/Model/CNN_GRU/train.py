import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Sliding 시계열 이미지 Dataset
class PacketSequenceDataset(Dataset):
    def __init__(self, image_dir, label, window_size=4, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label = label
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        seq_imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            seq_imgs.append(img)
        x_seq = torch.stack(seq_imgs, dim=0)  # (T, C, H, W)
        return x_seq, self.label

# ✅ CNN + GRU 모델
class CNN_GRU_AnomalyDetector(nn.Module):
    def __init__(self, input_channels=1, cnn_feat_dim=128, hidden_dim=64, window_size=4):
        super(CNN_GRU_AnomalyDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, cnn_feat_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)
        feats = self.cnn(x_seq)
        feats = feats.view(B, T, -1)
        out, _ = self.gru(feats)
        final = out[:, -1, :]
        return self.classifier(final).squeeze(1)

PREPROCESSING_TYPE = 'hilbert'

TRAIN_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train'

TEST_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train'
ATTACK_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/attack'


# ✅ 학습 및 평가 루프
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    window_size = 16
    batch_size = 1024 
    epochs = 10

    train_ds = PacketSequenceDataset(TRAIN_DATASET, label=0, window_size=window_size, transform=transform)

    # 50000개로 샘플링
    train_ds.image_paths = train_ds.image_paths[:50000]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = CNN_GRU_AnomalyDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            y = torch.zeros(x.size(0), dtype=torch.float32).to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), f'./AI/Model/CNN_GRU/Model/cnn_gru_anomaly_detector_{window_size}.pth')

    # Eval
    test_benign = PacketSequenceDataset(TEST_DATASET, label=0, window_size=window_size, transform=transform)
    test_attack = PacketSequenceDataset(ATTACK_DATASET, label=1, window_size=window_size, transform=transform)

    test_benign.image_paths = test_benign.image_paths[:50000]
    test_attack.image_paths = test_attack.image_paths[:50000]

    test_ds = torch.utils.data.ConcatDataset([test_benign, test_attack])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            scores = model(x).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y.numpy())

    preds = (np.array(all_scores) > 0.5).astype(int)
    print(classification_report(all_labels, preds, digits=4))
    print(f"ROC AUC: {roc_auc_score(all_labels, all_scores):.4f}")
