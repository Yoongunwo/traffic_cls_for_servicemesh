import os, re
import numpy as np
from glob import glob
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
DATA_DIR = './Data_CIC/Session_Windows_15'
H, W = 34, 44
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 2**12

# === 파일 이름 파싱 ===
def parse_name(fname):
    base = os.path.basename(fname)
    sid, idx = re.match(r'(.*)_(\d+)\.npy', base).groups()
    return sid, int(idx)

# === 시퀀스 로딩 ===
def load_image_paths_single_window(data_dir, is_attack=False, per_class=1000):
    label = int(is_attack)
    files = sorted(glob(os.path.join(data_dir, 'attack/*/*.npy' if is_attack else 'benign/*.npy')))[:per_class]
    samples = [[f] for f in files]  # 각 파일을 하나의 시퀀스로
    labels = [label] * len(samples)
    return samples, labels

# === CNN AutoEncoder ===
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class CNNDecoder(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 8x8
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 32x32
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid(),  # still ~32x32
            nn.AdaptiveAvgPool2d((34, 44))  # 정확한 크기로 맞춤
        )

    def forward(self, x):
        return self.net(x)

class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# === Dataset for AE 학습 ===
class BenignImageDataset(Dataset):
    def __init__(self, samples):
        self.frames = []
        for seq in samples:
            for p in seq:
                x = np.load(p)
                x = np.nan_to_num(x)
                x = np.clip(x, 0, 255).astype(np.float32) / 255.0
                for t in range(x.shape[0]):
                    vec = x[t]
                    if vec.shape[0] < H * W:
                        vec = np.pad(vec, (0, H * W - vec.shape[0]))
                    else:
                        vec = vec[:H * W]
                    img = vec.reshape(1, H, W)
                    self.frames.append(img)
        self.frames = torch.from_numpy(np.array(self.frames, dtype=np.float32))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

# === Feature Extractor ===
def extract_features(samples, cnn_model):
    cnn_model.eval()
    feats = []
    for seq in samples:
        imgs = []
        for path in seq:
            x = np.load(path)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.float32) / 255.0
            for t in range(x.shape[0]):
                vec = x[t]
                if vec.shape[0] < H * W:
                    vec = np.pad(vec, (0, H * W - vec.shape[0]))
                else:
                    vec = vec[:H * W]
                img = vec.reshape(1, H, W)
                imgs.append(torch.tensor(img, dtype=torch.float32))
        x_tensor = torch.stack(imgs).to(DEVICE)  # [T, 1, H, W]
        with torch.no_grad():
            feat = cnn_model(x_tensor).cpu().numpy()
        feats.append(np.mean(feat, axis=0))  # mean pooling
    return np.stack(feats)

# === 데이터 로딩 ===
print("Loading benign (train)...")
benign_train_paths, _ = load_image_paths_single_window(DATA_DIR, is_attack=False, per_class=50000)

print("Loading benign (test)...")
benign_test_paths, benign_test_labels = load_image_paths_single_window(DATA_DIR, is_attack=False, per_class=10000)

print("Loading attack...")
attack_paths, attack_labels = load_image_paths_single_window(DATA_DIR, is_attack=True, per_class=10000)

# === AutoEncoder 학습 ===
print("Training AutoEncoder...")
autoencoder = CNNAutoEncoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(f'./AI/Sliding_window_v4/SVM/Model/cnn_autoencoder_v3_{EPOCHS}_2.pth', map_location=DEVICE))
# optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
# train_loader = DataLoader(BenignImageDataset(benign_train_paths), batch_size=BATCH_SIZE, shuffle=True)

# for epoch in range(EPOCHS):
#     autoencoder.train()
#     total_loss = 0
#     for x in train_loader:
#         x = x.to(DEVICE)
#         out = autoencoder(x)
#         loss = criterion(out, x)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# torch.save(autoencoder.state_dict(), f'./AI/Sliding_window_v4/SVM/Model/cnn_autoencoder_v3_{EPOCHS}_2.pth')

cnn = autoencoder.encoder  # 학습된 encoder 사용

# === Feature 추출 및 평가 ===
print("Extracting features...")
X_train = extract_features(benign_train_paths, cnn)
X_test = extract_features(benign_test_paths + attack_paths, cnn)
y_test = np.array(benign_test_labels + attack_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ocsvm = OneClassSVM(
            kernel='rbf', 
            nu=0.01, 
            degree=4,
            gamma=25.0
        )

ocsvm.fit(X_train_scaled)

y_pred = ocsvm.predict(X_test_scaled)
y_pred = (y_pred == -1).astype(int)

print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
