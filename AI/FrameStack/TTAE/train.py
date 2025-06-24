import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import re

# ✅ FrameStack Dataset (input: [B, T, C, H, W])
class FrameStackDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        frames = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)  # [1, H, W]
            frames.append(img)
        stack = torch.stack(frames, dim=0)  # [T, 1, H, W]
        return stack, self.labels[idx + self.window_size - 1]

# ✅ TTAE Model
class TTAE(nn.Module):
    def __init__(self, T=4, embed_dim=128, patch_size=4, img_size=16):
        super().__init__()
        self.T = T
        self.HW = img_size // patch_size
        self.patch_dim = patch_size * patch_size
        self.flatten_dim = self.patch_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size),  # [B*T, D, H', W']
            nn.Flatten(2),  # [B*T, D, N]
            nn.Unflatten(1, (embed_dim,))
        )

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim * self.HW * self.HW, nhead=4, batch_first=True),
            num_layers=4
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * self.HW * self.HW, self.patch_dim * self.HW * self.HW),
            nn.Unflatten(1, (1, self.HW * 4, self.HW * 4)),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, T, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, 1, H, W]
        z = self.encoder(x)        # [B*T, D, N]
        z = z.view(B, T, -1)       # [B, T, D*N]
        z = self.temporal_transformer(z)  # [B, T, D*N]
        z = z[:, -1, :]            # 마지막 timestep → [B, D*N]
        recon = self.decoder(z)    # [B, 1, H, W]
        return recon  # ✅ [B, 1, H, W]

# ✅ 파일 정렬
def natural_key(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    return [os.path.join(path, f) for f in files], [label] * len(files)

# ✅ 학습 및 평가 함수
def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = loss_fn(recon, x[:, -1])  # 예측 대상: 마지막 프레임
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon = model(x)
            err = ((x[:, -1] - recon.squeeze(1)) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            scores.extend(err)
            labels.extend(y.numpy())

    threshold = np.percentile(scores, 95)
    preds = (np.array(scores) > threshold).astype(int)
    print(classification_report(labels, preds, digits=4))
    print("ROC AUC:", roc_auc_score(labels, scores))

# ✅ 실행
if __name__ == '__main__':
    ROOT = './Data/cic_data/Wednesday-workingHours/hilbert_seq'
    WIN = 4
    BATCH = 1024
    EPOCHS = 10

    print(f"FrameStack TTAE Window Size: {WIN} and Data: {ROOT}")

    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    benign_p = benign_p[:55000+WIN-1]
    benign_l = benign_l[:55000+WIN-1]
    attack_p = attack_p[:5000+WIN-1]
    attack_l = attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    benign_ds = FrameStackDataset(benign_p, benign_l, transform, WIN)
    attack_ds = FrameStackDataset(attack_p, attack_l, transform, WIN)

    indices = list(range(len(benign_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=42, shuffle=True)

    train_ds = Subset(benign_ds, train_idx)
    val_ds = Subset(benign_ds, val_idx)
    test_ds = ConcatDataset([val_ds, attack_ds])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    model = TTAE(T=WIN)
    train_and_evaluate(model, train_loader, test_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), EPOCHS)
    torch.save(model.state_dict(), './AI/FrameStack/TTAE/Model/tta_model_{WIN}.pth')
