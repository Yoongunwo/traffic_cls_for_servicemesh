import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

# ✅ 통계 이미지 데이터셋
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
            imgs.append(img)
        stack = torch.stack(imgs, dim=0).squeeze(1)  # [T, H, W]
        stat = torch.stack([
            stack.mean(dim=0),
            stack.std(dim=0),
            stack.min(dim=0).values,
            stack.max(dim=0).values
        ], dim=0)  # [4, H, W]

        return stat, self.labels[idx + self.window_size - 1]

# ✅ CNN-GRU 기반 Autoencoder
class CNN_GRU_StatAE(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 → 4
        )
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=64*4*4, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64*4*4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  # [B, 64, 4, 4]
        z = self.flatten(z).unsqueeze(1)  # [B, 1, 1024]
        _, h = self.gru(z)
        out = self.decoder(h[-1])
        return out
    
class CNN_GRU_StatAE32(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8
        )
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=64 * 8 * 8, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()  # 16 → 32
        )

    def forward(self, x):
        z = self.encoder(x)  # [B, 64, 8, 8]
        z = self.flatten(z).unsqueeze(1)  # [B, 1, 4096]
        _, h = self.gru(z)
        out = self.decoder(h[-1])
        return out

# ✅ 평가 함수
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

    scores = np.array(scores)
    labels = np.array(labels)

    # 1. Macro-F1 기준 threshold
    precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
    best_f1, best_thresh_f1 = 0, 0
    for t in thresholds_pr:
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = t

    # 2. Youden's J 기반 threshold
    fpr, tpr, thresholds_roc = roc_curve(labels, scores)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    best_thresh_roc = thresholds_roc[j_best_idx]

    for name, t in [("F1-macro", best_thresh_f1), ("Youden's J", best_thresh_roc)]:
        print(f"\n[Using {name} threshold = {t:.6f}]")
        preds = (scores > t).astype(int)
        print(classification_report(labels, preds, digits=4))
        print("ROC AUC:", roc_auc_score(labels, scores))

# ✅ 학습 함수
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

import re

def natural_key(string):
    # 문자열 내 숫자를 기준으로 정렬 (예: packet_2 < packet_10)
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label, stack_folder=False):
    image_paths = []
    labels = []
    if stack_folder:
        for subfolder in sorted(os.listdir(path)):
            subfolder_path = os.path.join(path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            for f in sorted(files, key=natural_key):
                image_paths.append(os.path.join(subfolder_path, f))
                labels.append(label)
    else:
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
        image_paths = [os.path.join(path, f) for f in files]
        labels = [label] * len(image_paths)

    return image_paths, labels

PREPROCESSING_TYPE = 'hilbert'
# ROOT = f'./Data/byte_16_hilbert_seq/save_front'
# ATTACK_ROOT = f'./Data/byte_16_hilbert_attack'
ROOT = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_32x32_seq'
BATCH_SIZE = 1024 * 8
WIN = 4
EPOCHS = 10
random_state = 42
SIZE = 32

def start_train():
    print(f"Statistic CNN+GRU Window size: {WIN}, Data: {ROOT}")
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    test_p, test_l = get_paths(os.path.join(ROOT, 'benign_val'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    # benign_p, benign_l = get_paths(os.path.join(ROOT, 'train'), 0)
    # attack_p, attack_l = get_paths(ATTACK_ROOT, 1, stack_folder=True)

    # benign_p, benign_l = benign_p[:55000+WIN-1], benign_l[:55000+WIN-1]
    # atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, WIN)
    test_ds = StatisticalChannelDataset(test_p, test_l, transform, WIN)
    atk_ds = StatisticalChannelDataset(attack_p, attack_l, transform, WIN)

    # total_len = len(benign_ds)
    # indices = list(range(total_len))
    # train_idx, val_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)

    # train_ds = torch.utils.data.Subset(benign_ds, train_idx)
    # val_ds = torch.utils.data.Subset(benign_ds, val_idx)
    # print(train_ds[0])
    
    test_ds = ConcatDataset([test_ds, atk_ds])
    train_loader = DataLoader(benign_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CNN_GRU_StatAE32(in_channels=4)
    train(model, train_loader, test_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=EPOCHS)

    # save
    torch.save(model.state_dict(), f'./AI/Statistic/CNN_GRU/Model/cnn_gru_statae_{PREPROCESSING_TYPE}_{WIN}_{SIZE}.pth')
    print(f"{WIN} Model saved successfully.")


# def start_evaluate():
#     model = CNN_GRU_StatAE(in_channels=4)
#     model.load_state_dict(torch.load(f'./AI/Statistic/CNN_GRU/Model/cnn_gru_statae_{PREPROCESSING_TYPE}_{WIN}.pth'))

#     benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
#     attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)
#     # transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
#     transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

#     val_p, val_l = benign_p[50000:55000], benign_l[50000:55000]
#     atk_p, atk_l = attack_p[:5000], attack_l[:5000]

#     val_ds = StatisticalChannelDataset(val_p, val_l, transform, WIN)
#     atk_ds = StatisticalChannelDataset(atk_p, atk_l, transform, WIN)

#     test_ds = ConcatDataset([val_ds, atk_ds])
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     evaluate(model, test_loader, device)

# ✅ 실행
if __name__ == "__main__":
    start_train()

