import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

from torch.utils.data import Dataset
from PIL import Image

class FeatureCNN(nn.Module):
    def __init__(self):
        super(FeatureCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128)
        )

    def forward(self, x):
        return self.encoder(x)
    
class FeatureCNN32(nn.Module):
    def __init__(self):
        super(FeatureCNN32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [B, 128, 4, 4]

            nn.Flatten(),                                 # [B, 128*4*4 = 2048]
            nn.Linear(128 * 4 * 4, 128)                   # 최종 feature vector
        )

    def forward(self, x):
        return self.encoder(x)
    
@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        f = model(x).cpu().numpy()
        feats.extend(f)
        labels.extend(y.numpy())
    return np.array(feats), np.array(labels)

DATA_PATHS = {
    'React+Nginx': f'./Data/32x32/save_front',
    'FastAPI': f'./Data/32x32/save_back',
    'PostgresSQL': f'./Data/32x32/postgres',
    'pgAdmin': f'./Data/32x32/pgadmin',
    'Jenkins': f'./Data/32x32/jenkins',
    'Prometheus': f'./Data/32x32/prometheus',
    'Grafana': f'./Data/32x32/grafana',
}
ATTACK_PATHS = {
    'brute_force': f'./Data/32x32/attack/brute_force',
    'kubernetes_enum': f'./Data/32x32/attack/kubernetes_enum',
    'kubernetes_escape': f'./Data/32x32/attack/kubernetes_escape',
    'kubernetes_manipulate': f'./Data/32x32/attack/kubernetes_manipulate',
    'remote_access': f'./Data/32x32/attack/remote_access',
}

safe_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]


def main():
    IMAGES = 50
    SEED =48  # ✅ 시드 고정
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    model = FeatureCNN32()
    
    model.to(device)

    all_feats = []
    all_labels = []
    label_map = {}
    label_idx = 0

    # ✅ 정상 데이터
    for name, path in DATA_PATHS.items():
        full_dataset = cnn_train.PacketImageDataset(path, transform, is_flat_structure=True, label=label_idx)

        dataloader = DataLoader(full_dataset, batch_size=IMAGES, shuffle=False)

        feats, labels = extract_features(model, dataloader, device)
        all_feats.append(feats)
        all_labels.append(labels)
        label_map[label_idx] = name
        label_idx += 1

    # 공격 데이터셋
    ATTACK_LABEL = label_idx
    attack_feats = []
    attack_labels = []

    for name, path in ATTACK_PATHS.items():
        full_dataset = cnn_train.PacketImageDataset(path, transform, is_flat_structure=True, label=ATTACK_LABEL)

        dataloader = DataLoader(full_dataset, batch_size=IMAGES, shuffle=False)

        feats, labels = extract_features(model, dataloader, device)
        attack_feats.append(feats)
        attack_labels.append(labels)

    if attack_feats:
        all_feats.append(np.concatenate(attack_feats, axis=0))
        all_labels.append(np.concatenate(attack_labels, axis=0))
        label_map[ATTACK_LABEL] = 'Attack'

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    label_color_map = {}
    for idx, label in label_map.items():
        if label == 'Attack':
            label_color_map[label] = '#d62728'  # 고정 빨강
        else:
            label_color_map[label] = safe_colors[idx % len(safe_colors)]

    # ✅ PCA 2D
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(all_feats)

    # ✅ 시각화
    plt.figure(figsize=(10, 8))
    num_classes = len(label_map)
    colors = cm.get_cmap('tab10', num_classes)

    for idx in range(num_classes):
        label = label_map[idx]
        mask = (all_labels == idx)
        plt.scatter(
            feats_2d[mask, 0], 
            feats_2d[mask, 1], 
            label=label_map[idx], 
            alpha=0.7, 
            s=30, 
            color=label_color_map.get(label, colors(idx))
            # color=color_map.get(label, 'gray'),  # 기본값은 회색
        )

    # plt.title("PCA of CNN-Extracted Features", fontsize=30)
    plt.xlabel("PCA Component 1", fontsize=24)
    plt.ylabel("PCA Component 2", fontsize=24)
    plt.legend(fontsize=24, loc='best', markerscale=2, 
               ncol=2, handletextpad=0, columnspacing=0,
               borderpad=0.1, borderaxespad=0.2)
    # plt.legend(fontsize=24, loc='best', markerscale=2, 
    #             handletextpad=0, columnspacing=0,
    #             borderpad=0.1, borderaxespad=0.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./Data_Distribution/feature_distribution.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

if __name__ == "__main__":
    main()
