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

current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

# ✅ 간단한 피처 추출 CNN
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

# ✅ 특징 추출 함수
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

# ✅ 데이터 경로
DATA_PATHS = {
    'front': './Data/byte_16/front_image/train',
    'back': './Data/byte_16/back_image/train',
    'postgres': './Data/byte_16/postgres/train',
    'pgadmin': './Data/byte_16/pgadmin/train',
    'jenkins': './Data/byte_16/jenkins/train',
    'prometheus': './Data/byte_16/prometheus/train',
    'grafana': './Data/byte_16/grafana/train',
}

def main():
    SEED = 42  # ✅ 시드 고정
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    model = FeatureCNN().to(device)

    all_feats = []
    all_labels = []
    label_map = {}

    for idx, (name, path) in enumerate(DATA_PATHS.items()):
        full_dataset = cnn_train.PacketImageDataset(path, transform, is_flat_structure=True, label=idx)
        
        # ✅ 무작위 50개 샘플링
        if len(full_dataset) < 50:
            print(f"⚠️ Warning: {name} has less than 50 samples. Using all {len(full_dataset)} samples.")
            subset_indices = list(range(len(full_dataset)))
        else:
            subset_indices = random.sample(range(len(full_dataset)), 50)

        subset = Subset(full_dataset, subset_indices)
        dataloader = DataLoader(subset, batch_size=50, shuffle=False)

        feats, labels = extract_features(model, dataloader, device)
        all_feats.append(feats)
        all_labels.append(labels)
        label_map[idx] = name

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ✅ PCA 2D
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(all_feats)

    # ✅ 시각화
    plt.figure(figsize=(10, 8))
    num_classes = len(label_map)
    colors = cm.get_cmap('tab10', num_classes)

    for idx in range(num_classes):
        mask = (all_labels == idx)
        plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], label=label_map[idx], alpha=0.7, s=30, color=colors(idx))

    plt.title("PCA of CNN-Extracted Features (50 samples per class)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./Data_Distribution/feature_distribution.png')

if __name__ == "__main__":
    main()
