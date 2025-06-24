# PatchCore 기반 이상 탐지 모델 구현 (PyTorch)
# - Backbone: Pretrained CNN (ex: resnet18)
# - Feature Embedding 추출 → Patch-wise Index 생성
# - Nearest Neighbor 기반 Anomaly Score 계산 및 저장 기능 포함

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image
from collections import Counter
import joblib

import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

# ✅ Custom Dataset
class PacketImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_flat_structure=True, label=0):
        self.transform = transform
        self.images = []
        self.labels = []

        if is_flat_structure:
            for file in os.listdir(root_dir):
                if file.endswith('.png'):
                    self.images.append(os.path.join(root_dir, file))
                    self.labels.append(label)
        else:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.png'):
                        self.images.append(os.path.join(subdir, file))
                        self.labels.append(label)

        print(f"Found {len(self.images)} images in {root_dir}")
        print(f"Label distribution: {dict(Counter(self.labels))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ✅ Feature Extractor using ResNet18
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, 2, 2]

    def forward(self, x):
        return self.features(x)

class FeatureExtractor32(nn.Module):
    def __init__(self):
        super(FeatureExtractor32, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 수정: stride 줄이기, MaxPool 제거
        resnet.conv1.stride = (1, 1)
        resnet.maxpool = nn.Identity()

        # 특성 추출 계층
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, H, W]

    def forward(self, x):
        return self.features(x)


# ✅ Flatten patches
def flatten_features(feats):
    b, c, h, w = feats.shape
    return feats.permute(0, 2, 3, 1).reshape(b, -1, c)  # [B, H*W, C]

def evaluate_patchcore(embedding_path, model_path, test_loader, device):
    model = FeatureExtractor32().to(device)
    model.eval()

    embedding_path = np.load(embedding_path)
    nn_model = joblib.load(model_path)
    
    all_labels, all_scores = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            feat = model(x)
            patches = flatten_features(feat).reshape(-1, 512).cpu().numpy()
            dists, _ = nn_model.kneighbors(patches)
            scores = dists.reshape(x.size(0), -1).mean(axis=1)
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    preds = [1 if s > np.percentile(all_scores, 95) else 0 for s in all_scores]
    print("\nPatch Core Evaluation:")
    print("Classification Report:\n", classification_report(all_labels, preds, digits=4, zero_division=0))    
    print("ROC AUC:", roc_auc_score(all_labels, preds))

def train_model(device, train_loader, epoches, model_dir, embedding_path, model_path):
    # ✅ Feature Extractor
    model = FeatureExtractor32().to(device)
    model.eval()

    # ✅ Extract Training Embeddings
    all_patches = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            feat = model(x)  # [B, C, H, W]
            patches = flatten_features(feat)  # [B, H*W, C]
            all_patches.append(patches.cpu())

    embeddings = torch.cat(all_patches, dim=0).reshape(-1, 512).numpy()
    print("✅ Feature bank shape:", embeddings.shape)

    # ✅ Save Feature Bank
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, embedding_path), embeddings)

    # ✅ Fit and Save Nearest Neighbors
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_model.fit(embeddings)
    joblib.dump(nn_model, os.path.join(model_dir, model_path))

TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/PatchCore/Model'
PATCHCORE_MODEL_PATH =  './AI/Model/PatchCore/Model/cic_feature_bank.npy'
NN_MODEL_PATH = './AI/Model/PatchCore/Model/cic_nn_model.pkl'

BATCH_SIZE = 4096 * 4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    normal_train = PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    # ✅ Feature Extractor
    model = FeatureExtractor().to(device)
    model.eval()

    # ✅ Extract Training Embeddings
    all_patches = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            feat = model(x)  # [B, C, H, W]
            patches = flatten_features(feat)  # [B, H*W, C]
            all_patches.append(patches.cpu())

    embeddings = torch.cat(all_patches, dim=0).reshape(-1, 512).numpy()
    print("✅ Feature bank shape:", embeddings.shape)

    # ✅ Save Feature Bank
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(PATCHCORE_MODEL_PATH, embeddings)

    # ✅ Fit and Save Nearest Neighbors
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_model.fit(embeddings)
    joblib.dump(nn_model, NN_MODEL_PATH)

    # ✅ Evaluation

    normal_test = PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_labels, all_scores = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            feat = model(x)
            patches = flatten_features(feat).reshape(-1, 512).cpu().numpy()
            dists, _ = nn_model.kneighbors(patches)
            scores = dists.reshape(x.size(0), -1).mean(axis=1)
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    # ✅ Classification Report
    preds = [1 if s > np.percentile(all_scores, 95) else 0 for s in all_scores]
    print("\n📌 Classification Report:")
    print(classification_report(all_labels, preds, digits=4))

if __name__ == '__main__':
    main()
