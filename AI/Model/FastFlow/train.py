# Re-execute the FastFlow full code after runtime reset

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import numpy as np
import os
from glob import glob
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

# ✅ SlidingConcatDataset for RGB (4x 16x16 → 32x32 RGB)
class SlidingConcatRGBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        top = torch.cat([imgs[0], imgs[1]], dim=2)
        bottom = torch.cat([imgs[2], imgs[3]], dim=2)
        full = torch.cat([top, bottom], dim=1)
        return full, self.labels[idx + self.window_size - 1]

# ✅ ResNet18 Backbone
class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

    def forward(self, x):
        return self.features(x)

# ✅ Simple Flow Block
class FlowBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        z = self.conv(x)
        log_det_jacobian = torch.sum(torch.log(torch.abs(self.conv.weight + 1e-6)))
        return z, log_det_jacobian

class FastFlow(nn.Module):
    def __init__(self, in_channels, steps=4):
        super().__init__()
        self.flows = nn.ModuleList([FlowBlock(in_channels) for _ in range(steps)])

    def forward(self, x):
        log_det = 0
        for flow in self.flows:
            x, log_jac = flow(x)
            log_det += log_jac
        return x, log_det

class FastFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18Backbone()
        self.flow = FastFlow(in_channels=256)

    def forward(self, x):
        feat = self.encoder(x)
        z, log_det = self.flow(feat)
        likelihood = -torch.mean(z ** 2, dim=[1, 2, 3]) + log_det
        return -likelihood

# ✅ Load paths + concat dataset
def get_concat_dataset(root_dir, label, transform):
    paths = sorted(glob(os.path.join(root_dir, "*.png")))
    labels = [label] * len(paths)
    return SlidingConcatRGBDataset(paths, labels, transform=transform)

def build_dataloaders(train_root, test_normal_root, test_attack_root, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = get_concat_dataset(train_root, 0, transform)
    test_norm = get_concat_dataset(test_normal_root, 0, transform)
    test_atk = get_concat_dataset(test_attack_root, 1, transform)
    test_ds = ConcatDataset([test_norm, test_atk])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)

# ✅ Training
def train_fastflow(model, dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, _ in dataloader:
            x = x.to(device)
            loss = model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[{epoch+1}] Avg Loss: {np.mean(losses):.4f}")


# ✅ Evaluation
def evaluate_fastflow(model, test_loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            score = model(x).cpu().numpy()
            scores.extend(score)
            labels.extend(y.numpy())

    scores = np.array(scores)
    labels = np.array(labels)
    preds = (scores > np.percentile(scores, 95)).astype(int)

    print("\nFastFlow Classification Report:")
    print(classification_report(labels, preds, digits=4))
    print(f"ROC AUC: {roc_auc_score(labels, scores):.4f}")
    plt.hist(scores, bins=50); plt.axvline(np.percentile(scores, 95), color='red'); plt.title("Anomaly Score Histogram"); plt.show()

BATCH_SIZE = 4096 * 4

# ✅ Run
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader = build_dataloaders(
    train_root="./Data/cic_data/Wednesday-workingHours/hilbert_seq/benign_train",
    test_normal_root="./Data/cic_data/Wednesday-workingHours/hilbert_seq/benign_test",
    test_attack_root="./Data/cic_data/Wednesday-workingHours/hilbert_seq/attack",
    batch_size=BATCH_SIZE
)
model = FastFlowModel().to(device)
train_fastflow(model, train_loader, device)
evaluate_fastflow(model, test_loader, device)