import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ✅ Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# ✅ ResNet Model
class SimpleResNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(32, 64, downsample=True)
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ✅ StatisticalChannelDataset
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

# ✅ Utilities

def natural_key(string):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
    full_paths = [os.path.join(path, f) for f in files]
    return full_paths, [label] * len(full_paths)

# ✅ Training & Evaluation

def train_resnet_model(root_dir, batch_size=512, win=4, epochs=10, img_size=32, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    benign_p, benign_l = get_paths(os.path.join(root_dir, 'benign'), 0)
    attack_p, attack_l = get_paths(os.path.join(root_dir, 'attack/LOIT'), 1)

    benign_p, benign_l = benign_p[:55000+win-1], benign_l[:55000+win-1]
    atk_p, atk_l = attack_p[:5000+win-1], attack_l[:5000+win-1]

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, win)
    atk_ds = StatisticalChannelDataset(attack_p, attack_l, transform, win)

    total_len = len(benign_ds)
    indices = list(range(total_len))
    train_idx, test_idx = train_test_split(indices, test_size=5000, random_state=random_state, shuffle=True)

    train_ds = Subset(benign_ds, train_idx)
    test_ds = Subset(benign_ds, test_idx)

    test_ds = ConcatDataset([test_ds, atk_ds])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleResNet(in_channels=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), f"./AI/Statistic/ResNet/Model/stat_resnet_{win}_{img_size}.pth")

    # Evaluation
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    print(classification_report(y_true, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_true, y_score))

def evaluate(root_dir, batch_size=512, win=4, img_size=32, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    benign_p, benign_l = get_paths(os.path.join(root_dir, 'benign'), 0)
    attack_p, attack_l = get_paths(os.path.join(root_dir, 'attack/LOIT'), 1)

    benign_p, benign_l = benign_p[50000:55000+win-1], benign_l[50000:55000+win-1]
    attack_p, attack_l = attack_p[:5000+win-1], attack_l[:5000+win-1]

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, win)
    atk_ds = StatisticalChannelDataset(attack_p, attack_l, transform, win)

    test_ds = ConcatDataset([benign_ds, atk_ds])

    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleResNet(in_channels=4).to(device)
    model.load_state_dict(torch.load(f"./AI/Statistic/ResNet/Model/stat_resnet_{win}_{img_size}.pth"))


    # Evaluation
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    print(classification_report(y_true, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_true, y_score))

# ✅ 실행 예시
if __name__ == '__main__':
    PREPROCESSING_TYPE = 'hilbert'
    ROOT = f'./Data_CIC/Fri_{PREPROCESSING_TYPE}_32'
    # train_resnet_model(ROOT, win=9, batch_size=1024*8)
    evaluate(ROOT, win=9, batch_size=1024*8)
