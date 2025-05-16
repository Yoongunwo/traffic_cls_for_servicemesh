import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
from sklearn.metrics import classification_report


# 현재 디렉토리 패스 추가
current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Model.CNN.train import PacketImageDataset  # 기존에 정의한 Dataset 사용
import AI.Model.CNN.train_v2 as cnn_train

# ✅ CBAM Attention Block 정의
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial Attention
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * sa
        return x

# ✅ CAE with CBAM 모델 정의
class AttentionCAE(nn.Module):
    def __init__(self):
        super(AttentionCAE, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            CBAMBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            CBAMBlock(64)
        )
        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            CBAMBlock(32),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ✅ 학습 함수 정의
def train_cae(model, dataloader, device, num_epochs=30, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, _ in dataloader:  # label은 사용하지 않음
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # ✅ 모델 저장
    os.makedirs("./AI/Model/CAE_Attention/Model", exist_ok=True)
    model_path = "./AI/Model/CAE_Attention/Model/front_attention_cae.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

def calculate_threshold(model, test_loader, device):
    # ✅ Threshold 계산 및 저장
    print("📊 Calculating threshold from training data...")
    model.eval()
    all_scores = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, images, reduction='none')
            loss = loss.view(loss.size(0), -1).mean(dim=1)
            all_scores.extend(loss.cpu().numpy())

    threshold = np.percentile(np.array(all_scores), 95)
    threshold_path = "./AI/Model/CAE_Attention/Model/front_threshold_attention_cae.npy"
    np.save(threshold_path, threshold)
    print(f"✅ Threshold saved to {threshold_path}: {threshold:.6f}")

def evaluate_attention_cae(model_path, test_loader, device, threshold=None):
    model = AttentionCAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, images, reduction='none')
            loss = loss.view(loss.size(0), -1).mean(dim=1)
            all_scores.extend(loss.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)

    if threshold is None:
        threshold = np.percentile(all_scores, 95)  # 이상 탐지용 임계값
        print(f"📌 Auto-calculated Threshold: {threshold:.6f}")

    preds = (all_scores > threshold).astype(int)
    print("\nCAE-Attention Classification Report:")
    print(classification_report(all_labels, preds, digits=4, zero_division=0))

# ✅ 메인 실행
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    normal_train = PacketImageDataset(
        './Data/byte_16/front_image/train', transform=transform, is_flat_structure=True
    )

    train_loader = DataLoader(normal_train, batch_size=1024, shuffle=True)
    model = AttentionCAE()
    train_cae(model, train_loader, device, num_epochs=50)

    normal_test = cnn_train.PacketImageDataset(
        './Data/byte_16/front_image/val', 
        transform, 
        is_flat_structure=True, 
        label=0
    )
    attack_test = cnn_train.PacketImageDataset(
        './Data/attack_to_byte_16', 
        transform, 
        is_flat_structure=False, 
        label=1
    )

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    calculate_threshold(model, test_loader, device)

if __name__ == '__main__':
    main()
