import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import Counter

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

from AI.Model.CNN import evaluate

# 커스텀 데이터셋 클래스
class PacketImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_flat_structure=True):
        self.transform = transform
        self.images = []
        self.labels = []
        
        if is_flat_structure:  # 이미지가 바로 있는 경우 (normal)
            for file in os.listdir(root_dir):
                if file.endswith('.png'):
                    self.images.append(os.path.join(root_dir, file))
                    self.labels.append(0)  # normal은 0
        else:  # 하위 폴더가 있는 경우 (attack)
            for subdir, _, files in os.walk(root_dir):
                folder_name = os.path.basename(subdir)
                if files:  # 파일이 있는 폴더만 처리
                    for file in files:
                        if file.endswith('.png'):
                            self.images.append(os.path.join(subdir, file))
                            self.labels.append(1)  # attack은 1
        
        print(f"Found {len(self.images)} images in {root_dir}")
        if len(self.images) > 0:
            print(f"Sample path: {self.images[0]}")
        print(f"Label distribution: {dict(Counter(self.labels))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

# CNN 모델 정의
class SimplePacketCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplePacketCNN, self).__init__()
        
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 두 번째 컨볼루션 블록
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 세 번째 컨볼루션 블록
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 완전 연결 계층
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

class RawImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def load_attack_subset(attack_root: str, split: str, transform):
    all_images = []
    all_labels = []

    for scenario in os.listdir(attack_root):
        scenario_dir = os.path.join(attack_root, scenario, split)
        if not os.path.isdir(scenario_dir):
            continue
        for file in os.listdir(scenario_dir):
            if file.endswith(".png"):
                all_images.append(os.path.join(scenario_dir, file))
                all_labels.append(1)  # attack 레이블

    print(f"[{split}] Loaded {len(all_images)} attack images from {attack_root}")
    return RawImageDataset(all_images, all_labels, transform=transform)


# 메인 실행 코드
def main():
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])
    
    # Normal 데이터셋 로드
    train_normal = PacketImageDataset('./Data/byte_16/jenkins/train', transform=transform, is_flat_structure=True)
    test_normal = PacketImageDataset('./Data/byte_16/jenkins/test', transform=transform, is_flat_structure=True)

    train_attack = load_attack_subset('./Data/attack_to_byte_16', 'train', transform)
    test_attack = load_attack_subset('./Data/attack_to_byte_16', 'test', transform)


    train_dataset = torch.utils.data.ConcatDataset([train_normal, train_attack])
   
    test_loader_normal = DataLoader(test_normal, batch_size=16384, shuffle=False)
    test_loader_attack = DataLoader(test_attack, batch_size=16384, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True)

    
    # 모델 초기화
    model = SimplePacketCNN().to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)
    torch.save(model.state_dict(), 'packet_classifier_jenkins_16_epoch50.pth')
    
    # 모델 평가
    evaluate.evaluate_model(model, test_loader_normal, test_loader_attack, device=device)

if __name__ == '__main__':
    main()