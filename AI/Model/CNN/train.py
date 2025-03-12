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
    normal_dataset = PacketImageDataset(
        './Data/save/save_packet_to_byte_16/front_image', 
        transform=transform,
        is_flat_structure=True
    )

    # Attack 데이터셋 로드
    attack_dataset = PacketImageDataset(
        './Data/attack/attack_to_byte_16', 
        transform=transform,
        is_flat_structure=False
    )

    # 각 데이터셋을 학습/테스트용으로 분할
    generator = torch.Generator().manual_seed(42)

    # Normal 데이터 분할
    normal_train_size = int(0.8 * len(normal_dataset))
    normal_test_size = len(normal_dataset) - normal_train_size
    normal_train_dataset, normal_test_dataset = torch.utils.data.random_split(
        normal_dataset, [normal_train_size, normal_test_size],
        generator=generator
    )

    # Attack 데이터 분할
    attack_train_size = int(0.8 * len(attack_dataset))
    attack_test_size = len(attack_dataset) - attack_train_size
    attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(
        attack_dataset, [attack_train_size, attack_test_size],
        generator=generator
    )

    # 학습 데이터셋 결합 (Normal + Attack)
    train_dataset = torch.utils.data.ConcatDataset([
        normal_train_dataset,
        attack_train_dataset
    ])
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True)
    test_normal_loader = DataLoader(normal_test_dataset, batch_size=16384, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=16384, shuffle=False)
    
    # 모델 초기화
    model = SimplePacketCNN().to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)
    torch.save(model.state_dict(), 'packet_classifier_front_16_epoch50.pth')
    
    # 모델 평가
    evaluate.evaluate_model(model, test_normal_loader, test_attack_loader, device=device)

if __name__ == '__main__':
    main()