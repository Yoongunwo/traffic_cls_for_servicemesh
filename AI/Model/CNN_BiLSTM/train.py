import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import sys

current_dir = os.getcwd()  
sys.path.append(current_dir)

from AI.Model.CNN import train


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(CNN_BiLSTM, self).__init__()

        # 1D CNN Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 1D CNN Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # BiLSTM Layer (2-Stack)
        self.lstm = nn.LSTM(
            input_size=32,  # CNN 마지막 output channel (32)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # BiLSTM 사용
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size * 2, num_classes)  # BiLSTM이 bidirectional이므로 hidden_size * 2
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # (batch, channels, seq_length)
        
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)  # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x, _ = self.lstm(x)  # BiLSTM 적용

        x = self.fc(x[:, -1, :])  # 마지막 시점의 출력만 사용
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 통계 계산
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Training complete!")

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 성능 평가
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy

def main():
    # Hyperparameters
    input_size = 32  # CNN 마지막 output channel 크기
    hidden_size = 64
    num_classes = 2
    num_epochs = 50
    batch_size = 8192*2
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 로드
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    normal_dataset = train.PacketImageDataset(
        './Data/save/save_packet_to_byte_16/front_image',
        transform=transform,
        is_flat_structure=True
    )

    attack_dataset = train.PacketImageDataset(
        './Data/attack/attack_to_byte_16',
        transform=transform,
        is_flat_structure=False
    )

    # 데이터 분할
    generator = torch.Generator().manual_seed(42)
    normal_train_size = int(0.8 * len(normal_dataset))
    normal_test_size = len(normal_dataset) - normal_train_size
    normal_train_dataset, normal_test_dataset = torch.utils.data.random_split(
        normal_dataset, [normal_train_size, normal_test_size],
        generator=generator
    )

    attack_train_size = int(0.8 * len(attack_dataset))
    attack_test_size = len(attack_dataset) - attack_train_size
    attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(
        attack_dataset, [attack_train_size, attack_test_size],
        generator=generator
    )

    # 데이터 로더
    train_loader = DataLoader(torch.utils.data.ConcatDataset([normal_train_dataset, attack_train_dataset]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch.utils.data.ConcatDataset([normal_test_dataset, attack_test_dataset]), batch_size=batch_size, shuffle=False)

    # 모델 정의
    model = CNN_BiLSTM(input_size, hidden_size, num_classes).to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 모델 저장
    torch.save(model.state_dict(), 'packet_classifier_epoch50.pth')
    torch.save(model.state_dict(), './AI/Model/CNN_BiLSTM/packet_classifier_epoch50.pth')

    # 모델 평가
    evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()
