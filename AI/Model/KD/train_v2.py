import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset

import os
import sys

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

# 🟢 Student Model: MobileNetV2 수정 (입력 크기 16x16에 맞춤)
class MobileNetV2_16x16(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2_16x16, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Stride 1로 변경
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    soft_targets = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                             F.softmax(teacher_logits / T, dim=1),
                             reduction='batchmean') * (T * T)
    hard_targets = F.cross_entropy(student_logits, labels)
    return alpha * soft_targets + (1 - alpha) * hard_targets

def main():
    batch_size = 8192*8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student = MobileNetV2_16x16().to(device)
    teacher = cnn_train.SimplePacketCNN().to(device)

    teacher.load_state_dict(torch.load("./AI/Model/CNN/packet_classifier_front_16_epoch50.pth"))
    teacher.eval()

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(num_output_channels=1),  # ✅ 입력을 Grayscale로 변환
        transforms.ToTensor(),
    ])

    train_normal = cnn_train.PacketImageDataset('./Data/byte_16/jenkins/train', transform=transform, is_flat_structure=True)
    test_normal = cnn_train.PacketImageDataset('./Data/byte_16/jenkins/test', transform=transform, is_flat_structure=True)

    train_attack = cnn_train.load_attack_subset('./Data/attack_to_byte_16', 'train', transform)
    test_attack = cnn_train.load_attack_subset('./Data/attack_to_byte_16', 'test', transform)

    # ✅ DataLoader 생성 (배치 크기 최적화)
    batch_size = 16384  # ✅ 메모리에 따라 조절
    train_dataset = torch.utils.data.ConcatDataset([train_normal, train_attack])
   
    test_loader_normal = DataLoader(test_normal, batch_size=batch_size, shuffle=False)
    test_loader_attack = DataLoader(test_attack, batch_size=batch_size, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ✅ Optimizer 설정
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    # ✅ Training Loop
    epochs = 50
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(images)  # Teacher의 예측값 가져오기

            student_outputs = student(images)  # Student의 예측값
            loss = distillation_loss(student_outputs, teacher_outputs, labels)  # Distillation Loss 계산

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # ✅ 모델 저장
    torch.save(student.state_dict(), "student_mobilenet_16x16.pth")
    print("\n✅ Student 모델 학습 완료 및 저장됨!")

if __name__ == "__main__":
    main()