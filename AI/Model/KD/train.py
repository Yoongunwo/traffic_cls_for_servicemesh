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
from collections import Counter

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

from AI.Model.CNN import train as cnn_train

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

    normal_dataset = cnn_train.PacketImageDataset(
        './Data/save/save_packet_to_byte_16/front_image',
        transform=transform,
        is_flat_structure=True
    )

    attack_dataset = cnn_train.PacketImageDataset(
        './Data/attack/attack_to_byte_16',
        transform=transform,
        is_flat_structure=False
    )

    generator = torch.Generator().manual_seed(42)

    normal_train_size = int(0.8 * len(normal_dataset))
    normal_test_size = len(normal_dataset) - normal_train_size
    normal_train_dataset, normal_test_dataset = torch.utils.data.random_split(
        normal_dataset, [normal_train_size, normal_test_size], generator=generator
    )

    attack_train_size = int(0.8 * len(attack_dataset))
    attack_test_size = len(attack_dataset) - attack_train_size
    attack_train_dataset, attack_test_dataset = torch.utils.data.random_split(
        attack_dataset, [attack_train_size, attack_test_size], generator=generator
    )

    # ✅ DataLoader 생성 (배치 크기 최적화)
    batch_size = 1024  # ✅ 메모리에 따라 조절
    train_loader = DataLoader(
        torch.utils.data.ConcatDataset([normal_train_dataset, attack_train_dataset]), 
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        torch.utils.data.ConcatDataset([normal_test_dataset, attack_test_dataset]), 
        batch_size=batch_size, shuffle=False
    )

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