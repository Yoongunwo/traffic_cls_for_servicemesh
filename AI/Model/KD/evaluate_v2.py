import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from sklearn.metrics import classification_report
import numpy as np
import os
import sys
from torchvision import models
import time

current_dir = os.getcwd()  
sys.path.append(current_dir)

from AI.Model.KD import train as kd_train
from AI.Model.CNN import train_v2 as cnn_train

# ✅ Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Student Model 로드
class MobileNetV2_16x16(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2_16x16, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Stride 1 변경
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# ✅ 모델 불러오기
# student = MobileNetV2_16x16().to(device)
student = MobileNetV2_16x16()
student.load_state_dict(torch.load("./AI/Model/KD/student_mobilenet_16x16.pth"))
student.eval()

# ✅ Transform 정의 (Grayscale 변환)
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# ✅ 테스트 데이터셋 로드 (공격 및 정상 데이터)

normal_test_dataset = cnn_train.PacketImageDataset('./Data/byte_16/jenkins/test', transform=transform, is_flat_structure=True)
attack_test_dataset = cnn_train.load_attack_subset('./Data/attack_to_byte_16', 'test', transform)


min_len = min(len(normal_test_dataset), len(attack_test_dataset))

normal_subset = Subset(normal_test_dataset, list(range(min_len)))
attack_subset = Subset(attack_test_dataset, list(range(min_len)))

# ✅ 병합된 균형 테스트 데이터셋
balanced_test_dataset = torch.utils.data.ConcatDataset([normal_subset, attack_subset])

# ✅ DataLoader
test_loader = DataLoader(
    balanced_test_dataset,
    batch_size=1024,
    shuffle=False
)

# ✅ DataLoader 
# batch_size = 1024  # 메모리 최적화
# test_loader = DataLoader(
#     torch.utils.data.ConcatDataset([normal_test_dataset, attack_test_dataset]), 
#     batch_size=batch_size, shuffle=False
# )

# ✅ 평가 지표 저장
all_labels = []
all_preds = []

# ✅ 테스트 루프 (모델 평가)
with torch.no_grad():
    for images, labels in test_loader:
        # images, labels = images.to(device), labels.to(device)
        
        start_time = time.time()
        outputs = student(images)
        preds = torch.argmax(outputs, dim=1)
        end_time = time.time()
        
        print(f"Elapsed Time: {end_time - start_time:.4f} sec")
        # break
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ✅ Classification Report 생성
report = classification_report(
    all_labels, 
    all_preds, 
    target_names=["Normal", "Attack"], 
    digits=4
)

print("\n📌 Classification Report:\n")
print(report)

# ✅ 결과를 저장 (파일 출력)
with open("evaluation_results.txt", "w") as f:
    f.write(report)

print("\n✅ 평가 완료! 결과가 'evaluation_results.txt'에 저장되었습니다.")


