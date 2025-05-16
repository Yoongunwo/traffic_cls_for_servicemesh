import torch
import time
from torchvision import models
import torch.nn as nn
import torch.nn.utils.prune as prune

import os
import sys

current_dir = os.getcwd()  
sys.path.append(current_dir)

# ✅ CNN 모델 정의
class SimplePacketCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimplePacketCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128 * 4 * 4, 256), 
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
    

# ✅ VGG 모델 정의
class VGGPacketCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGPacketCNN, self).__init__()
        self.vgg = models.vgg11(pretrained=False)

        # 첫 번째 입력층을 1채널로 변경 (기본 3채널 → 1채널)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # ✅ MaxPool2d 기본 설정 유지 (2x2)
        self.vgg.avgpool = nn.AdaptiveAvgPool2d(1)  # (N, 512, 1, 1)로 변환

        # Fully Connected Layer 수정
        self.vgg.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)  # 512 → 최종 클래스 개수
        )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = self.vgg.classifier(x)
        return x


# ✅ MobileNet 모델 정의 (32x32 입력 지원)
# class MobileNetPacketCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(MobileNetPacketCNN, self).__init__()
#         self.mobilenet = models.mobilenet_v2(pretrained=False)

#         # ✅ 입력 채널을 1개로 변경 & 첫 번째 Conv의 stride=2 설정
#         self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

#         self.mobilenet.classifier = nn.Sequential(
#             nn.Linear(1280, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         return self.mobilenet(x)

class MobileNetPacketCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetPacketCNN, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)

        # ✅ 첫 번째 Conv2d 수정 (stride=1로 변경, 채널 수 줄이기)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # ✅ BatchNorm2d도 16 채널로 변경 (이전 오류 해결)
        self.mobilenet.features[0][1] = nn.BatchNorm2d(16)

        # ✅ Fully Connected Layer 크기 줄이기
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(1280, 128),  # 1280 → 128로 축소
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)


# ✅ Pruning 적용 함수 (가중치 50% 제거)
def apply_pruning(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 실제로 pruning 적용
    return model


# ✅ Quantization 적용 함수 (FC 레이어 8bit 변환)
def apply_quantization(model):
    model.to("cpu")  # ✅ Pruning 후 반드시 CPU로 이동
    model.eval()  # ✅ Evaluation mode 설정

    # ✅ 모델을 TorchScript로 변환 (버그 방지)
    traced_model = torch.jit.trace(model, torch.randn(1, 1, 32, 32))
    
    # ✅ Quantization 적용
    quantized_model = torch.quantization.quantize_dynamic(
        traced_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return quantized_model

# ✅ 디바이스 설정 (GPU 사용 가능하면 GPU로 실행)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = SimplePacketCNN().to(device)
cnn_model.eval()

# 모델 로드
vgg_model = VGGPacketCNN().to(device)
vgg_model.eval()  # 추론 모드

# ✅ MobileNetV2 모델 로드
# mobilenet_model = MobileNetPacketCNN().to(device)
# mobilenet_model.eval()  # 추론 모드

mobilenet_model = MobileNetPacketCNN().to(device)
mobilenet_model.eval()

# ✅ Pruning 적용
pruned_model = apply_pruning(mobilenet_model)

# ✅ Quantization 적용 (CPU에서 실행)
quantized_model = apply_quantization(pruned_model).to("cpu")

def evaluate_model(model, input_tensor, device="cuda"):
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    return avg_time * 1000  # ms 단위 변환


# ✅ 입력 데이터 생성 (배치 크기 = 1024, 32x32 이미지)
input_tensor = torch.randn(1024, 1, 32, 32).to(device)

# ✅ 기존 MobileNetV2 속도 측정
mobilenet_time = evaluate_model(mobilenet_model, input_tensor, device)
print(f"✅ MobileNetV2 평균 추론 시간 (배치=1024): {mobilenet_time:.4f} ms")

# ✅ Pruned MobileNetV2 속도 측정
pruned_time = evaluate_model(pruned_model, input_tensor, device)
print(f"✅ Pruned MobileNetV2 평균 추론 시간 (배치=1024): {pruned_time:.4f} ms")

# ✅ Quantized MobileNetV2 속도 측정 (CPU에서 실행)
quantized_time = evaluate_model(quantized_model, input_tensor.to("cpu"), "cpu")
print(f"✅ Pruned + Quantized MobileNetV2 평균 추론 시간 (배치=1024, CPU): {quantized_time:.4f} ms")