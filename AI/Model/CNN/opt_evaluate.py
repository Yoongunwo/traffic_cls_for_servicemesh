import torch
import torch.nn.utils.prune as prune
import torch.quantization
import time
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

current_dir = os.getcwd()  
sys.path.append(current_dir)

from AI.Model.CNN import train


# ✅ Pruning 적용 함수 (CPU로 이동 추가)
def apply_pruning(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # 실제로 pruning 적용
    model.to("cpu")  # ✅ Pruning 후 CPU로 이동
    return model


# ✅ Quantization 적용 함수
def apply_quantization(model):
    model.eval()  # ✅ Evaluation mode 설정
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8  # FC 레이어를 8비트로 변환
    )
    return quantized_model


# ✅ 모델 평가 함수 (CPU & GPU 지원)
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.shape[0]  # 현재 배치 크기
            total_images += batch_size

            start_time = time.time()
            _ = model(images)
            end_time = time.time()

            total_time += (end_time - start_time)

    avg_batch_time = (total_time / len(test_loader)) * 1000  # ms 단위 변환
    avg_sample_time = (total_time / total_images) * 1000  # ms 단위 변환 (1장 처리 시간)
    return avg_batch_time, avg_sample_time


# ✅ 메인 실행 코드
def main():
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
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

    test_normal_loader = DataLoader(normal_test_dataset, batch_size=256, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=256, shuffle=False)

    # ✅ 기존 학습된 모델 로드
    model = train.SimplePacketCNN().to(device)
    model.load_state_dict(torch.load('./AI/Model/CNN/packet_classifier_front_16.pth'))
    model.eval()

    # ✅ 기본 모델 평가 (GPU)
    base_batch_time_gpu, base_sample_time_gpu = evaluate_model(model, test_normal_loader, device)
    print(f"✅ 기본 모델 평균 배치 추론 시간 (GPU): {base_batch_time_gpu:.4f} ms")
    print(f"✅ 기본 모델 평균 1장 처리 시간 (GPU): {base_sample_time_gpu:.4f} ms")

    # ✅ 기본 모델 평가 (CPU)
    base_batch_time_cpu, base_sample_time_cpu = evaluate_model(model, test_normal_loader, "cpu")
    print(f"✅ 기본 모델 평균 배치 추론 시간 (CPU): {base_batch_time_cpu:.4f} ms")
    print(f"✅ 기본 모델 평균 1장 처리 시간 (CPU): {base_sample_time_cpu:.4f} ms")

    # ✅ Pruning 적용 (CPU로 이동됨)
    pruned_model = apply_pruning(model)
    pruned_batch_time, pruned_sample_time = evaluate_model(pruned_model, test_normal_loader, "cpu")
    print(f"✅ Pruned 모델 평균 배치 추론 시간 (CPU): {pruned_batch_time:.4f} ms")
    print(f"✅ Pruned 모델 평균 1장 처리 시간 (CPU): {pruned_sample_time:.4f} ms")

    # ✅ Quantization 적용 (CPU에서 실행)
    quantized_model = apply_quantization(pruned_model)
    quantized_batch_time, quantized_sample_time = evaluate_model(quantized_model, test_normal_loader, "cpu")
    print(f"✅ Pruned + Quantized 모델 평균 배치 추론 시간 (CPU): {quantized_batch_time:.4f} ms")
    print(f"✅ Pruned + Quantized 모델 평균 1장 처리 시간 (CPU): {quantized_sample_time:.4f} ms")

    # ✅ 최적화 결과 비교
    print("\n🔹 최적화 결과:")
    print(f"기본 모델 (GPU) : 배치 {base_batch_time_gpu:.4f} ms | 1장 {base_sample_time_gpu:.4f} ms")
    print(f"기본 모델 (CPU) : 배치 {base_batch_time_cpu:.4f} ms | 1장 {base_sample_time_cpu:.4f} ms")
    print(f"Pruned 모델  (CPU) : 배치 {pruned_batch_time:.4f} ms | 1장 {pruned_sample_time:.4f} ms")
    print(f"Pruned + Quantized 모델 (CPU) : 배치 {quantized_batch_time:.4f} ms | 1장 {quantized_sample_time:.4f} ms")


if __name__ == '__main__':
    main()
