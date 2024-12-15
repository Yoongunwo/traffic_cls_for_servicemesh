import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

from AI.Model.CNN import train

def evaluate_model(model, test_normal_loader, test_attack_loader, device='cuda'):
    
    model.eval()
    
    # 예측값과 실제값을 저장할 리스트
    all_predictions = []
    all_labels = []
    
    # 정상 트래픽 평가
    normal_correct = 0
    normal_total = 0
    
    print("\nEvaluating Normal Traffic:")
    print(f"Number of normal test samples: {len(test_normal_loader.dataset)}")
    
    with torch.no_grad():
        for images, _ in test_normal_loader:
            images = images.to(device)
            labels = torch.zeros(images.size(0), dtype=torch.long).to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # CPU로 이동하여 리스트에 추가
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            normal_total += labels.size(0)
            normal_correct += (predicted == labels).sum().item()
    
    # 공격 트래픽 평가
    attack_correct = 0
    attack_total = 0
    
    print("\nEvaluating Attack Traffic:")
    print(f"Number of attack test samples: {len(test_attack_loader.dataset)}")
    
    with torch.no_grad():
        for images, _ in test_attack_loader:
            images = images.to(device)
            labels = torch.ones(images.size(0), dtype=torch.long).to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # CPU로 이동하여 리스트에 추가
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            attack_total += labels.size(0)
            attack_correct += (predicted == labels).sum().item()
    
    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_predictions)
    
    # True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm.ravel()
    
    # 메트릭스 계산
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = 100 * fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # 결과 출력
    print("\nConfusion Matrix:")
    print(f"True Negatives (정상을 정상으로): {tn}")
    print(f"False Positives (정상을 공격으로): {fp}")
    print(f"False Negatives (공격을 정상으로): {fn}")
    print(f"True Positives (공격을 공격으로): {tp}")
    
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall (Detection Rate): {recall:.2f}%")
    print(f"F1 Score: {f1_score:.2f}%")
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_front.png')
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate
    }

# 메인 실행 코드
def main():
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
    normal_dataset = train.PacketImageDataset(
        './Data/save/save_packet_to_byte/back_image', 
        transform=transform,
        is_flat_structure=True
    )

    # Attack 데이터셋 로드
    attack_dataset = train.PacketImageDataset(
        './Data/attack/attack_to_byte', 
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
    test_normal_loader = DataLoader(normal_test_dataset, batch_size=32, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=32, shuffle=False)
                                                    
    model = train.SimplePacketCNN().to(device)
    model.load_state_dict(torch.load('./packet_classifier.pth'))

    evaluate_model(model, test_normal_loader, test_attack_loader, device=device)

if __name__ == '__main__':
    main()