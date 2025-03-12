import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import sys

current_dir = os.getcwd()  
sys.path.append(current_dir)

from AI.Model.CNN import train

from AI.Model.CNN_BiLSTM import train as cnn_bilstm_train


def evaluate_model(model, test_normal_loader, test_attack_loader, device='cuda'):
    model.eval()

    all_scores = []
    all_labels = []

    # 🔹 정상 데이터 평가 (Label: 0)
    with torch.no_grad():
        for images, _ in test_normal_loader:
            images = images.to(device)
            labels = torch.zeros(images.size(0), dtype=torch.long).cpu().numpy()  # 정상 데이터 (0)

            outputs = model(images)
            scores = torch.softmax(outputs, dim=1)[:, 1]  # 공격(1) 클래스 확률 사용

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels)

    # 🔹 공격 데이터 평가 (Label: 1)
    with torch.no_grad():
        for images, _ in test_attack_loader:
            images = images.to(device)
            labels = torch.ones(images.size(0), dtype=torch.long).cpu().numpy()  # 공격 데이터 (1)

            outputs = model(images)
            scores = torch.softmax(outputs, dim=1)[:, 1]  # 공격(1) 클래스 확률 사용

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels)

    # 🔹 ROC 커브 및 AUC 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    # 🔹 최적 임계값 찾기 (Youden's J Statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n🔹 Optimal Threshold: {optimal_threshold:.6f}")
    print(f"🔹 At optimal threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")

    # 🔹 최적 임계값으로 예측값 생성
    predictions_optimal = (np.array(all_scores) > optimal_threshold).astype(int)

    # 🔹 Confusion Matrix 계산
    cm_optimal = confusion_matrix(all_labels, predictions_optimal)

    # True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm_optimal.ravel()

    # 🔹 공격 데이터(1) 기준 Precision, Recall, F1-score
    attack_precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    attack_recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    attack_f1 = 2 * attack_precision * attack_recall / (attack_precision + attack_recall) if (attack_precision + attack_recall) > 0 else 0

    # 🔹 정상 데이터(0) 기준 Precision, Recall, F1-score
    normal_precision = 100 * tn / (tn + fn) if (tn + fn) > 0 else 0
    normal_recall = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0

    # 🔹 결과 출력
    print("\nConfusion Matrix (Optimal Threshold):")
    print(f"True Negatives (정상을 정상으로): {tn}")
    print(f"False Positives (정상을 공격으로): {fp}")
    print(f"False Negatives (공격을 정상으로): {fn}")
    print(f"True Positives (공격을 공격으로): {tp}")

    print("\nMetrics (Attack 기준 - Positive)")
    print(f"Precision: {attack_precision:.2f}%")
    print(f"Recall: {attack_recall:.2f}%")
    print(f"F1 Score: {attack_f1:.2f}%")

    print("\nMetrics (Normal 기준 - Negative)")
    print(f"Precision: {normal_precision:.2f}%")
    print(f"Recall: {normal_recall:.2f}%")
    print(f"F1 Score: {normal_f1:.2f}%")

    # 🔹 Confusion Matrix 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_optimal.png')
    plt.close()

    # 🔹 ROC Curve 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    return {
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm_optimal,
        'roc_auc': roc_auc,
        'attack_precision': attack_precision,
        'attack_recall': attack_recall,
        'attack_f1': attack_f1,
        'normal_precision': normal_precision,
        'normal_recall': normal_recall,
        'normal_f1': normal_f1
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

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

    test_normal_loader = DataLoader(normal_test_dataset, batch_size=16384, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=16384, shuffle=False)

    model = cnn_bilstm_train.CNN_BiLSTM(input_size=32, hidden_size=64, num_classes=2).to(device)
    model.load_state_dict(torch.load('./AI/Model/CNN_BiLSTM/packet_classifier_epoch50.pth'))

    results = evaluate_model(model, test_normal_loader, test_attack_loader, device=device)

    print(f"\n🔹 Final Optimal Threshold: {results['optimal_threshold']:.6f}")
    print(f"🔹 ROC AUC: {results['roc_auc']:.4f}")

if __name__ == '__main__':
    main()
