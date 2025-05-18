import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
from sklearn.metrics import classification_report, roc_auc_score

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

import AI.Model.CAE.model as autoencoder_model
import AI.Model.CAE.train as train

def evaluate_model(model, test_loader, threshold, device='cpu'):
    model.eval()
    anomaly_scores = []
    all_labels = []
    
    with torch.no_grad():
        for images_, labels in test_loader:
            data = images_.to(device)
            outputs = model(data)
            
            # 재구성 오차 계산
            errors = torch.mean((data - outputs) ** 2, dim=(1,2,3))
            anomaly_scores.extend(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    anomaly_scores = np.array(anomaly_scores)
    all_labels = np.array(all_labels)

    if len(anomaly_scores) == 0 or len(all_labels) == 0:
        print("🚨 No data to evaluate. Check test_loader or preprocessing.")
        return [], []

    predictions = (anomaly_scores > threshold).astype(int)

    print("CAE Classification Report:")
    print(classification_report(all_labels, predictions, digits=4, zero_division=0))
    print("CAE AUC:", roc_auc_score(all_labels, anomaly_scores))
    return predictions, anomaly_scores

def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

def plot_reconstruction(model, test_loader, device='cpu', num_images=5):
    model.eval()
    with torch.no_grad():
        # 테스트 데이터에서 이미지 가져오기
        images, _ = next(iter(test_loader))
        images = images[:num_images].to(device)
        
        # 이미지 재구성
        reconstructed = model(images)
        
        # 시각화
        plt.figure(figsize=(12, 4))
        for i in range(num_images):
            # 원본 이미지
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
            
            # 재구성된 이미지
            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
                
        plt.savefig('reconstruction_examples.png')
        plt.close()

def plot_roc_curve(model, test_normal_loader, test_attack_loader, threshold, device='cpu'):
    model.eval()
    all_labels = []
    all_scores = []
    
    # 정상 데이터 처리
    with torch.no_grad():
        for images, _ in test_normal_loader:  # 이미지와 레이블 분리
            images = images.to(device)        # 이미지만 device로 이동
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))
            all_scores.extend(errors.cpu().numpy())
            all_labels.extend([0] * images.size(0))  # batch.size(0)를 images.size(0)로 변경

    # 공격 데이터 처리
    with torch.no_grad():
        for images, _ in test_attack_loader:  # 이미지와 레이블 분리
            images = images.to(device)        # 이미지만 device로 이동
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))
            all_scores.extend(errors.cpu().numpy())
            all_labels.extend([1] * images.size(0))  # batch.size(0)를 images.size(0)로 변경

    # ROC 커브 계산
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    # ROC 커브 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    return roc_auc

def plot_confusion_matrix(model, test_normal_loader, test_attack_loader, threshold, device='cpu'):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_predictions = []
    all_labels = []
    normal_errors = []
    attack_errors = []
    
    # 정상 데이터 처리
    with torch.no_grad():
        for images, labels in test_normal_loader:
            images = images.to(device)
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))
            
            # Append errors
            normal_errors.extend(errors.cpu().numpy())

            predictions = (errors > threshold).cpu().numpy().astype(int)
            all_predictions.extend(predictions)
            all_labels.extend([0] * images.size(0)) 
            # all_labels.extend(labels.numpy())

    # 공격 데이터 처리
    with torch.no_grad():
        for images, labels in test_attack_loader:
            images = images.to(device)
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))

            # Append errors
            attack_errors.extend(errors.cpu().numpy())

            predictions = (errors > threshold).cpu().numpy().astype(int)
            all_predictions.extend(predictions)
            all_labels.extend([1] * images.size(0))
            # all_labels.extend(labels.numpy())
    
    # return metrics
    print(f"Normal errors - min: {np.min(normal_errors):.6f}, max: {np.max(normal_errors):.6f}, mean: {np.mean(normal_errors):.6f}, std: {np.std(normal_errors):.6f}")
    print(f"Attack errors - min: {np.min(attack_errors):.6f}, max: {np.max(attack_errors):.6f}, mean: {np.mean(attack_errors):.6f}, std: {np.std(attack_errors):.6f}")
    print(f"Current threshold: {threshold:.6f}")
    
    # 더 나은 임계값 계산해보기
    all_errors = np.concatenate([normal_errors, attack_errors])
    all_true_labels = np.concatenate([[0] * len(normal_errors), [1] * len(attack_errors)])
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_errors)
    
    # Youden's J statistic으로 최적 임계값 찾기 (specificity + sensitivity - 1을 최대화)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Calculated optimal threshold: {optimal_threshold:.6f}")
    print(f"At optimal threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    
    # 혼동 행렬 계산을 위해 최적 임계값으로 예측 다시 생성
    all_predictions_optimal = (np.array(all_errors) > optimal_threshold).astype(int)
    
    # 원래 임계값으로의 혼동 행렬
    cm_original = confusion_matrix(all_labels, all_predictions)
    
    # 최적 임계값으로의 혼동 행렬
    cm_optimal = confusion_matrix(all_true_labels, all_predictions_optimal)
    
    # 시각화 - 원래 임계값
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix (Threshold: {threshold:.6f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_original.png')
    plt.close()
    
    # 시각화 - 최적 임계값
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.6f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_optimal.png')
    plt.close()
    
    # 원래 임계값으로의 성능 메트릭 계산
    tn, fp, fn, tp = cm_original.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_original = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    # 최적 임계값으로의 성능 메트릭 계산
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
    accuracy_opt = (tp_opt + tn_opt) / (tp_opt + tn_opt + fp_opt + fn_opt)
    precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
    recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
    f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0
    
    metrics_optimal = {
        'accuracy': accuracy_opt,
        'precision': precision_opt,
        'recall': recall_opt,
        'f1_score': f1_opt,
        'true_negatives': int(tn_opt),
        'false_positives': int(fp_opt),
        'false_negatives': int(fn_opt),
        'true_positives': int(tp_opt),
        'optimal_threshold': optimal_threshold
    }
    
    print("\nOriginal Threshold Metrics:")
    print(f"Accuracy: {metrics_original['accuracy']:.4f}")
    print(f"Precision: {metrics_original['precision']:.4f}")
    print(f"Recall: {metrics_original['recall']:.4f}")
    print(f"F1 Score: {metrics_original['f1_score']:.4f}")
    
    print("\nOptimal Threshold Metrics:")
    print(f"Accuracy: {metrics_optimal['accuracy']:.4f}")
    print(f"Precision: {metrics_optimal['precision']:.4f}")
    print(f"Recall: {metrics_optimal['recall']:.4f}")
    print(f"F1 Score: {metrics_optimal['f1_score']:.4f}")
    
    return metrics_original, metrics_optimal

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])
    
    # 데이터셋 로드
    normal_dataset = autoencoder_model.PacketImageDataset(
        './Data/save/save_packet_to_byte_16/front_image', 
        transform=transform,
        is_flat_structure=True
    )

    # Attack 데이터셋 로드
    attack_test_dataset = autoencoder_model.PacketImageDataset(
        './Data/attack/attack_to_byte_16', 
        transform=transform,
        is_flat_structure=False
    )
    
    # 데이터로더 생성
    # 각 데이터셋을 학습/테스트용으로 분할
    generator = torch.Generator().manual_seed(42)

    # Normal 데이터 분할
    normal_train_size = int(0.8 * len(normal_dataset))
    normal_test_size = len(normal_dataset) - normal_train_size
    normal_train_dataset, normal_test_dataset = torch.utils.data.random_split(
        normal_dataset, [normal_train_size, normal_test_size],
        generator=generator
    )
    print(f"Normal Train: {len(normal_train_dataset)}")
    print(f"Normal Test: {len(normal_test_dataset)}")
    print(f"Attack Test: {len(attack_test_dataset)}")

    train_loader = DataLoader(normal_train_dataset, batch_size=1024*16, shuffle=True)
    test_normal_loader = DataLoader(normal_test_dataset, batch_size=1024*16, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=1024*16, shuffle=False)
    
    # 모델 로드
    model = autoencoder_model.ConvAutoencoder().to(device)
    (torch.load('./AI/Model/CAE/autoencoder_model_16.pth'))
    
    checkpoint = torch.load('./AI/Model/CAE/autoencoder_model_16.pth')
    model = autoencoder_model.ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # state_dict만 로드
    threshold = checkpoint['threshold']  # 저장된 임계값 사용
    
    # 모델 평가
    evaluate_model(model, test_normal_loader, threshold, device=device)
    
    # ROC 커브 그리기
    # plot_roc_curve(model, test_normal_loader, test_attack_loader, threshold, device=device)
    
    # # 혼동 행렬 및 성능 메트릭 계산
    # metrics = plot_confusion_matrix(model, test_normal_loader, test_attack_loader, threshold, device=device)

    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1_score']:.4f}")
    roc_auc = plot_roc_curve(model, test_normal_loader, test_attack_loader, threshold, device=device)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # 혼동 행렬 및 성능 메트릭 계산
    metrics_original, metrics_optimal = plot_confusion_matrix(model, test_normal_loader, test_attack_loader, threshold, device=device)
    
    # 최적 임계값으로 모델 다시 평가
    if metrics_optimal['f1_score'] > metrics_original['f1_score']:
        print(f"\nUsing optimal threshold: {metrics_optimal['optimal_threshold']:.6f}")
        print(f"Precision: {metrics_optimal['precision']:.4f}")
        print(f"Recall: {metrics_optimal['recall']:.4f}")
        print(f"F1 Score: {metrics_optimal['f1_score']:.4f}")
        
        # 최적 임계값으로 새 체크포인트 저장
        checkpoint['threshold'] = metrics_optimal['optimal_threshold']
        torch.save(checkpoint, './AI/Model/CAE/autoencoder_model_16_optimal.pth')
        print("Saved model with optimal threshold")
    else:
        print(f"\nKeeping original threshold: {threshold:.6f}")
        print(f"Precision: {metrics_original['precision']:.4f}")
        print(f"Recall: {metrics_original['recall']:.4f}")
        print(f"F1 Score: {metrics_original['f1_score']:.4f}")

    attack_precision_original = metrics_original['true_positives'] / (metrics_original['true_positives'] + metrics_original['false_negatives']) if (metrics_original['true_positives'] + metrics_original['false_negatives']) > 0 else 0
    attack_recall_original = metrics_original['true_positives'] / (metrics_original['true_positives'] + metrics_original['false_positives']) if (metrics_original['true_positives'] + metrics_original['false_positives']) > 0 else 0

    attack_precision_optimal = metrics_optimal['true_positives'] / (metrics_optimal['true_positives'] + metrics_optimal['false_negatives']) if (metrics_optimal['true_positives'] + metrics_optimal['false_negatives']) > 0 else 0
    attack_recall_optimal = metrics_optimal['true_positives'] / (metrics_optimal['true_positives'] + metrics_optimal['false_positives']) if (metrics_optimal['true_positives'] + metrics_optimal['false_positives']) > 0 else 0

    print("\n🔹 공격 기준 Precision & Recall (Original Threshold)")
    print(f"Attack Precision: {attack_precision_original:.4f}")
    print(f"Attack Recall: {attack_recall_original:.4f}")

    print("\n🔹 공격 기준 Precision & Recall (Optimal Threshold)")
    print(f"Attack Precision: {attack_precision_optimal:.4f}")
    print(f"Attack Recall: {attack_recall_optimal:.4f}")

    normal_precision_optimal = metrics_optimal['true_negatives'] / (metrics_optimal['true_negatives'] + metrics_optimal['false_positives']) if (metrics_optimal['true_negatives'] + metrics_optimal['false_positives']) > 0 else 0
    normal_recall_optimal = metrics_optimal['true_negatives'] / (metrics_optimal['true_negatives'] + metrics_optimal['false_negatives']) if (metrics_optimal['true_negatives'] + metrics_optimal['false_negatives']) > 0 else 0

    print("\n🔹 정상 기준 Precision & Recall (Optimal Threshold)")
    print(f"Normal Precision: {normal_precision_optimal:.4f}")
    print(f"Normal Recall: {normal_recall_optimal:.4f}")

if __name__ == '__main__':
    main()