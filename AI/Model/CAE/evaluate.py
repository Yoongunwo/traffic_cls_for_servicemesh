import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

import AI.Model.CAE.model as autoencoder_model
import AI.Model.CAE.train as train

def evaluate_model(model, test_loader, threshold, device='cpu'):
    model.eval()
    anomaly_scores = []
    
    with torch.no_grad():
        for images_, _ in test_loader:
            data = images_.to(device)
            outputs = model(data)
            
            # 재구성 오차 계산
            errors = torch.mean((data - outputs) ** 2, dim=(1,2,3))
            anomaly_scores.extend(errors.cpu().numpy())
    
    # 이상 탐지
    predictions = (np.array(anomaly_scores) > threshold).astype(int)
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
    
    # 정상 데이터 처리
    with torch.no_grad():
        for images, labels in test_normal_loader:
            images = images.to(device)
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))
            predictions = (errors > threshold).cpu().numpy().astype(int)
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    # 공격 데이터 처리
    with torch.no_grad():
        for images, labels in test_attack_loader:
            images = images.to(device)
            outputs = model(images)
            errors = torch.mean((images - outputs) ** 2, dim=(1,2,3))
            predictions = (errors > threshold).cpu().numpy().astype(int)
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 성능 메트릭 계산
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    return metrics

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    # 데이터셋 로드
    normal_dataset = autoencoder_model.PacketImageDataset(
        './Data/save/save_packet_to_byte/front_image', 
        transform=transform,
        is_flat_structure=True
    )

    # Attack 데이터셋 로드
    attack_test_dataset = autoencoder_model.PacketImageDataset(
        './Data/attack/attack_to_byte', 
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
    (torch.load('./AI/Model/CAE/autoencoder_model.pth'))
    
    checkpoint = torch.load('./AI/Model/CAE/autoencoder_model.pth')
    model = autoencoder_model.ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # state_dict만 로드
    threshold = checkpoint['threshold']  # 저장된 임계값 사용
    
    # 모델 평가
    evaluate_model(model, test_normal_loader, threshold, device=device)
    
    # ROC 커브 그리기
    plot_roc_curve(model, test_normal_loader, test_attack_loader, threshold, device=device)
    
    # 혼동 행렬 및 성능 메트릭 계산
    plot_confusion_matrix(model, test_normal_loader, test_attack_loader, threshold, device=device)

if __name__ == '__main__':
    main()