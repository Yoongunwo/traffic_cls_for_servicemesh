import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import sys

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

import AI.Model.CAE.evaluate as evaluate
import AI.Model.CAE.model as autoencoder_model

def train_autoencoder(model, train_loader, num_epochs=50, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            # 입력 데이터를 device로 이동
            data = images.to(device)
            
            # 기울기 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(data)
            loss = criterion(outputs, data)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 에포크당 평균 손실
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return train_losses

def calculate_threshold(model, train_loader, device='cpu', percentile=95):
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for images, _ in train_loader:
            data = images.to(device)
            outputs = model(data)
            
            # 각 이미지의 재구성 오차 계산
            errors = torch.mean((data - outputs) ** 2, dim=(1,2,3))
            reconstruction_errors.extend(errors.cpu().numpy())
    
    # 임계값 계산 (예: 95 퍼센타일)
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold

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

    train_loader = DataLoader(normal_train_dataset, batch_size=1024*16, shuffle=True)
    test_normal_loader = DataLoader(normal_test_dataset, batch_size=1024*16, shuffle=False)
    test_attack_loader = DataLoader(attack_test_dataset, batch_size=1024*16, shuffle=False)


    # 모델 초기화
    model = autoencoder_model.ConvAutoencoder().to(device)
    
    # 모델 학습
    train_losses = train_autoencoder(model, train_loader, num_epochs=50, device=device)
    
    # 학습 그래프 그리기
    evaluate.plot_training_loss(train_losses)
    
    # 재구성 예제 시각화
    evaluate.plot_reconstruction(model, test_normal_loader, device)
    
    # 임계값 계산
    threshold = calculate_threshold(model, train_loader, device)
    print(f"Anomaly threshold: {threshold:.4f}")
    
    # 테스트 데이터 평가
    normal_predictions, normal_scores = evaluate.evaluate_model(model, test_normal_loader, threshold, device)
    attack_predictions, attack_scores = evaluate.evaluate_model(model, test_attack_loader, threshold, device)
    
    # 결과 출력
    roc_auc = evaluate.plot_roc_curve(model, test_normal_loader, test_attack_loader, threshold, device)
    
    # 결과 출력
    print("\nResults:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Normal data detected as anomaly: {normal_predictions.mean()*100:.2f}%")
    print(f"Attack data detected as anomaly: {attack_predictions.mean()*100:.2f}%")
    
    # 결과 저장
    results = {
        'roc_auc': roc_auc,
        'threshold': threshold,
        'normal_detection_rate': normal_predictions.mean(),
        'attack_detection_rate': attack_predictions.mean(),
    }
    
    import json
    with open('detection_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'roc_auc': roc_auc,
    }, 'autoencoder_model.pth')
    
if __name__ == '__main__':
    main()