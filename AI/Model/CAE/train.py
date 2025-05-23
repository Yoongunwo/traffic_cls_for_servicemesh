import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import os
import sys

current_dir = os.getcwd()  # C:\Users\gbwl3\Desktop\SourceCode\k8s_research
sys.path.append(current_dir)

import AI.Model.CAE.evaluate as evaluate
import AI.Model.CAE.model as autoencoder_model
from AI.Model.CNN import train_v2 as cnn_train

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

TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

MODEL_DIR = './AI/Model/CAE/Model'
CNN_MODEL_PATH = './AI/Model/CAE/Model/cic_autoencoder_epoch50.pth'
THRESH_HOLD_PATH = './AI/Model/CAE/Model/cic_autoencoder_threshold.npy'

BATCH_SIZE = 4096*32

def train_model(device, train_loader, epoches, model_dir, model_path, threshold_path):
    # initialize model
    model = autoencoder_model.ConvAutoencoder().to(device)
    
    # training
    train_autoencoder(model, train_loader, num_epochs=epoches, device=device)
    
    # save
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_path))

    # 임계값 계산
    threshold = calculate_threshold(model, train_loader, device=device, percentile=95)
    # save threshold
    np.save(os.path.join(model_dir, threshold_path), threshold)

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])
    
    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 초기화
    model = autoencoder_model.ConvAutoencoder().to(device)
    
    # 모델 학습
    train_losses = train_autoencoder(model, train_loader, num_epochs=50, device=device)
    
    # save
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), CNN_MODEL_PATH)

    # 임계값 계산
    threshold = calculate_threshold(model, train_loader, device=device, percentile=95)
    # save threshold
    np.save(THRESH_HOLD_PATH, threshold)

    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 평가
    evaluate.evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=threshold
    )
    
    
if __name__ == '__main__':
    main()