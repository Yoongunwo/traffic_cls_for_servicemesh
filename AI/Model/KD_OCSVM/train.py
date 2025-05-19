import joblib
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import os
import sys

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train
from AI.Model.OCSVM import train as ocs_train

# ✅ Student 모델 (경량화 CNN)
class TinyFeatureCNN(nn.Module):
    def __init__(self):
        super(TinyFeatureCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128)
        )
    def forward(self, x):
        return self.encoder(x)

# ✅ Feature 추출 함수
@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        f = model(x).cpu().numpy()
        feats.extend(f)
        labels.extend(y.numpy())
    return np.array(feats), np.array(labels)

# ✅ KD 학습 함수
def train_kd(student, teacher, dataloader, device, epochs=50):
    student.to(device)
    teacher.to(device)
    teacher.eval()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            with torch.no_grad():
                teacher_feat = teacher(x)
            student_feat = student(x)
            loss = criterion(student_feat, teacher_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[KD Epoch {epoch+1}/{epochs}] Loss: {total_loss / len(dataloader):.4f}")

def evaluate(kd_model_path, model_path, test_loader, device):
    student = TinyFeatureCNN()
    student.load_state_dict(torch.load(kd_model_path, weights_only=True))
    student.eval()
    student.to(device)

    feats_test, labels_test = extract_features(student, test_loader, device)
    clf = joblib.load(model_path)
    preds = clf.predict(feats_test)
    preds = [0 if p == 1 else 1 for p in preds]  # 1 : anomaly, 0 : normal

    print("\n✅ KD + Tiny-CNN + OC-SVM:")
    print(classification_report(labels_test, preds, digits=4))


TRAIN_DATASET = './Data/cic_data/Wednesday-workingHours/benign_train'
TEST_DATASET = './Data/cic_data/Wednesday-workingHours/benign_test'
ATTACK_DATASET = './Data/cic_data/Wednesday-workingHours/attack'

TEACHER_MODEL_PATH = './AI/Model/OCSVM/Model/cic_ocsvm_deep_cnn_epoch50.pth'

MODEL_DIR = './AI/Model/KD_OCSVM/Model'
CNN_MODEL_PATH = './AI/Model/KD_OCSVM/Model/cic_tiny_deep_cnn_student.pth'
OCSVM_MODEL_PATH = './AI/Model/KD_OCSVM/Model/cic_tiny_deep_ocsvm.pkl'

BATCH_SIZE = 4096 * 8
EPOCHS = 50

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)
    
    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE, shuffle=True)

    # ✅ 모델 정의 및 KD 학습
    teacher = ocs_train.DeepFeatureCNN()
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, weights_only=True))
    teacher.eval()

    student = TinyFeatureCNN()
    train_kd(student, teacher, train_loader, device, epochs=EPOCHS)

    feats_train, _ = extract_features(student, train_loader, device)

    clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    clf.fit(feats_train)

    os.makedirs(MODEL_DIR, exist_ok=True)                                                                                                                                             
    torch.save(student.state_dict(), CNN_MODEL_PATH)
    joblib.dump(clf, OCSVM_MODEL_PATH)


    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=False, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    feats_test, labels_test = extract_features(student, test_loader, device)


    preds = clf.predict(feats_test)
    preds = [0 if p == 1 else 1 for p in preds]  # 이상이면 1

    print("\n✅ KD + Tiny-CNN + OC-SVM 결과")                                                               
    print(classification_report(labels_test, preds, digits=4))

if __name__ == '__main__':
    main()                                                                                                               
