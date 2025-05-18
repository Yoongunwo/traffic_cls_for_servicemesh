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
    preds = [0 if p == 1 else 1 for p in preds]  # 이상이면 1

    print("\n✅ KD + Tiny-CNN + OC-SVM test 결과")
    print(classification_report(labels_test, preds, digits=4))

# ✅ 메인 실행
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_train = cnn_train.PacketImageDataset('./Data/byte_16/front_image/train', transform, is_flat_structure=True, label=0)
    normal_test = cnn_train.PacketImageDataset('./Data/byte_16/front_image/test', transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset('./Data/attack_to_byte_16', transform, is_flat_structure=False, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    train_loader = DataLoader(normal_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # ✅ 모델 정의 및 KD 학습
    # teacher = ocs_train.FeatureCNN()
    teacher = ocs_train.DeepFeatureCNN()
    # teacher.load_state_dict(torch.load('./AI/Model/OCSVM/Model/front_ocsvm_cnn_epoch50.pth', weights_only=True))
    teacher.load_state_dict(torch.load('./AI/Model/OCSVM/Model/front_ocsvm_deep_cnn_epoch50.pth', weights_only=True))
    teacher.eval()

    student = TinyFeatureCNN()
    train_kd(student, teacher, train_loader, device, epochs=100)

    # ✅ Feature 추출 및 One-Class SVM 학습
    feats_train, _ = extract_features(student, train_loader, device)
    feats_test, labels_test = extract_features(student, test_loader, device)

    clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    clf.fit(feats_train)

    preds = clf.predict(feats_test)
    preds = [0 if p == 1 else 1 for p in preds]  # 이상이면 1

    print("\n✅ KD + Tiny-CNN + OC-SVM 결과")                                                               
    print(classification_report(labels_test, preds, digits=4))

    os.makedirs('./AI/Model/KD_OCSVM/Model', exist_ok=True)                                                                                                                                             
    torch.save(student.state_dict(), './AI/Model/KD_OCSVM/Model/tiny_deep_cnn_student.pth')
    joblib.dump(clf, './AI/Model/KD_OCSVM/Model/tiny_deep_ocsvm.pkl')

    evaluate('./AI/Model/KD_OCSVM/Model/tiny_deep_cnn_student.pth', './AI/Model/KD_OCSVM/Model/tiny_deep_ocsvm.pkl', test_loader, device)

if __name__ == '__main__':
    main()                                                                                                               
