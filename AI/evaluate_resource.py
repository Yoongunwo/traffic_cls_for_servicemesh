import os
import sys
import numpy as np
from sklearn import svm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from sklearn import svm
import joblib

import psutil
import time

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train
from AI.Model.OCSVM import train as ocs_train

from AI.Model.KD_OCSVM import train as kd_ocsvm_train

def measure_resource(featureCNN, svm_model, dataloader):
    process = psutil.Process(os.getpid())

    cpu_usages = []
    mem_usages = []

    start_time = time.time()

    with torch.no_grad():
        for x, _ in dataloader:

            # 리소스 측정 시작
            cpu = psutil.cpu_percent(interval=None)
            mem = process.memory_info().rss / (1024 ** 2)  # MB

            # 예측
            features = featureCNN(x).cpu().numpy()
            _ = svm_model.predict(features)

            # 리소스 측정 종료 (단일 패킷 기준)
            cpu_usages.append(cpu)
            mem_usages.append(mem)

    elapsed = time.time() - start_time

    # 최종 출력
    print(f"Inference Time: {elapsed:.4f}s")
    print(f"📊 Average CPU Usage: {np.mean(cpu_usages):.2f}%")
    print(f"📊 Average Memory Usage: {np.mean(mem_usages):.2f} MB")


def measure_latency(featureCNN, svm_model, dataloader):
    with torch.no_grad():

        # ⏱️ 시작 시간
        start_time = time.time()

        # 📌 실제 추론 수행
        predict_ocsvm(featureCNN, svm_model, dataloader)

        elapsed = time.time() - start_time
        print(f"Inference Time: {elapsed:.4f}s")


def predict_ocsvm(featureCNN, svm_model, dataloader):
    for x, y in dataloader:
        f = featureCNN(x).numpy()

        prediction = svm_model.predict(f)

        prediction = np.where(prediction == 1, 0, 1) 



def main():
    device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_test = cnn_train.PacketImageDataset('./Data/byte_16/front_image/test', transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset('./Data/attack_to_byte_16', transform, is_flat_structure=False, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # one class svm
    featureCNN = ocs_train.FeatureCNN()
    featureCNN.load_state_dict(torch.load('./AI/Model/OCSVM/Model/front_ocsvm_cnn_epoch50.pth', weights_only=True))
    featureCNN.to(device)
    featureCNN.eval()

    oneclassSVM = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oneclassSVM = joblib.load('./AI/Model/OCSVM/Model/front_ocsvm.pkl')

    print("\nOne Class SVM")
    measure_resource(featureCNN, oneclassSVM, test_loader)
    measure_latency(featureCNN, oneclassSVM, test_loader)
    
    # deallocate model
    del featureCNN
    del oneclassSVM

    # KD-OCSVM
    kd_ocsvm = kd_ocsvm_train.TinyFeatureCNN()
    kd_ocsvm.load_state_dict(torch.load('./AI/Model/KD_OCSVM/Model/tiny_cnn_student.pth', weights_only=True))
    kd_ocsvm.to(device)
    kd_ocsvm.eval()

    oneclassSVM = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oneclassSVM = joblib.load('./AI/Model/KD_OCSVM/Model/tiny_ocsvm.pkl')

    print("\nKD-OCSVM")
    measure_resource(kd_ocsvm, oneclassSVM, test_loader)
    measure_latency(kd_ocsvm, oneclassSVM, test_loader)
    
if __name__ == '__main__':
    main()