import os
import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from collections import Counter

from sklearn import svm
import joblib

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train
from AI.Model.OCSVM import train as ocs_train
from AI.Model.Deep_SVDD import train as svdd_train
from AI.Model.f_AnoGAN import train as f_anogan_train
from AI.Model.CAE_Attention import train as cae_attention_train


import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # one class svm
    featureCNN = ocs_train.FeatureCNN()
    featureCNN.load_state_dict(torch.load('./AI/Model/OCSVM/Model/front_ocsvm_cnn_epoch50.pth', weights_only=True))
    featureCNN.to(device)
    featureCNN.eval()

    feats_test, labels_test = ocs_train.extract_features(featureCNN, test_loader, device)

    oneclassSVM = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oneclassSVM = joblib.load('./AI/Model/OCSVM/Model/front_ocsvm.pkl')
    
    predictions = oneclassSVM.predict(feats_test)
    predictions = np.where(predictions == 1, 0, 1)  
    print("One Class SVM Predictions:")
    print(classification_report(labels_test, predictions, digits=4))

    # # deep svdd
    featureCNN = svdd_train.FeatureCNN()
    featureCNN.load_state_dict(torch.load('./AI/Model/Deep_SVDD/Model/front_deep_svdd_model.pth', weights_only=True))
    featureCNN.eval()
    featureCNN.to(device)

    feats_test, labels_test = svdd_train.extract_features(featureCNN, test_loader, device)

    center = np.load('./AI/Model/Deep_SVDD/Model/front_deep_svdd_center.npy')
    threshold = np.load('./AI/Model/Deep_SVDD/Model/front_deep_svdd_threshold.npy')

    dists = np.linalg.norm(feats_test - center, axis=1)

    preds = (dists > threshold).astype(int)

    print("Deep SVDD Predictions:")
    print(classification_report(labels_test, preds, digits=4))

    # f-AnoGAN
    G = f_anogan_train.Generator()
    E = f_anogan_train.Encoder()

    G.load_state_dict(torch.load('./AI/Model/f_AnoGAN/Model/front_generator_epoch50.pth', weights_only=True))
    E.load_state_dict(torch.load('./AI/Model/f_AnoGAN/Model/front_encoder_epoch50.pth', weights_only=True))
    threshold = np.load('./AI/Model/f_AnoGAN/Model/front_threshold.npy').item()


    G.to(device)
    E.to(device)
    G.eval()
    E.eval()

    all_scores = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            z_hat = E(imgs)
            x_gen = G(z_hat)
            score = F.l1_loss(imgs, x_gen, reduction='none')
            score = score.view(score.size(0), -1).mean(dim=1)
            batch_preds = (score > threshold).int().cpu().numpy()
            all_scores.extend(score.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend(batch_preds)

    print("f-AnoGAN Predictions:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    # 예: f-AnoGAN or CAE-Attention
    plt.hist(all_scores, bins=100)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    # cae-attention
    cae_attention_train.evaluate_attention_cae(
        model_path='./AI/Model/CAE_Attention/Model/front_attention_cae.pth',
        test_loader=test_loader,
        device=device,
        threshold=threshold
    )

    # PatchCore

    # CAE

    # CNN_BiLSTM


if __name__ == '__main__':
    main()
