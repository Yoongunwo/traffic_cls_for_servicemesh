import os
import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, roc_auc_score
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
from AI.Model.PatchCore import train as patchcore_train

from AI.Model.CAE import model as cae_model
from AI.Model.CAE import evaluate as cae_evaluate

from AI.Model.CNN_BiLSTM_AE import train as cnn_bilstm_ae_train

from AI.Model.KD_OCSVM import train as kd_ocsvm_train

from AI import train as com_train

import matplotlib.pyplot as plt

TEST = 'cic'
PREPROCESSING_TYPE = 'zigzag'
METHOD = '_window4'

# TEST_DATASET = f'./Data/byte_16_{PREPROCESSING_TYPE}_seq/save_front/test'
# ATTACK_DATASET = f'./Data/byte_16_{PREPROCESSING_TYPE}_attack/'

TEST_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train'
ATTACK_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/attack'

# ocsvm
OCSVM_MODEL_PATH = f'./AI/Model/OCSVM/Model/{TEST}_{PREPROCESSING_TYPE}_ocsvm_cnn{METHOD}.pth'
OCSVM_SVM_MODEL_PATH = f'./AI/Model/OCSVM/Model/{TEST}_{PREPROCESSING_TYPE}_ocsvm{METHOD}.pkl'

# PatchCore
PATCHCORE_EMBEDDING_PATH=f'./AI/Model/PatchCore/Model/{TEST}_{PREPROCESSING_TYPE}_feature_bank{METHOD}.npy'
PATCHCORE_MODEL_PATH=f'./AI/Model/PatchCore/Model/{TEST}_{PREPROCESSING_TYPE}_nn_model{METHOD}.pkl'

# f-AnoGAN
F_ANOGAN_G_PATH = f'./AI/Model/f_AnoGAN/Model/{TEST}_{PREPROCESSING_TYPE}_generator{METHOD}.pth'
F_ANOGAN_E_PATH = f'./AI/Model/f_AnoGAN/Model/{TEST}_{PREPROCESSING_TYPE}_encoder{METHOD}.pth'
F_ANOGAN_THRESHOLD_PATH = f'./AI/Model/f_AnoGAN/Model/{TEST}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy'

# Deep SVDD
DEEP_SVDD_MODEL_PATH = f'./AI/Model/Deep_SVDD/Model/{TEST}_{PREPROCESSING_TYPE}_deep_svdd{METHOD}.pth'
DEEP_SVDD_CENTER_PATH = f'./AI/Model/Deep_SVDD/Model/{TEST}_{PREPROCESSING_TYPE}_center{METHOD}.npy'
DEEP_SVDD_THRESHOLD_PATH = f'./AI/Model/Deep_SVDD/Model/{TEST}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy'

# CNN-BiLSTM-AE
CNN_BILSTM_AE_MODEL_PATH = f'./AI/Model/CNN_BiLSTM_AE/Model/{TEST}_{PREPROCESSING_TYPE}_cnn_bilstm_ae{METHOD}.pth'
CNN_BILSTM_AE_THRESHOLD_PATH = f'./AI/Model/CNN_BiLSTM_AE/Model/{TEST}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy'

# CAE-Attention
CAE_ATTENTION_MODEL_PATH = f'./AI/Model/CAE_Attention/Model/{TEST}_{PREPROCESSING_TYPE}_attention_cae{METHOD}.pth'
CAE_ATTENTION_THRESHOLD_PATH = f'./AI/Model/CAE_Attention/Model/{TEST}_{PREPROCESSING_TYPE}_threshold_attention_cae{METHOD}.npy'

# CAE
CAE_MODEL_PATH = f'./AI/Model/CAE/Model/{TEST}_{PREPROCESSING_TYPE}_autoencoder{METHOD}.pth'
CAE_THRESHOLD_PATH = f'./AI/Model/CAE/Model/{TEST}_{PREPROCESSING_TYPE}_autoencoder_threshold{METHOD}.npy'

BATCH_SIZE = 1024 * 4
SIZE = 32

MAX_EVAL = 20000

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])

    # ✅ 데이터 로딩
    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=True, label=1)
    if TEST == 'cic':
        limited_indices = list(range(min(len(normal_test), MAX_EVAL)))
        normal_test = Subset(normal_test, limited_indices)
        attack_test = Subset(attack_test, limited_indices)
        print(f"Evaluating on {len(normal_test)} samples.")

    if SIZE == 32:
        # 원래 Dataset 객체에서 경로와 라벨 추출
        original_dataset = normal_test.dataset if isinstance(normal_test, Subset) else normal_test
        selected_indices = normal_test.indices if isinstance(normal_test, Subset) else list(range(len(normal_test)))
        image_paths = [original_dataset.images[i] for i in selected_indices]
        labels = [original_dataset.labels[i] for i in selected_indices]
        normal_test = com_train.SlidingConcatDataset(image_paths, labels, transform=transform)

        original_dataset = attack_test.dataset if isinstance(attack_test, Subset) else attack_test
        selected_indices = attack_test.indices if isinstance(attack_test, Subset) else list(range(len(attack_test)))
        image_paths = [original_dataset.images[i] for i in selected_indices]
        labels = [original_dataset.labels[i] for i in selected_indices]
        attack_test = com_train.SlidingConcatDataset(image_paths, labels, transform=transform)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # one class svm
    featureCNN = ocs_train.DeepFeatureCNN32() if SIZE == 32 else ocs_train.DeepFeatureCNN()
    featureCNN.load_state_dict(torch.load(OCSVM_MODEL_PATH, weights_only=True))
    featureCNN.to(device)
    featureCNN.eval()

    feats_test, labels_test = ocs_train.extract_features(featureCNN, test_loader, device)

    oneclassSVM = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oneclassSVM = joblib.load(OCSVM_SVM_MODEL_PATH)
    
    predictions = oneclassSVM.predict(feats_test)
    predictions = np.where(predictions == 1, 0, 1)  
    print("One Class SVM Predictions:")
    print(classification_report(labels_test, predictions, digits=4))
    print("One Class SVM AUC:", roc_auc_score(labels_test, predictions))

    
    # KD-OCSVM
    # kd_ocsvm = kd_ocsvm_train.TinyFeatureCNN()
    # kd_ocsvm.load_state_dict(torch.load('./AI/Model/KD_OCSVM/Model/tiny_deep_cnn_student.pth', weights_only=True))
    # kd_ocsvm.to(device)
    # kd_ocsvm.eval()

    # feats_test, labels_test = kd_ocsvm_train.extract_features(kd_ocsvm, test_loader, device)
    # oneclassSVM = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    # oneclassSVM = joblib.load('./AI/Model/KD_OCSVM/Model/tiny_deep_ocsvm.pkl')
    # predictions = oneclassSVM.predict(feats_test)
    # predictions = np.where(predictions == 1, 0, 1)
    # print("KD-OCSVM Predictions:")
    # print(classification_report(labels_test, predictions, digits=4))
    # print("KD-OCSVM AUC:", roc_auc_score(labels_test, predictions), "\n")

    # # deep svdd
    # featureCNN = svdd_train.FeatureCNN32() if SIZE == 32 else svdd_train.FeatureCNN()
    # featureCNN.load_state_dict(torch.load(DEEP_SVDD_MODEL_PATH, weights_only=True))
    # featureCNN.eval()
    # featureCNN.to(device)

    # feats_test, labels_test = svdd_train.extract_features(featureCNN, test_loader, device)

    # center = np.load(DEEP_SVDD_CENTER_PATH)
    # threshold = np.load(DEEP_SVDD_THRESHOLD_PATH)

    # dists = np.linalg.norm(feats_test - center, axis=1)

    # preds = (dists > threshold).astype(int)

    # print("Deep SVDD Predictions:")
    # print(classification_report(labels_test, preds, digits=4))
    # print("Deep SVDD AUC:", roc_auc_score(labels_test, preds))

    # # f-AnoGAN
    # G = f_anogan_train.Generator32() if SIZE == 32 else f_anogan_train.Generator()
    # E = f_anogan_train.Encoder32() if SIZE == 32 else f_anogan_train.Encoder()

    # G.load_state_dict(torch.load(F_ANOGAN_G_PATH, weights_only=True))
    # E.load_state_dict(torch.load(F_ANOGAN_E_PATH, weights_only=True))
    # threshold = np.load(F_ANOGAN_THRESHOLD_PATH).item()


    # G.to(device)
    # E.to(device)
    # G.eval()
    # E.eval()

    # print("f-AnoGAN Predictions:")

    # f_anogan_train.evaluate(
    #     encoder=E,
    #     generator=G,
    #     dataloader=test_loader,
    #     device=device
    # )

    # cae-attention
    # cae_attention_train.evaluate_attention_cae(
    #     model_path=CAE_ATTENTION_MODEL_PATH,
    #     threshold_path=CAE_ATTENTION_THRESHOLD_PATH,
    #     test_loader=test_loader,
    #     device=device,
    # )

    # CAE
    # cae = cae_model.ConvAutoencoder32() if SIZE == 32 else cae_model.ConvAutoencoder()
    # cae.load_state_dict(torch.load(CAE_MODEL_PATH, weights_only=True))
    # cae.eval()
    # cae.to(device)
    # try:
    #     threshold = np.load(CAE_THRESHOLD_PATH).item()
    # except:
    #     threshold = None

    # cae_evaluate.evaluate_model(
    #     model=cae,
    #     test_loader=test_loader,
    #     device=device,
    #     threshold=threshold
    # )

    # CNN_BiLSTM_AE
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # ⬅️ flatten: (1,16,16) → (256,)
    ])

    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=False, label=1)

    if SIZE == 32:
        limited_indices = list(range(min(len(normal_test), MAX_EVAL)))
        normal_test = Subset(normal_test, limited_indices)
        attack_test = Subset(attack_test, limited_indices)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    cnn_bilsm_ae = cnn_bilstm_ae_train.CNN_BiLSTM_Autoencoder32(input_dim=256) if SIZE == 32 \
        else cnn_bilstm_ae_train.CNN_BiLSTM_Autoencoder(input_dim=256)
    
    cnn_bilsm_ae.load_state_dict(torch.load(CNN_BILSTM_AE_MODEL_PATH, weights_only=True))
    cnn_bilsm_ae.eval()
    cnn_bilsm_ae.to(device)

    threshold = np.load(CNN_BILSTM_AE_THRESHOLD_PATH).item()
    cnn_bilstm_ae_train.evaluate(
        model=cnn_bilsm_ae,
        dataloader=test_loader,
        threshold=threshold,
        device=device,
    )

    # PatchCore
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    normal_test = cnn_train.PacketImageDataset(TEST_DATASET, transform, is_flat_structure=True, label=0)
    attack_test = cnn_train.PacketImageDataset(ATTACK_DATASET, transform, is_flat_structure=False, label=1)

    min_len = min(len(normal_test), len(attack_test))
    test_dataset = torch.utils.data.ConcatDataset([
        Subset(normal_test, list(range(min_len))),
        Subset(attack_test, list(range(min_len)))
    ])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    patchcore_train.evaluate_patchcore(
        embedding_path=PATCHCORE_EMBEDDING_PATH,
        model_path=PATCHCORE_MODEL_PATH,
        test_loader=test_loader,
        device=device
    )

if __name__ == '__main__':
    main()