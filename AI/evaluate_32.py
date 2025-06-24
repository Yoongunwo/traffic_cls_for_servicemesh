import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

import re

current_dir = os.getcwd()
sys.path.append(current_dir)

from AI.Statistic.CNN_GRU import train as cnn_gru_train
from AI.Statistic.Deep_CNN_AE import train as deep_cnn_ae_train
from AI.Statistic.CNN_Transformer import train as cnn_transformer_train
from AI.Statistic.MAE import train_v2 as mae_train
from AI.Statistic.ConvLSTM_AE import train as conv_lstm_ae_train

from AI.FrameStack.AE import train as frame_stack_ae_train

# ✅ 통계 이미지 데이터셋
class StatisticalChannelDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        stack = torch.stack(imgs, dim=0).squeeze(1)  # [T, H, W]
        stat = torch.stack([
            stack.mean(dim=0),
            stack.std(dim=0),
            stack.min(dim=0).values,
            stack.max(dim=0).values
        ], dim=0)  # [4, H, W]

        return stat, self.labels[idx + self.window_size - 1]

def natural_key(string):
    # 문자열 내 숫자를 기준으로 정렬 (예: packet_2 < packet_10)
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def get_paths(path, label, stack_folder=False):
    image_paths = []
    labels = []
    if stack_folder:
        for subfolder in sorted(os.listdir(path)):
            subfolder_path = os.path.join(path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            for f in sorted(files, key=natural_key):
                image_paths.append(os.path.join(subfolder_path, f))
                labels.append(label)
    else:
        files = sorted([f for f in os.listdir(path) if f.endswith('.png')], key=natural_key)
        image_paths = [os.path.join(path, f) for f in files]
        labels = [label] * len(image_paths)

    return image_paths, labels

def get_statistical_data():
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack/sA'), 1)

    benign_p, benign_l = benign_p[50000:55000+WIN-1], benign_l[50000:55000+WIN-1]
    atk_p, atk_l = attack_p[:5000+WIN-1], attack_l[:5000+WIN-1]

    transform = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

    benign_ds = StatisticalChannelDataset(benign_p, benign_l, transform, WIN)
    atk_ds = StatisticalChannelDataset(atk_p, atk_l, transform, WIN)

    print(f"Train dataset size: {len(benign_ds)}")
    print(f"Test dataset size: {len(atk_ds)}")

    test_ds = ConcatDataset([benign_ds, atk_ds])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return test_loader

def get_stackframe_data():
    benign_p, benign_l = get_paths(os.path.join(ROOT, 'benign_train'), 0)
    test_p, test_l = get_paths(os.path.join(ROOT, 'benign_val'), 0)
    attack_p, attack_l = get_paths(os.path.join(ROOT, 'attack'), 1)

    transform = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor()])

    benign_ds = frame_stack_ae_train.FrameStackDataset(benign_p, benign_l, transform, WIN)
    test_ds = frame_stack_ae_train.FrameStackDataset(test_p, test_l, transform, WIN)
    atk_ds = frame_stack_ae_train.FrameStackDataset(attack_p, attack_l, transform, WIN)

    train_loader = DataLoader(benign_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_ds = ConcatDataset([test_ds, atk_ds])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return train_loader, test_loader

PREPROCESSING_TYPE = 'hilbert'
# ROOT = f'./Data/byte_16_hilbert_seq/save_front'
# ATTACK_ROOT = f'./Data/byte_16_hilbert_attack'
ROOT = f'./Data_CIC/Fri_{PREPROCESSING_TYPE}_32/'
BATCH_SIZE = 1024 * 4
WIN = 16
random_state = 42
SIZE = 32
EPOCHS = 10

def start_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_statistical_data()

    # print(f"Statistic CNN+GRU Window size: {WIN}, Data: {ROOT}")
    # model = cnn_gru_train.CNN_GRU_StatAE32(in_channels=4)
    # model.load_state_dict(torch.load(f'./AI/Statistic/CNN_GRU/Model/cnn_gru_statae_{PREPROCESSING_TYPE}_{WIN}_{SIZE}.pth'))
    # model.to(device)
    # cnn_gru_train.evaluate(model, test_loader, device)

    # print(f"Statistic Deep CNN AE Window size: {WIN}, Data: {ROOT}")
    # deep_cnn_ae_model = deep_cnn_ae_train.DeepStatCNN_AE32()
    # deep_cnn_ae_model.load_state_dict(torch.load(f'./AI/Statistic/Deep_CNN_AE/Model/deep_stat_cnn_ae_{WIN}_{EPOCHS}_{SIZE}.pth'))
    # deep_cnn_ae_model.to(device)
    # deep_cnn_ae_train.evaluate(deep_cnn_ae_model, test_loader, device)

    print(f"Statistic CNN Transformer Window size: {WIN}, Data: {ROOT}")
    cnn_transformer_model = cnn_transformer_train.CNNTransformerAE()
    cnn_transformer_model.load_state_dict(torch.load(f'./AI/Statistic/CNN_Transformer/Model/stat_cnn_transformer_ae_{WIN}_{EPOCHS}_{SIZE}.pth'))
    cnn_transformer_model.to(device)
    cnn_transformer_train.evaluate(cnn_transformer_model, test_loader, device)

    # print(f"Statistic MAE Window size: {WIN}, Data: {ROOT}")
    # mae_model = mae_train.MAE(in_ch=4, img_size=SIZE)
    # mae_model.load_state_dict(torch.load(f'./AI/Statistic/MAE/Model/mae_model_{WIN}_{SIZE}.pth'))
    # mae_model.to(device)
    # mae_train.evaluate(mae_model, test_loader, device)

    # print(f"Statistic ConvLSTM AE Window size: {WIN}, Data: {ROOT}")
    # conv_lstm_ae_model = conv_lstm_ae_train.ConvLSTM_AE()
    # conv_lstm_ae_train.train(conv_lstm_ae_model, train_loader, test_loader, device, epochs=EPOCHS)

    # train_loader, test_loader = get_stackframe_data()

    # print(f"Frame Stack AE Window size: {WIN}, Data: {ROOT}")
    # frame_stack_ae_model = frame_stack_ae_train.FrameStackAE()
    # frame_stack_ae_train.train_and_evaluate(frame_stack_ae_model, train_loader, test_loader, device, epochs=EPOCHS)
    # torch.save(frame_stack_ae_model.state_dict(), f'./AI/FrameStack/AE/Model/frame_stack_ae_{PREPROCESSING_TYPE}_{WIN}_{SIZE}.pth')


# ✅ 실행
if __name__ == "__main__":
    start_evaluate()

