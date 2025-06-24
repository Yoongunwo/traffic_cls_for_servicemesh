import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

current_dir = os.getcwd() 
sys.path.append(current_dir)

from AI.Model.CNN import train_v2 as cnn_train

from AI.Model.OCSVM import train as ocs_train
from AI.Model.Deep_SVDD import train as svdd_train
from AI.Model.f_AnoGAN import train as f_anogan_train
from AI.Model.CAE_Attention import train as cae_attention_train
from AI.Model.PatchCore import train as patchcore_train
from AI.Model.CAE import train as cae_train
from AI.Model.CNN_BiLSTM_AE import train as cnn_bilstm_ae_train

from torch.utils.data import Subset

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SlidingConcatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1

    def __getitem__(self, idx):
        imgs = []
        lbls = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('L')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            lbls.append(self.labels[idx + i])

        # 🧩 16x16 → 32x32로 조합: [0,1]
        #                           [2,3]
        top = torch.cat([imgs[0], imgs[1]], dim=2)
        bottom = torch.cat([imgs[2], imgs[3]], dim=2)
        final_img = torch.cat([top, bottom], dim=1)

        # 레이블은 가장 마지막 프레임 기준
        return final_img, lbls[-1]
    
class SlidingConcatDataset_for_cnnbilsmae(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        assert len(image_paths) == len(labels)
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
                img = self.transform(img)  # (1, 16, 16)
            imgs.append(img)
        
        # 2x2 식 배열로 합치기: 1, 2 (top), 3, 4 (bottom)
        top = torch.cat([imgs[0], imgs[1]], dim=2)   # (1, 16, 32)
        bottom = torch.cat([imgs[2], imgs[3]], dim=2)  # (1, 16, 32)
        full = torch.cat([top, bottom], dim=1)        # (1, 32, 32)
        
        return full.view(-1), self.labels[idx]
    
class SlidingConcatDataset_for_patchcore(Dataset):
    def __init__(self, image_paths, labels, transform=None, window_size=4):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths) - self.window_size + 1
    
    def __getitem__(self, idx):
        imgs = []
        for i in range(self.window_size):
            img = Image.open(self.image_paths[idx + i]).convert('RGB')
            if self.transform:
                img = self.transform(img)  # (1, 16, 16)
            imgs.append(img)

        # 4x4 배열로 붙이기 → (1, 64, 64)
        sqrt_w = int(self.window_size ** 0.5)
        rows = []
        for i in range(0, self.window_size, sqrt_w):
            row = torch.cat(imgs[i:i+sqrt_w], dim=2)  # (1, H, W*4)
            rows.append(row)
        full = torch.cat(rows, dim=1)  # (1, H*4, W*4) = (1, 64, 64)

        return full, self.labels[idx]

DATA_TYPE = 'cic'
PREPROCESSING_TYPE = 'zigzag'

# TRAIN_DATASET = f'./Data/byte_16_{PREPROCESSING_TYPE}/save_front/train'
# TRAIN_DATASET = f'./Data/byte_16_{PREPROCESSING_TYPE}_seq/save_front/train'
TRAIN_DATASET = f'./Data/cic_data/Wednesday-workingHours/{PREPROCESSING_TYPE}_seq/benign_train'

METHOD = '_window4'

BATCH_SIZE = 512 * 4
EPOCHES = 10
MAX_TRAIN = 50000

SIZE = 32

def train_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])
    
        
    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    # print(normal_train.images[:5])

    # if DATA_TYPE == 'cic':
    limited_indices = list(range(min(len(normal_train), MAX_TRAIN)))
    normal_train = Subset(normal_train, limited_indices)
    print(f"Training on {len(normal_train)} samples.")

    # for cic dataset

    if SIZE == 32:
        # 원래 Dataset 객체에서 경로와 라벨 추출
        original_dataset = normal_train.dataset if isinstance(normal_train, Subset) else normal_train
        selected_indices = normal_train.indices if isinstance(normal_train, Subset) else list(range(len(normal_train)))

        image_paths = [original_dataset.images[i] for i in selected_indices]
        labels = [original_dataset.labels[i] for i in selected_indices]

        normal_train = SlidingConcatDataset(image_paths, labels, transform=transform)
            

    train_loader = DataLoader(normal_train, batch_size=BATCH_SIZE*4, shuffle=True)

    # CAE
    # cae_train.train_model(device, train_loader, epoches=EPOCHES, 
    #                       model_dir='./AI/Model/CAE/Model', 
    #                       model_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_autoencoder{METHOD}.pth',
    #                       threshold_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy')
    
    # # CAE-Attention
    # cae_attention_train.train_model(device, train_loader, epoches=EPOCHES,
    #                     model_dir='./AI/Model/CAE_Attention/Model',
    #                     model_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_attention_cae{METHOD}.pth',
    #                     threshold_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_threshold_attention_cae{METHOD}.npy')
    
    
    # # Deep-SVDD
    # svdd_train.train_model(device, train_loader, epoches=EPOCHES,
    #                     model_dir='./AI/Model/Deep_SVDD/Model',
    #                     model_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_deep_svdd{METHOD}.pth',
    #                     center_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_center{METHOD}.npy',
    #                     threshold_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy')
    
    # # f-AnoGAN
    # f_anogan_train.train_model(device, train_loader, epoches=EPOCHES,
    #                     model_dir='./AI/Model/f_AnoGAN/Model',
    #                     gen_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_generator{METHOD}.pth',
    #                     disc_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_discriminator{METHOD}.pth',
    #                     encoder_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_encoder{METHOD}.pth',
    #                     threshold_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy')
    
    # # OCSVM
    ocs_train.train_model(device, train_loader, epoches=EPOCHES,
                        model_dir='./AI/Model/OCSVM/Model',
                        cnn_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_ocsvm_cnn{METHOD}.pth',
                        ocsvm_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_ocsvm{METHOD}.pkl')

    # CNN-BiLSTM-AE
    # transform = transforms.Compose([
    #     transforms.Resize((16, 16)),
    #     transforms.Grayscale(),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.view(-1))  # ⬅️ flatten: (1,16,16) → (256,)
    # ])

    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Grayscale(),
        transforms.ToTensor(),  # 결과: (1, 16, 16)
    ])

    normal_train = cnn_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    if SIZE == 32:
        image_paths = normal_train.images
        labels = normal_train.labels

        normal_train = SlidingConcatDataset_for_cnnbilsmae(image_paths, labels, transform=transform)

    limited_indices = list(range(min(len(normal_train), MAX_TRAIN)))
    normal_train = torch.utils.data.Subset(normal_train, limited_indices)
    print(f"Training on {len(normal_train)} samples.")

    train_loader = DataLoader(normal_train, batch_size=1024, shuffle=False)

    cnn_bilstm_ae_train.train_model(device, train_loader, epoches=EPOCHES,
                        model_dir='./AI/Model/CNN_BiLSTM_AE/Model',
                        model_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_cnn_bilstm_ae{METHOD}.pth',
                        threshold_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_threshold{METHOD}.npy')

    # PathCore
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    normal_train = patchcore_train.PacketImageDataset(TRAIN_DATASET, transform, is_flat_structure=True, label=0)

    if DATA_TYPE == 'cic':
        limited_indices = list(range(min(len(normal_train), MAX_TRAIN)))
        normal_train = Subset(normal_train, limited_indices)
        print(f"Training on {len(normal_train)} samples.")

    if SIZE == 32:
        original_dataset = normal_train.dataset if isinstance(normal_train, Subset) else normal_train
        selected_indices = normal_train.indices if isinstance(normal_train, Subset) else list(range(len(normal_train)))

        image_paths = [original_dataset.images[i] for i in selected_indices]
        labels = [original_dataset.labels[i] for i in selected_indices]

        normal_train = SlidingConcatDataset_for_patchcore(image_paths, labels, transform=transform)

    train_loader = DataLoader(normal_train, batch_size=512, shuffle=False)


    patchcore_train.train_model(device, train_loader, epoches=EPOCHES,
                                model_dir='./AI/Model/PatchCore/Model',
                                embedding_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_feature_bank{METHOD}.npy',
                                model_path=f'{DATA_TYPE}_{PREPROCESSING_TYPE}_nn_model{METHOD}.pkl')


if __name__ == "__main__":
    train_models()