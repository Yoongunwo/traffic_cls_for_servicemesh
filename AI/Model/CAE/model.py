import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
from collections import Counter

class PacketImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_flat_structure=True):
        self.transform = transform
        self.images = []
        self.labels = []
        
        if is_flat_structure:  # 이미지가 바로 있는 경우 (normal)
            for file in os.listdir(root_dir):
                if file.endswith('.png'):
                    self.images.append(os.path.join(root_dir, file))
                    self.labels.append(0)  # normal은 0
        else:  # 하위 폴더가 있는 경우 (attack)
            for subdir, _, files in os.walk(root_dir):
                folder_name = os.path.basename(subdir)
                if files:  # 파일이 있는 폴더만 처리
                    for file in files:
                        if file.endswith('.png'):
                            self.images.append(os.path.join(subdir, file))
                            self.labels.append(1)  # attack은 1
        
        print(f"Found {len(self.images)} images in {root_dir}")
        if len(self.images) > 0:
            print(f"Sample path: {self.images[0]}")
        print(f"Label distribution: {dict(Counter(self.labels))}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 32x32 -> 16x16
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)               # 16x16 -> 8x8
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x