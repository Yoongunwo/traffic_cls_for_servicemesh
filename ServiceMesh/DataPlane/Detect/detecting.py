import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import Counter

from Model import model as cnn_model

def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = cnn_model.SimplePacketCNN().to(device)
    state_dict = torch.load(
        './Detect/packet_classifier.pth',
        map_location=device,  # 또는 map_location='cpu'
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model.eval()
    return model
